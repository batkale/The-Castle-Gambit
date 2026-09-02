"""Fit the evaluation weights to game results -- Texel tuning, done as regression.

    python tools/tune.py --verify          # prove the feature extractor is exact
    python tools/tune.py --fit             # build the dataset and fit
    python tools/tune.py --report          # what changed, in centipawns

The evaluation is a *linear* function of its weights: every term is a weight multiplied by a
count. So instead of nudging weights one at a time and re-scoring the corpus each time, we
extract a feature vector per position once and fit the whole 838-parameter vector with
gradient descent.

That only works if the features are exactly right, which is what `--verify` proves: for a
random sample of positions, `features . weights` must equal `evaluate()` to the centipawn.
If that holds, the model and the engine are the same function and the fit is meaningful.

The objective is the standard one: push a sigmoid of the score towards the result of the game
the position came from, 1 for a white win, 0.5 for a draw, 0 for a black win.

Development only: never enters the submission zip.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

from cg_core import (
    BISHOP,
    FILE_BB,
    KING,
    KNIGHT,
    KNIGHT_ATTACKS,
    ONE,
    PAWN,
    ROOK,
    S_OCC,
    S_OCC_W,
    S_SIDE,
    WHITE,
    ZERO,
    bishop_attacks,
    king_square,
    load_fen,
    lsb,
    new_state,
    njit,
    popcount,
    queen_attacks,
    rook_attacks,
)
from cg_eval import (
    BISHOP_PAIR_EG,
    BISHOP_PAIR_MG,
    DOUBLED_EG,
    DOUBLED_MG,
    FRONT_SPAN,
    ISOLATED_EG,
    ISOLATED_MASK,
    ISOLATED_MG,
    KING_ATTACK_WEIGHT,
    KING_DANGER,
    KING_ZONE,
    MOBILITY_EG,
    MOBILITY_MG,
    PASSED_EG,
    PASSED_MASK,
    PASSED_MG,
    PHASE_WEIGHT,
    PIECE_EG,
    PIECE_MG,
    PSQT_EG,
    PSQT_MG,
    ROOK_OPEN_EG,
    ROOK_OPEN_MG,
    ROOK_SEMI_EG,
    ROOK_SEMI_MG,
    TEMPO,
    TOTAL_PHASE,
    evaluate,
)

# ---------------------------------------------------------------- weight vector layout
# One block of counts serves both tapers, because the midgame and endgame terms multiply the
# same quantities; only the phase blend differs.
I_PIECE = 0  # 6   material by piece type
I_PSQT = 6  # 384  piece-square, piece_type * 64 + square
I_MOBILITY = 390  # 6
I_PASSED = 396  # 8   by rank
I_DOUBLED = 404
I_ISOLATED = 405
I_PAIR = 406
I_ROOK_OPEN = 407
I_ROOK_SEMI = 408
BLOCK = 409

N_DANGER = len(KING_DANGER)
N_WEIGHTS = BLOCK * 2 + N_DANGER  # midgame block, endgame block, king danger (midgame only)


def pack_weights() -> np.ndarray:
    """The engine's current weights as one vector, in the layout above."""
    w = np.zeros(N_WEIGHTS, dtype=np.float64)
    for block, piece, psqt, mobility, passed, doubled, isolated, pair, rook_open, rook_semi in (
        (0, PIECE_MG, PSQT_MG, MOBILITY_MG, PASSED_MG, DOUBLED_MG, ISOLATED_MG,
         BISHOP_PAIR_MG, ROOK_OPEN_MG, ROOK_SEMI_MG),
        (BLOCK, PIECE_EG, PSQT_EG, MOBILITY_EG, PASSED_EG, DOUBLED_EG, ISOLATED_EG,
         BISHOP_PAIR_EG, ROOK_OPEN_EG, ROOK_SEMI_EG),
    ):
        w[block + I_PIECE : block + I_PIECE + 6] = piece
        w[block + I_PSQT : block + I_PSQT + 384] = psqt.reshape(-1)
        w[block + I_MOBILITY : block + I_MOBILITY + 6] = mobility
        w[block + I_PASSED : block + I_PASSED + 8] = passed
        w[block + I_DOUBLED] = doubled
        w[block + I_ISOLATED] = isolated
        w[block + I_PAIR] = pair
        w[block + I_ROOK_OPEN] = rook_open
        w[block + I_ROOK_SEMI] = rook_semi
    w[2 * BLOCK :] = KING_DANGER
    return w


@njit(cache=False)
def extract(state, ply, counts):
    """Fill `counts` with the feature vector, mirroring cg_eval.evaluate term for term.

    Returns (phase, white danger index, black danger index). Counts are from white's point of
    view: white adds, black subtracts, so the vector describes the position, not the mover.
    """
    for i in range(BLOCK):
        counts[i] = 0.0
    phase = 0
    occupancy = state[ply, S_OCC]
    white_king = king_square(state, ply, 0)
    black_king = king_square(state, ply, 1)
    danger_white = 0
    danger_black = 0

    for colour in range(2):
        sign = 1 if colour == WHITE else -1
        base = 6 * colour
        own = state[ply, S_OCC_W + colour]
        own_pawns = state[ply, base + PAWN]
        enemy_pawns = state[ply, 6 * (1 - colour) + PAWN]
        enemy_king = black_king if colour == WHITE else white_king
        zone = KING_ZONE[1 - colour, enemy_king]
        king_pressure = 0

        for piece_type in range(6):
            pieces = state[ply, base + piece_type]
            while pieces != ZERO:
                square = lsb(pieces)
                pieces &= pieces - ONE
                relative = square if colour == WHITE else square ^ 56
                phase += PHASE_WEIGHT[piece_type]
                counts[I_PIECE + piece_type] += sign
                counts[I_PSQT + piece_type * 64 + relative] += sign

                if piece_type == PAWN:
                    file = square & 7
                    if (PASSED_MASK[colour, square] & enemy_pawns) == ZERO:
                        counts[I_PASSED + (relative >> 3)] += sign
                    if (ISOLATED_MASK[file] & own_pawns) == ZERO:
                        counts[I_ISOLATED] += sign
                    if (FRONT_SPAN[colour, square] & own_pawns) != ZERO:
                        counts[I_DOUBLED] += sign
                    continue

                if piece_type == KING:
                    continue

                if piece_type == KNIGHT:
                    attacks = KNIGHT_ATTACKS[square]
                elif piece_type == BISHOP:
                    attacks = bishop_attacks(square, occupancy)
                elif piece_type == ROOK:
                    attacks = rook_attacks(square, occupancy)
                else:
                    attacks = queen_attacks(square, occupancy)

                counts[I_MOBILITY + piece_type] += sign * popcount(attacks & ~own)

                hits = popcount(attacks & zone)
                if hits > 0:
                    king_pressure += KING_ATTACK_WEIGHT[piece_type] * hits

                if piece_type == ROOK:
                    file_mask = FILE_BB[square & 7]
                    if (file_mask & own_pawns) == ZERO:
                        if (file_mask & enemy_pawns) == ZERO:
                            counts[I_ROOK_OPEN] += sign
                        else:
                            counts[I_ROOK_SEMI] += sign

        if popcount(state[ply, base + BISHOP]) >= 2:
            counts[I_PAIR] += sign

        if king_pressure > N_DANGER - 1:
            king_pressure = N_DANGER - 1
        if colour == WHITE:
            danger_white = king_pressure
        else:
            danger_black = king_pressure

    if phase > TOTAL_PHASE:
        phase = TOTAL_PHASE
    return phase, danger_white, danger_black


def score_from_features(counts, phase, danger_white, danger_black, w) -> int:
    """The same score the engine computes, rebuilt from features. White's point of view."""
    midgame = float(counts @ w[:BLOCK]) + w[2 * BLOCK + danger_white] - w[2 * BLOCK + danger_black]
    endgame = float(counts @ w[BLOCK : 2 * BLOCK])
    return int(
        (round(midgame) * phase + round(endgame) * (TOTAL_PHASE - phase)) // TOTAL_PHASE
    )


def verify(sample: int, paths: list[Path]) -> int:
    """features . weights must equal evaluate(), exactly, or the fit means nothing."""
    w = pack_weights()
    state = new_state()
    counts = np.zeros(BLOCK, dtype=np.float64)
    lines = list(read_positions(paths, limit=sample))
    if not lines:
        raise SystemExit("no positions found; run tools/selfplay.py first")

    bad = 0
    for fen, _ in lines:
        load_fen(state, 0, fen)
        phase, dw, db = extract(state, 0, counts)
        mine = score_from_features(counts, phase, dw, db, w)
        engine = int(evaluate(state, 0))
        theirs = engine - TEMPO if state[0, S_SIDE] == WHITE else TEMPO - engine
        if mine != theirs:
            bad += 1
            if bad <= 5:
                print(f"  MISMATCH {mine} vs {theirs}  {fen}")
    print(f"verified {len(lines):,} positions, {bad} mismatches")
    return bad


def read_positions(paths: list[Path], limit: int = 0):
    seen = 0
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                fen, _, label = line.rstrip("\n").rpartition(";")
                if not fen:
                    continue
                yield fen, float(label)
                seen += 1
                if limit and seen >= limit:
                    return


def build_dataset(paths: list[Path], cache: Path, limit: int = 0) -> dict:
    state = new_state()
    counts = np.zeros(BLOCK, dtype=np.float64)
    indices: list[np.ndarray] = []
    values: list[np.ndarray] = []
    indptr = [0]
    phases, dangers_w, dangers_b, labels = [], [], [], []

    started = time.perf_counter()
    for number, (fen, label) in enumerate(read_positions(paths, limit), 1):
        load_fen(state, 0, fen)
        phase, dw, db = extract(state, 0, counts)
        nz = np.nonzero(counts)[0]
        indices.append(nz.astype(np.int32))
        values.append(counts[nz].astype(np.float32))
        indptr.append(indptr[-1] + nz.size)
        phases.append(phase)
        dangers_w.append(dw)
        dangers_b.append(db)
        labels.append(label)
        if number % 100_000 == 0:
            print(f"  {number:,} positions  ({time.perf_counter() - started:.0f}s)", flush=True)

    data = {
        "indices": np.concatenate(indices),
        "values": np.concatenate(values),
        "indptr": np.array(indptr, dtype=np.int64),
        "phase": np.array(phases, dtype=np.float32),
        "danger_w": np.array(dangers_w, dtype=np.int32),
        "danger_b": np.array(dangers_b, dtype=np.int32),
        "label": np.array(labels, dtype=np.float32),
    }
    np.savez_compressed(cache, **data)
    print(f"cached {len(labels):,} positions to {cache} ({time.perf_counter() - started:.0f}s)")
    return data


@njit(cache=False)
def scores_and_gradient(indices, values, indptr, phase, danger_w, danger_b, label, w, scale,
                        gradient):
    """Mean squared error against the sigmoid of the score, and its gradient.

    One pass: the score of every position, then the residual pushed back through the same
    sparse rows. Written as a loop rather than a matrix product because the phase blend
    differs per row, so there is no single dense matrix to multiply by.
    """
    n = label.size
    gradient[:] = 0.0
    total = 0.0
    for row in range(n):
        ph = phase[row] / TOTAL_PHASE
        eg_weight = 1.0 - ph
        midgame = 0.0
        endgame = 0.0
        for k in range(indptr[row], indptr[row + 1]):
            i = indices[k]
            v = values[k]
            midgame += v * w[i]
            endgame += v * w[BLOCK + i]
        midgame += w[2 * BLOCK + danger_w[row]] - w[2 * BLOCK + danger_b[row]]
        score = midgame * ph + endgame * eg_weight

        prediction = 1.0 / (1.0 + np.exp(-scale * score))
        error = prediction - label[row]
        total += error * error
        # d/dw of (p - y)^2 through the sigmoid
        common = 2.0 * error * prediction * (1.0 - prediction) * scale
        for k in range(indptr[row], indptr[row + 1]):
            i = indices[k]
            v = values[k]
            gradient[i] += common * v * ph
            gradient[BLOCK + i] += common * v * eg_weight
        gradient[2 * BLOCK + danger_w[row]] += common * ph
        gradient[2 * BLOCK + danger_b[row]] -= common * ph
    for i in range(gradient.size):
        gradient[i] /= n
    return total / n


def find_scale(data, w) -> float:
    """The sigmoid steepness that best explains the results at the current weights."""
    gradient = np.zeros(N_WEIGHTS)
    best, best_scale = None, 1.0 / 400.0
    for candidate in np.linspace(0.5, 3.0, 11) / 400.0:
        error = scores_and_gradient(
            data["indices"], data["values"], data["indptr"], data["phase"],
            data["danger_w"], data["danger_b"], data["label"], w, candidate, gradient,
        )
        if best is None or error < best:
            best, best_scale = error, candidate
    print(f"sigmoid scale {best_scale * 400:.2f}/400, starting error {best:.6f}")
    return best_scale


def fit(data, epochs: int, learning_rate: float, anchor_strength: float) -> np.ndarray:
    start = pack_weights()
    w = start.copy()
    scale = find_scale(data, w)

    # material and piece-square tables overlap: adding a constant to one and subtracting it
    # from the other leaves the score unchanged. An anchor towards the hand-written values
    # keeps the fit from wandering along those directions where the data says nothing.
    frozen = np.zeros(N_WEIGHTS, dtype=bool)
    frozen[I_PIECE + KING] = True  # one king each: the term cancels and is unidentifiable
    frozen[BLOCK + I_PIECE + KING] = True
    frozen[2 * BLOCK] = True  # king danger at zero pressure is the reference point

    gradient = np.zeros(N_WEIGHTS)
    moment1 = np.zeros(N_WEIGHTS)
    moment2 = np.zeros(N_WEIGHTS)
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8

    print(f"fitting {N_WEIGHTS} weights on {data['label'].size:,} positions")
    for epoch in range(1, epochs + 1):
        error = scores_and_gradient(
            data["indices"], data["values"], data["indptr"], data["phase"],
            data["danger_w"], data["danger_b"], data["label"], w, scale, gradient,
        )
        gradient += anchor_strength * (w - start)
        gradient[frozen] = 0.0

        moment1 = beta1 * moment1 + (1 - beta1) * gradient
        moment2 = beta2 * moment2 + (1 - beta2) * gradient * gradient
        step = learning_rate * (moment1 / (1 - beta1**epoch)) / (
            np.sqrt(moment2 / (1 - beta2**epoch)) + epsilon
        )
        w -= step
        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:>4}  error {error:.6f}  |step| {np.abs(step).max():.3f}")
    return w


def normalise(w: np.ndarray) -> np.ndarray:
    """Move the average of each piece-square table into the material value for that piece.

    The two are interchangeable, so the fit leaves them split arbitrarily. Folding makes the
    numbers readable: a knight is worth what PIECE says, and the table says where it likes
    to stand.
    """
    w = w.copy()
    for block in (0, BLOCK):
        for piece_type in range(6):
            lo = block + I_PSQT + piece_type * 64
            table = w[lo : lo + 64]
            mean = table.mean()
            table -= mean
            w[block + I_PIECE + piece_type] += mean
    w[2 * BLOCK : 2 * BLOCK + N_DANGER] -= w[2 * BLOCK]
    return w


def write_weights(w: np.ndarray, destination: Path) -> None:
    w = normalise(w)
    rounded = np.rint(w).astype(np.int32)
    blocks = {}
    for name, block in (("mg", 0), ("eg", BLOCK)):
        blocks[f"piece_{name}"] = rounded[block + I_PIECE : block + I_PIECE + 6]
        blocks[f"psqt_{name}"] = rounded[block + I_PSQT : block + I_PSQT + 384].reshape(6, 64)
        blocks[f"mobility_{name}"] = rounded[block + I_MOBILITY : block + I_MOBILITY + 6]
        blocks[f"passed_{name}"] = rounded[block + I_PASSED : block + I_PASSED + 8]
        blocks[f"scalars_{name}"] = np.array(
            [
                rounded[block + I_DOUBLED],
                rounded[block + I_ISOLATED],
                rounded[block + I_PAIR],
                rounded[block + I_ROOK_OPEN],
                rounded[block + I_ROOK_SEMI],
            ],
            dtype=np.int32,
        )
    blocks["king_danger"] = rounded[2 * BLOCK :]
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destination, **blocks)
    print(f"\nwrote {destination}")


def report(w: np.ndarray) -> None:
    start = normalise(pack_weights())
    tuned = normalise(w)
    names = ["pawn", "knight", "bishop", "rook", "queen", "king"]
    print("\nmaterial, midgame -> endgame")
    for i, name in enumerate(names[:5]):
        print(
            f"  {name:<7} {start[I_PIECE + i]:>7.0f} -> {tuned[I_PIECE + i]:>7.0f}"
            f"   |   {start[BLOCK + I_PIECE + i]:>7.0f} -> {tuned[BLOCK + I_PIECE + i]:>7.0f}"
        )
    labels = [
        ("doubled pawn", I_DOUBLED), ("isolated pawn", I_ISOLATED),
        ("bishop pair", I_PAIR), ("rook, open file", I_ROOK_OPEN),
        ("rook, half-open", I_ROOK_SEMI),
    ]
    print("\nterms, midgame -> endgame")
    for name, index in labels:
        print(
            f"  {name:<17} {start[index]:>6.0f} -> {tuned[index]:>6.0f}"
            f"   |   {start[BLOCK + index]:>6.0f} -> {tuned[BLOCK + index]:>6.0f}"
        )
    print("\npassed pawn by rank, endgame")
    print("  before " + " ".join(f"{start[BLOCK + I_PASSED + r]:>5.0f}" for r in range(8)))
    print("  after  " + " ".join(f"{tuned[BLOCK + I_PASSED + r]:>5.0f}" for r in range(8)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit evaluation weights to game results.")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--cache", type=Path, default=Path("data/features.npz"))
    parser.add_argument("--out", type=Path, default=Path("weights/eval.npz"))
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.6)
    parser.add_argument("--anchor", type=float, default=2e-7)
    parser.add_argument("--rebuild", action="store_true")
    arguments = parser.parse_args()

    paths = sorted(arguments.data.glob("sp*.txt"))
    if arguments.verify or not arguments.fit:
        if verify(2000, paths) != 0:
            raise SystemExit("feature extraction does not match the engine; not fitting")
        if not arguments.fit:
            return

    if arguments.cache.exists() and not arguments.rebuild:
        print(f"loading {arguments.cache}")
        data = dict(np.load(arguments.cache))
    else:
        data = build_dataset(paths, arguments.cache, arguments.limit)

    w = fit(data, arguments.epochs, arguments.learning_rate, arguments.anchor)
    report(w)
    write_weights(w, arguments.out)


if __name__ == "__main__":
    main()
