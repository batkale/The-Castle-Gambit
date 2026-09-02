"""Hand-crafted evaluation, tapered between a midgame and an endgame score.

Every term is carried as a (midgame, endgame) pair and blended at the end by a phase count
that falls from 24 to 0 as pieces come off. That is what lets one function say both "castle
and keep the king behind pawns" and "walk the king to the centre", which are the same
position evaluated at different times.

The weights here are starting values, chosen by hand to be sane rather than optimal. They
are laid out so tools/tune.py can fit them to real game results later; nothing else in the
engine reads them, so retuning is a data change, not a code change.

Scores are returned from the side to move's point of view, the negamax convention.
"""

import os
from pathlib import Path

import numpy as np

from cg_core import (
    BISHOP,
    CACHE,
    FILE_BB,
    KING,
    KING_ATTACKS,
    KNIGHT,
    KNIGHT_ATTACKS,
    ONE,
    PAWN,
    QUEEN,
    ROOK,
    S_OCC,
    S_OCC_W,
    S_SIDE,
    WHITE,
    ZERO,
    U,
    bishop_attacks,
    king_square,
    lsb,
    njit,
    popcount,
    queen_attacks,
    rook_attacks,
)

# ---------------------------------------------------------------- material and phase
PIECE_MG = np.array([100, 325, 340, 510, 985, 0], dtype=np.int32)
PIECE_EG = np.array([120, 305, 330, 550, 1000, 0], dtype=np.int32)
PHASE_WEIGHT = np.array([0, 1, 1, 2, 4, 0], dtype=np.int32)
TOTAL_PHASE = 24

# what a piece is worth when deciding whether a capture sequence is worth entering
SEE_VALUE = np.array([100, 325, 340, 510, 985, 20000], dtype=np.int32)


def _table(rows: str) -> np.ndarray:
    """Read a table written the way a board is drawn, rank 8 at the top, into a1-first order."""
    values = [int(v) for v in rows.split()]
    assert len(values) == 64, f"expected 64 entries, got {len(values)}"
    board = np.zeros(64, dtype=np.int32)
    for index, value in enumerate(values):
        board[(7 - index // 8) * 8 + index % 8] = value
    return board


# fmt: off
PAWN_MG = _table("""
      0   0   0   0   0   0   0   0
     45  50  50  50  50  50  50  45
     10  12  22  32  32  22  12  10
      4   6  12  26  26  10   6   4
      0   0   4  20  20   0   0   0
      4  -4  -8   4   4 -10  -4   4
      6  10  10 -18 -18  12  10   6
      0   0   0   0   0   0   0   0
""")
PAWN_EG = _table("""
      0   0   0   0   0   0   0   0
     90  90  88  85  85  88  90  90
     50  52  48  42  42  48  52  50
     22  22  18  16  16  18  22  22
     10  10   8   8   8   8  10  10
      4   4   4   6   6   4   4   4
      2   2   2   2   2   2   2   2
      0   0   0   0   0   0   0   0
""")
KNIGHT_MG = _table("""
    -55 -40 -30 -28 -28 -30 -40 -55
    -38 -18   0   4   4   0 -18 -38
    -28   6  16  20  20  16   6 -28
    -28   4  20  26  26  20   4 -28
    -28   0  18  24  24  18   0 -28
    -30   8  16  18  18  16   8 -30
    -38 -18   0   8   8   0 -18 -38
    -55 -38 -30 -28 -28 -30 -38 -55
""")
KNIGHT_EG = _table("""
    -50 -38 -24 -20 -20 -24 -38 -50
    -38 -18  -4   0   0  -4 -18 -38
    -24  -4   8  14  14   8  -4 -24
    -20   0  14  20  20  14   0 -20
    -20   0  14  20  20  14   0 -20
    -24  -4   8  14  14   8  -4 -24
    -38 -18  -4   0   0  -4 -18 -38
    -50 -38 -24 -20 -20 -24 -38 -50
""")
BISHOP_MG = _table("""
    -18  -9  -11  -8  -8 -11  -9 -18
     -8   4    0   0   0   0   4  -8
     -5   6    8   8   8   8   6  -5
     -5   2   10  14  14  10   2  -5
     -5   4   10  14  14  10   4  -5
     -5  10    9   8   8   9  10  -5
     -5  14    4   2   2   4  14  -5
    -18  -8   -8  -9  -9  -8  -8 -18
""")
BISHOP_EG = _table("""
    -18  -9  -8  -5  -5  -8  -9 -18
     -9   0   2   4   4   2   0  -9
     -8   2   8  10  10   8   2  -8
     -5   4  10  14  14  10   4  -5
     -5   4  10  14  14  10   4  -5
     -8   2   8  10  10   8   2  -8
     -9   0   2   4   4   2   0  -9
    -18  -9  -8  -5  -5  -8  -9 -18
""")
ROOK_MG = _table("""
      0   0   2   4   4   2   0   0
      6  10  10  10  10  10  10   6
     -4   0   0   0   0   0   0  -4
     -4   0   0   0   0   0   0  -4
     -4   0   0   0   0   0   0  -4
     -4   0   0   0   0   0   0  -4
     -4   0   0   0   0   0   0  -4
     -2   0   2   6   6   4   0  -2
""")
ROOK_EG = _table("""
     10  10  10  10  10  10  10  10
     12  14  14  14  14  14  14  12
      6   6   6   6   6   6   6   6
      2   2   2   2   2   2   2   2
      0   0   0   0   0   0   0   0
     -2  -2  -2  -2  -2  -2  -2  -2
     -4  -4  -4  -4  -4  -4  -4  -4
      0   0   0   0   0   0   0   0
""")
QUEEN_MG = _table("""
    -18 -10  -8  -4  -4  -8 -10 -18
    -10   0   4   0   0   4   0 -10
     -8   4   4   4   4   4   4  -8
     -4   0   4   4   4   4   0  -4
     -2   0   4   4   4   4   0  -4
     -8   6   6   6   6   6   4  -8
    -10   0   6   0   0   0   0 -10
    -18 -10  -8  -4  -4  -8 -10 -18
""")
QUEEN_EG = _table("""
    -30 -18 -14  -8  -8 -14 -18 -30
    -18  -4   0   4   4   0  -4 -18
    -14   0   8  10  10   8   0 -14
     -8   4  10  16  16  10   4  -8
     -8   4  10  16  16  10   4  -8
    -14   0   8  10  10   8   0 -14
    -18  -4   0   4   4   0  -4 -18
    -30 -18 -14  -8  -8 -14 -18 -30
""")
KING_MG = _table("""
    -40 -50 -50 -60 -60 -50 -50 -40
    -40 -50 -50 -60 -60 -50 -50 -40
    -40 -50 -50 -60 -60 -50 -50 -40
    -40 -50 -50 -60 -60 -50 -50 -40
    -28 -38 -40 -50 -50 -40 -38 -28
    -18 -22 -25 -28 -28 -25 -22 -18
     16  16  -6 -12 -12  -6  16  16
     18  30   8  -8   0  -8  32  18
""")
KING_EG = _table("""
    -50 -32 -20 -10 -10 -20 -32 -50
    -26 -10   6  14  14   6 -10 -26
    -16   8  22  28  28  22   8 -16
    -12  12  28  34  34  28  12 -12
    -12  12  28  34  34  28  12 -12
    -16   8  22  28  28  22   8 -16
    -24  -8   6  14  14   6  -8 -24
    -50 -30 -20 -14 -14 -20 -30 -50
""")
# fmt: on

PSQT_MG = np.stack([PAWN_MG, KNIGHT_MG, BISHOP_MG, ROOK_MG, QUEEN_MG, KING_MG])
PSQT_EG = np.stack([PAWN_EG, KNIGHT_EG, BISHOP_EG, ROOK_EG, QUEEN_EG, KING_EG])

# ---------------------------------------------------------------- positional weights
MOBILITY_MG = np.array([0, 4, 5, 3, 2, 0], dtype=np.int32)
MOBILITY_EG = np.array([0, 4, 5, 6, 5, 0], dtype=np.int32)

PASSED_MG = np.array([0, 4, 8, 18, 34, 60, 96, 0], dtype=np.int32)
PASSED_EG = np.array([0, 10, 20, 38, 66, 108, 164, 0], dtype=np.int32)

DOUBLED_MG, DOUBLED_EG = -10, -24
ISOLATED_MG, ISOLATED_EG = -14, -16
BACKWARD_MG, BACKWARD_EG = -8, -10
BISHOP_PAIR_MG, BISHOP_PAIR_EG = 28, 48
ROOK_OPEN_MG, ROOK_OPEN_EG = 26, 12
ROOK_SEMI_MG, ROOK_SEMI_EG = 12, 6
TEMPO = 12

# King danger grows faster than linearly in the number of attackers: two pieces aimed at a
# king are worth much more than twice one. Indexed by the weighted attacker count.
KING_DANGER = np.array(
    [0, 0, 8, 22, 44, 74, 112, 152, 194, 236, 274, 306, 332, 352, 366, 376, 382, 386, 388, 390],
    dtype=np.int32,
)
KING_ATTACK_WEIGHT = np.array([0, 2, 2, 3, 5, 0], dtype=np.int32)

# ---------------------------------------------------------------- derived masks
PASSED_MASK = np.zeros((2, 64), dtype=U)
FRONT_SPAN = np.zeros((2, 64), dtype=U)
ISOLATED_MASK = np.zeros(8, dtype=U)
KING_ZONE = np.zeros((2, 64), dtype=U)


def _build_masks() -> None:
    for file in range(8):
        adjacent = ZERO
        if file > 0:
            adjacent |= FILE_BB[file - 1]
        if file < 7:
            adjacent |= FILE_BB[file + 1]
        ISOLATED_MASK[file] = adjacent

    for square in range(64):
        rank, file = divmod(square, 8)
        ahead_white = ZERO
        ahead_black = ZERO
        for r in range(rank + 1, 8):
            ahead_white |= U(0xFF) << U(r * 8)
        for r in range(0, rank):
            ahead_black |= U(0xFF) << U(r * 8)
        own_and_neighbours = FILE_BB[file] | ISOLATED_MASK[file]
        PASSED_MASK[0][square] = ahead_white & own_and_neighbours
        PASSED_MASK[1][square] = ahead_black & own_and_neighbours
        FRONT_SPAN[0][square] = ahead_white & FILE_BB[file]
        FRONT_SPAN[1][square] = ahead_black & FILE_BB[file]

        # the squares an attack on the king is measured over: the ring, plus one rank
        # further out in front, where an attack usually arrives from
        zone = KING_ATTACKS[square] | (ONE << U(square))
        KING_ZONE[0][square] = zone | (zone << U(8))
        KING_ZONE[1][square] = zone | (zone >> U(8))


_build_masks()


# ---------------------------------------------------------------- tuned weights
# tools/tune.py fits these to game results and writes weights/eval.npz. Loading has to happen
# here, at import: numba folds these tables into the compiled code as constants the first
# time a jitted function runs, so anything assigned after that would be ignored silently.


def _load_tuned() -> str:
    """Overwrite the hand-written tables from weights/eval.npz, if it is there."""
    global DOUBLED_MG, DOUBLED_EG, ISOLATED_MG, ISOLATED_EG
    global BISHOP_PAIR_MG, BISHOP_PAIR_EG, ROOK_OPEN_MG, ROOK_OPEN_EG
    global ROOK_SEMI_MG, ROOK_SEMI_EG

    override = os.environ.get("CG_WEIGHTS")
    if override in ("none", ""):
        # an explicit opt-out, so an A/B test can pit the hand-written weights against a
        # tuned file that is sitting in its default location
        return "hand-written (forced)"
    path = Path(override) if override else Path(__file__).resolve().parent / "weights" / "eval.npz"
    if not path.exists():
        return "hand-written"
    data = np.load(path)
    for table, key in (
        (PIECE_MG, "piece_mg"), (PIECE_EG, "piece_eg"),
        (PSQT_MG, "psqt_mg"), (PSQT_EG, "psqt_eg"),
        (MOBILITY_MG, "mobility_mg"), (MOBILITY_EG, "mobility_eg"),
        (PASSED_MG, "passed_mg"), (PASSED_EG, "passed_eg"),
        (KING_DANGER, "king_danger"),
    ):
        table[:] = data[key]
    (DOUBLED_MG, ISOLATED_MG, BISHOP_PAIR_MG, ROOK_OPEN_MG, ROOK_SEMI_MG) = (
        int(v) for v in data["scalars_mg"]
    )
    (DOUBLED_EG, ISOLATED_EG, BISHOP_PAIR_EG, ROOK_OPEN_EG, ROOK_SEMI_EG) = (
        int(v) for v in data["scalars_eg"]
    )
    return str(path)


WEIGHT_SOURCE = _load_tuned()


@njit(cache=CACHE)
def evaluate(state, ply):
    """Static score in centipawns, from the point of view of the side to move."""
    midgame = 0
    endgame = 0
    phase = 0
    occupancy = state[ply, S_OCC]

    # kept as scalars rather than a two-element array: allocating anything per leaf costs
    # more than the whole rest of this function
    white_king = king_square(state, ply, 0)
    black_king = king_square(state, ply, 1)

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
                # black reads the same tables through a vertically mirrored square
                relative = square if colour == WHITE else square ^ 56
                phase += PHASE_WEIGHT[piece_type]
                midgame += sign * (PIECE_MG[piece_type] + PSQT_MG[piece_type, relative])
                endgame += sign * (PIECE_EG[piece_type] + PSQT_EG[piece_type, relative])

                if piece_type == PAWN:
                    file = square & 7
                    if (PASSED_MASK[colour, square] & enemy_pawns) == ZERO:
                        rank = relative >> 3
                        midgame += sign * PASSED_MG[rank]
                        endgame += sign * PASSED_EG[rank]
                    if (ISOLATED_MASK[file] & own_pawns) == ZERO:
                        midgame += sign * ISOLATED_MG
                        endgame += sign * ISOLATED_EG
                    if (FRONT_SPAN[colour, square] & own_pawns) != ZERO:
                        midgame += sign * DOUBLED_MG
                        endgame += sign * DOUBLED_EG
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

                moves = popcount(attacks & ~own)
                midgame += sign * MOBILITY_MG[piece_type] * moves
                endgame += sign * MOBILITY_EG[piece_type] * moves

                hits = popcount(attacks & zone)
                if hits > 0:
                    king_pressure += KING_ATTACK_WEIGHT[piece_type] * hits

                if piece_type == ROOK:
                    file_mask = FILE_BB[square & 7]
                    if (file_mask & own_pawns) == ZERO:
                        if (file_mask & enemy_pawns) == ZERO:
                            midgame += sign * ROOK_OPEN_MG
                            endgame += sign * ROOK_OPEN_EG
                        else:
                            midgame += sign * ROOK_SEMI_MG
                            endgame += sign * ROOK_SEMI_EG

        if popcount(state[ply, base + BISHOP]) >= 2:
            midgame += sign * BISHOP_PAIR_MG
            endgame += sign * BISHOP_PAIR_EG

        if king_pressure > 19:
            king_pressure = 19
        midgame += sign * KING_DANGER[king_pressure]

    if phase > TOTAL_PHASE:
        phase = TOTAL_PHASE
    score = (midgame * phase + endgame * (TOTAL_PHASE - phase)) // TOTAL_PHASE
    if state[ply, S_SIDE] == WHITE:
        return score + TEMPO
    return -score + TEMPO


@njit(cache=CACHE)
def is_pawn_endgame(state, ply):
    """True when neither side has a piece. Null-move pruning is unsafe here, because
    zugzwang is the whole content of such positions."""
    for colour in range(2):
        base = 6 * colour
        if (
            state[ply, base + KNIGHT]
            | state[ply, base + BISHOP]
            | state[ply, base + ROOK]
            | state[ply, base + QUEEN]
        ) != ZERO:
            return False
    return True


@njit(cache=CACHE)
def has_non_pawn_material(state, ply, colour):
    base = 6 * colour
    return (
        state[ply, base + KNIGHT]
        | state[ply, base + BISHOP]
        | state[ply, base + ROOK]
        | state[ply, base + QUEEN]
    ) != ZERO
