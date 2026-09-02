"""Generate labelled positions for tuning, by having the engine play itself.

    python tools/selfplay.py --games 4000 --workers 9

Each game starts from a few random plies, so the openings are varied, and is then played out
by the engine on a fixed node budget. Every position reached is written out with the result
of the game it came from: 1.0 white won, 0.5 drawn, 0.0 black won.

Two filters decide what is worth recording, and both matter for what comes next:

* the position must be quiet -- not in check, and its static evaluation must already equal
  what a quiescence search returns. Tuning fits the *static* evaluation, so a position whose
  score only makes sense after a capture sequence would be fitting the wrong function;
* the opening plies are skipped, because a random move is not evidence about anything.

Positions are appended as `fen;result` lines, one file per worker, so a run can be stopped
and resumed and nothing is lost.

Development only: never enters the submission zip.
"""

import argparse
import os
import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

NODES_PER_MOVE = 6000
PLY_CAP = 240
OPENING_PLIES = 6

# Self-play from level starts teaches almost nothing about what a piece is worth: both sides
# share an evaluation, so material imbalances are rare and the ones that occur are already
# decided. Removing a piece before the game starts manufactures the imbalances the fit needs
# -- a rook against two minors, a queen against three pieces, a pawn down in a good structure.
IMBALANCE_RATE = 0.4


def _handicap(board, rng) -> None:
    """Take one piece off the board, so the game is played at a material imbalance."""
    import chess

    colour = rng.choice([chess.WHITE, chess.BLACK])
    candidates = [
        square
        for square, piece in board.piece_map().items()
        if piece.color == colour and piece.piece_type != chess.KING
    ]
    if candidates:
        board.remove_piece_at(rng.choice(candidates))
        board.clear_stack()


def play_games(count: int, seed: int, out: Path, report_every: int = 25) -> None:
    import chess

    from cg_core import load_fen, move_to_uci, new_state
    from cg_eval import evaluate
    from cg_search import INFINITY, new_workspace, quiesce, search

    rng = random.Random(seed)
    state, work = new_state(), new_workspace()
    written = games = 0
    started = time.perf_counter()

    with out.open("a", encoding="utf-8") as handle:
        for _ in range(count):
            board = chess.Board()
            for _ in range(OPENING_PLIES):  # a random, varied start
                moves = list(board.legal_moves)
                if not moves:
                    break
                board.push(rng.choice(moves))
            if rng.random() < IMBALANCE_RATE:
                _handicap(board, rng)
            # removing a piece can open a line onto a king, which would leave the side that
            # is not to move in check -- an illegal position python-chess would still play
            if not board.is_valid() or board.outcome(claim_draw=True) is not None:
                continue

            pending: list[str] = []
            while board.outcome(claim_draw=True) is None and len(board.move_stack) < PLY_CAP:
                fen = board.fen()
                load_fen(state, 0, fen)

                # quiet means the static score already survives a capture search
                far = time.perf_counter() + 60.0
                if not board.is_check():
                    static = evaluate(state, 0)
                    if quiesce(work, state, 0, -INFINITY, INFINITY, far) == static:
                        pending.append(fen)

                work.tt[:] = 0  # each move decided on its own, so positions stay independent
                move, _, _, _, _ = search(work, state, 0, 63, far, far, NODES_PER_MOVE)
                if move == 0:
                    break
                board.push(chess.Move.from_uci(move_to_uci(int(move))))

            outcome = board.outcome(claim_draw=True)
            if outcome is None or outcome.winner is None:
                label = 0.5
            else:
                label = 1.0 if outcome.winner == chess.WHITE else 0.0

            for fen in pending:
                handle.write(f"{fen};{label}\n")
            written += len(pending)
            games += 1
            if games % report_every == 0:
                handle.flush()
                rate = games / (time.perf_counter() - started)
                print(
                    f"  worker {seed}: {games} games, {written:,} positions, {rate:.2f} games/s",
                    flush=True,
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-play position generator.")
    parser.add_argument("--games", type=int, default=4000, help="total across all workers")
    parser.add_argument("--workers", type=int, default=9)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--worker", type=int, default=-1, help="internal: run as one worker")
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    arguments.out.mkdir(parents=True, exist_ok=True)

    if arguments.worker >= 0:
        play_games(arguments.games, arguments.seed, arguments.out / f"sp{arguments.worker}.txt")
        return

    share = max(1, arguments.games // arguments.workers)
    print(f"{arguments.workers} workers x {share} games (numba compiles once per worker)")
    children = []
    for index in range(arguments.workers):
        children.append(
            subprocess.Popen(
                [
                    sys.executable, "-W", "ignore", __file__,
                    "--worker", str(index),
                    "--games", str(share),
                    "--seed", str(arguments.seed * 1000 + index),
                    "--out", str(arguments.out),
                ],
                env={**os.environ, "PYTHONPATH": os.getcwd()},
            )
        )
    started = time.perf_counter()
    for child in children:
        child.wait()

    total = 0
    for path in sorted(arguments.out.glob("sp*.txt")):
        with path.open(encoding="utf-8") as handle:
            total += sum(1 for _ in handle)
    print(f"\n{total:,} positions in {arguments.out} after {time.perf_counter() - started:.0f}s")


if __name__ == "__main__":
    main()
