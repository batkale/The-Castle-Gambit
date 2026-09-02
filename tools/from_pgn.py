"""Turn a PGN archive into tuning positions.

    python tools/from_pgn.py games.pgn --out data --min-rating 2000

Self-play is a weak teacher for material values: both sides share an evaluation, the openings
are level, and material imbalances are rare and usually already decided. Real games between
strong players carry the imbalances the fit needs -- a rook for two minors, a queen for three
pieces, a pawn structure held against a bishop pair.

The rules allow this explicitly: the ban covers engines shipped inside the submission, not
what the weights are learned from. A Lichess monthly dump filtered to higher-rated games is
the usual source, and it is plain PGN.

Output matches tools/selfplay.py, `fen;result` per line, so tools/tune.py reads either.

Development only: never enters the submission zip.
"""

import argparse
import bz2
import gzip
import sys
import time
from pathlib import Path

import chess
import chess.pgn

sys.path.insert(0, ".")

RESULTS = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}
SKIP_OPENING_PLIES = 12  # book moves say nothing about evaluation
SKIP_FINAL_PLIES = 4  # the last few moves of a lost game are noise


def open_maybe_compressed(path: Path):
    if path.suffix == ".bz2":
        return bz2.open(path, "rt", encoding="utf-8", errors="replace")
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("rt", encoding="utf-8", errors="replace")


def rating_of(headers, key: str) -> int:
    try:
        return int(headers.get(key, "0"))
    except ValueError:
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract tuning positions from a PGN.")
    parser.add_argument("pgn", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--name", default="pgn0")
    parser.add_argument("--min-rating", type=int, default=2000)
    parser.add_argument("--max-games", type=int, default=0)
    parser.add_argument("--every", type=int, default=2, help="keep one in N quiet positions")
    arguments = parser.parse_args()

    from cg_core import load_fen, new_state
    from cg_eval import evaluate
    from cg_search import INFINITY, new_workspace, quiesce

    arguments.out.mkdir(parents=True, exist_ok=True)
    destination = arguments.out / f"{arguments.name}.txt"
    state, work = new_state(), new_workspace()
    games = kept = written = 0
    started = time.perf_counter()

    with open_maybe_compressed(arguments.pgn) as source, destination.open(
        "w", encoding="utf-8"
    ) as handle:
        while True:
            game = chess.pgn.read_game(source)
            if game is None:
                break
            games += 1
            if arguments.max_games and games > arguments.max_games:
                break

            label = RESULTS.get(game.headers.get("Result", "*"))
            if label is None:
                continue
            if min(
                rating_of(game.headers, "WhiteElo"), rating_of(game.headers, "BlackElo")
            ) < arguments.min_rating:
                continue

            moves = list(game.mainline_moves())
            if len(moves) < SKIP_OPENING_PLIES + SKIP_FINAL_PLIES + 4:
                continue
            kept += 1

            board = game.board()
            for ply, move in enumerate(moves):
                board.push(move)
                if ply < SKIP_OPENING_PLIES or ply >= len(moves) - SKIP_FINAL_PLIES:
                    continue
                if ply % arguments.every or board.is_check():
                    continue
                fen = board.fen()
                load_fen(state, 0, fen)
                far = time.perf_counter() + 60.0
                # the same quietness test tuning assumes: the static score must already
                # survive a capture search, or we would be fitting the wrong function
                if quiesce(work, state, 0, -INFINITY, INFINITY, far) == evaluate(state, 0):
                    handle.write(f"{fen};{label}\n")
                    written += 1

            if games % 2000 == 0:
                handle.flush()
                print(
                    f"  {games:,} games read, {kept:,} used, {written:,} positions "
                    f"({time.perf_counter() - started:.0f}s)",
                    flush=True,
                )

    print(f"\n{written:,} positions from {kept:,} of {games:,} games -> {destination}")


if __name__ == "__main__":
    main()
