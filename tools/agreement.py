"""Measure the engine against strong human play, move by move.

    python tools/agreement.py world-championship.pgn --nodes 50000 --workers 8

For every position in the archive, the engine searches and its choice is compared with the
move actually played. Two numbers come out:

* **agreement** -- how often the engine picks the same move. Not a target to maximise: a
  strong engine disagrees with humans regularly and is often right. It is a tracking signal,
  and a fall after a change is worth explaining;
* **regret** -- when the engine disagrees, how much worse it thinks the played move was, in
  centipawns. This is the more useful of the two, because it separates "a different move of
  equal value" from "the engine thinks Kasparov blundered a rook".

Both are broken down by game phase. A version that agrees in the middlegame but not the
endgame has an endgame evaluation problem, which SPRT alone would never localise: SPRT says
which of two versions is better, never what either is bad at.

Large regrets are printed at the end. Against world champions those are the engine's bugs far
more often than theirs, which makes them the most valuable output here.

Development only: never enters the submission zip.
"""

import argparse
import bz2
import gzip
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import chess
import chess.pgn

sys.path.insert(0, ".")

from cg_search import MATE_BOUND

SKIP_OPENING_PLIES = 12  # book, not judgement
SKIP_FINAL_PLIES = 6  # a resigned position tells us nothing
PHASE_WEIGHT = {chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4}


def open_maybe_compressed(path: Path):
    if path.suffix == ".bz2":
        return bz2.open(path, "rt", encoding="utf-8", errors="replace")
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("rt", encoding="utf-8", errors="replace")


def phase_of(board: chess.Board) -> str:
    total = sum(
        weight * len(board.pieces(piece, colour))
        for piece, weight in PHASE_WEIGHT.items()
        for colour in (chess.WHITE, chess.BLACK)
    )
    if total >= 18:
        return "opening"
    if total >= 8:
        return "middlegame"
    return "endgame"


def run_worker(arguments) -> None:
    from cg_core import load_fen, move_to_uci, new_state
    from cg_search import new_workspace, search

    state, work = new_state(), new_workspace()
    results = []

    with open_maybe_compressed(arguments.pgn) as source:
        index = -1
        while True:
            game = chess.pgn.read_game(source)
            if game is None:
                break
            index += 1
            if index % arguments.workers != arguments.worker:
                continue
            if arguments.max_games and index >= arguments.max_games:
                break

            moves = list(game.mainline_moves())
            board = game.board()
            for ply, played in enumerate(moves):
                if SKIP_OPENING_PLIES <= ply < len(moves) - SKIP_FINAL_PLIES:
                    far = time.perf_counter() + 120.0
                    work.tt[:] = 0
                    load_fen(state, 0, board.fen())
                    best, best_score, depth, _, _ = search(
                        work, state, 0, 63, far, far, arguments.nodes
                    )
                    if best:
                        chosen = move_to_uci(int(best))
                        regret = 0
                        reply_score = 0
                        if chosen != played.uci():
                            # score the human move by searching the position it leads to
                            after = board.copy()
                            after.push(played)
                            work.tt[:] = 0
                            load_fen(state, 0, after.fen())
                            _, reply_score, _, _, _ = search(
                                work, state, 0, 63, far, far, arguments.nodes
                            )
                            regret = int(best_score) + int(reply_score)
                        # a mate score is not a centipawn quantity: "mate in 6 rather than
                        # a draw" would otherwise land as 29,000cp and swamp every mean.
                        # Counted separately instead.
                        mate = abs(int(best_score)) > MATE_BOUND or (
                            chosen != played.uci() and abs(int(reply_score)) > MATE_BOUND
                        )
                        results.append(
                            {
                                "phase": phase_of(board),
                                "agree": chosen == played.uci(),
                                "mate": bool(mate),
                                "regret": max(0, regret),
                                "depth": int(depth),
                                "played": played.uci(),
                                "engine": chosen,
                                "fen": board.fen(),
                                "white": game.headers.get("White", "?"),
                                "black": game.headers.get("Black", "?"),
                                "event": game.headers.get("Event", "?"),
                                "ply": ply,
                            }
                        )
                board.push(played)

    out = arguments.out / f"agree{arguments.worker}.jsonl"
    with out.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row) + "\n")


def summarise(paths: list[Path], show: int) -> None:
    rows = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle)
    if not rows:
        raise SystemExit("no positions scored")

    mates = [r for r in rows if r.get("mate") or r["regret"] > 20000]
    scored = [r for r in rows if not (r.get("mate") or r["regret"] > 20000)]
    print(f"\n{len(rows):,} positions from strong human play")
    print(f"{len(mates):,} turn on a forced mate, which is not a centipawn quantity, and are")
    print("counted apart rather than allowed to swamp the averages\n")

    print(f"{'phase':<12}{'positions':>10}{'agreement':>11}{'mean':>8}{'median':>8}{'>100cp':>8}")
    for phase in ("opening", "middlegame", "endgame", "all"):
        subset = scored if phase == "all" else [r for r in scored if r["phase"] == phase]
        if not subset:
            continue
        regrets = sorted(r["regret"] for r in subset)
        agree = sum(r["agree"] for r in subset) / len(subset)
        mean = sum(regrets) / len(subset)
        # the median is the honest middle: a handful of tactical moments drag the mean
        median = regrets[len(regrets) // 2]
        over = sum(r > 100 for r in regrets) / len(subset)
        print(f"{phase:<12}{len(subset):>10,}{agree:>10.1%}{mean:>8.1f}{median:>8.0f}{over:>7.1%}")

    worst = sorted(scored, key=lambda r: -r["regret"])[:show]
    print(f"\nthe {show} positions where the engine most disagrees:")
    print("(deep-search these before believing either side; a disagreement that grows with")
    print("depth is a real finding, one that shrinks was the engine being shallow)\n")
    for row in worst:
        print(f"  {row['regret']:>5}cp  played {row['played']}, engine wants {row['engine']}")
        print(f"         {row['white']} - {row['black']}, {row['event']}, ply {row['ply']}")
        print(f"         {row['fen']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Move agreement against a PGN archive.")
    parser.add_argument("pgn", type=Path)
    parser.add_argument("--nodes", type=int, default=50000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--worker", type=int, default=-1, help="internal")
    parser.add_argument("--max-games", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("data/agreement"))
    parser.add_argument("--show", type=int, default=8)
    arguments = parser.parse_args()

    arguments.out.mkdir(parents=True, exist_ok=True)
    if arguments.worker >= 0:
        run_worker(arguments)
        return

    for stale in arguments.out.glob("agree*.jsonl"):
        stale.unlink()

    started = time.perf_counter()
    children = [
        subprocess.Popen(
            [
                sys.executable, "-W", "ignore", __file__, str(arguments.pgn),
                "--worker", str(index),
                "--workers", str(arguments.workers),
                "--nodes", str(arguments.nodes),
                "--max-games", str(arguments.max_games),
                "--out", str(arguments.out),
            ],
            env={**os.environ, "PYTHONPATH": os.getcwd()},
        )
        for index in range(arguments.workers)
    ]
    for child in children:
        child.wait()
    print(f"scored in {time.perf_counter() - started:.0f}s")
    summarise(sorted(arguments.out.glob("agree*.jsonl")), arguments.show)


if __name__ == "__main__":
    main()
