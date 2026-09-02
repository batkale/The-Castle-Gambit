"""A/B test two weight sets, with a stopping rule that knows when it has seen enough.

    python tools/match.py --a "" --b weights/eval.npz --games 2000

`--a` and `--b` are paths to weight files; an empty string means the hand-written weights.
Each engine runs in its own process, because numba folds the weight tables into the compiled
code as constants -- one process is one weight set, permanently.

Three things make the result mean something:

* **Paired openings.** Every opening is played twice, once with each engine as white, so a
  lucky opening helps both sides equally and the variance drops.
* **Fixed nodes per move, not fixed time.** The machine is running nine of these at once; a
  clock would measure system load as much as chess.
* **SPRT.** Rather than playing a round number of games and squinting at the score, the test
  accumulates a log-likelihood ratio between "no better" and "better by `--elo1`" and stops
  as soon as one is enough more likely than the other. A change worth 5 Elo usually needs
  thousands of games; this finds out with as few as it can.

Development only: never enters the submission zip.
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

import chess

sys.path.insert(0, ".")

RUNNER = Path(__file__).resolve().parent / "engine_server.py"
PLY_CAP = 300
PIECE_VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


class Engine:
    """One long-lived engine process, pinned to one weight set."""

    def __init__(self, weights: str, nodes: int) -> None:
        environment = {**os.environ, "CG_FIXED_NODES": str(nodes), "PYTHONPATH": os.getcwd()}
        # "none" forces the hand-written weights. Clearing the variable is not enough:
        # cg_eval falls back to weights/eval.npz, so both sides would load the same file and
        # the match would silently be engine against itself.
        environment["CG_WEIGHTS"] = str(Path(weights).resolve()) if weights else "none"
        self.process = subprocess.Popen(
            [sys.executable, "-W", "ignore", str(RUNNER), os.getcwd()],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            env=environment, text=True, bufsize=1,
        )
        if not json.loads(self._readline()).get("ready"):
            raise RuntimeError("engine failed to start")

    def _readline(self) -> str:
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("engine died")
        return line

    def send(self, message: dict) -> dict:
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()
        return json.loads(self._readline())

    def new_game(self) -> None:
        self.send({"newgame": True})

    def move(self, fen: str) -> str:
        return self.send({"fen": fen, "time_left_ms": 120_000})["move"]

    def stop(self) -> None:
        try:
            self.process.stdin.close()
            self.process.wait(timeout=5)
        except Exception:
            self.process.kill()


def adjudicate(board: chess.Board) -> float:
    balance = sum(
        value * (len(board.pieces(piece, chess.WHITE)) - len(board.pieces(piece, chess.BLACK)))
        for piece, value in PIECE_VALUE.items()
    )
    return 1.0 if balance > 0 else 0.0 if balance < 0 else 0.5


def play(white: Engine, black: Engine, opening: str) -> float:
    """One game. Returns white's score."""
    board = chess.Board(opening)
    white.new_game()
    black.new_game()
    while True:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            if outcome.winner is None:
                return 0.5
            return 1.0 if outcome.winner == chess.WHITE else 0.0
        if len(board.move_stack) >= PLY_CAP:
            return adjudicate(board)
        engine = white if board.turn == chess.WHITE else black
        uci = engine.move(board.fen())
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:  # a loss, exactly as the platform would score it
            return 0.0 if board.turn == chess.WHITE else 1.0
        board.push(move)


def elo_to_score(elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def log_likelihood_ratio(wins: int, draws: int, losses: int, elo0: float, elo1: float) -> float:
    """How much more likely the results are under "better by elo1" than under "better by elo0".

    The normal approximation used by every engine test framework: model the per-game score as
    a random variable, and compare the two hypotheses about its mean.
    """
    total = wins + draws + losses
    if total == 0 or wins == 0 or losses == 0:
        return 0.0
    w, d = wins / total, draws / total
    score = w + d / 2.0
    variance = (w + d / 4.0) - score * score
    if variance <= 0:
        return 0.0
    s0, s1 = elo_to_score(elo0), elo_to_score(elo1)
    return (s1 - s0) * (2 * score - s0 - s1) / (2 * variance / total)


def elo_estimate(wins: int, draws: int, losses: int) -> tuple[float, float]:
    total = wins + draws + losses
    score = (wins + draws / 2) / total
    if score <= 0 or score >= 1:
        return (math.inf if score >= 1 else -math.inf), 0.0
    elo = -400.0 * math.log10(1.0 / score - 1.0)
    w, d = wins / total, draws / total
    variance = (w + d / 4.0) - score * score
    deviation = math.sqrt(max(variance, 1e-12) / total)
    # the derivative of the elo curve at this score, for a rough 95% interval
    margin = 1.96 * deviation * 400.0 / (math.log(10) * score * (1.0 - score))
    return elo, margin


def make_openings(count: int, plies: int, seed: int) -> list[str]:
    """Random but not silly: a few random plies, keeping only roughly level positions."""
    rng = random.Random(seed)
    openings = []
    while len(openings) < count:
        board = chess.Board()
        for _ in range(plies):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if board.outcome(claim_draw=True) is not None:
            continue
        balance = sum(
            value * (len(board.pieces(piece, chess.WHITE)) - len(board.pieces(piece, chess.BLACK)))
            for piece, value in PIECE_VALUE.items()
        )
        if balance == 0:  # material-level starts; the engines supply the imbalance
            openings.append(board.fen())
    return openings


class Tally:
    def __init__(self) -> None:
        self.wins = self.draws = self.losses = 0
        self.lock = threading.Lock()
        self.done = threading.Event()

    def add(self, score: float) -> None:
        with self.lock:
            if score == 1.0:
                self.wins += 1
            elif score == 0.0:
                self.losses += 1
            else:
                self.draws += 1

    def counts(self) -> tuple[int, int, int]:
        with self.lock:
            return self.wins, self.draws, self.losses


def worker(index: int, arguments, openings: list[str], tally: Tally, cursor: list[int]) -> None:
    a = Engine(arguments.a, arguments.nodes)
    b = Engine(arguments.b, arguments.nodes)
    try:
        while not tally.done.is_set():
            with tally.lock:
                position = cursor[0]
                cursor[0] += 1
            if position >= len(openings) * 2:
                return
            opening = openings[position // 2]
            # the same opening is played from both sides, so luck cancels
            if position % 2 == 0:
                tally.add(play(b, a, opening))
            else:
                tally.add(1.0 - play(a, b, opening))
    except Exception as error:
        print(f"  worker {index} stopped: {type(error).__name__}: {error}", flush=True)
    finally:
        a.stop()
        b.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="SPRT match between two weight sets.")
    parser.add_argument("--a", default="", help="baseline weights, empty for hand-written")
    parser.add_argument("--b", default="weights/eval.npz", help="candidate weights")
    parser.add_argument("--games", type=int, default=2000)
    parser.add_argument("--nodes", type=int, default=20000, help="nodes per move, both sides")
    parser.add_argument("--workers", type=int, default=5, help="pairs of processes")
    parser.add_argument("--elo0", type=float, default=0.0)
    parser.add_argument("--elo1", type=float, default=8.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--fixed",
        action="store_true",
        help="play every game, ignoring the SPRT bounds. Slower, but the elo it reports is "
        "unbiased: stopping the moment a bound is crossed means stopping on a favourable "
        "fluctuation, which inflates the estimate even though the accept/reject is sound.",
    )
    arguments = parser.parse_args()

    lower = math.log(arguments.beta / (1 - arguments.alpha))
    upper = math.log((1 - arguments.beta) / arguments.alpha)

    openings = make_openings(arguments.games // 2 + 1, 8, arguments.seed)
    print(f"B = {arguments.b or 'hand-written'}   vs   A = {arguments.a or 'hand-written'}")
    print(f"{arguments.nodes:,} nodes/move, {len(openings)} openings played from both sides")
    print(f"SPRT elo({arguments.elo0}, {arguments.elo1})  bounds [{lower:.2f}, {upper:.2f}]\n")

    tally = Tally()
    cursor = [0]
    threads = [
        threading.Thread(target=worker, args=(i, arguments, openings, tally, cursor), daemon=True)
        for i in range(arguments.workers)
    ]
    for thread in threads:
        thread.start()

    started = time.perf_counter()
    last = 0
    while any(thread.is_alive() for thread in threads):
        time.sleep(5)
        wins, draws, losses = tally.counts()
        total = wins + draws + losses
        if total == last:
            continue
        last = total
        llr = log_likelihood_ratio(wins, draws, losses, arguments.elo0, arguments.elo1)
        elo, margin = elo_estimate(wins, draws, losses)
        rate = total / max(time.perf_counter() - started, 1e-9)
        print(
            f"  {total:>5} games  +{wins} ={draws} -{losses}   "
            f"elo {elo:+.1f} +/- {margin:.1f}   llr {llr:+.2f}   {rate:.2f} games/s",
            flush=True,
        )
        crossed = llr >= upper or llr <= lower
        if (crossed and not arguments.fixed) or total >= arguments.games:
            tally.done.set()
            break

    tally.done.set()
    for thread in threads:
        thread.join(timeout=20)

    wins, draws, losses = tally.counts()
    llr = log_likelihood_ratio(wins, draws, losses, arguments.elo0, arguments.elo1)
    elo, margin = elo_estimate(wins, draws, losses)
    print(f"\n+{wins} ={draws} -{losses}   elo {elo:+.1f} +/- {margin:.1f}   llr {llr:+.2f}")
    if arguments.fixed:
        print("fixed-length run: the elo above is an unbiased estimate")
    elif llr >= upper:
        print(f"ACCEPTED: B is better than A by at least {arguments.elo0} elo")
    elif llr <= lower:
        print(f"REJECTED: B is not better than A by {arguments.elo1} elo")
    else:
        print("INCONCLUSIVE: ran out of games before the test decided")


if __name__ == "__main__":
    main()
