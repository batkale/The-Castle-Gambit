"""The Castel Gambit -- entry point for AI Chessathon.

    get_move(fen, time_left_ms) -> uci

The engine is a classical alpha-beta searcher with a hand-written evaluation, compiled by
numba. Nothing is downloaded, nothing is shelled out to, and no third-party engine is
involved: cg_core.py generates moves, cg_eval.py scores positions, cg_search.py searches.

This file is deliberately the thin part. It owns three things the search should not:

* the clock. It decides how long this move gets and hands the search two deadlines.
* the game's position history, so repetition is visible to the search. We are only shown
  positions on our own turn, but we also know the move we played, so replaying it locally
  recovers the ply in between and the history stays complete.
* the safety net. Whatever happens inside, this function returns a legal UCI move. An
  illegal move or an exception loses the game outright, so the returned move is checked
  against python-chess before it goes out, and any failure falls back to a legal move.
"""

import os

# numba must not start a thread pool: the container has one core and threads past the first
# take time away from the search. Set before numba is imported anywhere.
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# numba writes its compilation cache next to the source by default, and the filesystem is
# read-only apart from /tmp.
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(os.environ.get("TMPDIR", "/tmp"), "cg"))

import sys
import time

import chess

import cg_search
from cg_core import (
    HAVE_NUMBA,
    S_KEY,
    load_fen,
    move_to_uci,
    new_state,
)

# ---------------------------------------------------------------- clock
INCREMENT_MS = 500  # fixed by the event: 120 s + 0.5 s
OVERHEAD_MS = 45  # json, the pipe and the referee's own bookkeeping
DIVISOR = 24  # slice of the remaining clock this move may have
MAX_FRACTION = 0.30  # never stake more than this much of what is left on one move
MAX_DEPTH = 63

# Testing hook, unset in rated play. With a node budget instead of a clock, an A/B match is
# deterministic and immune to how loaded the machine is, which is what makes a few thousand
# games mean something. tools/match.py sets it.
FIXED_NODES = int(os.environ.get("CG_FIXED_NODES", "0"))

STATE = new_state()
SCRATCH = new_state()
# numba freezes module-level arrays read-only, so everything the search writes to lives
# here and is passed in. Allocated once, at import, and reused for every move.
WORK = cg_search.new_workspace()
GAME_KEYS: list[int] = []

_last_report = ""


def _budget(time_left_ms: int) -> tuple[float, float]:
    """Soft and hard limits in seconds.

    The soft limit is what the deepening loop aims at: it will not open a new depth past it.
    The hard limit is where the tree unwinds mid-iteration. Dividing the remaining clock
    rather than spending a constant means the budget shrinks on its own as the game goes on
    and converges on the increment, so there is no move at which we can flag.
    """
    available = max(0.0, time_left_ms - OVERHEAD_MS)
    if available <= 0:
        return 0.0, 0.0
    soft = available / DIVISOR + 0.75 * INCREMENT_MS
    hard = min(available * MAX_FRACTION, soft * 3.5)
    soft = min(soft, hard)
    return soft / 1000.0, hard / 1000.0


def _fallback(fen: str) -> str:
    """A legal move, whatever went wrong. Prefers a capture so it is not actively terrible."""
    board = chess.Board(fen)
    best, best_value = None, -1
    for move in board.legal_moves:
        value = 0
        if board.is_capture(move):
            captured = board.piece_type_at(move.to_square)
            value = 1 if captured is None else (1, 3, 3, 5, 9, 0)[captured - 1]
        if value > best_value:
            best, best_value = move, value
    return best.uci() if best is not None else "0000"


def get_move(fen: str, time_left_ms: int) -> str:
    global _last_report
    started = time.perf_counter()
    try:
        load_fen(STATE, 0, fen)
        root_key = int(STATE[0, S_KEY])

        history_length = len(GAME_KEYS)
        for index, key in enumerate(GAME_KEYS):
            WORK.repetition[index] = key

        if FIXED_NODES:
            soft = hard = 3600.0
            node_limit = FIXED_NODES
        else:
            soft, hard = _budget(time_left_ms)
            # a node ceiling in case the clock check ever fails; far above anything one move
            # reaches, so it only ever fires as a backstop
            node_limit = 1 << 40
        deadline = started + hard

        move, score, depth, nodes, seldepth = cg_search.search(
            WORK,
            STATE,
            history_length,
            MAX_DEPTH,
            started + soft,
            deadline,
            node_limit,
        )

        uci = move_to_uci(int(move)) if move else ""
        board = chess.Board(fen)
        if not uci or chess.Move.from_uci(uci) not in board.legal_moves:
            uci = _fallback(fen)
            print(f"fallback: search returned {uci!r}", file=sys.stderr)

        elapsed = time.perf_counter() - started
        _last_report = (
            f"d{depth}/{seldepth} {score:+d}cp {nodes:,}n "
            f"{elapsed:.2f}s {nodes / max(elapsed, 1e-6):,.0f}nps clock={time_left_ms}ms"
        )
        print(_last_report, file=sys.stderr)

        # record both plies: the position we were shown, and the one our move creates. That
        # keeps the history contiguous even though we never see the opponent's turn.
        GAME_KEYS.append(root_key)
        played = chess.Move.from_uci(uci)
        board.push(played)
        load_fen(SCRATCH, 0, board.fen())
        GAME_KEYS.append(int(SCRATCH[0, S_KEY]))
        return uci

    except Exception as error:  # never let anything reach the runner
        print(f"error: {type(error).__name__}: {error}", file=sys.stderr)
        try:
            return _fallback(fen)
        except Exception:
            return "0000"


def _warm_up() -> None:
    """Compile every jitted function before the clock starts.

    numba compiles per signature on first call, so this runs a real search over a handful of
    positions: quiet, tactical, an endgame and a position in check. Between them they reach
    every branch that matters, which is the point -- a branch compiled on the clock is a
    branch paid for twice.
    """
    positions = (
        chess.STARTING_FEN,
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
        "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
    )
    for fen in positions:
        load_fen(STATE, 0, fen)
        limit = time.perf_counter() + 30.0
        cg_search.search(WORK, STATE, 0, 6, limit, limit, 1 << 40)
    # reset the tables the warm-up dirtied so the first real move starts clean
    WORK.tt[:] = 0
    WORK.history[:] = 0
    WORK.counters[:] = 0


_started_at = time.perf_counter()
_warm_up()
print(
    f"ready in {time.perf_counter() - _started_at:.1f}s (numba={HAVE_NUMBA})",
    file=sys.stderr,
)
