"""Correctness gate for the move generator.

An illegal move loses a rated game outright, so this is the test that matters most. It runs
two independent checks:

* a walk over thousands of random positions, comparing our legal move set against
  python-chess square by square, plus the incrementally maintained zobrist key against a
  recomputation. This finds bugs fast and covers promotion, en passant and castling because
  random play reaches them;
* perft on the six standard positions, whose node counts are published and unambiguous.

    python tools/test_movegen.py [--perft-depth N] [--positions N]
"""

import argparse
import random
import sys
import time

import chess

sys.path.insert(0, ".")

from cg_core import (
    compute_key,
    gen_captures,
    gen_moves,
    legal_moves,
    load_fen,
    make_move,
    move_to_uci,
    new_move_buffer,
    new_state,
)

PERFT_SUITE = [
    ("startpos", chess.STARTING_FEN, [20, 400, 8902, 197281, 4865609]),
    (
        "kiwipete",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        [48, 2039, 97862, 4085603],
    ),
    ("endgame", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", [14, 191, 2812, 43238, 674624]),
    (
        "promotions",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        [6, 264, 9467, 422333],
    ),
    (
        "cramped",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        [44, 1486, 62379, 2103487],
    ),
    (
        "steady",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        [46, 2079, 89890, 3894594],
    ),
]


def perft(state, moves, ply: int, depth: int) -> int:
    total = gen_moves(state, ply, moves)
    if depth == 1:  # count legal children without descending into them
        count = 0
        for i in range(total):
            if make_move(state, ply, int(moves[ply, i])):
                count += 1
        return count
    nodes = 0
    for i in range(total):
        move = int(moves[ply, i])
        if make_move(state, ply, move):
            nodes += perft(state, moves, ply + 1, depth - 1)
    return nodes


def compare_positions(count: int, seed: int) -> int:
    """Play random games, checking every position we pass through."""
    rng = random.Random(seed)
    state, moves = new_state(), new_move_buffer()
    failures = checked = 0
    board = chess.Board()

    while checked < count:
        if board.is_game_over(claim_draw=False) or board.ply() > 160:
            board = chess.Board()
            continue

        fen = board.fen()
        load_fen(state, 0, fen)

        mine = sorted(move_to_uci(m) for m in legal_moves(state, 0, moves))
        theirs = sorted(m.uci() for m in board.legal_moves)
        if mine != theirs:
            failures += 1
            print(f"\n  MOVE MISMATCH {fen}")
            print(f"    only mine:   {sorted(set(mine) - set(theirs))}")
            print(f"    only theirs: {sorted(set(theirs) - set(mine))}")

        if int(state[0, 19]) != int(compute_key(state, 0)):
            failures += 1
            print(f"\n  KEY MISMATCH after load {fen}")

        # The quiescence generator is pseudo-legal like the main one, so filter it the same
        # way before comparing. What has to hold is that it finds every legal capture and
        # queen promotion, and invents nothing that is not one.
        captures = gen_captures(state, 0, moves)
        capture_set = {
            move_to_uci(int(moves[0, i]))
            for i in range(captures)
            if make_move(state, 0, int(moves[0, i]))
        }
        # the contract: every legal capture that is not an underpromotion, plus every legal
        # queen promotion. Underpromotions are left to the main search; a knight promotion
        # that matters is nearly always found a ply earlier anyway.
        legal_captures = {
            m.uci()
            for m in board.legal_moves
            if (board.is_capture(m) and m.promotion in (None, chess.QUEEN))
            or m.promotion == chess.QUEEN
        }
        if capture_set != legal_captures:
            failures += 1
            print(f"\n  CAPTURE MISMATCH {fen}")
            print(f"    only mine:   {sorted(capture_set - legal_captures)}")
            print(f"    only theirs: {sorted(legal_captures - capture_set)}")

        # the incremental key must survive a real move
        chosen = rng.choice(list(board.legal_moves))
        engine_move = next(
            m for m in legal_moves(state, 0, moves) if move_to_uci(m) == chosen.uci()
        )
        make_move(state, 0, engine_move)
        if int(state[1, 19]) != int(compute_key(state, 1)):
            failures += 1
            print(f"\n  KEY MISMATCH after {chosen.uci()} from {fen}")

        board.push(chosen)
        checked += 1
        if checked % 250 == 0:
            print(f"  {checked} positions checked, {failures} failures", flush=True)

    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", type=int, default=1500)
    parser.add_argument("--perft-depth", type=int, default=3)
    parser.add_argument("--seed", type=int, default=11)
    arguments = parser.parse_args()

    print(f"comparing {arguments.positions} random positions against python-chess")
    failures = compare_positions(arguments.positions, arguments.seed)
    print(f"  -> {failures} failures\n")

    print(f"perft to depth {arguments.perft_depth}")
    state, moves = new_state(), new_move_buffer()
    for name, fen, expected in PERFT_SUITE:
        load_fen(state, 0, fen)
        for depth in range(1, min(arguments.perft_depth, len(expected)) + 1):
            started = time.perf_counter()
            got = perft(state, moves, 0, depth)
            want = expected[depth - 1]
            elapsed = time.perf_counter() - started
            status = "ok " if got == want else "BAD"
            rate = got / elapsed if elapsed > 0 else 0
            print(
                f"  {status} {name:<11} depth {depth}: {got:>9,} "
                f"(want {want:>9,}) {elapsed:6.2f}s {rate:>9,.0f} nps"
            )
            if got != want:
                failures += 1

    print("\nPASS" if failures == 0 else f"\nFAIL ({failures} failures)")
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
