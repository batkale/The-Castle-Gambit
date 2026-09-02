"""Measure the two numbers that decide whether this engine is competitive.

    python tools/bench.py

* **Init time.** Importing agent.py compiles every jitted function. The platform allows 60
  seconds before the clock starts and a miss is scored as a loss, so this needs headroom,
  not just a pass.
* **Nodes per second.** Everything else in the engine trades against this. Pure Python runs
  around 5k nps and is only good for correctness work; numba should be two orders up.

Run this wherever numba actually loads. It prints a verdict on both numbers.
"""

import sys
import time

sys.path.insert(0, ".")

INIT_BUDGET_S = 60.0
POSITIONS = [
    ("opening", "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
    ("middlegame", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("tactical", "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1"),
    ("endgame", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
    ("pawn endgame", "8/8/4k3/8/2p5/8/B2P2K1/8 w - - 0 1"),
]


def main() -> None:
    started = time.perf_counter()
    import agent

    init = time.perf_counter() - started

    from cg_core import HAVE_NUMBA, load_fen, new_state
    from cg_search import new_workspace, search

    print(f"\nnumba: {HAVE_NUMBA}")
    verdict = "OK" if init < INIT_BUDGET_S * 0.6 else "TIGHT" if init < INIT_BUDGET_S else "OVER"
    print(f"init: {init:.1f}s of the {INIT_BUDGET_S:.0f}s budget  [{verdict}]")
    if not HAVE_NUMBA:
        print("  (pure Python: this is the correctness fallback, not the real speed)")

    print(f"\n{'position':<14} {'depth':>6} {'score':>8} {'nodes':>12} {'nps':>11}")
    state = new_state()
    work = new_workspace()
    total_nodes = 0
    total_time = 0.0
    for name, fen in POSITIONS:
        load_fen(state, 0, fen)
        now = time.perf_counter()
        _, score, depth, nodes, seldepth = search(work, state, 0, 63, now + 3.0, now + 3.5, 1 << 40)
        elapsed = time.perf_counter() - now
        total_nodes += nodes
        total_time += elapsed
        print(
            f"{name:<14} {depth:>3}/{seldepth:<2} {score:>+8} {nodes:>12,} "
            f"{nodes / max(elapsed, 1e-9):>11,.0f}"
        )

    rate = total_nodes / max(total_time, 1e-9)
    print(f"\noverall: {rate:,.0f} nps over {total_nodes:,} nodes")
    if HAVE_NUMBA and rate < 200_000:
        print("  slower than expected for compiled code; check that njit really applied")

    # a move under the real clock, end to end, is the number that actually matters
    print("\nfull move at 120s on the clock:")
    move = agent.get_move(POSITIONS[1][1], 120_000)
    print(f"  played {move}")


if __name__ == "__main__":
    main()
