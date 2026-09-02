# The Castel Gambit

An entry for [AI Chessathon](https://aichessathon.com). A classical alpha-beta engine with a
hand-written evaluation, written in the subset of Python that numba compiles.

No third-party engine is involved. Move generation, evaluation and search are all in this
repository, and the only shipped constants that were not typed by hand are the magic
multipliers in `cg_magics.py`, which `tools/gen_magics.py` searches for from a fixed seed.

## What ships

`harness/package.py` puts every root-level `*.py` into the zip. Currently 73 KB unzipped
against a 50 MB cap, so there is room for a learned evaluation later.

| File | Role |
|---|---|
| `agent.py` | `get_move(fen, time_left_ms)`. Clock, repetition history, safety net. |
| `cg_core.py` | Bitboards, magic sliding attacks, move generation, make-move, Zobrist. |
| `cg_eval.py` | Tapered evaluation: material, piece-square tables, mobility, pawns, king safety. |
| `cg_search.py` | Iterative deepening PVS, transposition table, null move, LMR, quiescence. |
| `cg_magics.py` | Generated. Magic multipliers, one table per square. |

`tools/` is development only and stays out of the zip.

## Design

**Board.** 12 piece bitboards plus a mailbox, all in one row of a `uint64` array. Make-move
copies the row forward instead of undoing in place: 84 words of memcpy is cheaper than
getting unmake right, and unmake bugs surface as illegal moves in rated games. Sliding
attacks use fancy magic bitboards, 841 KiB of tables built at import.

**Legality.** Moves are generated pseudo-legally and rejected by make-move when they leave
the king in check. Castling is the exception, since its transit square cannot be tested
after the fact. `agent.py` then checks the chosen move against python-chess before returning
it, so a search bug degrades into a weak move rather than a forfeit.

**Search.** Negamax with alpha-beta, principal variation search, and a transposition table
that survives across moves. Ordering is TT move, then captures by MVV-LVA with SEE deciding
good from bad, then killers, counter-moves and history. Null-move pruning, late move
reductions, futility and reverse futility, check extensions, mate distance pruning.

**Clock.** Each move gets a slice of the remaining clock rather than a constant, so the
budget shrinks by itself and converges on the increment. Three independent stops: the
deepening loop will not open a depth it cannot finish, the tree unwinds at a hard deadline
checked every 1024 nodes, and a node ceiling backstops both.

**Repetition.** The referee claims threefold automatically, so a won position can be drawn
without warning. We are only shown positions on our own turn, but we also know the move we
played, so replaying it locally recovers the ply in between and the history stays complete.

## Verify

```bash
python tools/test_movegen.py --positions 3000 --perft-depth 4
```

Perft on the six standard positions plus a walk over random positions comparing every legal
move against python-chess. This is the test that matters: an illegal move loses outright.

```bash
python tools/bench.py
```

Init time against the 60 s budget, and nodes per second. Run it where numba loads.

```bash
python -m harness.arena --opponent baselines/minimax --games 8 --base-ms 10000
python -m harness.package
```

## Play it

```bash
python tools/play_web.py
```

Opens a board on http://127.0.0.1:8800. Click a piece, click where it goes. Pick your colour
and how long the engine gets per move; it shows the depth, score and node count behind each
reply. It drives the real `agent.get_move`, including the time management, so what you are
playing is what gets uploaded.

## Status

Verified with numba on Python 3.12, the platform's versions:

| Measure | Result |
|---|---|
| Init, against the 60 s budget | **28-31 s** |
| Search speed | **1.6M nps** (3.6M nps raw move generation) |
| Depth at 3 s | 13-14 middlegame, 19-25 endgames |
| Perft, 6 positions to depth 5 | 16.3M nodes, **exact** |
| Random positions vs python-chess | 4,000, zero move or key diffs |
| Full game at 120 s + 0.5 s | won by mate, 48.6 s left on the clock |
| `ruff` and `mypy --strict` | clean |

The same code runs without numba at about 4.7k nps, which is the fallback the correctness
work was done against, not a configuration to play in.

Init is the tightest constraint and it is only half spent, so watch it. Two ways of buying
it back were measured and rejected: `cache=True` segfaults on a warm start and writes a
96 MB cache against a 256 MB scratch budget, and `NUMBA_OPT` below 3 changes compile time by
under a second. If the platform's core is much slower than a laptop's, the fix is to compile
less, not to compile differently.
