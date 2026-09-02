"""Engine process for tools/match.py.

The platform's runner starts a fresh process per game. A few thousand test games cannot
afford that, because each start pays the numba compile. So this speaks the same protocol plus
one extra message, {"newgame": true}, which resets exactly the state a fresh process would
have had: the position history, the transposition table and the ordering heuristics.

Development only: never enters the submission zip.
"""

import json
import os
import sys

protocol = os.fdopen(os.dup(1), "w")
os.dup2(2, 1)

sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else ".")
import agent  # noqa: E402

protocol.write(json.dumps({"ready": True}) + "\n")
protocol.flush()

for line in sys.stdin:
    request = json.loads(line)
    if request.get("newgame"):
        agent.GAME_KEYS.clear()
        agent.WORK.tt[:] = 0
        agent.WORK.history[:] = 0
        agent.WORK.counters[:] = 0
        reply = {"ok": True}
    else:
        reply = {"move": agent.get_move(request["fen"], request["time_left_ms"])}
    protocol.write(json.dumps(reply) + "\n")
    protocol.flush()
