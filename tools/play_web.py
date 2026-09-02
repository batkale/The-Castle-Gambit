"""Play against the engine in a browser.

    python tools/play_web.py

Serves a board on http://127.0.0.1:8800. Click a piece, click where it goes. The engine
answers on the same process the harness would run, so what you are playing is exactly what
gets uploaded -- same search, same evaluation, same clock code.

Development only: this file lives in tools/ and never enters the submission zip.
"""

import argparse
import json
import sys
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import chess

sys.path.insert(0, ".")

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>The Castle Gambit</title>
<style>
  :root {
    --bg: #14181b; --panel: #1c2226; --line: #2b343a; --text: #e6ebe8;
    --muted: #8d9a93; --light: #ebe8dd; --dark: #6f8f6a; --accent: #7fd4a3;
    --from: #d8c46a; --hint: rgba(40,60,45,.42);
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--text); min-height: 100vh;
    font: 15px/1.55 ui-sans-serif, system-ui, "Segoe UI", sans-serif;
    display: flex; align-items: flex-start; justify-content: center; gap: 28px;
    padding: 20px; flex-wrap: wrap;
  }
  h1 { font-size: 1.15rem; margin: 0 0 2px; font-weight: 600; letter-spacing: -.01em; }
  .sub { color: var(--muted); font-size: .82rem; margin: 0 0 18px; }

  #board {
    display: grid; grid-template: repeat(8, 1fr) / repeat(8, 1fr);
    width: min(88vw, 72vmin, 560px); aspect-ratio: 1;
    border: 1px solid var(--line); border-radius: 4px; overflow: hidden;
    user-select: none; touch-action: manipulation;
  }
  .sq {
    display: flex; align-items: center; justify-content: center;
    position: relative; cursor: pointer; font-size: min(10vw, 8.4vmin, 64px);
    line-height: 1;
  }
  .sq.l { background: var(--light); }
  .sq.d { background: var(--dark); }
  .sq.sel { box-shadow: inset 0 0 0 4px var(--from); }
  .sq.last { box-shadow: inset 0 0 0 4px rgba(216,196,106,.5); }
  .sq .glyph { position: relative; z-index: 2; }
  .sq.w .glyph { color: #fbfdfc; text-shadow: 0 1px 0 #4a5450, 0 0 3px rgba(0,0,0,.55); }
  .sq.b .glyph { color: #24292b; text-shadow: 0 1px 0 rgba(255,255,255,.22); }
  .dot::after {
    content: ""; position: absolute; inset: 0; margin: auto;
    width: 26%; height: 26%; border-radius: 50%; background: var(--hint); z-index: 1;
  }
  .cap::after {
    content: ""; position: absolute; inset: 6%; border-radius: 50%;
    border: 6px solid var(--hint); z-index: 1;
  }
  .coord {
    position: absolute; font-size: 10px; font-weight: 600; opacity: .48; z-index: 3;
    font-family: ui-monospace, monospace;
  }
  .coord.f { right: 3px; bottom: 1px; }
  .coord.r { left: 3px; top: 1px; }
  .sq.l .coord { color: #4b5a4a; } .sq.d .coord { color: #dfe8dc; }

  aside { width: 288px; display: flex; flex-direction: column; gap: 14px; }
  .card { background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 14px 16px; }
  .card h2 {
    font-size: .68rem; letter-spacing: .12em; text-transform: uppercase;
    color: var(--muted); margin: 0 0 10px; font-weight: 600;
    font-family: ui-monospace, monospace;
  }
  .row { display: flex; justify-content: space-between; gap: 12px; font-size: .88rem; padding: 2px 0; }
  .row span:last-child { font-family: ui-monospace, monospace; color: var(--accent); font-variant-numeric: tabular-nums; }
  #status { font-size: .95rem; min-height: 1.5em; }
  #status.over { color: var(--from); font-weight: 600; }

  .controls { display: flex; gap: 8px; flex-wrap: wrap; }
  button, select {
    background: #232b30; color: var(--text); border: 1px solid var(--line);
    border-radius: 4px; padding: 7px 12px; font: inherit; font-size: .85rem; cursor: pointer;
  }
  button:hover, select:hover { border-color: var(--accent); }
  button:disabled { opacity: .45; cursor: default; border-color: var(--line); }
  label { font-size: .8rem; color: var(--muted); display: flex; align-items: center; gap: 7px; }

  #moves {
    font-family: ui-monospace, monospace; font-size: .8rem; line-height: 1.75;
    max-height: 210px; overflow-y: auto; color: var(--muted);
  }
  #moves b { color: var(--text); font-weight: 500; }

  #promo {
    position: fixed; inset: 0; background: rgba(8,10,11,.72); display: none;
    align-items: center; justify-content: center; z-index: 50;
  }
  #promo.on { display: flex; }
  #promo div { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 18px; display: flex; gap: 10px; }
  #promo button { font-size: 2.4rem; padding: 6px 14px; line-height: 1; }
</style>
</head>
<body>

<main>
  <h1>The Castle Gambit</h1>
  <p class="sub">Click a piece, then its destination.</p>
  <div id="board"></div>
</main>

<aside>
  <div class="card">
    <h2>Game</h2>
    <div id="status">Loading engine&hellip;</div>
  </div>

  <div class="card">
    <h2>New game</h2>
    <div class="controls">
      <label>Play <select id="colour"><option value="white">White</option><option value="black">Black</option></select></label>
      <label>Engine <select id="think">
        <option value="500">0.5s</option>
        <option value="1000" selected>1s</option>
        <option value="3000">3s</option>
        <option value="10000">10s</option>
      </select></label>
    </div>
    <div class="controls" style="margin-top:10px">
      <button id="new">Start</button>
      <button id="undo">Take back</button>
    </div>
  </div>

  <div class="card">
    <h2>Last search</h2>
    <div class="row"><span>Depth</span><span id="i-depth">&mdash;</span></div>
    <div class="row"><span>Score, engine&rsquo;s view</span><span id="i-score">&mdash;</span></div>
    <div class="row"><span>Nodes</span><span id="i-nodes">&mdash;</span></div>
    <div class="row"><span>Speed</span><span id="i-nps">&mdash;</span></div>
    <div class="row"><span>Time</span><span id="i-time">&mdash;</span></div>
  </div>

  <div class="card">
    <h2>Moves</h2>
    <div id="moves">&mdash;</div>
  </div>
</aside>

<div id="promo"><div></div></div>

<script>
const GLYPH = {p:"\\u265F",n:"\\u265E",b:"\\u265D",r:"\\u265C",q:"\\u265B",k:"\\u265A"};
const boardEl = document.getElementById("board");
const promoEl = document.getElementById("promo");
let state = null, selected = null, flipped = false, busy = false;

function parseFen(fen) {
  const squares = new Array(64).fill(null);
  const rows = fen.split(" ")[0].split("/");
  for (let r = 0; r < 8; r++) {
    let file = 0;
    for (const ch of rows[r]) {
      if (ch >= "1" && ch <= "8") { file += +ch; continue; }
      squares[(7 - r) * 8 + file] = ch;
      file++;
    }
  }
  return squares;
}

function name(i) { return "abcdefgh"[i & 7] + (1 + (i >> 3)); }

function render() {
  if (!state) return;
  const squares = parseFen(state.fen);
  const targets = new Set();
  if (selected !== null) {
    for (const uci of state.legal) if (uci.slice(0, 2) === name(selected)) targets.add(uci.slice(2, 4));
  }
  boardEl.innerHTML = "";
  for (let row = 7; row >= 0; row--) {
    for (let col = 0; col < 8; col++) {
      const r = flipped ? 7 - row : row, c = flipped ? 7 - col : col;
      const i = r * 8 + c, sq = name(i);
      const cell = document.createElement("div");
      cell.className = "sq " + ((r + c) % 2 ? "l" : "d");
      const piece = squares[i];
      if (piece) {
        cell.classList.add(piece === piece.toUpperCase() ? "w" : "b");
        const g = document.createElement("span");
        g.className = "glyph";
        g.textContent = GLYPH[piece.toLowerCase()];
        cell.appendChild(g);
      }
      if (i === selected) cell.classList.add("sel");
      if (state.last && (sq === state.last.slice(0, 2) || sq === state.last.slice(2, 4))) cell.classList.add("last");
      if (targets.has(sq)) cell.classList.add(piece ? "cap" : "dot");
      if (r === (flipped ? 7 : 0)) {
        const f = document.createElement("span"); f.className = "coord f"; f.textContent = "abcdefgh"[c]; cell.appendChild(f);
      }
      if (c === (flipped ? 7 : 0)) {
        const k = document.createElement("span"); k.className = "coord r"; k.textContent = r + 1; cell.appendChild(k);
      }
      cell.onclick = () => click(i);
      boardEl.appendChild(cell);
    }
  }
}

function click(i) {
  if (busy || !state || state.over) return;
  const sq = name(i);
  if (selected !== null) {
    const options = state.legal.filter(u => u.slice(0, 2) === name(selected) && u.slice(2, 4) === sq);
    if (options.length === 1) { selected = null; send(options[0]); return; }
    if (options.length > 1) { askPromotion(options); return; }
  }
  const mine = state.legal.some(u => u.slice(0, 2) === sq);
  selected = mine ? i : null;
  render();
}

function askPromotion(options) {
  const holder = promoEl.firstElementChild;
  holder.innerHTML = "";
  for (const uci of options) {
    const b = document.createElement("button");
    b.textContent = GLYPH[uci[4]];
    b.onclick = () => { promoEl.classList.remove("on"); selected = null; send(uci); };
    holder.appendChild(b);
  }
  promoEl.classList.add("on");
}

async function post(path, body) {
  busy = true;
  document.getElementById("status").textContent = "Thinking\\u2026";
  try {
    const r = await fetch(path, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body || {})});
    state = await r.json();
    selected = null;
    paint();
  } catch (e) {
    document.getElementById("status").textContent = "Lost the engine: " + e;
  } finally { busy = false; }
}

const send = uci => post("/api/move", {uci});

function paint() {
  render();
  const s = document.getElementById("status");
  s.textContent = state.status;
  s.classList.toggle("over", !!state.over);
  const info = state.info;
  const set = (id, v) => document.getElementById(id).textContent = v;
  if (info) {
    set("i-depth", info.depth + "/" + info.seldepth);
    set("i-score", (info.score > 0 ? "+" : "") + (info.score / 100).toFixed(2));
    set("i-nodes", info.nodes.toLocaleString());
    set("i-nps", Math.round(info.nps / 1000).toLocaleString() + "k/s");
    set("i-time", info.seconds.toFixed(2) + "s");
  }
  const m = document.getElementById("moves");
  if (!state.san.length) { m.innerHTML = "&mdash;"; }
  else {
    let out = "";
    for (let i = 0; i < state.san.length; i += 2) {
      out += "<b>" + (i / 2 + 1) + ".</b> " + state.san[i] + " " + (state.san[i + 1] || "") + "<br>";
    }
    m.innerHTML = out;
    m.scrollTop = m.scrollHeight;
  }
  document.getElementById("undo").disabled = state.san.length < 1;
}

document.getElementById("new").onclick = () => {
  const colour = document.getElementById("colour").value;
  flipped = colour === "black";
  post("/api/new", {colour, think_ms: +document.getElementById("think").value});
};
document.getElementById("undo").onclick = () => post("/api/undo", {});

post("/api/new", {colour: "white", think_ms: 1000});
</script>
</body>
</html>
"""


class Game:
    """One game against the engine, held in the server process."""

    def __init__(self) -> None:
        self.board = chess.Board()
        self.human = chess.WHITE
        self.think_ms = 1000
        self.info: dict[str, float] | None = None
        self.last = ""

    def clock_for(self) -> int:
        """The clock to hand the agent so its own budget lands on think_ms.

        agent._budget spends (time_left - overhead) / 24 plus three quarters of the
        increment, so inverting it keeps the real time-management code on the path rather
        than bypassing it with a fixed movetime.
        """
        import agent

        floor = 0.75 * agent.INCREMENT_MS
        target = max(self.think_ms, floor + 25)
        return int((target - floor) * agent.DIVISOR + agent.OVERHEAD_MS)

    def reset(self, colour: str, think_ms: int) -> None:
        import agent

        self.board = chess.Board()
        self.human = chess.WHITE if colour == "white" else chess.BLACK
        self.think_ms = max(400, int(think_ms))
        self.info = None
        self.last = ""
        agent.GAME_KEYS.clear()
        agent.WORK.tt[:] = 0
        agent.WORK.history[:] = 0
        agent.WORK.counters[:] = 0

    def engine_move(self) -> None:
        import agent

        started = time.perf_counter()
        uci = agent.get_move(self.board.fen(), self.clock_for())
        seconds = time.perf_counter() - started
        move = chess.Move.from_uci(uci)
        if move not in self.board.legal_moves:
            raise RuntimeError(f"engine returned an illegal move: {uci}")
        self.board.push(move)
        self.last = uci
        report = agent._last_report
        self.info = {
            "depth": _field(report, "d", 0),
            "seldepth": _field(report, "d", 1),
            "score": _score(report),
            "nodes": int(agent.WORK.control[0]),
            "nps": int(agent.WORK.control[0] / max(seconds, 1e-9)),
            "seconds": seconds,
        }

    def payload(self) -> dict[str, object]:
        outcome = self.board.outcome(claim_draw=True)
        over = outcome is not None
        if over:
            if outcome.winner is None:
                status = f"Draw by {outcome.termination.name.replace('_', ' ').lower()}."
            elif outcome.winner == self.human:
                status = "You win."
            else:
                status = "Engine wins."
        elif self.board.turn == self.human:
            status = "Your move." + (" You are in check." if self.board.is_check() else "")
        else:
            status = "Engine to move."
        return {
            "fen": self.board.fen(),
            "legal": [m.uci() for m in self.board.legal_moves] if not over else [],
            "san": _san_list(self.board),
            "info": self.info,
            "status": status,
            "over": over,
            "last": self.last,
        }


def _san_list(board: chess.Board) -> list[str]:
    replay = chess.Board()
    out = []
    for move in board.move_stack:
        out.append(replay.san(move))
        replay.push(move)
    return out


def _field(report: str, prefix: str, index: int) -> int:
    """Pull the depth pair out of the agent's own stderr line, e.g. 'd15/27'."""
    for token in report.split():
        if token.startswith(prefix) and "/" in token:
            try:
                return int(token[len(prefix) :].split("/")[index])
            except ValueError:
                return 0
    return 0


def _score(report: str) -> int:
    for token in report.split():
        if token.endswith("cp"):
            try:
                return int(token[:-2])
            except ValueError:
                return 0
    return 0


GAME = Game()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args: object) -> None:  # quiet; the engine prints its own lines
        pass

    def do_GET(self) -> None:
        if self.path != "/":
            self.send_error(404)
            return
        body = PAGE.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        request = json.loads(self.rfile.read(length) or b"{}")
        try:
            if self.path == "/api/new":
                GAME.reset(request.get("colour", "white"), request.get("think_ms", 1000))
                if GAME.board.turn != GAME.human:
                    GAME.engine_move()
            elif self.path == "/api/undo":
                for _ in range(2):  # take back the pair, so it is your move again
                    if GAME.board.move_stack:
                        GAME.board.pop()
                GAME.last = ""
                GAME.info = None
            elif self.path == "/api/move":
                move = chess.Move.from_uci(request["uci"])
                if move in GAME.board.legal_moves:
                    GAME.board.push(move)
                    GAME.last = move.uci()
                    if GAME.board.outcome(claim_draw=True) is None:
                        GAME.engine_move()
            else:
                self.send_error(404)
                return
            payload = GAME.payload()
        except Exception as error:  # surface it in the page rather than dying silently
            payload = GAME.payload()
            payload["status"] = f"{type(error).__name__}: {error}"
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser(description="Play against the engine in a browser.")
    parser.add_argument("--port", type=int, default=8800)
    parser.add_argument("--no-open", action="store_true")
    arguments = parser.parse_args()

    print("compiling the engine, about 20 seconds ...", flush=True)
    started = time.perf_counter()
    import agent  # noqa: F401  - importing is what compiles it

    print(f"ready in {time.perf_counter() - started:.1f}s", flush=True)

    url = f"http://127.0.0.1:{arguments.port}"
    print(f"open {url}  (ctrl-c to stop)", flush=True)
    if not arguments.no_open:
        webbrowser.open(url)
    ThreadingHTTPServer(("127.0.0.1", arguments.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
