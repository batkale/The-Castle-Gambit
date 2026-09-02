"""Iterative deepening principal-variation search.

The whole search runs inside numba, including the deepening loop, so the only Python on the
move path is setting up the call and reading the answer out.

Everything the search writes to lives in a `Workspace`, allocated once at import and threaded
through as an argument. That is not a style choice: numba freezes module-level arrays as
read-only constants, so a table the search mutates cannot be a global. Read-only tables --
attack tables, piece-square tables, the reduction schedule -- stay global, where numba can
fold them in as constants.

Three independent things stop the search, because flagging is a loss and one mechanism
failing quietly is not worth the risk:

* the deepening loop refuses to start a depth it does not expect to finish;
* the tree checks the wall clock every 1024 nodes and unwinds past the hard deadline;
* a node limit backstops both, in case the clock check ever misbehaves.

An aborted iteration is discarded and the previous one's move is played, which is the point
of iterative deepening.
"""

import time
from collections import namedtuple

import numpy as np

from cg_core import (
    CACHE,
    EMPTY,
    EPCAP,
    HAVE_NUMBA,
    MAX_MOVES,
    MAX_PLY,
    ONE,
    PAWN,
    PROMO,
    QUEEN,
    S_HALF,
    S_KEY,
    S_MAIL,
    S_OCC,
    S_SIDE,
    ZERO,
    U,
    attackers_to,
    bishop_attacks,
    gen_captures,
    gen_moves,
    in_check,
    lsb,
    make_move,
    make_null,
    njit,
    rook_attacks,
)
from cg_eval import SEE_VALUE, evaluate, has_non_pawn_material

MATE = 30000
MATE_BOUND = 29000
INFINITY = 32000
NULL_MOVE = np.int64(-1)  # marks the null-move recursion; typed so it is not a literal

TT_BITS = 22
TT_SIZE = 1 << TT_BITS
TT_MASK = U(TT_SIZE - 1)
EXACT, LOWER, UPPER = 0, 1, 2

STACK_DEPTH = MAX_PLY + 8

# control slots
NODES, ABORTED, NODE_LIMIT, SELDEPTH, ROOT_MOVE = 0, 1, 2, 3, 4

# Late move reductions: deeper searches and later moves get cut back harder. Read-only, so
# it stays a global and numba treats it as a constant.
LMR = np.zeros((64, 64), dtype=np.int32)
for _depth in range(1, 64):
    for _played in range(1, 64):
        LMR[_depth, _played] = int(0.80 + np.log(_depth) * np.log(_played) / 2.30)

FUTILITY = np.array([0, 110, 210, 320, 440, 570, 710, 860], dtype=np.int32)

Workspace = namedtuple(
    "Workspace",
    "tt moves order killers history counters repetition control see_gains",
)


def new_workspace() -> Workspace:
    """Everything the search mutates. Allocated once; nothing in the tree allocates."""
    return Workspace(
        # one entry is two adjacent words, so a probe touches a single cache line
        tt=np.zeros(TT_SIZE * 2, dtype=U),
        moves=np.zeros((STACK_DEPTH, MAX_MOVES), dtype=np.int32),
        order=np.zeros((STACK_DEPTH, MAX_MOVES), dtype=np.int32),
        killers=np.zeros((STACK_DEPTH, 2), dtype=np.int32),
        history=np.zeros((2, 64, 64), dtype=np.int32),
        counters=np.zeros((2, 64, 64), dtype=np.int32),
        repetition=np.zeros(2048, dtype=U),
        control=np.zeros(8, dtype=np.int64),
        see_gains=np.zeros((STACK_DEPTH, 32), dtype=np.int32),
    )


if HAVE_NUMBA:  # pragma: no cover - exercised only where numba loads
    from numba import objmode

    @njit(cache=CACHE)
    def _now():
        """Wall clock inside compiled code. objmode costs a microsecond or so, which is why
        the caller only asks every 1024 nodes."""
        with objmode(seconds="float64"):
            seconds = time.perf_counter()
        return seconds

else:

    def _now():
        return time.perf_counter()


# ---------------------------------------------------------------- transposition table
@njit(cache=CACHE)
def tt_store(tt, key, move, score, depth, flag, ply):
    index = np.int64((key & TT_MASK) * U(2))
    stored = tt[index]
    stored_depth = np.int64((tt[index + 1] >> U(32)) & U(255))
    # keep the deeper result unless this is a different position, which is the common case
    if stored == key and stored_depth > depth + 2:
        return
    adjusted = score
    if score > MATE_BOUND:
        adjusted = score + ply
    elif score < -MATE_BOUND:
        adjusted = score - ply
    packed = (
        U(move & 0xFFFF)
        | (U(adjusted + 32768) << U(16))
        | (U(depth & 255) << U(32))
        | (U(flag) << U(40))
    )
    tt[index] = key
    tt[index + 1] = packed


@njit(cache=CACHE)
def tt_probe(tt, key):
    """Returns (hit, move, score, depth, flag). Score still needs the ply correction."""
    index = np.int64((key & TT_MASK) * U(2))
    if tt[index] != key:
        return False, np.int64(0), np.int64(0), np.int64(0), np.int64(0)
    packed = tt[index + 1]
    move = np.int64(packed & U(0xFFFF))
    score = np.int64((packed >> U(16)) & U(0xFFFF)) - 32768
    depth = np.int64((packed >> U(32)) & U(255))
    flag = np.int64((packed >> U(40)) & U(255))
    return True, move, score, depth, flag


# ---------------------------------------------------------------- static exchange evaluation
@njit(cache=CACHE)
def see(ws, state, ply, move):
    """Net material after the capture sequence on the target square, in centipawns.

    The swap-off list algorithm: repeatedly take with the cheapest remaining attacker and
    decide, walking back up, whether each side would actually have entered the exchange.
    """
    origin = move & 63
    target = (move >> 6) & 63
    kind = (move >> 14) & 3

    if kind == EPCAP:
        captured_value = SEE_VALUE[PAWN]
    else:
        captured = np.int64(state[ply, S_MAIL + target])
        captured_value = 0 if captured == EMPTY else SEE_VALUE[captured % 6]

    attacker = np.int64(state[ply, S_MAIL + origin])
    attacker_value = SEE_VALUE[attacker % 6]
    if kind == PROMO:
        promoted = ((move >> 12) & 3) + 1
        captured_value += SEE_VALUE[promoted] - SEE_VALUE[PAWN]
        attacker_value = SEE_VALUE[promoted]

    gains = ws.see_gains[ply]
    gains[0] = captured_value
    depth = 0

    occupancy = state[ply, S_OCC] & ~(ONE << U(origin))
    if kind == EPCAP:
        captured_square = target - 8 if state[ply, S_SIDE] == 0 else target + 8
        occupancy &= ~(ONE << U(captured_square))
    side = np.int64(state[ply, S_SIDE])

    attackers = attackers_to(state, ply, target, occupancy) & occupancy

    while True:
        depth += 1
        if depth >= 31:
            break
        side = 1 - side
        gains[depth] = attacker_value - gains[depth - 1]

        # cheapest attacker of the side now to move
        base = 6 * side
        found = -1
        for piece_type in range(6):
            candidates = state[ply, base + piece_type] & attackers & occupancy
            if candidates != ZERO:
                found = lsb(candidates)
                attacker_value = SEE_VALUE[piece_type]
                break
        if found < 0:
            break

        occupancy &= ~(ONE << U(found))
        # taking a piece off a line can reveal a slider behind it
        diagonal = state[ply, 2] | state[ply, 4] | state[ply, 8] | state[ply, 10]
        straight = state[ply, 3] | state[ply, 4] | state[ply, 9] | state[ply, 10]
        attackers |= bishop_attacks(target, occupancy) & diagonal
        attackers |= rook_attacks(target, occupancy) & straight
        attackers &= occupancy

    # walk back down: a side only enters the exchange if it comes out ahead
    while depth > 1:
        depth -= 1
        if gains[depth - 1] > -gains[depth]:
            gains[depth - 1] = -gains[depth]
    return gains[0]


# ---------------------------------------------------------------- draw detection
@njit(cache=CACHE)
def is_repetition(ws, state, ply, history_length):
    """A position seen before, either earlier in this game or higher up this search line.

    One repetition inside the tree is treated as a draw. That is standard: if a line can be
    repeated once it can usually be repeated twice, and waiting for the third occurrence
    costs depth for nothing.
    """
    key = state[ply, S_KEY]
    halfmove = np.int64(state[ply, S_HALF])
    index = history_length + ply - 2
    limit = history_length + ply - halfmove
    if limit < 0:
        limit = 0
    while index >= limit:
        if ws.repetition[index] == key:
            return True
        index -= 2
    return False


@njit(cache=CACHE)
def is_material_draw(state, ply):
    """King versus king, and king and a single minor versus king. Anything richer is left
    to the search."""
    if (state[ply, PAWN] | state[ply, 6 + PAWN]) != ZERO:
        return False
    if (state[ply, 3] | state[ply, 4] | state[ply, 9] | state[ply, 10]) != ZERO:
        return False
    minors = 0
    for index in (1, 2, 7, 8):
        piece = state[ply, index]
        while piece != ZERO:
            piece &= piece - ONE
            minors += 1
    return minors <= 1


# ---------------------------------------------------------------- move ordering
@njit(cache=CACHE)
def score_moves(ws, state, ply, count, tt_move, previous_move, quiescence):
    side = np.int64(state[ply, S_SIDE])
    for index in range(count):
        move = np.int64(ws.moves[ply, index])
        origin = move & 63
        target = (move >> 6) & 63
        kind = (move >> 14) & 3
        captured = np.int64(state[ply, S_MAIL + target])

        if move == tt_move:
            ws.order[ply, index] = 1 << 24
            continue

        if kind == PROMO and ((move >> 12) & 3) == 3:
            ws.order[ply, index] = (1 << 22) + 800
            continue

        if captured != EMPTY or kind == EPCAP:
            victim = SEE_VALUE[PAWN] if kind == EPCAP else SEE_VALUE[captured % 6]
            attacker = SEE_VALUE[np.int64(state[ply, S_MAIL + origin]) % 6]
            # most valuable victim, least valuable attacker; SEE only decides the sign,
            # because it is far more expensive than the subtraction
            if quiescence or victim >= attacker or see(ws, state, ply, move) >= 0:
                ws.order[ply, index] = (1 << 22) + victim * 8 - attacker
            else:
                ws.order[ply, index] = -(1 << 22) + victim * 8 - attacker
            continue

        if move == ws.killers[ply, 0]:
            ws.order[ply, index] = (1 << 21) + 200
        elif move == ws.killers[ply, 1]:
            ws.order[ply, index] = (1 << 21) + 100
        elif previous_move > 0 and move == ws.counters[
            side, previous_move & 63, (previous_move >> 6) & 63
        ]:
            ws.order[ply, index] = 1 << 21
        else:
            ws.order[ply, index] = ws.history[side, origin, target]


@njit(cache=CACHE)
def pick_move(ws, ply, index, count):
    """Selection sort, one step at a time. Most nodes cut off after a couple of moves, so
    sorting the whole list up front would be wasted work."""
    best = index
    for candidate in range(index + 1, count):
        if ws.order[ply, candidate] > ws.order[ply, best]:
            best = candidate
    if best != index:
        keep_order = ws.order[ply, index]
        ws.order[ply, index] = ws.order[ply, best]
        ws.order[ply, best] = keep_order
        keep_move = ws.moves[ply, index]
        ws.moves[ply, index] = ws.moves[ply, best]
        ws.moves[ply, best] = keep_move


# ---------------------------------------------------------------- quiescence
@njit(cache=CACHE)
def quiesce(ws, state, ply, alpha, beta, hard_deadline):
    ws.control[NODES] += 1
    if ply > ws.control[SELDEPTH]:
        ws.control[SELDEPTH] = ply
    # the cheap mask test runs first, so the clock is only actually read once every
    # 1024 nodes -- a couple of milliseconds apart at compiled speed
    if ws.control[NODES] & 1023 == 0 and (
        _now() > hard_deadline or ws.control[NODES] > ws.control[NODE_LIMIT]
    ):
        ws.control[ABORTED] = 1
        return 0
    if ws.control[ABORTED] == 1 or ply >= MAX_PLY - 2:
        return evaluate(state, ply)

    stand_pat = evaluate(state, ply)
    if stand_pat >= beta:
        return stand_pat
    if stand_pat > alpha:
        alpha = stand_pat

    count = gen_captures(state, ply, ws.moves)
    score_moves(ws, state, ply, count, np.int64(0), np.int64(0), True)

    best = stand_pat
    for index in range(count):
        pick_move(ws, ply, index, count)
        move = np.int64(ws.moves[ply, index])

        # delta pruning: a capture that cannot drag the score near alpha is not worth a node
        target = (move >> 6) & 63
        captured = np.int64(state[ply, S_MAIL + target])
        gain = SEE_VALUE[PAWN] if captured == EMPTY else SEE_VALUE[captured % 6]
        if ((move >> 14) & 3) == PROMO:
            gain += SEE_VALUE[QUEEN]
        if stand_pat + gain + 180 < alpha:
            continue
        if see(ws, state, ply, move) < 0:
            continue

        if not make_move(state, ply, move):
            continue
        score = -quiesce(ws, state, ply + 1, -beta, -alpha, hard_deadline)
        if ws.control[ABORTED] == 1:
            return 0
        if score > best:
            best = score
            if score > alpha:
                alpha = score
                if alpha >= beta:
                    break
    return best


# ---------------------------------------------------------------- main search
@njit(cache=CACHE)
def negamax(ws, state, ply, depth, alpha, beta, history_length, previous_move, hard_deadline):
    ws.control[NODES] += 1
    if ws.control[NODES] & 1023 == 0 and (
        _now() > hard_deadline or ws.control[NODES] > ws.control[NODE_LIMIT]
    ):
        ws.control[ABORTED] = 1
        return 0
    if ws.control[ABORTED] == 1:
        return 0

    root = ply == 0
    pv_node = beta - alpha > 1

    if not root:
        if np.int64(state[ply, S_HALF]) >= 100 or is_material_draw(state, ply):
            return 0
        if is_repetition(ws, state, ply, history_length):
            return 0
        # mate distance pruning: nothing found here can beat a mate already proven above
        if alpha < -MATE + ply:
            alpha = -MATE + ply
        if beta > MATE - ply - 1:
            beta = MATE - ply - 1
        if alpha >= beta:
            return alpha

    ws.repetition[history_length + ply] = state[ply, S_KEY]
    checked = in_check(state, ply)
    if checked:
        depth += 1  # a forced sequence is cheap to look at and expensive to guess about

    if depth <= 0 or ply >= MAX_PLY - 4:
        return quiesce(ws, state, ply, alpha, beta, hard_deadline)

    key = state[ply, S_KEY]
    hit, tt_move, tt_score, tt_depth, tt_flag = tt_probe(ws.tt, key)
    if hit:
        if tt_score > MATE_BOUND:
            tt_score -= ply
        elif tt_score < -MATE_BOUND:
            tt_score += ply
        if not pv_node and tt_depth >= depth:
            if tt_flag == EXACT:
                return tt_score
            if tt_flag == LOWER and tt_score >= beta:
                return tt_score
            if tt_flag == UPPER and tt_score <= alpha:
                return tt_score
    else:
        tt_move = 0

    static = evaluate(state, ply)

    if not pv_node and not checked:
        # reverse futility: so far ahead that giving away depth * a pawn still holds beta
        if depth <= 7 and static - FUTILITY[depth] >= beta and abs(beta) < MATE_BOUND:
            return static - FUTILITY[depth]

        # null move: hand the opponent a free move; if we are still winning, so is the
        # real move we have not looked at yet. Skipped without pieces, where zugzwang
        # makes the assumption false.
        side = np.int64(state[ply, S_SIDE])
        if (
            depth >= 3
            and static >= beta
            and has_non_pawn_material(state, ply, side)
            and previous_move != -1
        ):
            reduction = 3 + depth // 6
            make_null(state, ply)
            score = -negamax(
                ws,
                state,
                ply + 1,
                depth - reduction,
                -beta,
                -beta + 1,
                history_length,
                NULL_MOVE,
                hard_deadline,
            )
            if ws.control[ABORTED] == 1:
                return 0
            if score >= beta:
                return beta if abs(score) > MATE_BOUND else score

    count = gen_moves(state, ply, ws.moves)
    score_moves(ws, state, ply, count, tt_move, previous_move, False)

    best_score = -INFINITY
    best_move = 0
    played = 0
    flag = UPPER

    for index in range(count):
        pick_move(ws, ply, index, count)
        move = np.int64(ws.moves[ply, index])
        target = (move >> 6) & 63
        origin = move & 63
        kind = (move >> 14) & 3
        is_capture = np.int64(state[ply, S_MAIL + target]) != EMPTY or kind == EPCAP
        quiet = not is_capture and kind != PROMO

        if (
            not pv_node
            and not checked
            and quiet
            and played > 0
            and depth <= 7
            and best_score > -MATE_BOUND
        ):
            # futility: a quiet move this far below alpha is very unlikely to raise it
            if static + FUTILITY[depth] < alpha:
                continue
            # late move pruning: deep in a bad-looking node, stop trying quiet moves
            if played > 4 + depth * depth:
                continue

        if not make_move(state, ply, move):
            continue
        played += 1

        gives_check = in_check(state, ply + 1)
        new_depth = depth - 1

        if played == 1:
            score = -negamax(
                ws,
                state,
                ply + 1,
                new_depth,
                -beta,
                -alpha,
                history_length,
                move,
                hard_deadline,
            )
        else:
            reduction = 0
            if depth >= 3 and quiet and not checked and not gives_check:
                capped_depth = depth if depth < 63 else 63
                capped_played = played if played < 63 else 63
                reduction = LMR[capped_depth, capped_played]
                if pv_node:
                    reduction -= 1
                if ws.history[np.int64(state[ply, S_SIDE]), origin, target] > 4000:
                    reduction -= 1
                if reduction < 0:
                    reduction = 0
                if reduction > new_depth - 1:
                    reduction = new_depth - 1 if new_depth > 1 else 0

            score = -negamax(
                ws,
                state,
                ply + 1,
                new_depth - reduction,
                -alpha - 1,
                -alpha,
                history_length,
                move,
                hard_deadline,
            )
            # the reduced or null-window search beat alpha, so it has to be redone properly
            if score > alpha and (reduction > 0 or pv_node):
                score = -negamax(
                    ws,
                    state,
                    ply + 1,
                    new_depth,
                    -beta,
                    -alpha,
                    history_length,
                    move,
                    hard_deadline,
                )

        if ws.control[ABORTED] == 1:
            return 0

        if score > best_score:
            best_score = score
            best_move = move
            if root:
                ws.control[ROOT_MOVE] = move
            if score > alpha:
                alpha = score
                flag = EXACT
                if alpha >= beta:
                    if quiet:
                        if ws.killers[ply, 0] != move:
                            ws.killers[ply, 1] = ws.killers[ply, 0]
                            ws.killers[ply, 0] = move
                        side = np.int64(state[ply, S_SIDE])
                        ws.history[side, origin, target] += depth * depth
                        if ws.history[side, origin, target] > 1 << 20:
                            for a in range(64):
                                for b in range(64):
                                    ws.history[side, a, b] >>= 1
                        if previous_move > 0:
                            ws.counters[
                                side, previous_move & 63, (previous_move >> 6) & 63
                            ] = move
                    flag = LOWER
                    break

    if played == 0:
        return -MATE + ply if checked else 0

    tt_store(ws.tt, key, best_move, best_score, depth, flag, ply)
    return best_score


@njit(cache=CACHE)
def search(ws, state, history_length, max_depth, soft_deadline, hard_deadline, node_limit):
    """Iterative deepening with aspiration windows.

    Returns the best move, its score, the depth that completed, the nodes searched and how
    deep the quiescence reached.
    """
    ws.control[NODES] = 0
    ws.control[ABORTED] = 0
    ws.control[NODE_LIMIT] = node_limit
    ws.control[SELDEPTH] = 0
    ws.control[ROOT_MOVE] = 0  # a root move from the previous position must never leak in
    for ply in range(STACK_DEPTH):
        ws.killers[ply, 0] = 0
        ws.killers[ply, 1] = 0

    best_move = 0
    best_score = 0
    completed = 0

    # every one of these is an int64, never a compile-time literal, so negamax and everything
    # it calls compile exactly once instead of once per distinct constant
    root_ply = np.int64(0)
    no_move = np.int64(0)
    wide = np.int64(INFINITY)

    # a legal move to fall back on before any search has finished
    count = gen_moves(state, root_ply, ws.moves)
    for index in range(count):
        if make_move(state, root_ply, np.int64(ws.moves[0, index])):
            best_move = np.int64(ws.moves[0, index])
            break
    if best_move == 0:  # no legal move: mate or stalemate, the caller handles it
        return np.int64(0), np.int64(0), np.int64(0), ws.control[NODES], np.int64(0)

    alpha = np.int64(-INFINITY)
    beta = np.int64(INFINITY)
    for depth in range(1, max_depth + 1):
        if depth >= 5:
            window = 24
            while True:
                score = negamax(
                    ws, state, root_ply, depth, alpha, beta, history_length, no_move, hard_deadline
                )
                if ws.control[ABORTED] == 1:
                    break
                if score <= alpha:
                    alpha = alpha - window if alpha - window > -INFINITY else -INFINITY
                    window *= 3
                elif score >= beta:
                    beta = beta + window if beta + window < INFINITY else INFINITY
                    window *= 3
                else:
                    break
        else:
            score = negamax(
                ws, state, root_ply, depth, -wide, wide, history_length, no_move, hard_deadline
            )

        if ws.control[ABORTED] == 1:
            break

        if ws.control[ROOT_MOVE] != 0:
            best_move = ws.control[ROOT_MOVE]
        best_score = score
        completed = depth
        alpha = score - 24
        beta = score + 24

        if abs(score) > MATE_BOUND:  # a forced mate is not going to be improved on
            break
        if _now() > soft_deadline:
            break

    return best_move, best_score, completed, ws.control[NODES], ws.control[SELDEPTH]
