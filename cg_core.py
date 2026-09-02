"""Bitboard position, move generation and make-move.

Everything here is written in the subset of Python numba compiles: flat numpy arrays, no
objects, no allocation inside the hot loops. Without numba the same code still runs, just
about a hundred times slower, which is what makes it testable on a machine that cannot load
the numba binaries.

Two conventions are load bearing:

* squares are indexed a1=0 .. h8=63, the same as python-chess, so moving a position across
  that boundary is a straight copy with no mirroring;
* every bitboard value is a numpy uint64 and every shift count is cast to uint64 too. Mixing
  a uint64 with a plain int is well defined under numpy 2 but numba promotes it to float64,
  so the casts are not decoration.

A position lives in one row of a state array, and make-move copies the row forward rather
than undoing in place. Copying 84 words is cheaper than getting unmake right, and unmake
bugs are the kind that surface as an illegal move in a rated game.
"""

import os

import numpy as np

try:
    from numba import njit

    HAVE_NUMBA = True
except Exception:  # pragma: no cover - the platform always has numba
    HAVE_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore[misc]
        """No-op stand-in so the engine still imports and runs as plain Python."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def decorate(function):
            return function

        return decorate


from cg_magics import (
    BISHOP_BITS,
    BISHOP_MAGIC,
    BISHOP_OFFSET,
    BISHOP_TABLE_SIZE,
    ROOK_BITS,
    ROOK_MAGIC,
    ROOK_OFFSET,
    ROOK_TABLE_SIZE,
)

# Persisting compiled code between processes looked like free init time and is not: with
# cache=True a warm start segfaults inside numba on this build, and the cache itself is 96 MB
# against a 256 MB scratch budget. Measured, rejected, left switchable so it can be retried
# on the platform's own numba without editing code. Do not turn this on without re-testing a
# warm start -- a segfault mid-game is a loss.
CACHE = os.environ.get("CG_NUMBA_CACHE", "0") == "1"

U = np.uint64
ONE = U(1)
ZERO = U(0)

WHITE, BLACK = 0, 1
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 0, 1, 2, 3, 4, 5
EMPTY = 12  # mailbox filler for an unoccupied square

# ---------------------------------------------------------------- state row layout
# 0..11  piece bitboards, indexed colour * 6 + piece type
S_OCC_W, S_OCC_B, S_OCC = 12, 13, 14
S_SIDE = 15  # 0 white to move, 1 black
S_CASTLE = 16  # bit 0 white O-O, 1 white O-O-O, 2 black O-O, 3 black O-O-O
S_EP = 17  # en passant target square, 64 when there is none
S_HALF = 18  # halfmove clock, for the fifty-move rule
S_KEY = 19  # zobrist hash
S_MAIL = 20  # 20..83, piece index per square or EMPTY
STATE_WIDTH = 84

NO_EP = U(64)

MAX_PLY = 128
MAX_MOVES = 256

# ---------------------------------------------------------------- move layout, 16 bits
# 0..5 from, 6..11 to, 12..13 promotion piece, 14..15 move kind
QUIET, PROMO, EPCAP, CASTLE = 0, 1, 2, 3


@njit(cache=CACHE)
def move_from(move):
    return move & 63


@njit(cache=CACHE)
def move_to(move):
    return (move >> 6) & 63


@njit(cache=CACHE)
def move_promo(move):
    """Promotion piece type, KNIGHT..QUEEN. Only meaningful when the kind is PROMO."""
    return ((move >> 12) & 3) + 1


@njit(cache=CACHE)
def move_kind(move):
    return (move >> 14) & 3


def encode_move(origin: int, target: int, promo_index: int = 0, kind: int = QUIET) -> int:
    return origin | (target << 6) | (promo_index << 12) | (kind << 14)


# ---------------------------------------------------------------- table construction
# This runs once at import in plain Python. It is a few hundred thousand operations, well
# inside the init budget, and keeping it out of numba saves compiling code that runs once.


def _mask64(value: int) -> int:
    return value & 0xFFFFFFFFFFFFFFFF


def _ray(square: int, directions, blockers: int, mask_only: bool) -> int:
    rank, file = divmod(square, 8)
    attacks = 0
    for dr, df in directions:
        r, f = rank + dr, file + df
        while 0 <= r < 8 and 0 <= f < 8:
            if mask_only and not (0 <= r + dr < 8 and 0 <= f + df < 8):
                break
            bit = 1 << (r * 8 + f)
            attacks |= bit
            if not mask_only and blockers & bit:
                break
            r, f = r + dr, f + df
    return attacks


_ROOK_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
_BISHOP_DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
_KNIGHT_DELTAS = ((2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1))
_KING_DELTAS = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))


def _leaper(square: int, deltas) -> int:
    rank, file = divmod(square, 8)
    attacks = 0
    for dr, df in deltas:
        r, f = rank + dr, file + df
        if 0 <= r < 8 and 0 <= f < 8:
            attacks |= 1 << (r * 8 + f)
    return attacks


def _submasks(mask: int):
    """Every submask of mask, via the Carry-Rippler trick."""
    sub = 0
    while True:
        yield sub
        sub = (sub - mask) & mask
        if sub == 0:
            return


KNIGHT_ATTACKS = np.zeros(64, dtype=U)
KING_ATTACKS = np.zeros(64, dtype=U)
PAWN_ATTACKS = np.zeros((2, 64), dtype=U)
ROOK_MASK = np.zeros(64, dtype=U)
BISHOP_MASK = np.zeros(64, dtype=U)
ROOK_TABLE = np.zeros(ROOK_TABLE_SIZE, dtype=U)
BISHOP_TABLE = np.zeros(BISHOP_TABLE_SIZE, dtype=U)
BETWEEN = np.zeros((64, 64), dtype=U)
FILE_BB = np.zeros(8, dtype=U)
RANK_BB = np.zeros(8, dtype=U)


def _build_tables() -> None:
    for square in range(64):
        KNIGHT_ATTACKS[square] = U(_leaper(square, _KNIGHT_DELTAS))
        KING_ATTACKS[square] = U(_leaper(square, _KING_DELTAS))
        PAWN_ATTACKS[WHITE][square] = U(_leaper(square, ((1, 1), (1, -1))))
        PAWN_ATTACKS[BLACK][square] = U(_leaper(square, ((-1, 1), (-1, -1))))

        rook_mask = _ray(square, _ROOK_DIRS, 0, True)
        bishop_mask = _ray(square, _BISHOP_DIRS, 0, True)
        ROOK_MASK[square] = U(rook_mask)
        BISHOP_MASK[square] = U(bishop_mask)

        rook_shift = 64 - int(ROOK_BITS[square])
        base = int(ROOK_OFFSET[square])
        magic = int(ROOK_MAGIC[square])
        for occ in _submasks(rook_mask):
            ROOK_TABLE[base + (_mask64(occ * magic) >> rook_shift)] = U(
                _ray(square, _ROOK_DIRS, occ, False)
            )

        bishop_shift = 64 - int(BISHOP_BITS[square])
        base = int(BISHOP_OFFSET[square])
        magic = int(BISHOP_MAGIC[square])
        for occ in _submasks(bishop_mask):
            BISHOP_TABLE[base + (_mask64(occ * magic) >> bishop_shift)] = U(
                _ray(square, _BISHOP_DIRS, occ, False)
            )

    for i in range(8):
        FILE_BB[i] = U(0x0101010101010101 << i)
        RANK_BB[i] = U(0xFF << (i * 8))

    # squares strictly between two aligned squares; zero for pairs that do not line up
    for a in range(64):
        ra, fa = divmod(a, 8)
        for b in range(64):
            rb, fb = divmod(b, 8)
            if a == b or not (ra == rb or fa == fb or abs(ra - rb) == abs(fa - fb)):
                continue
            dr = (rb > ra) - (rb < ra)
            df = (fb > fa) - (fb < fa)
            path, r, f = 0, ra + dr, fa + df
            while (r, f) != (rb, fb):
                path |= 1 << (r * 8 + f)
                r, f = r + dr, f + df
            BETWEEN[a][b] = U(path)


_build_tables()

# Shifts and magics live in uint64 arrays so numba never has to widen them mid-expression.
ROOK_SHIFT = (64 - ROOK_BITS).astype(U)
BISHOP_SHIFT = (64 - BISHOP_BITS).astype(U)
ROOK_MAGIC_U = ROOK_MAGIC.astype(U)
BISHOP_MAGIC_U = BISHOP_MAGIC.astype(U)
ROOK_BASE = ROOK_OFFSET.astype(np.int64)
BISHOP_BASE = BISHOP_OFFSET.astype(np.int64)

# ---------------------------------------------------------------- zobrist keys
_rng = np.random.default_rng(0x5EEDC4571E)
ZOB_PIECE = _rng.integers(0, 1 << 64, size=(12, 64), dtype=np.uint64)
ZOB_SIDE = U(int(_rng.integers(0, 1 << 64, dtype=np.uint64)))
ZOB_CASTLE = _rng.integers(0, 1 << 64, size=16, dtype=np.uint64)
ZOB_EP_FILE = _rng.integers(0, 1 << 64, size=9, dtype=np.uint64)  # index 8 means "no ep"


# ---------------------------------------------------------------- bit primitives
_DEBRUIJN = 0x03F79D71B4CB0A89
DEBRUIJN = U(_DEBRUIJN)
DEBRUIJN_INDEX = np.zeros(64, dtype=np.int64)
for _i in range(64):
    DEBRUIJN_INDEX[_mask64((1 << _i) * _DEBRUIJN) >> 58] = _i

_M1, _M2, _M4 = U(0x5555555555555555), U(0x3333333333333333), U(0x0F0F0F0F0F0F0F0F)
_H01 = U(0x0101010101010101)
_TWO, _FOUR, _EIGHT, _FIFTYSIX, _FIFTYEIGHT = U(2), U(4), U(8), U(56), U(58)


@njit(cache=CACHE)
def lsb(bb):
    """Index of the least significant set bit, by de Bruijn multiplication. Undefined for
    zero, which no caller passes."""
    return DEBRUIJN_INDEX[((bb & (ZERO - bb)) * DEBRUIJN) >> _FIFTYEIGHT]


@njit(cache=CACHE)
def popcount(bb):
    """SWAR population count: constant time, no loop for LLVM to unroll badly."""
    bb = bb - ((bb >> ONE) & _M1)
    bb = (bb & _M2) + ((bb >> _TWO) & _M2)
    bb = (bb + (bb >> _FOUR)) & _M4
    return np.int64((bb * _H01) >> _FIFTYSIX)


@njit(cache=CACHE)
def bishop_attacks(square, occupancy):
    index = ((occupancy & BISHOP_MASK[square]) * BISHOP_MAGIC_U[square]) >> BISHOP_SHIFT[square]
    return BISHOP_TABLE[BISHOP_BASE[square] + np.int64(index)]


@njit(cache=CACHE)
def rook_attacks(square, occupancy):
    index = ((occupancy & ROOK_MASK[square]) * ROOK_MAGIC_U[square]) >> ROOK_SHIFT[square]
    return ROOK_TABLE[ROOK_BASE[square] + np.int64(index)]


@njit(cache=CACHE)
def queen_attacks(square, occupancy):
    return bishop_attacks(square, occupancy) | rook_attacks(square, occupancy)


@njit(cache=CACHE)
def attackers_to(state, ply, square, occupancy):
    """Every piece of either colour that attacks `square` given `occupancy`."""
    bishops = state[ply, BISHOP] | state[ply, 6 + BISHOP]
    rooks = state[ply, ROOK] | state[ply, 6 + ROOK]
    queens = state[ply, QUEEN] | state[ply, 6 + QUEEN]
    bishops |= queens
    rooks |= queens
    return (
        (PAWN_ATTACKS[BLACK, square] & state[ply, PAWN])
        | (PAWN_ATTACKS[WHITE, square] & state[ply, 6 + PAWN])
        | (KNIGHT_ATTACKS[square] & (state[ply, KNIGHT] | state[ply, 6 + KNIGHT]))
        | (KING_ATTACKS[square] & (state[ply, KING] | state[ply, 6 + KING]))
        | (bishop_attacks(square, occupancy) & bishops)
        | (rook_attacks(square, occupancy) & rooks)
    )


@njit(cache=CACHE)
def is_attacked(state, ply, square, by_colour):
    """Whether `by_colour` attacks `square`. Cheaper than attackers_to when only a yes or no
    is wanted, because each family can bail out as soon as it hits."""
    base = 6 * by_colour
    occupancy = state[ply, S_OCC]
    if PAWN_ATTACKS[1 - by_colour, square] & state[ply, base + PAWN]:
        return True
    if KNIGHT_ATTACKS[square] & state[ply, base + KNIGHT]:
        return True
    if KING_ATTACKS[square] & state[ply, base + KING]:
        return True
    if bishop_attacks(square, occupancy) & (state[ply, base + BISHOP] | state[ply, base + QUEEN]):
        return True
    return (
        rook_attacks(square, occupancy)
        & (state[ply, base + ROOK] | state[ply, base + QUEEN])
    ) != ZERO


@njit(cache=CACHE)
def king_square(state, ply, colour):
    return lsb(state[ply, 6 * colour + KING])


@njit(cache=CACHE)
def in_check(state, ply):
    side = np.int64(state[ply, S_SIDE])
    return is_attacked(state, ply, king_square(state, ply, side), 1 - side)


# ---------------------------------------------------------------- move generation
@njit(cache=CACHE)
def castling_clear(state, ply, path, origin, transit, enemy):
    """Whether a castle is playable: the path empty, and neither the king square nor
    the square it steps over attacked. The destination is left to make_move, which
    rejects any move that leaves the king in check."""
    if (state[ply, S_OCC] & path) != ZERO:
        return False
    if is_attacked(state, ply, origin, enemy):
        return False
    return not is_attacked(state, ply, transit, enemy)


# Rights are cleared by touching a square: moving off it or capturing onto it. One table
# indexed by both the origin and the target covers king moves, rook moves and rook captures
# without a special case for any of them.
CASTLE_MASK = np.full(64, 15, dtype=np.int64)
CASTLE_MASK[0] = 13  # a1 rook: white loses O-O-O
CASTLE_MASK[7] = 14  # h1 rook: white loses O-O
CASTLE_MASK[4] = 12  # e1 king: white loses both
CASTLE_MASK[56] = 7  # a8 rook
CASTLE_MASK[63] = 11  # h8 rook
CASTLE_MASK[60] = 3  # e8 king


@njit(cache=CACHE)
def gen_moves(state, ply, moves):
    """Every pseudo-legal move, written into moves[ply]. Returns how many.

    Pseudo-legal means the king may be left in check; make_move rejects those. Castling is
    the exception, because its intermediate square cannot be tested after the fact, so the
    transit squares are checked here.
    """
    side = np.int64(state[ply, S_SIDE])
    base = 6 * side
    count = 0
    occupancy = state[ply, S_OCC]
    own = state[ply, S_OCC_W + side]
    enemy = state[ply, S_OCC_W + (1 - side)]
    empty = ~occupancy

    # ---- pawns
    pawns = state[ply, base + PAWN]
    if side == WHITE:
        step = 8
        single = (pawns << _EIGHT) & empty
        double = ((single & RANK_BB[2]) << _EIGHT) & empty
        last_rank = RANK_BB[7]
    else:
        step = -8
        single = (pawns >> _EIGHT) & empty
        double = ((single & RANK_BB[5]) >> _EIGHT) & empty
        last_rank = RANK_BB[0]

    promotions = single & last_rank
    quiet_pushes = single & ~last_rank
    while quiet_pushes != ZERO:
        target = lsb(quiet_pushes)
        quiet_pushes &= quiet_pushes - ONE
        moves[ply, count] = (target - step) | (target << 6)
        count += 1
    while promotions != ZERO:
        target = lsb(promotions)
        promotions &= promotions - ONE
        origin = target - step
        for promo in range(4):
            moves[ply, count] = origin | (target << 6) | (promo << 12) | (PROMO << 14)
            count += 1
    while double != ZERO:
        target = lsb(double)
        double &= double - ONE
        moves[ply, count] = (target - 2 * step) | (target << 6)
        count += 1

    workers = pawns
    while workers != ZERO:
        origin = lsb(workers)
        workers &= workers - ONE
        targets = PAWN_ATTACKS[side, origin] & enemy
        while targets != ZERO:
            target = lsb(targets)
            targets &= targets - ONE
            if (ONE << U(target)) & last_rank:
                for promo in range(4):
                    moves[ply, count] = origin | (target << 6) | (promo << 12) | (PROMO << 14)
                    count += 1
            else:
                moves[ply, count] = origin | (target << 6)
                count += 1

    ep_square = np.int64(state[ply, S_EP])
    if ep_square != 64:
        # pawns of ours that attack the target: the attack relation runs both ways, so the
        # enemy-coloured attack set of the target square finds them
        takers = PAWN_ATTACKS[1 - side, ep_square] & pawns
        while takers != ZERO:
            origin = lsb(takers)
            takers &= takers - ONE
            moves[ply, count] = origin | (ep_square << 6) | (EPCAP << 14)
            count += 1

    # ---- knights, then the three sliders, then the king
    workers = state[ply, base + KNIGHT]
    while workers != ZERO:
        origin = lsb(workers)
        workers &= workers - ONE
        targets = KNIGHT_ATTACKS[origin] & ~own
        while targets != ZERO:
            target = lsb(targets)
            targets &= targets - ONE
            moves[ply, count] = origin | (target << 6)
            count += 1

    workers = state[ply, base + BISHOP]
    while workers != ZERO:
        origin = lsb(workers)
        workers &= workers - ONE
        targets = bishop_attacks(origin, occupancy) & ~own
        while targets != ZERO:
            target = lsb(targets)
            targets &= targets - ONE
            moves[ply, count] = origin | (target << 6)
            count += 1

    workers = state[ply, base + ROOK]
    while workers != ZERO:
        origin = lsb(workers)
        workers &= workers - ONE
        targets = rook_attacks(origin, occupancy) & ~own
        while targets != ZERO:
            target = lsb(targets)
            targets &= targets - ONE
            moves[ply, count] = origin | (target << 6)
            count += 1

    workers = state[ply, base + QUEEN]
    while workers != ZERO:
        origin = lsb(workers)
        workers &= workers - ONE
        targets = queen_attacks(origin, occupancy) & ~own
        while targets != ZERO:
            target = lsb(targets)
            targets &= targets - ONE
            moves[ply, count] = origin | (target << 6)
            count += 1

    origin = king_square(state, ply, side)
    targets = KING_ATTACKS[origin] & ~own
    while targets != ZERO:
        target = lsb(targets)
        targets &= targets - ONE
        moves[ply, count] = origin | (target << 6)
        count += 1

    # ---- castling
    rights = np.int64(state[ply, S_CASTLE])
    if side == WHITE:
        if rights & 1 and castling_clear(state, ply, U(0x60), 4, 5, BLACK):
            moves[ply, count] = 4 | (6 << 6) | (CASTLE << 14)
            count += 1
        if rights & 2 and castling_clear(state, ply, U(0x0E), 4, 3, BLACK):
            moves[ply, count] = 4 | (2 << 6) | (CASTLE << 14)
            count += 1
    else:
        if rights & 4 and castling_clear(state, ply, U(0x6000000000000000), 60, 61, WHITE):
            moves[ply, count] = 60 | (62 << 6) | (CASTLE << 14)
            count += 1
        if rights & 8 and castling_clear(state, ply, U(0x0E00000000000000), 60, 59, WHITE):
            moves[ply, count] = 60 | (58 << 6) | (CASTLE << 14)
            count += 1

    return count


@njit(cache=CACHE)
def gen_captures(state, ply, moves):
    """Captures, en passant and queen promotions only: the quiescence move set."""
    side = np.int64(state[ply, S_SIDE])
    base = 6 * side
    count = 0
    occupancy = state[ply, S_OCC]
    own = state[ply, S_OCC_W + side]
    enemy = state[ply, S_OCC_W + (1 - side)]

    pawns = state[ply, base + PAWN]
    if side == WHITE:
        step = 8
        pushes = (pawns << _EIGHT) & ~occupancy & RANK_BB[7]
        last_rank = RANK_BB[7]
    else:
        step = -8
        pushes = (pawns >> _EIGHT) & ~occupancy & RANK_BB[0]
        last_rank = RANK_BB[0]

    while pushes != ZERO:  # promotions are captures of a sort: they change material
        target = lsb(pushes)
        pushes &= pushes - ONE
        moves[ply, count] = (target - step) | (target << 6) | (3 << 12) | (PROMO << 14)
        count += 1

    workers = pawns
    while workers != ZERO:
        origin = lsb(workers)
        workers &= workers - ONE
        targets = PAWN_ATTACKS[side, origin] & enemy
        while targets != ZERO:
            target = lsb(targets)
            targets &= targets - ONE
            if (ONE << U(target)) & last_rank:
                moves[ply, count] = origin | (target << 6) | (3 << 12) | (PROMO << 14)
            else:
                moves[ply, count] = origin | (target << 6)
            count += 1

    ep_square = np.int64(state[ply, S_EP])
    if ep_square != 64:
        takers = PAWN_ATTACKS[1 - side, ep_square] & pawns
        while takers != ZERO:
            origin = lsb(takers)
            takers &= takers - ONE
            moves[ply, count] = origin | (ep_square << 6) | (EPCAP << 14)
            count += 1

    for piece_type in range(KNIGHT, KING + 1):
        workers = state[ply, base + piece_type]
        while workers != ZERO:
            origin = lsb(workers)
            workers &= workers - ONE
            if piece_type == KNIGHT:
                targets = KNIGHT_ATTACKS[origin]
            elif piece_type == BISHOP:
                targets = bishop_attacks(origin, occupancy)
            elif piece_type == ROOK:
                targets = rook_attacks(origin, occupancy)
            elif piece_type == QUEEN:
                targets = queen_attacks(origin, occupancy)
            else:
                targets = KING_ATTACKS[origin]
            targets &= enemy & ~own
            while targets != ZERO:
                target = lsb(targets)
                targets &= targets - ONE
                moves[ply, count] = origin | (target << 6)
                count += 1

    return count


@njit(cache=CACHE)
def make_move(state, ply, move):
    """Apply `move` to the position at `ply`, writing the result at `ply + 1`.

    Returns False when the move leaves the mover in check, in which case the row at ply + 1
    is garbage and the caller simply moves on to the next move.
    """
    for i in range(STATE_WIDTH):
        state[ply + 1, i] = state[ply, i]

    side = np.int64(state[ply, S_SIDE])
    opponent = 1 - side
    origin = move_from(move)
    target = move_to(move)
    kind = move_kind(move)
    piece = np.int64(state[ply, S_MAIL + origin])
    piece_type = piece - 6 * side
    key = state[ply, S_KEY]
    halfmove = np.int64(state[ply, S_HALF]) + 1

    previous_ep = np.int64(state[ply, S_EP])
    if previous_ep == 64:
        key ^= ZOB_EP_FILE[8]
    else:
        key ^= ZOB_EP_FILE[previous_ep & 7]

    state[ply + 1, piece] &= ~(ONE << U(origin))
    state[ply + 1, S_MAIL + origin] = EMPTY
    key ^= ZOB_PIECE[piece, origin]

    if kind == EPCAP:
        captured_square = target - 8 if side == WHITE else target + 8
        captured = np.int64(state[ply, S_MAIL + captured_square])
        state[ply + 1, captured] &= ~(ONE << U(captured_square))
        state[ply + 1, S_MAIL + captured_square] = EMPTY
        key ^= ZOB_PIECE[captured, captured_square]
        halfmove = 0
    else:
        captured = np.int64(state[ply, S_MAIL + target])
        if captured != EMPTY:
            state[ply + 1, captured] &= ~(ONE << U(target))
            key ^= ZOB_PIECE[captured, target]
            halfmove = 0

    landing = piece
    if kind == PROMO:
        landing = 6 * side + move_promo(move)
    state[ply + 1, landing] |= ONE << U(target)
    state[ply + 1, S_MAIL + target] = landing
    key ^= ZOB_PIECE[landing, target]

    if piece_type == PAWN:
        halfmove = 0

    if kind == CASTLE:
        if target == 6:
            rook_origin, rook_target = 7, 5
        elif target == 2:
            rook_origin, rook_target = 0, 3
        elif target == 62:
            rook_origin, rook_target = 63, 61
        else:
            rook_origin, rook_target = 56, 59
        rook = 6 * side + ROOK
        state[ply + 1, rook] &= ~(ONE << U(rook_origin))
        state[ply + 1, rook] |= ONE << U(rook_target)
        state[ply + 1, S_MAIL + rook_origin] = EMPTY
        state[ply + 1, S_MAIL + rook_target] = rook
        key ^= ZOB_PIECE[rook, rook_origin] ^ ZOB_PIECE[rook, rook_target]

    previous_rights = np.int64(state[ply, S_CASTLE])
    rights = previous_rights & CASTLE_MASK[origin] & CASTLE_MASK[target]
    key ^= ZOB_CASTLE[previous_rights] ^ ZOB_CASTLE[rights]
    state[ply + 1, S_CASTLE] = rights

    next_ep = 64
    if piece_type == PAWN and (target - origin == 16 or origin - target == 16):
        next_ep = (target + origin) >> 1
        key ^= ZOB_EP_FILE[next_ep & 7]
    else:
        key ^= ZOB_EP_FILE[8]
    state[ply + 1, S_EP] = next_ep

    state[ply + 1, S_HALF] = halfmove
    state[ply + 1, S_SIDE] = opponent
    key ^= ZOB_SIDE
    state[ply + 1, S_KEY] = key

    white = ZERO
    black = ZERO
    for i in range(6):
        white |= state[ply + 1, i]
        black |= state[ply + 1, 6 + i]
    state[ply + 1, S_OCC_W] = white
    state[ply + 1, S_OCC_B] = black
    state[ply + 1, S_OCC] = white | black

    return not is_attacked(state, ply + 1, king_square(state, ply + 1, side), opponent)


@njit(cache=CACHE)
def make_null(state, ply):
    """Pass the move to the opponent. Used by null-move pruning, never when in check."""
    for i in range(STATE_WIDTH):
        state[ply + 1, i] = state[ply, i]
    key = state[ply, S_KEY] ^ ZOB_SIDE
    previous_ep = np.int64(state[ply, S_EP])
    if previous_ep == 64:
        key ^= ZOB_EP_FILE[8] ^ ZOB_EP_FILE[8]
    else:
        key ^= ZOB_EP_FILE[previous_ep & 7] ^ ZOB_EP_FILE[8]
    state[ply + 1, S_EP] = 64
    state[ply + 1, S_SIDE] = 1 - np.int64(state[ply, S_SIDE])
    state[ply + 1, S_HALF] = np.int64(state[ply, S_HALF]) + 1
    state[ply + 1, S_KEY] = key


@njit(cache=CACHE)
def compute_key(state, ply):
    """Zobrist key from scratch. make_move keeps one incrementally; this is what the tests
    check it against."""
    key = ZERO
    for square in range(64):
        piece = np.int64(state[ply, S_MAIL + square])
        if piece != EMPTY:
            key ^= ZOB_PIECE[piece, square]
    key ^= ZOB_CASTLE[np.int64(state[ply, S_CASTLE])]
    ep_square = np.int64(state[ply, S_EP])
    if ep_square == 64:
        key ^= ZOB_EP_FILE[8]
    else:
        key ^= ZOB_EP_FILE[ep_square & 7]
    if state[ply, S_SIDE] == BLACK:
        key ^= ZOB_SIDE
    return key


def new_state() -> np.ndarray:
    return np.zeros((MAX_PLY + 8, STATE_WIDTH), dtype=U)


def new_move_buffer() -> np.ndarray:
    return np.zeros((MAX_PLY + 8, MAX_MOVES), dtype=np.int32)


# ---------------------------------------------------------------- FEN and UCI
# Called once per move rather than once per node, so plain Python is fast enough and the
# engine stays independent of python-chess for anything on the hot path.

SQUARE_NAMES = tuple(f"{chr(97 + f)}{r + 1}" for r in range(8) for f in range(8))
_PIECE_LETTERS = "pnbrqk"


def load_fen(state: np.ndarray, ply: int, fen: str) -> None:
    state[ply, :] = 0
    for square in range(64):
        state[ply, S_MAIL + square] = EMPTY

    fields = fen.split()
    for row_index, row in enumerate(fields[0].split("/")):
        rank = 7 - row_index
        file = 0
        for character in row:
            if character.isdigit():
                file += int(character)
                continue
            colour = WHITE if character.isupper() else BLACK
            piece = 6 * colour + _PIECE_LETTERS.index(character.lower())
            square = rank * 8 + file
            state[ply, piece] |= ONE << U(square)
            state[ply, S_MAIL + square] = piece
            file += 1

    state[ply, S_SIDE] = WHITE if fields[1] == "w" else BLACK
    rights = 0
    for bit, flag in enumerate("KQkq"):
        if flag in fields[2]:
            rights |= 1 << bit
    state[ply, S_CASTLE] = rights
    if len(fields) > 3 and fields[3] != "-":
        state[ply, S_EP] = (ord(fields[3][0]) - 97) + 8 * (int(fields[3][1]) - 1)
    else:
        state[ply, S_EP] = 64
    state[ply, S_HALF] = int(fields[4]) if len(fields) > 4 else 0

    white = ZERO
    black = ZERO
    for i in range(6):
        white |= state[ply, i]
        black |= state[ply, 6 + i]
    state[ply, S_OCC_W] = white
    state[ply, S_OCC_B] = black
    state[ply, S_OCC] = white | black
    state[ply, S_KEY] = compute_key(state, ply)


def move_to_uci(move: int) -> str:
    text = SQUARE_NAMES[move & 63] + SQUARE_NAMES[(move >> 6) & 63]
    if ((move >> 14) & 3) == PROMO:
        text += "nbrq"[(move >> 12) & 3]
    return text


def legal_moves(state: np.ndarray, ply: int, moves: np.ndarray) -> list[int]:
    """Filtered move list for the boundary code. The search never calls this: inside the
    tree, generation and legality are fused so a rejected move costs nothing extra."""
    total = gen_moves(state, ply, moves)
    return [int(moves[ply, i]) for i in range(total) if make_move(state, ply, int(moves[ply, i]))]

