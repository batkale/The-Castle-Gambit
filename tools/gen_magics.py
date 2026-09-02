"""Search for magic multipliers for rook and bishop attack lookup.

Run once; the constants it prints are pasted into cg_magics.py. Keeping the search here
rather than at import means startup is instant and deterministic. The numbers are found by
random trial on this machine -- there is nothing borrowed about them.

    python tools/gen_magics.py > cg_magics.py
"""

import random

import numpy as np

U64 = np.uint64
FULL = (1 << 64) - 1

ROOK_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
BISHOP_DIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def rays(square: int, dirs: tuple[tuple[int, int], ...], blockers: int, stop_early: bool) -> int:
    """Squares attacked from `square`, stopping on (and including) the first blocker."""
    rank, file = divmod(square, 8)
    attacks = 0
    for dr, df in dirs:
        r, f = rank + dr, file + df
        while 0 <= r < 8 and 0 <= f < 8:
            # for the relevance mask we drop the final square of each ray: a blocker sitting
            # on the board edge never changes what lies beyond it
            if stop_early and not (0 <= r + dr < 8 and 0 <= f + df < 8):
                break
            bit = 1 << (r * 8 + f)
            attacks |= bit
            if not stop_early and blockers & bit:
                break
            r, f = r + dr, f + df
    return attacks


def subsets(mask: int) -> list[int]:
    """Every submask of `mask`, via the Carry-Rippler trick."""
    out, sub = [], 0
    while True:
        out.append(sub)
        sub = (sub - mask) & mask
        if sub == 0:
            return out


def find(
    square: int, dirs: tuple[tuple[int, int], ...], rng: random.Random
) -> tuple[int, int, int]:
    mask = rays(square, dirs, 0, True)
    bits = bin(mask).count("1")
    occupancies = subsets(mask)
    occs = np.array(occupancies, dtype=U64)
    atts = np.array([rays(square, dirs, occ, False) for occ in occupancies], dtype=U64)
    shift = U64(64 - bits)
    size = 1 << bits

    while True:
        # a sparse candidate (few set bits) spreads the index better than a dense one
        magic = U64(rng.getrandbits(64) & rng.getrandbits(64) & rng.getrandbits(64))
        with np.errstate(over="ignore"):
            index = (occs * magic) >> shift
        table = np.zeros(size, dtype=U64)
        table[index] = atts
        # a collision between two different attack sets shows up as a mismatch on read-back;
        # two occupancies sharing an attack set may safely share a slot
        if np.array_equal(table[index], atts):
            return int(magic), bits, size


def emit(name: str, dirs: tuple[tuple[int, int], ...], rng: random.Random) -> int:
    magics, bit_counts, offsets, total = [], [], [], 0
    for square in range(64):
        magic, bits, size = find(square, dirs, rng)
        magics.append(magic)
        bit_counts.append(bits)
        offsets.append(total)
        total += size
    print(f"{name}_MAGIC = np.array([")
    for i in range(0, 64, 4):
        print("    " + " ".join(f"0x{m:016X}," for m in magics[i : i + 4]))
    print("], dtype=np.uint64)")
    for label, data in (("BITS", bit_counts), ("OFFSET", offsets)):
        print(f"{name}_{label} = np.array([")
        for i in range(0, 64, 8):
            print("    " + " ".join(f"{v}," for v in data[i : i + 8]))
        print("], dtype=np.int64)")
    print(f"{name}_TABLE_SIZE = {total}")
    print()
    return total


def main() -> None:
    rng = random.Random(0x0C0FFEE)  # fixed seed: the same numbers come out every run
    print('"""Magic multipliers found by tools/gen_magics.py. Generated file, do not edit."""')
    print()
    print("import numpy as np")
    print()
    rook = emit("ROOK", ROOK_DIRS, rng)
    bishop = emit("BISHOP", BISHOP_DIRS, rng)
    print(f"# rook table {rook} entries, bishop table {bishop} entries", end="")
    print(f" -- {(rook + bishop) * 8 / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
