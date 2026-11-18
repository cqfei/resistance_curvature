"""Optional fused CUDA kernels for bounded-width sparse elimination.

One program owns a whole sparse column. Columns in the same elimination
level are independent; only their Schur updates can share destination slots.
The dense Schur complement is handled by PyTorch's Cholesky routines.
"""

import triton
import triton.language as tl


@triton.jit
def _factor_level(V, STARTS, PIVOTS, PAIR_STARTS, TARGETS, B: tl.constexpr):
    owner = tl.program_id(0)
    col = tl.load(PIVOTS + owner)
    start = tl.load(STARTS + col)
    end = tl.load(STARTS + col + 1)
    width = end - start - 1
    pos = tl.arange(0, B)
    diagonal = tl.sqrt(tl.load(V + start))
    entries = tl.load(V + start + 1 + pos, pos < width, 0) / diagonal
    i = pos[:, None]
    j = pos[None, :]
    mask = (i < width) & (j < width) & (i <= j)
    pair_start = tl.load(PAIR_STARTS + owner)
    pair = pair_start + i * width - i * (i + 1) // 2 + j
    target = tl.load(TARGETS + pair, mask, 0)
    update = -(entries[:, None] * entries[None, :])
    tl.atomic_add(V + target, update, mask, sem="relaxed")
    # Scalar diagonal loads can be replicated across warps. All warps must
    # have read the original pivot before any warp overwrites it with sqrt.
    tl.debug_barrier()
    tl.store(V + start + 1 + pos, entries, pos < width)
    tl.store(V + start, diagonal)


@triton.jit
def _selected_level(V, Z, STARTS, PIVOTS, PAIR_STARTS, TARGETS, B: tl.constexpr):
    owner = tl.program_id(0)
    col = tl.load(PIVOTS + owner)
    start = tl.load(STARTS + col)
    width = tl.load(STARTS + col + 1) - start - 1
    pos = tl.arange(0, B)
    diagonal = tl.load(V + start)
    unit = tl.load(V + start + 1 + pos, pos < width, 0) / diagonal
    i = pos[:, None]
    j = pos[None, :]
    first = tl.minimum(i, j)
    second = tl.maximum(i, j)
    pair = tl.load(PAIR_STARTS + owner) + first * width - first * (first + 1) // 2 + second
    mask = (i < width) & (j < width)
    target = tl.load(TARGETS + pair, mask, 0)
    inverse_block = tl.load(Z + target, mask, 0)
    off_diagonal = -tl.sum(inverse_block * unit[None, :], 1)
    inverse_diagonal = 1.0 / (diagonal * diagonal) - tl.sum(unit * off_diagonal, 0)
    tl.store(Z + start + 1 + pos, off_diagonal, pos < width)
    tl.store(Z + start, inverse_diagonal)


def factor_level(values, plan, level):
    _factor_level[(level.pivots.numel(),)](
        values, plan.column_starts, level.pivots, level.pair_starts,
        level.targets, B=triton.next_power_of_2(max(1, plan.degree_cap)), num_warps=4,
    )


def selected_level(values, selected, plan, level):
    _selected_level[(level.pivots.numel(),)](
        values, selected, plan.column_starts, level.pivots, level.pair_starts,
        level.targets, B=triton.next_power_of_2(max(1, plan.degree_cap)), num_warps=4,
    )
