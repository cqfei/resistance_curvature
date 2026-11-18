"""Exact sparse-prefix Cholesky with a dense Schur-complement tail.

Only columns with at most ``degree_cap`` active neighbours are eliminated
sparsely.  Remaining vertices are handled together with dense Cholesky.
This bounds retained sparse update indices without changing the matrix or
introducing a second approximation beyond the requested diagonal shift.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass, fields, replace
from heapq import heapify, heappop, heappush
from math import isfinite

import torch

try:
    from . import _hybrid_cuda
except (ImportError, ModuleNotFoundError):
    _hybrid_cuda = None


def _indices(values):
    if isinstance(values, array) and values:
        return torch.frombuffer(values, dtype=torch.int64)
    return torch.tensor(values, dtype=torch.int64)


def _move(record, device):
    return replace(record, **{
        field.name: value.to(device)
        for field in fields(record)
        if isinstance(value := getattr(record, field.name), torch.Tensor)
    })


@dataclass(frozen=True)
class _Level:
    pivots: torch.Tensor
    diagonals: torch.Tensor
    entries: torch.Tensor
    rows: torch.Tensor
    owners: torch.Tensor
    left: torch.Tensor
    right: torch.Tensor
    targets: torch.Tensor
    pair_starts: torch.Tensor

    def to(self, device):
        return _move(self, device)


@dataclass(frozen=True)
class HybridPlan:
    n: int
    permutation: torch.Tensor
    inverse_permutation: torch.Tensor
    edge_rows: torch.Tensor
    edge_cols: torch.Tensor
    edge_slots: torch.Tensor
    diagonal_slots: torch.Tensor
    column_starts: torch.Tensor
    schedule: tuple[_Level, ...]
    sparse_columns: int
    tail_offset: int
    tail_size: int
    degree_cap: int

    @property
    def nnz(self):
        """Number of allocated scalar slots, including the full dense tail."""
        return self.tail_offset + self.tail_size * self.tail_size

    @property
    def early_nnz(self):
        return self.tail_offset

    @property
    def levels(self):
        return len(self.schedule) + int(self.tail_size > 0)

    def to(self, device):
        moved = _move(self, device)
        return replace(moved, schedule=tuple(level.to(device) for level in self.schedule))


@dataclass(frozen=True)
class HybridFactor:
    plan: HybridPlan
    values: torch.Tensor
    info: torch.Tensor


def symbolic_cholesky(n, edge_rows, edge_cols, ordering="minimum_degree", *, degree_cap=32):
    """Prepare exact low-degree elimination and the remaining dense tail.

    Input contains one entry per undirected edge and no self loops.  The CPU
    plan may be reused while edge endpoints and their input order stay fixed.
    """
    if n < 1:
        raise ValueError("n must be positive")
    if ordering != "minimum_degree":
        raise ValueError("the hybrid path requires minimum_degree ordering")
    if not isinstance(degree_cap, int) or degree_cap < 0:
        raise ValueError("degree_cap must be a nonnegative integer")
    edge_rows = torch.as_tensor(edge_rows, dtype=torch.int64, device="cpu")
    edge_cols = torch.as_tensor(edge_cols, dtype=torch.int64, device="cpu")
    if edge_rows.ndim != 1 or edge_rows.shape != edge_cols.shape:
        raise ValueError("edge endpoints must be two equally sized vectors")
    row_nodes, col_nodes = edge_rows.tolist(), edge_cols.tolist()
    adjacency = [set() for _ in range(n)]
    for row, col in zip(row_nodes, col_nodes):
        if not (0 <= row < n and 0 <= col < n) or row == col:
            raise ValueError("edges must connect two distinct valid nodes")
        if col in adjacency[row]:
            raise ValueError("each undirected edge must appear once")
        adjacency[row].add(col)
        adjacency[col].add(row)

    queue = [(len(neighbors), node) for node, neighbors in enumerate(adjacency)]
    heapify(queue)
    alive = [True] * n
    permutation, original_columns = [], []
    while queue:
        degree, node = heappop(queue)
        if not alive[node] or degree != len(adjacency[node]):
            continue
        if degree > degree_cap:
            break
        alive[node] = False
        neighbors = adjacency[node]
        permutation.append(node)
        original_columns.append(tuple(neighbors))
        for neighbor in neighbors:
            # Touch at most degree_cap neighbours. Rebuilding this whole set
            # would copy a high-degree hub once for every eliminated leaf.
            updated = adjacency[neighbor]
            updated.discard(node)
            updated.update(neighbors)
            updated.discard(neighbor)
            heappush(queue, (len(updated), neighbor))
        adjacency[node] = set()

    sparse_columns = len(permutation)
    permutation.extend(node for node in range(n) if alive[node])
    inverse = [0] * n
    for index, node in enumerate(permutation):
        inverse[node] = index
    columns = [sorted(inverse[node] for node in column) for column in original_columns]
    tail_size = n - sparse_columns

    slots, diagonal_slots = [], []
    offset = 0
    for col, rows in enumerate(columns):
        diagonal_slots.append(offset)
        slots.append({row: offset + i for i, row in enumerate((col, *rows))})
        offset += len(rows) + 1
    tail_offset = offset
    column_starts = list(diagonal_slots) + [tail_offset]
    diagonal_slots.extend(tail_offset + i * (tail_size + 1) for i in range(tail_size))

    def lookup(row, col):
        lower, upper = min(row, col), max(row, col)
        if lower < sparse_columns:
            return slots[lower][upper]
        return tail_offset + (upper - sparse_columns) * tail_size + lower - sparse_columns

    permuted_rows = [inverse[row] for row in row_nodes]
    permuted_cols = [inverse[col] for col in col_nodes]
    edge_slots = [lookup(row, col) for row, col in zip(permuted_rows, permuted_cols)]

    heights = [0] * sparse_columns
    for column, rows in enumerate(columns):
        if rows and rows[0] < sparse_columns:
            parent = rows[0]
            heights[parent] = max(heights[parent], heights[column] + 1)
    groups = [[] for _ in range(max(heights) + 1)] if heights else []
    for column, height in enumerate(heights):
        groups[height].append(column)
    schedule = []
    for group in groups:
        diagonals, entries, rows, owners = [], [], [], []
        left, right, targets = array("q"), array("q"), array("q")
        pair_starts = []
        for owner, column in enumerate(group):
            neighbors = columns[column]
            start = len(entries)
            diagonals.append(slots[column][column])
            entries.extend(slots[column][row] for row in neighbors)
            rows.extend(neighbors)
            owners.extend([owner] * len(neighbors))
            pair_starts.append(len(targets))
            for first, col in enumerate(neighbors):
                for second in range(first, len(neighbors)):
                    left.append(start + second)
                    right.append(start + first)
                    targets.append(lookup(col, neighbors[second]))
        pair_starts.append(len(targets))
        schedule.append(_Level(*(
            _indices(values)
            for values in (group, diagonals, entries, rows, owners, left, right, targets, pair_starts)
        )))

    return HybridPlan(
        n=n, permutation=_indices(permutation), inverse_permutation=_indices(inverse),
        edge_rows=_indices(permuted_rows), edge_cols=_indices(permuted_cols),
        edge_slots=_indices(edge_slots), diagonal_slots=_indices(diagonal_slots),
        column_starts=_indices(column_starts),
        schedule=tuple(schedule), sparse_columns=sparse_columns, tail_offset=tail_offset,
        tail_size=tail_size, degree_cap=degree_cap,
    )


def factor_laplacian(plan, edge_weights, epsilon, *, check=True):
    """Factor P(L+epsilon I)P^T exactly using the sparse/dense split."""
    if edge_weights.ndim != 1 or edge_weights.shape != plan.edge_slots.shape:
        raise ValueError("edge_weights must have one value per symbolic edge")
    if edge_weights.device != plan.permutation.device:
        raise ValueError("move the symbolic plan to the edge weights' device first")
    if edge_weights.dtype not in (torch.float32, torch.float64):
        raise ValueError("edge weights must use float32 or float64")
    if not isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    degrees = edge_weights.new_zeros(plan.n)
    degrees.index_add_(0, plan.edge_rows, edge_weights)
    degrees.index_add_(0, plan.edge_cols, edge_weights)
    values = edge_weights.new_zeros(plan.nnz)
    values[plan.diagonal_slots] = degrees + epsilon
    values[plan.edge_slots] = -edge_weights
    for level in plan.schedule:
        if values.is_cuda and _hybrid_cuda is not None:
            _hybrid_cuda.factor_level(values, plan, level)
            continue
        diagonal = values[level.diagonals].sqrt()
        values[level.diagonals] = diagonal
        if level.entries.numel():
            entries = values[level.entries] / diagonal[level.owners]
            values[level.entries] = entries
            values.index_add_(0, level.targets, -(entries[level.left] * entries[level.right]))
    if plan.tail_size:
        tail = values[plan.tail_offset:].view(plan.tail_size, plan.tail_size)
        tail_factor, info = torch.linalg.cholesky_ex(tail, upper=False)
        tail.copy_(tail_factor)
    else:
        info = torch.zeros((), dtype=torch.int32, device=edge_weights.device)
    if check:
        diagonal = values[plan.diagonal_slots]
        if not bool((info == 0) & (diagonal.isfinite() & (diagonal > 0)).all()):
            raise RuntimeError("non-positive or non-finite hybrid Cholesky pivot")
    return HybridFactor(plan, values, info)


def edge_resistances(factor):
    """Get original-edge inverse entries by a dense tail and Takahashi sweep."""
    plan, values = factor.plan, factor.values
    selected = values.new_zeros(plan.nnz)
    if plan.tail_size:
        tail_factor = values[plan.tail_offset:].view(plan.tail_size, plan.tail_size)
        tail_inverse = torch.cholesky_inverse(tail_factor, upper=False)
        selected[plan.tail_offset:].copy_(tail_inverse.reshape(-1))
    for level in reversed(plan.schedule):
        if values.is_cuda and _hybrid_cuda is not None:
            _hybrid_cuda.selected_level(values, selected, plan, level)
            continue
        diagonal = values[level.diagonals]
        inverse_diagonal = diagonal.reciprocal().square()
        if level.entries.numel():
            unit_entries = values[level.entries] / diagonal[level.owners]
            inverse_block = selected[level.targets]
            dot_products = torch.zeros_like(unit_entries)
            dot_products.index_add_(0, level.left, inverse_block * unit_entries[level.right])
            mirror = torch.where(level.left == level.right, 0, inverse_block * unit_entries[level.left])
            dot_products.index_add_(0, level.right, mirror)
            off_diagonal = -dot_products
            selected[level.entries] = off_diagonal
            inverse_diagonal.index_add_(0, level.owners, -(unit_entries * off_diagonal))
        selected[level.diagonals] = inverse_diagonal
    diagonal = selected[plan.diagonal_slots]
    return diagonal[plan.edge_rows] + diagonal[plan.edge_cols] - 2 * selected[plan.edge_slots]
