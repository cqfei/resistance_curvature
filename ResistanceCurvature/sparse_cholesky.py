"""Fill-aware graph Laplacian Cholesky with PyTorch CPU or CUDA operations.

The reusable symbolic plan contains only structural factor entries and
Schur-update indices. Numeric factorization and selected inversion never
allocate an n-by-n matrix. Full inversion explicitly produces that matrix.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass, fields, replace
from heapq import heapify, heappop, heappush
from math import isfinite

import torch


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

    def to(self, device):
        return _move_tensors(self, device)


def _move_tensors(record, device):
    tensors = {
        field.name: value.to(device)
        for field in fields(record)
        if isinstance(value := getattr(record, field.name), torch.Tensor)
    }
    return replace(record, **tensors)


@dataclass(frozen=True)
class SparseCholeskyPlan:
    """Structure in elimination order, with input-edge lookup indices."""

    n: int
    permutation: torch.Tensor
    inverse_permutation: torch.Tensor
    edge_rows: torch.Tensor
    edge_cols: torch.Tensor
    edge_slots: torch.Tensor
    diagonal_slots: torch.Tensor
    factor_coordinates: torch.Tensor
    schedule: tuple[_Level, ...]

    @property
    def nnz(self):
        return self.factor_coordinates.shape[1]

    @property
    def levels(self):
        return len(self.schedule)

    def to(self, device):
        moved = _move_tensors(self, device)
        return replace(moved, schedule=tuple(level.to(device) for level in self.schedule))


@dataclass(frozen=True)
class SparseCholeskyFactor:
    """Packed lower triangular factor C of P(L + epsilon I)P^T."""

    plan: SparseCholeskyPlan
    values: torch.Tensor

    def to_sparse_csr(self):
        return torch.sparse_coo_tensor(
            self.plan.factor_coordinates,
            self.values,
            (self.plan.n, self.plan.n),
            device=self.values.device,
        ).coalesce().to_sparse_csr()


def _elimination(adjacency, ordering):
    if ordering == "minimum_degree":
        queue = [(len(neighbors), node) for node, neighbors in enumerate(adjacency)]
        heapify(queue)
    alive = [True] * len(adjacency)
    permutation = []
    columns = []
    for step in range(len(adjacency)):
        node = step
        if ordering == "minimum_degree":
            degree, node = heappop(queue)
            while not alive[node] or degree != len(adjacency[node]):
                degree, node = heappop(queue)
        alive[node] = False
        neighbors = adjacency[node]
        permutation.append(node)
        columns.append(tuple(neighbors))
        for neighbor in neighbors:
            updated = (adjacency[neighbor] | neighbors) - {node, neighbor}
            adjacency[neighbor] = updated
            if ordering == "minimum_degree":
                heappush(queue, (len(updated), neighbor))
    inverse = [0] * len(permutation)
    for index, node in enumerate(permutation):
        inverse[node] = index
    return permutation, inverse, [sorted(inverse[node] for node in column) for column in columns]


def _index_tensor(values):
    if isinstance(values, array) and values:
        return torch.frombuffer(values, dtype=torch.int64)
    return torch.tensor(values, dtype=torch.int64)


def _build_schedule(columns, slots):
    heights = [0] * len(columns)
    for column, rows in enumerate(columns):
        if rows:
            parent = rows[0]
            heights[parent] = max(heights[parent], heights[column] + 1)
    groups = [[] for _ in range(max(heights) + 1)]
    for column, height in enumerate(heights):
        groups[height].append(column)

    schedule = []
    for group in groups:
        diagonals, entries, rows, owners = [], [], [], []
        left, right, targets = array("q"), array("q"), array("q")
        for owner, column in enumerate(group):
            neighbors = columns[column]
            start = len(entries)
            diagonals.append(slots[column][column])
            entries.extend(slots[column][row] for row in neighbors)
            rows.extend(neighbors)
            owners.extend([owner] * len(neighbors))
            for first, col in enumerate(neighbors):
                target_slots = slots[col]
                for second in range(first, len(neighbors)):
                    left.append(start + second)
                    right.append(start + first)
                    targets.append(target_slots[neighbors[second]])
        schedule.append(_Level(*(
            _index_tensor(values)
            for values in (group, diagonals, entries, rows, owners, left, right, targets)
        )))
    return tuple(schedule)


def symbolic_cholesky(n, edge_rows, edge_cols, ordering="minimum_degree"):
    """Analyze a graph with one entry per undirected edge and no self-loops.

    The analysis runs on CPU. Call plan.to(device) before numeric work. Treat
    all supplied edges as structural nonzeros. The plan may be reused while
    the structural edge list and ordering remain fixed.
    """
    if n < 1:
        raise ValueError("n must be positive")
    if ordering not in {"natural", "minimum_degree"}:
        raise ValueError("ordering must be 'natural' or 'minimum_degree'")
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

    permutation, inverse, columns = _elimination(adjacency, ordering)
    slots = []
    diagonal_slots, factor_rows, factor_cols = [], [], []
    for col, rows in enumerate(columns):
        start = len(factor_rows)
        diagonal_slots.append(start)
        factor_rows.extend((col, *rows))
        factor_cols.extend([col] * (len(rows) + 1))
        slots.append({row: start + i for i, row in enumerate((col, *rows))})

    permuted_rows = [inverse[row] for row in row_nodes]
    permuted_cols = [inverse[col] for col in col_nodes]
    edge_slots = [
        slots[min(row, col)][max(row, col)]
        for row, col in zip(permuted_rows, permuted_cols)
    ]
    return SparseCholeskyPlan(
        n=n,
        permutation=_index_tensor(permutation),
        inverse_permutation=_index_tensor(inverse),
        edge_rows=_index_tensor(permuted_rows),
        edge_cols=_index_tensor(permuted_cols),
        edge_slots=_index_tensor(edge_slots),
        diagonal_slots=_index_tensor(diagonal_slots),
        factor_coordinates=torch.tensor([factor_rows, factor_cols], dtype=torch.int64),
        schedule=_build_schedule(columns, slots),
    )


def factor_laplacian(plan, edge_weights, epsilon, *, check=True):
    """Factor the shifted Laplacian of the nonnegative undirected weights."""
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
        diagonal = values[level.diagonals].sqrt()
        values[level.diagonals] = diagonal
        if level.entries.numel():
            entries = values[level.entries] / diagonal[level.owners]
            values[level.entries] = entries
            values.index_add_(0, level.targets, -(entries[level.left] * entries[level.right]))
    if check:
        diagonal = values[plan.diagonal_slots]
        if not bool((diagonal.isfinite() & (diagonal > 0)).all()):
            raise RuntimeError("non-positive or non-finite sparse Cholesky pivot")
    return SparseCholeskyFactor(plan, values)


def full_inverse(factor, *, scratch_elements=2_000_000):
    """Solve all unit RHSs with sparse triangular sweeps, then unpermute."""
    plan, values = factor.plan, factor.values
    if scratch_elements < 1:
        raise ValueError("scratch_elements must be positive")
    largest_level = max(max(level.entries.numel(), level.pivots.numel()) for level in plan.schedule)
    block_size = max(1, scratch_elements // largest_level)
    result = values.new_empty((plan.n, plan.n))
    # Blocking RHSs limits the size of edge-by-RHS gathers on a small GPU.
    for start in range(0, plan.n, block_size):
        stop = min(start + block_size, plan.n)
        rhs = values.new_zeros((plan.n, stop - start))
        original_columns = torch.arange(start, stop, device=values.device)
        rhs[plan.inverse_permutation[original_columns], original_columns - start] = 1
        for level in plan.schedule:
            diagonal = values[level.diagonals, None]
            solved = rhs[level.pivots] / diagonal
            rhs[level.pivots] = solved
            if level.entries.numel():
                updates = values[level.entries, None] * solved[level.owners]
                rhs.index_add_(0, level.rows, -updates)
        for level in reversed(plan.schedule):
            solved = rhs[level.pivots]
            if level.entries.numel():
                updates = values[level.entries, None] * rhs[level.rows]
                solved.index_add_(0, level.owners, -updates)
            rhs[level.pivots] = solved / values[level.diagonals, None]
        result[:, start:stop] = rhs[plan.inverse_permutation]
    return result


def edge_resistances(factor):
    """Compute shifted effective resistances by selected inversion."""
    plan, values = factor.plan, factor.values
    selected = values.new_zeros(plan.nnz)
    for level in reversed(plan.schedule):
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
    return (diagonal[plan.edge_rows] + diagonal[plan.edge_cols]) - 2 * selected[plan.edge_slots]


def full_inverse_cuda_csr(factor):
    """Form the full inverse using PyTorch's CUDA CSR triangular solves.

    Requires the SparseCsrCUDA dispatch of torch.triangular_solve, verified
    with PyTorch 2.9.1. Use full_inverse for the portable implementation.
    """
    plan, values = factor.plan, factor.values
    if values.device.type != "cuda":
        raise ValueError("the CSR triangular solve path requires CUDA")
    lower = factor.to_sparse_csr()
    rhs = values.new_zeros((plan.n, plan.n))
    rhs[torch.arange(plan.n, device=values.device), plan.permutation] = 1
    forward = torch.triangular_solve(rhs, lower, upper=False).solution
    backward = torch.triangular_solve(forward, lower, upper=False, transpose=True).solution
    return backward[plan.inverse_permutation]
