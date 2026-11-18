"""Regression checks for exact sparse-prefix / dense-tail elimination."""

import random
import unittest

import torch

from ResistanceCurvature import hybrid_cholesky as hybrid
from ResistanceCurvature.edge_curvature import dense_edge_curvature, selected_curvature
from ResistanceCurvature.resistance_curvature import _advance_flow


def tree_core_graph(n, components=4, core_size=20, seed=731):
    """Disconnected weighted cores with random tree attachments and some chords."""
    random_state = random.Random(seed)
    edges = set()
    for component in range(components):
        start = component * n // components
        stop = (component + 1) * n // components
        core_stop = min(start + core_size, stop)
        for row in range(start, core_stop):
            for col in range(row + 1, core_stop):
                edges.add((row, col))
        for node in range(core_stop, stop):
            parent = random_state.randrange(start, node)
            edges.add((parent, node))
            if node % 7 == 0:
                edges.add((random_state.randrange(start, node), node))
    ordered = sorted(edges)
    rows = torch.tensor([edge[0] for edge in ordered], dtype=torch.int64)
    cols = torch.tensor([edge[1] for edge in ordered], dtype=torch.int64)
    weights = torch.tensor([0.7 + 0.6 * random_state.random() for _ in ordered], dtype=torch.float64)
    return rows, cols, weights


def evaluate(plan, rows, cols, weights):
    factor = hybrid.factor_laplacian(plan, weights, 1e-3)
    resistance = hybrid.edge_resistances(factor)
    nodes, edges = selected_curvature(resistance, rows, cols, weights, plan.n)
    return factor, resistance, nodes, edges


class HybridCholeskyTest(unittest.TestCase):
    def assert_close(self, actual, expected):
        tolerance = dict(rtol=1e-8, atol=1e-9)
        if actual.dtype == torch.float32:
            tolerance = dict(rtol=5e-3, atol=5e-5)
        torch.testing.assert_close(actual, expected, **tolerance)

    @torch.no_grad()
    def test_exact_blocks_and_edge_resistances(self):
        rows, cols, initial = tree_core_graph(72, components=3, core_size=10)
        for dtype in (torch.float64, torch.float32):
            weights = initial.to(dtype)
            reference = dense_edge_curvature(72, rows, cols, weights, 1e-3)
            for cap in (0, 1, 4, 16):
                with self.subTest(dtype=dtype, degree_cap=cap):
                    plan = hybrid.symbolic_cholesky(72, rows, cols, degree_cap=cap).to("cpu")
                    factor, resistance, nodes, edges = evaluate(plan, rows, cols, weights)
                    for actual, expected in zip((resistance, nodes, edges), reference[:3]):
                        self.assert_close(actual, expected)

                    # Reconstruct the block factor, including cross-block entries.
                    lower = weights.new_zeros((72, 72))
                    for level in plan.schedule:
                        lower[level.pivots, level.pivots] = factor.values[level.diagonals]
                        lower[level.rows, level.pivots[level.owners]] = factor.values[level.entries]
                    if plan.tail_size:
                        lower[plan.sparse_columns:, plan.sparse_columns:] = factor.values[
                            plan.tail_offset:
                        ].view(plan.tail_size, plan.tail_size)
                    matrix = weights.new_zeros((72, 72))
                    matrix[rows, cols] = matrix[cols, rows] = -weights
                    matrix.diagonal().copy_(-matrix.sum(1) + 1e-3)
                    permuted = matrix[plan.permutation][:, plan.permutation]
                    self.assert_close(lower @ lower.T, permuted)
                    self.assertEqual(plan.sparse_columns + plan.tail_size, 72)

    def test_parallel_levels_read_and_write_disjoint_factor_slots(self):
        # A broad level is important: the original CUDA race was missed on
        # small graphs with only a few simultaneously active columns.
        rows, cols, _ = tree_core_graph(2708)
        for cap in (8, 16, 32):
            with self.subTest(degree_cap=cap):
                plan = hybrid.symbolic_cholesky(2708, rows, cols, degree_cap=cap)
                total_width = 0
                total_pairs = 0
                for level in plan.schedule:
                    sources = set(level.diagonals.tolist()) | set(level.entries.tolist())
                    self.assertFalse(sources & set(level.targets.tolist()))
                    self.assertEqual(level.pair_starts.numel(), level.pivots.numel() + 1)
                    self.assertEqual(level.pair_starts[-1].item(), level.targets.numel())
                    for owner, column in enumerate(level.pivots.tolist()):
                        width = int(plan.column_starts[column + 1] - plan.column_starts[column] - 1)
                        start = level.pair_starts[owner].item()
                        stop = level.pair_starts[owner + 1].item()
                        self.assertLessEqual(width, cap)
                        self.assertEqual(stop - start, width * (width + 1) // 2)
                        self.assertEqual(len(set(level.targets[start:stop].tolist())), stop - start)
                        total_width += width
                        total_pairs += stop - start
                self.assertLessEqual(2 * total_pairs, (cap + 1) * total_width)
                self.assertEqual(plan.column_starts.numel(), plan.sparse_columns + 1)

    @torch.no_grad()
    def test_reused_plan_flow_matches_dense_after_every_step(self):
        rows, cols, initial = tree_core_graph(60, components=2, core_size=10, seed=19)
        for dtype in (torch.float64, torch.float32):
            plan = hybrid.symbolic_cholesky(60, rows, cols, degree_cap=4)
            for normalized in (False, True):
                dense_weights = initial.to(dtype)
                sparse_weights = dense_weights.clone()
                for iteration in range(4):
                    with self.subTest(dtype=dtype, normalized=normalized, iteration=iteration):
                        _, resistance, nodes, edges = evaluate(plan, rows, cols, sparse_weights)
                        dense = dense_edge_curvature(60, rows, cols, dense_weights, 1e-3)
                        for actual, expected in zip((resistance, nodes, edges), dense[:3]):
                            self.assert_close(actual, expected)
                        self.assert_close(sparse_weights, dense_weights)
                        sparse_weights = _advance_flow(sparse_weights, edges, 0.025, normalized)
                        dense_weights = _advance_flow(dense_weights, dense[2], 0.025, normalized)

    @torch.no_grad()
    def test_empty_edges_and_isolated_vertices(self):
        rows = torch.empty(0, dtype=torch.int64)
        weights = torch.empty(0, dtype=torch.float64)
        for n in (1, 9):
            for cap in (0, 16):
                plan = hybrid.symbolic_cholesky(n, rows, rows, degree_cap=cap)
                factor, resistance, nodes, edges = evaluate(plan, rows, rows, weights)
                self.assertEqual(plan.tail_size, 0)
                self.assertEqual(resistance.numel(), 0)
                self.assertEqual(edges.numel(), 0)
                self.assertTrue(torch.equal(nodes, torch.ones(n, dtype=weights.dtype)))
                self.assert_close(factor.values, torch.full((n,), 1e-3 ** 0.5, dtype=weights.dtype))

    @unittest.skipUnless(torch.cuda.is_available() and hybrid._hybrid_cuda is not None,
                         "requires CUDA and the optional Triton kernels")
    @torch.no_grad()
    def test_cuda_broad_levels_repeated_factor_and_curvature(self):
        rows, cols, initial = tree_core_graph(2708)
        weights = initial.float()
        cpu_plan = hybrid.symbolic_cholesky(2708, rows, cols, degree_cap=16)
        cpu_factor, expected_r, expected_nodes, expected_edges = evaluate(cpu_plan, rows, cols, weights)
        plan = cpu_plan.to("cuda")
        cuda_rows, cuda_cols, cuda_weights = rows.cuda(), cols.cuda(), weights.cuda()
        expected = [value.cuda() for value in (cpu_factor.values, expected_r, expected_nodes, expected_edges)]
        for repeat in range(8):
            with self.subTest(repeat=repeat):
                factor, resistance, nodes, edges = evaluate(plan, cuda_rows, cuda_cols, cuda_weights)
                actual = (factor.values, resistance, nodes, edges)
                for value, reference in zip(actual, expected):
                    self.assert_close(value, reference)
                self.assertLess((factor.values - expected[0]).abs().max().item(), 2e-3)
                self.assertLess((resistance - expected[1]).abs().max().item(), 1e-3)
                self.assertLess((edges - expected[3]).abs().max().item(), 5e-3)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    @torch.no_grad()
    def test_cuda_flow_matches_dense(self):
        rows, cols, initial = tree_core_graph(120, components=3, core_size=12, seed=81)
        plan = hybrid.symbolic_cholesky(120, rows, cols, degree_cap=8).to("cuda")
        rows, cols = rows.cuda(), cols.cuda()
        for normalized in (False, True):
            dense_weights = initial.float().cuda()
            sparse_weights = dense_weights.clone()
            for iteration in range(4):
                with self.subTest(normalized=normalized, iteration=iteration):
                    _, resistance, nodes, edges = evaluate(plan, rows, cols, sparse_weights)
                    dense = dense_edge_curvature(120, rows, cols, dense_weights, 1e-3)
                    for actual, expected in zip((resistance, nodes, edges), dense[:3]):
                        self.assert_close(actual, expected)
                    self.assert_close(sparse_weights, dense_weights)
                    sparse_weights = _advance_flow(sparse_weights, edges, 0.025, normalized)
                    dense_weights = _advance_flow(dense_weights, dense[2], 0.025, normalized)


if __name__ == "__main__":
    unittest.main()
