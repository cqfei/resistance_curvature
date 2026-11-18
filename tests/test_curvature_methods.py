import unittest

import torch

from ResistanceCurvature import ResistanceCurvature
from ResistanceCurvature.resistance_curvature import ResistanceCurvature as ModuleCalculator


METHODS = ("dense_pinv", "dense_cholesky", "sparse_cholesky")


class CurvatureMethodsTest(unittest.TestCase):
    def setUp(self):
        self.calculator = ResistanceCurvature(device=torch.device("cpu"))
        self.n = 6
        self.rows = torch.tensor([0, 0, 1, 1, 2, 3, 4], dtype=torch.int64)
        self.cols = torch.tensor([1, 2, 2, 3, 4, 4, 5], dtype=torch.int64)
        self.weights = torch.tensor([1.0, 0.8, 1.2, 0.6, 0.9, 1.1, 0.7], dtype=torch.float64)

    def matrix(self, n, rows, cols, weights):
        result = weights.new_zeros((n, n))
        result[rows, cols] = result[cols, rows] = weights
        return result

    def assert_close(self, actual, expected):
        tolerance = dict(rtol=1e-8, atol=1e-9)
        if actual.dtype == torch.float32:
            tolerance = dict(rtol=5e-3, atol=5e-5)
        torch.testing.assert_close(actual, expected, **tolerance)

    def test_package_export(self):
        self.assertIs(ResistanceCurvature, ModuleCalculator)

    @torch.no_grad()
    def test_matrix_and_edge_interfaces_for_all_methods(self):
        for dtype in (torch.float64, torch.float32):
            weights = self.weights.to(dtype)
            adjacency = self.matrix(self.n, self.rows, self.cols, weights)
            for method in METHODS:
                with self.subTest(dtype=dtype, method=method):
                    node_matrix, edge_matrix = self.calculator.cal_curvature(adjacency, method=method)
                    node_vector, edge_vector = self.calculator.cal_curvature_edges(
                        self.n, self.rows, self.cols, weights, method=method
                    )
                    self.assertEqual(node_vector.shape, (self.n,))
                    self.assertEqual(edge_vector.shape, weights.shape)
                    self.assertEqual(node_vector.dtype, dtype)
                    self.assertEqual(edge_vector.dtype, dtype)
                    self.assert_close(node_vector, node_matrix)
                    self.assert_close(edge_vector, edge_matrix[self.rows, self.cols])
                    self.assertTrue(torch.equal(edge_matrix, edge_matrix.T))
                    self.assertFalse(bool(edge_matrix[adjacency == 0].any()))

    @torch.no_grad()
    def test_legacy_flags_and_explicit_method_override(self):
        adjacency = self.matrix(self.n, self.rows, self.cols, self.weights)
        for disturbed, method in ((False, "dense_pinv"), (True, "dense_cholesky")):
            with self.subTest(method=method):
                legacy = self.calculator.cal_curvature(adjacency, disturbed, 1e-3)
                explicit = self.calculator.cal_curvature(adjacency, method=method)
                overridden = self.calculator.cal_curvature(
                    adjacency, not disturbed, 1e-3, method=method
                )
                for expected, actual, override in zip(legacy, explicit, overridden):
                    self.assertTrue(torch.equal(expected, actual))
                    self.assertTrue(torch.equal(expected, override))
                for normalized in (False, True):
                    legacy_flow = self.calculator.cal_curvature_flow(
                        adjacency, 3, 0.05, normalized, torch.float64, disturbed, 1e-3
                    )
                    explicit_flow = self.calculator.cal_curvature_flow(
                        adjacency, 3, 0.05, normalized, torch.float64, method=method
                    )
                    self.assertTrue(torch.equal(legacy_flow, explicit_flow))

    @torch.no_grad()
    def test_sparse_dense_curvature_and_weights_after_each_flow_step(self):
        plan = self.calculator.prepare_sparse_plan(self.n, self.rows, self.cols)
        for dtype in (torch.float64, torch.float32):
            initial_weights = self.weights.to(dtype)
            initial_matrix = self.matrix(self.n, self.rows, self.cols, initial_weights)
            for normalized in (False, True):
                for steps in (1, 2, 3):
                    with self.subTest(dtype=dtype, normalized=normalized, steps=steps):
                        options = dict(n_iter=steps, step=0.05, flow_norm=normalized)
                        dense = self.calculator.cal_curvature_flow(
                            initial_matrix, dtype=dtype, method="dense_cholesky", **options
                        )
                        sparse_weights = self.calculator.cal_curvature_flow_edges(
                            self.n, self.rows, self.cols, initial_weights, plan=plan, **options
                        )
                        sparse_matrix = self.calculator.cal_curvature_flow(
                            initial_matrix, dtype=dtype, method="sparse_cholesky", **options
                        )
                        self.assert_close(sparse_weights, dense[self.rows, self.cols])
                        self.assert_close(sparse_matrix, dense)

                        # Each method evaluates curvature on its own updated weights.
                        dense_nodes, dense_edges = self.calculator.cal_curvature(
                            dense, method="dense_cholesky"
                        )
                        sparse_nodes, sparse_edges = self.calculator.cal_curvature_edges(
                            self.n, self.rows, self.cols, sparse_weights, plan=plan
                        )
                        self.assert_close(sparse_nodes, dense_nodes)
                        self.assert_close(sparse_edges, dense_edges[self.rows, self.cols])

    @torch.no_grad()
    def test_dense_edge_flow_matches_dense_matrix_flow(self):
        adjacency = self.matrix(self.n, self.rows, self.cols, self.weights)
        for method in ("dense_pinv", "dense_cholesky"):
            for normalized in (False, True):
                with self.subTest(method=method, normalized=normalized):
                    options = dict(n_iter=3, step=0.05, flow_norm=normalized, method=method)
                    matrix = self.calculator.cal_curvature_flow(
                        adjacency, dtype=torch.float64, **options
                    )
                    vector = self.calculator.cal_curvature_flow_edges(
                        self.n, self.rows, self.cols, self.weights, **options
                    )
                    self.assert_close(vector, matrix[self.rows, self.cols])

    @torch.no_grad()
    def test_disconnected_graph_and_isolated_node(self):
        rows = torch.tensor([0, 0, 1, 3], dtype=torch.int64)
        cols = torch.tensor([1, 2, 2, 4], dtype=torch.int64)
        weights = torch.tensor([1.0, 0.8, 1.2, 0.6], dtype=torch.float64)
        adjacency = self.matrix(6, rows, cols, weights)
        dense_nodes, dense_matrix = self.calculator.cal_curvature(
            adjacency, method="dense_cholesky"
        )
        for method in METHODS:
            with self.subTest(method=method):
                nodes, edges = self.calculator.cal_curvature_edges(
                    6, rows, cols, weights, method=method
                )
                self.assertEqual(nodes[-1].item(), 1.0)
                self.assertTrue(bool(nodes.isfinite().all()))
                self.assertTrue(bool(edges.isfinite().all()))
                if method == "sparse_cholesky":
                    self.assert_close(nodes, dense_nodes)
                    self.assert_close(edges, dense_matrix[rows, cols])

    @torch.no_grad()
    def test_empty_graph_and_zero_round_dtype(self):
        rows, cols, weights = self.rows[:0], self.cols[:0], self.weights[:0]
        adjacency = torch.zeros((3, 3), dtype=torch.float64)
        for method in METHODS:
            with self.subTest(method=method):
                nodes, edges = self.calculator.cal_curvature_edges(
                    3, rows, cols, weights, method=method
                )
                self.assertTrue(torch.equal(nodes, torch.ones(3, dtype=weights.dtype)))
                self.assertEqual(edges.numel(), 0)
                for normalized in (False, True):
                    flowed = self.calculator.cal_curvature_flow(
                        adjacency, n_iter=3, flow_norm=normalized, method=method
                    )
                    self.assertTrue(torch.equal(flowed, adjacency.float()))
                zero_round = self.calculator.cal_curvature_flow(
                    adjacency, n_iter=0, dtype=torch.float32, method=method
                )
                self.assertEqual(zero_round.dtype, torch.float32)
                self.assertTrue(torch.equal(zero_round, adjacency.float()))

    @torch.no_grad()
    def test_plan_reuse_and_mismatched_edge_order(self):
        plan = self.calculator.prepare_sparse_plan(self.n, self.rows, self.cols)
        changed_weights = self.weights * 1.2
        cached = self.calculator.cal_curvature_edges(
            self.n, self.rows, self.cols, changed_weights, plan=plan
        )
        fresh = self.calculator.cal_curvature_edges(
            self.n, self.rows, self.cols, changed_weights
        )
        for actual, expected in zip(cached, fresh):
            self.assert_close(actual, expected)
        with self.assertRaisesRegex(ValueError, "plan must match"):
            self.calculator.cal_curvature_edges(
                self.n, self.rows.flip(0), self.cols.flip(0), changed_weights.flip(0), plan=plan
            )
        with self.assertRaisesRegex(ValueError, "plan must match"):
            self.calculator.cal_curvature_edges(
                self.n + 1, self.rows, self.cols, changed_weights, plan=plan
            )

    @torch.no_grad()
    def test_natural_order_plan_remains_compatible(self):
        plan = self.calculator.prepare_sparse_plan(
            self.n, self.rows, self.cols, ordering="natural"
        )
        sparse = self.calculator.cal_curvature_edges(
            self.n, self.rows, self.cols, self.weights, plan=plan
        )
        dense = self.calculator.cal_curvature_edges(
            self.n, self.rows, self.cols, self.weights, method="dense_cholesky"
        )
        for actual, expected in zip(sparse, dense):
            self.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
