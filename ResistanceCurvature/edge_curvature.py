"""Dense and sparse regularized resistance curvature on original graph edges.

Pass an undirected, loop-free graph as two int64 endpoint vectors and a
floating-point weight vector on the same device. List each positive-weight
edge once. Both functions return edge resistance, node curvature, edge
curvature in input order, and a scalar tensor indicating factorization success.
"""

import torch

if __package__:
    from .sparse_cholesky import edge_resistances, factor_laplacian
    from . import hybrid_cholesky
else:
    from sparse_cholesky import edge_resistances, factor_laplacian
    import hybrid_cholesky


def selected_curvature(resistance, rows, columns, weights, n):
    """Apply resistance curvature, minimum-resistance scaling, and leaf adjustment."""
    if not resistance.numel():
        return weights.new_ones(n), resistance

    contribution = resistance * weights
    weighted_resistance = weights.new_zeros(n)
    weighted_resistance.index_add_(0, rows, contribution)
    weighted_resistance.index_add_(0, columns, contribution)
    nodes = 1 - 0.5 * weighted_resistance

    reciprocal = torch.where(resistance != 0, resistance.reciprocal(), 0.0)
    minimum = torch.where(resistance > 0, resistance, float("inf")).amin()
    edges = 2 * (nodes[rows] + nodes[columns]) * reciprocal
    edges = edges * minimum / 2
    degree = torch.zeros(n, device=weights.device, dtype=torch.int64)
    present = (edges != 0).long()
    degree.index_add_(0, rows, present)
    degree.index_add_(0, columns, present)
    leaf_edge = (degree[rows] == 1) | (degree[columns] == 1)
    return nodes, torch.where(leaf_edge, edges.abs(), edges)


def dense_edge_curvature(n, rows, cols, weights, epsilon=1e-3):
    """Build the full shifted Laplacian and inverse before gathering edge entries."""
    shifted = weights.new_zeros((n, n))
    shifted[rows, cols] = shifted[cols, rows] = -weights
    degrees = weights.new_zeros(n)
    degrees.index_add_(0, rows, weights)
    degrees.index_add_(0, cols, weights)
    shifted.diagonal().copy_(degrees + epsilon)

    factor, info = torch.linalg.cholesky_ex(shifted)
    inverse = torch.cholesky_inverse(factor)
    diagonal = inverse.diagonal()
    resistance = (diagonal[rows] + diagonal[cols]) - (inverse[rows, cols] + inverse[cols, rows])
    nodes, edges = selected_curvature(resistance, rows, cols, weights, n)
    return resistance, nodes, edges, info == 0


def sparse_edge_curvature(plan, rows, cols, weights, epsilon=1e-3):
    """Use selected inversion with a plan for the same endpoints and edge order."""
    backend = hybrid_cholesky if isinstance(plan, hybrid_cholesky.HybridPlan) else None
    factor = (backend.factor_laplacian if backend else factor_laplacian)(plan, weights, epsilon, check=False)
    pivots = factor.values[plan.diagonal_slots]
    resistance = (backend.edge_resistances if backend else edge_resistances)(factor)
    nodes, edges = selected_curvature(resistance, rows, cols, weights, plan.n)
    valid = (pivots.isfinite() & (pivots > 0)).all()
    if backend:
        valid = valid & (factor.info == 0)
    return resistance, nodes, edges, valid
