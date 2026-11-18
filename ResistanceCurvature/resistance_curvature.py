import torch
from typing import Tuple

if __package__:
    from .edge_curvature import sparse_edge_curvature
    from .hybrid_cholesky import HybridPlan as SparseCholeskyPlan, symbolic_cholesky
    from .sparse_cholesky import symbolic_cholesky as _legacy_symbolic_cholesky
else:
    from edge_curvature import sparse_edge_curvature
    from hybrid_cholesky import HybridPlan as SparseCholeskyPlan, symbolic_cholesky
    from sparse_cholesky import symbolic_cholesky as _legacy_symbolic_cholesky


def _resolve_method(method, use_disturb):
    if method is None:
        return "dense_cholesky" if use_disturb else "dense_pinv"
    if method not in ("dense_pinv", "dense_cholesky", "sparse_cholesky"):
        raise ValueError(f"Unknown curvature method: {method!r}")
    return method


def _matrix_edges(weight_matrix):
    if weight_matrix.ndim != 2 or weight_matrix.shape[0] != weight_matrix.shape[1]:
        raise ValueError("sparse Cholesky requires a square weight matrix")
    if not torch.equal(weight_matrix, weight_matrix.T) or bool(weight_matrix.diagonal().ne(0).any()):
        raise ValueError("sparse Cholesky requires a symmetric matrix with a zero diagonal")
    rows, cols = weight_matrix.nonzero(as_tuple=True)
    upper = rows < cols
    rows, cols = rows[upper], cols[upper]
    return rows, cols, weight_matrix[rows, cols]


def _edge_matrix(n, rows, cols, values):
    matrix = values.new_zeros((n, n))
    matrix[rows, cols] = matrix[cols, rows] = values
    return matrix


def _validate_edges(n, rows, cols, weights):
    if n < 1 or rows.ndim != 1 or rows.shape != cols.shape or rows.shape != weights.shape:
        raise ValueError("provide a positive node count and three equally sized edge vectors")
    if rows.dtype != torch.int64 or cols.dtype != torch.int64:
        raise ValueError("edge endpoints must use torch.int64")
    if rows.device != weights.device or cols.device != weights.device:
        raise ValueError("edge endpoints and weights must be on the same device")
    if weights.dtype not in (torch.float32, torch.float64):
        raise ValueError("edge weights must use torch.float32 or torch.float64")
    valid = ((rows >= 0) & (rows < n) & (cols >= 0) & (cols < n) & (rows != cols)
             & weights.isfinite() & (weights > 0))
    if not bool(valid.all()):
        raise ValueError("each edge must connect two distinct valid nodes and have a finite positive weight")


def _prepare_edge_plan(n, rows, cols, plan):
    if plan is None:
        return symbolic_cholesky(n, rows, cols).to(rows.device)
    if (plan.n != n or plan.permutation.device != rows.device
            or not torch.equal(plan.inverse_permutation[rows], plan.edge_rows)
            or not torch.equal(plan.inverse_permutation[cols], plan.edge_cols)):
        raise ValueError("the sparse plan must match the node count, device and ordered edge list")
    return plan


def _validate_unique_edges(n, rows, cols):
    keys = torch.minimum(rows, cols) * n + torch.maximum(rows, cols)
    if keys.unique().numel() != rows.numel():
        raise ValueError("each undirected edge must appear once")


def _checked_sparse_curvature(plan, rows, cols, weights, disturb_factor):
    resistance, nodes, edges, valid = sparse_edge_curvature(plan, rows, cols, weights, disturb_factor)
    valid = valid & ((resistance > 0) & resistance.isfinite()).all()
    valid = valid & nodes.isfinite().all() & edges.isfinite().all()
    if not bool(valid):
        raise RuntimeError("sparse curvature failed: check shifted Laplacian definiteness and numerical precision")
    return nodes, edges


def _advance_flow(weights, curvature, step, normalize):
    conductance = torch.where(weights != 0, 1 / weights, 0.0)
    if normalize:
        total = conductance.sum()
        mean = torch.where(total != 0, (curvature * conductance).sum() / total, 0.0)
        conductance -= step * conductance * (curvature - mean)
    else:
        conductance -= step * curvature * conductance
    return torch.where(conductance != 0, 1 / conductance, 0.0)


class ResistanceCurvature():
    """
        A class to compute resistance curvature for graphs using PyTorch.

        This class implements methods to calculate node and edge curvatures based on
        resistance distance metrics derived from graph Laplacians.
    """
    def __init__(self,device: torch.device = None):
        """
        Initialize the ResistanceCurvature calculator.

        Args:
            device: The computation device (e.g., 'cpu' or 'cuda').
        """
        self.device=device if device is not None else \
                 torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def cal_laplacian(self, weight_matrix: torch.Tensor) -> torch.Tensor:
        """
            Compute the unnormalized graph Laplacian matrix.

            Laplacian L = D - W, where D is degree matrix and W is weight matrix.

            Args:
                weight_matrix: Input weight matrix.

            Returns:
                Laplacian matrix of the graph.
        """
        degree_matrix=torch.diag(torch.sum(weight_matrix,dim=1))
        return degree_matrix-weight_matrix

    def cal_laplacian_inverse_disturb(self,
                                    weight_matrix: torch.Tensor,
                                    disturb_factor: float = 1e-3) -> torch.Tensor:
        """
        Compute the inverse of Laplacian with small diagonal disturbance for numerical stability.

        Args:
            weight_matrix: Input weight matrix.
            disturb_factor: Small positive value to add to diagonal for numerical stability.

        Returns:
            Inverse of the disturbed Laplacian matrix.
        """
        laplacian = self.cal_laplacian(weight_matrix)
        disturbed_laplacian = laplacian + disturb_factor * torch.eye(
            laplacian.shape[0], device=laplacian.device
        )
        # Cholesky decomposition for efficient inversion
        L = torch.linalg.cholesky(disturbed_laplacian)
        return torch.cholesky_inverse(L)

    def cal_laplacian_pseudo_inverse(self, weight_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute the Moore-Penrose pseudo-inverse of the Laplacian matrix.

        Args:
            weight_matrix: Input weight matrix.

        Returns:
            Pseudo-inverse of the Laplacian matrix.
        """
        laplacian = self.cal_laplacian(weight_matrix)
        return torch.linalg.pinv(laplacian)
    def cal_resistance_matrix(self, laplacian_inverse: torch.Tensor, affinity_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute the resistance distance matrix from Laplacian inverse.

        Resistance distance between nodes i and j is:
        R_ij = L^+_ii + L^+_jj - L^+_ij - L^+_ji

        Args:
            laplacian_inverse: Inverse or pseudo-inverse of Laplacian matrix.
            affinity_matrix: Binary adjacency matrix to mask non-existent edges.

        Returns:
            Resistance distance matrix masked by adjacency.
        """
        diag = torch.diag(laplacian_inverse).reshape(1, laplacian_inverse.shape[0])
        resistance=(diag + diag.T) - (laplacian_inverse + laplacian_inverse.T)
        return resistance*affinity_matrix
    def cal_node_curvature(self,weight_matrix: torch.Tensor, resistence_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute node curvature from resistance distances.

        Node curvature: k_i = 1 - 0.5 * sum_j w_ij R_ij

        Args:
            weight_matrix: Original weight matrix.
            resistance_matrix: Resistance distance matrix.

        Returns:
            Vector of node curvatures.
        """
        return 1-0.5*torch.sum(resistence_matrix*weight_matrix,dim=1)

    def cal_edge_curvature(self,node_curvature: torch.Tensor,resistance_matrix: torch.Tensor,affinity_matrix: torch.Tensor)-> torch.Tensor:
        """
        Compute edge curvature from node curvatures and resistance distances.

        Edge curvature: k_ij = 2(k_i + k_j)/R_ij for existing edges

        Args:
            node_curvature: Node curvature values.
            resistance_matrix: Resistance distance matrix.
            affinity_matrix: Binary adjacency matrix.

        Returns:
            Matrix of edge curvatures.
        """
        # Broadcast node curvatures to pairwise matrix
        k_i = node_curvature.unsqueeze(1)
        k_j = node_curvature.unsqueeze(0)
        pairwise_sum = k_i + k_j

        # Compute reciprocal of resistance with masking
        inv_resistance = torch.where(
            resistance_matrix != 0,
            1.0 / resistance_matrix,
            torch.zeros_like(resistance_matrix)
        )
        return 2 * pairwise_sum * inv_resistance * affinity_matrix
    def adjust_weights_for_leaf_nodes_matrix(self,matrix):
        """
        Adjust weights for edges connected to leaf nodes (degree=1).

        Ensures weights connected to leaf nodes are positive by flipping signs if negative.

        Args:
            matrix: Weight matrix to adjust.

        Returns:
            Adjusted weight matrix.
        """
        degrees = torch.sum(matrix != 0, dim=1)
        is_leaf = degrees == 1

        # Find edges connected to leaf nodes
        is_leaf = is_leaf.unsqueeze(1)
        leaf_edges = is_leaf | is_leaf.T

        # Flip sign of negative weights connected to leaves
        to_adjust = (matrix < 0) & leaf_edges
        matrix[to_adjust] = -matrix[to_adjust]
        return matrix

    def cal_curvature(self, weight_matrix: torch.Tensor, use_disturb: bool = True,
                      disturb_factor: float = 1e-3, *, method: str = None
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return node curvature and a dense edge-curvature matrix.

        An explicit method takes precedence over the legacy use_disturb flag.
        Use cal_curvature_edges with sparse_cholesky for a sparse input/output path.
        """
        method = _resolve_method(method, use_disturb)
        if method == "sparse_cholesky":
            rows, cols, weights = _matrix_edges(weight_matrix)
            n = weight_matrix.shape[0]
            nodes, edges = self.cal_curvature_edges(n, rows, cols, weights, disturb_factor=disturb_factor)
            return nodes, _edge_matrix(n, rows, cols, edges)

        affinity_matrix = (weight_matrix != 0).float()
        return self._cal_curvature_dense(weight_matrix, affinity_matrix, method, disturb_factor)

    def _cal_curvature_dense(self, weight_matrix, affinity_matrix, method, disturb_factor):
        if method == "dense_cholesky":
            laplacian_inverse = self.cal_laplacian_inverse_disturb(weight_matrix, disturb_factor)
        else:
            laplacian_inverse = self.cal_laplacian_pseudo_inverse(weight_matrix)

        resistance = self.cal_resistance_matrix(laplacian_inverse, affinity_matrix)
        nodes = self.cal_node_curvature(weight_matrix, resistance)
        edges = self.cal_edge_curvature(nodes, resistance, affinity_matrix)
        positive = resistance[resistance > 0]
        if positive.numel():
            edges = edges * positive.min() / 2
        elif not bool(affinity_matrix.any()):
            return nodes, edges
        else:
            raise RuntimeError("the graph has no positive edge resistance to scale curvature")
        return nodes, self.adjust_weights_for_leaf_nodes_matrix(edges)

    def prepare_sparse_plan(self, n, rows, cols, *, ordering="minimum_degree", degree_cap=32) -> SparseCholeskyPlan:
        """Prepare bounded-width sparse elimination and a dense Schur tail.

        ``degree_cap`` limits the active degree of sparsely eliminated columns.
        It controls execution and storage, without changing the requested shift.
        """
        if ordering == "natural":
            return _legacy_symbolic_cholesky(n, rows, cols, ordering=ordering).to(rows.device)
        return symbolic_cholesky(n, rows, cols, ordering=ordering, degree_cap=degree_cap).to(rows.device)

    def cal_curvature_edges(self, n, rows, cols, edge_weights, *,
                            method="sparse_cholesky", disturb_factor=1e-3,
                            plan: SparseCholeskyPlan = None):
        """Return (node_curvature, edge_curvature) in node and input-edge order.

        Supply int64 endpoint vectors and FP32/FP64 positive weights on the
        same device. List each undirected edge once and omit self-loops.
        """
        method = _resolve_method(method, True)
        _validate_edges(n, rows, cols, edge_weights)
        if method == "sparse_cholesky":
            plan = _prepare_edge_plan(n, rows, cols, plan)
            return _checked_sparse_curvature(plan, rows, cols, edge_weights, disturb_factor)

        if plan is not None:
            raise ValueError("a sparse plan is only used by sparse_cholesky")
        _validate_unique_edges(n, rows, cols)
        nodes, matrix = self.cal_curvature(
            _edge_matrix(n, rows, cols, edge_weights), method=method, disturb_factor=disturb_factor
        )
        return nodes, matrix[rows, cols]

    def cal_curvature_flow(self, weight_matrix: torch.Tensor, n_iter: int = 1,
                           step: float = 1.0, flow_norm: bool = False,
                           dtype: torch.dtype = torch.float32, use_disturb: bool = True,
                           disturb_factor: float = 1e-3, *, method: str = None
                           ) -> torch.Tensor:
        """Run curvature flow, returning a dense matrix of updated weights."""
        method = _resolve_method(method, use_disturb)
        weight_matrix = weight_matrix.to(dtype)
        if n_iter <= 0:
            return weight_matrix
        if method == "sparse_cholesky":
            rows, cols, weights = _matrix_edges(weight_matrix)
            n = weight_matrix.shape[0]
            weights = self.cal_curvature_flow_edges(
                n, rows, cols, weights, n_iter=n_iter, step=step, flow_norm=flow_norm,
                disturb_factor=disturb_factor,
            )
            return _edge_matrix(n, rows, cols, weights)

        affinity_matrix = (weight_matrix != 0).float()
        for _ in range(n_iter):
            _, curvature = self._cal_curvature_dense(weight_matrix, affinity_matrix, method, disturb_factor)
            weight_matrix = _advance_flow(weight_matrix, curvature, step, flow_norm)
        return weight_matrix

    def cal_curvature_flow_edges(self, n, rows, cols, edge_weights, *, n_iter=1,
                                 step=1.0, flow_norm=False, method="sparse_cholesky",
                                 disturb_factor=1e-3, plan: SparseCholeskyPlan = None):
        """Run curvature flow and return edge weights in the original order.

        The sparse path keeps the initial edge pattern and reuses one plan.
        It preserves the input weight dtype and checks factorization each round.
        """
        method = _resolve_method(method, True)
        _validate_edges(n, rows, cols, edge_weights)
        if method != "sparse_cholesky":
            if plan is not None:
                raise ValueError("a sparse plan is only used by sparse_cholesky")
            _validate_unique_edges(n, rows, cols)
            matrix = self.cal_curvature_flow(
                _edge_matrix(n, rows, cols, edge_weights), n_iter=n_iter, step=step,
                flow_norm=flow_norm, dtype=edge_weights.dtype, method=method,
                disturb_factor=disturb_factor,
            )
            return matrix[rows, cols]

        if n_iter <= 0 or not edge_weights.numel():
            return edge_weights
        plan = _prepare_edge_plan(n, rows, cols, plan)
        for _ in range(n_iter):
            _, curvature = _checked_sparse_curvature(plan, rows, cols, edge_weights, disturb_factor)
            edge_weights = _advance_flow(edge_weights, curvature, step, flow_norm)
        return edge_weights
