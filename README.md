# ResistanceCurvature

A PyTorch package for calculating resistance curvature and resistance curvature flow.

## Features

- CPU and CUDA computation with float32 or float64 weights.
- Three methods for computing curvature.

  | `method` | Laplacian computation |
  | --- | --- |
  | `dense_pinv` | SVD-based Moore–Penrose pseudoinverse |
  | `dense_cholesky` | Dense Cholesky factorization and full inverse of `L + epsilon I` |
  | `sparse_cholesky` | Sparse Cholesky factorization and selected inversion of `L + epsilon I` |

- Edge-list interfaces for sparse input and output.
- Reusable symbolic plans for fixed graph topology and curvature flow.

## Installation

You can install the package using pip:

```bash
pip install ResistanceCurvature
```

To install the latest version from this repository:

```bash
python -m pip install git+https://github.com/cqfei/resistance_curvature.git
```

The package requires Python 3.8 or newer and PyTorch 1.10 or newer.
For CUDA, install the appropriate PyTorch build for your environment.

## Simple Example

```python
import torch
from ResistanceCurvature.resistance_curvature import ResistanceCurvature

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rc = ResistanceCurvature(device=device)
adj_tensor = torch.tensor([[0, 1, 1],
                     [1, 0, 1],
                     [1, 1, 0],], dtype=torch.float32, device=device)
node_curvature, edge_curvature = rc.cal_curvature(adj_tensor)
print("Node Curvature:", node_curvature)
print("Edge Curvature:", edge_curvature)

```

## Selecting a method

The matrix interface accepts an optional keyword-only `method`:

```python
node_curvature, edge_curvature = rc.cal_curvature(
    adj_tensor,
    method="sparse_cholesky",
    disturb_factor=1e-3,
)
```

Use `method="dense_pinv"` for the pseudoinverse, or
`method="dense_cholesky"` for the dense regularized inverse.
The original calls with `use_disturb=True` or `use_disturb=False`
select `dense_cholesky` or `dense_pinv`, respectively. An explicit
`method` takes precedence.

Both Cholesky methods compute regularized effective resistances. For the same
positive `disturb_factor`, their selected inverse entries agree in exact
arithmetic. Floating-point results can differ slightly. The regularized results
approximate those obtained from the unperturbed Laplacian pseudoinverse.

The matrix interface returns an `n x n` edge-curvature matrix. Use the edge-list
interface below to retain sparse input and output.

## Sparse edge-list interface

List each undirected edge once, use node IDs from `0` to `n - 1`, and omit
self-loops. Endpoints must be `torch.int64`, and weights must be finite,
positive `torch.float32` or `torch.float64` values. Put endpoints and weights
on the same device.

```python
import torch
from ResistanceCurvature import ResistanceCurvature

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rc = ResistanceCurvature(device=device)
n = 4
rows = torch.tensor([0, 0, 1, 2], dtype=torch.int64, device=device)
cols = torch.tensor([1, 2, 2, 3], dtype=torch.int64, device=device)
weights = torch.tensor([1.0, 0.8, 1.2, 0.7], dtype=torch.float64, device=device)

with torch.inference_mode():
    plan = rc.prepare_sparse_plan(n, rows, cols)
    node_curvature, edge_curvature = rc.cal_curvature_edges(
        n, rows, cols, weights,
        method="sparse_cholesky",
        disturb_factor=1e-3,
        plan=plan,
    )
```

`node_curvature` has length `n`. `edge_curvature` has one entry per input edge,
in the same order. Isolated nodes are supported. The `plan` argument is optional
for a single call.

The sparse path uses a minimum-degree elimination order, computes a sparse
Cholesky factor, and applies a Takahashi-type selected inversion recurrence.
It computes the diagonal entries, original-edge entries, and auxiliary inverse
entries required by the recurrence. This edge-list path avoids allocating
the full inverse. Its time and memory depend on elimination fill-in and the
precomputed update indices.

The returned edge curvature includes the original implementation's
minimum-positive-resistance scaling and leaf-node sign adjustment.

## Curvature flow

Reuse the symbolic plan while the topology, edge order, and device remain fixed:

```python
with torch.inference_mode():
    updated_weights = rc.cal_curvature_flow_edges(
        n, rows, cols, weights,
        n_iter=5,
        step=0.05,
        flow_norm=True,
        method="sparse_cholesky",
        disturb_factor=1e-3,
        plan=plan,
    )
```

`updated_weights` follows the input edge order and preserves the input dtype.
Each round recomputes the numerical factorization and selected inverse entries.
The sparse path checks numerical factorization and curvature results each round.
Choose flow parameters that keep the shifted matrix positive definite.

The matrix interface also accepts `method`:

```python
updated_matrix = rc.cal_curvature_flow(
    adj_tensor,
    n_iter=5,
    step=0.05,
    flow_norm=True,
    dtype=adj_tensor.dtype,
    method="sparse_cholesky",
    disturb_factor=1e-3,
)
```

## Running the checks

From a checkout of the repository:

```bash
python -m unittest discover -s tests -v
```

The tests compare matrix and edge-list interfaces, legacy flags, sparse and dense
regularized curvatures, and curvature-flow weights after each iteration.

## Algorithm references

- Davis, T. A. (2006). *Direct methods for sparse linear systems*. Society for
  Industrial and Applied Mathematics. [Book DOI](https://doi.org/10.1137/1.9780898718881)
- Takahashi, K., Fagan, J., & Chin, M. (1973). Formation of a sparse bus impedance
  matrix and its application to short circuit study. In *Proceedings of the 8th
  PICA Conference*, Minneapolis, MN.
- Lin, L., Yang, C., Meza, J. C., Lu, J., Ying, L., & E, W. (2011).
  SelInv—An algorithm for selected inversion of a sparse symmetric matrix.
  *ACM Transactions on Mathematical Software*, 37(4), Article 40.
  [Article DOI](https://doi.org/10.1145/1916461.1916464)
