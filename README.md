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

The default sparse path uses **exact sparse-prefix elimination with a dense
Schur-complement tail**. It is a hybrid factorization of the same
`L + epsilon I`; the split introduces no additional approximation.

1. Eliminate minimum-degree vertices while their active degree is at most
   `degree_cap` (32 by default). Symbolic analysis retains updates only for
   these bounded-width columns. Adjacency sets are updated in place so
   eliminating leaves does not repeatedly copy a large hub's neighbours.
2. Factor the remaining Schur complement with dense Cholesky and invert this
   smaller block. This replaces the long, high-fill part of scalar elimination
   with matrix operations.
3. Apply the Takahashi recurrence backwards through the sparse prefix to
   recover the diagonal and original-edge inverse entries needed for curvature.

On Linux CUDA installations with Triton, each sparse level uses one fused
factorization kernel and one fused selected-inversion kernel. CPU and CUDA
installations without Triton use the portable PyTorch implementation. The
fused kernels were verified with PyTorch 2.8.0, CUDA 12.8 and Triton 3.4.0.
The original fully sparse plan remains supported for existing low-level
callers, and explicitly requesting `ordering="natural"` retains that path.

For a reused plan, let `q` be the number of sparse columns, `r=n-q` the tail
size, `F_p=q+sum(b_j)` and `U_p=sum(b_j*(b_j+1)/2)` over the sparse prefix.
Plan and numerical storage are `O(n+m+F_p+U_p+r^2)`, rather than retaining
the full factorization's potentially enormous update count. Since `b_j <= B`
for the fixed cap `B`, `U_p <= (B+1)*(F_p-q)/2`. With fixed `B`, numerical
arithmetic is `O(n+m+r^3)` and storage is `O(n+m+r^2)`. If `B` is treated as
a variable, the padded CUDA kernels use `O(n+m+q*B^2+r^3)` arithmetic.
The one-time CPU ordering and symbolic preparation are measured separately.
The tail size depends on fill-in, not just the original average degree.
In the worst case `r=n`, the method has dense cubic time and quadratic space;
it does not guarantee a speedup for every graph.

Set a different split without changing epsilon or the mathematical result:

```python
plan = rc.prepare_sparse_plan(n, rows, cols, degree_cap=32)
node_curvature, edge_curvature = rc.cal_curvature_edges(
    n, rows, cols, weights, method="sparse_cholesky", plan=plan,
    disturb_factor=1e-3,
)
```

Reuse this plan across calls or curvature-flow steps with the same topology.
Each step still recomputes the numerical factor and selected inverse for the
current weights. Use the edge-list interface to avoid a dense input/output.
FP32 execution orders differ, so results agree up to floating-point error;
use FP64 when stricter numerical agreement is needed.

On five real graphs, the following measurements used one RTX 3090 (24 GB),
FP32, unit edge weights, epsilon=0.001, two warmups and the median of seven
synchronized calls. Both columns time the public `cal_curvature_edges` API;
the sparse plan is reused. Memory is the maximum PyTorch reserved memory
observed across the public calls and subsequent low-level CUDA Graph tests,
including the graph's retained allocations (MiB).

| Graph | Dense ms | Hybrid sparse ms | Speedup | Dense peak MiB | Hybrid peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cora | 11.246 | 2.949 | 3.81x | 302 | 6 |
| CiteSeer | 16.442 | 2.805 | 5.86x | 452 | 6 |
| PubMed | 1145.335 | 6.962 | 164.50x | 15216 | 90 |
| Amazon-Photo | 101.936 | 23.827 | 4.28x | 2304 | 280 |
| Coauthor-CS | 1054.054 | 39.752 | 26.52x | 13166 | 442 |

These are steady-state measurements, not first-call speedups. Input upload,
symbolic preparation and the first public call took approximately
0.75/0.75/1.17/1.30/1.59 seconds for the hybrid path in table order,
versus 0.24/0.25/1.30/0.34/1.21 seconds for dense. Those measurements
exclude Python import, input-file loading and CUDA context initialization;
a fresh Triton compiler cache can add compilation cost. Reusing the plan
and process amortizes preparation and library initialization.
All five graphs' FP64 resistances and node/edge curvatures were also compared
at rtol=1e-8, atol=1e-9. All passed. In FP32 all edge-curvature comparisons
passed rtol=5e-3, atol=5e-5; some near-zero node curvatures in Cora/CiteSeer
differed by up to 6.11e-5, so that tighter node check did not pass everywhere.

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
