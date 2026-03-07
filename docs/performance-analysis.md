# NTM Performance Analysis

## 1. Current Performance Profile

NTM copy task: **~0.66s/epoch** vs PyTorch **~0.2s/epoch** = ~3.3x gap.

Optimizations applied (from 1.38s baseline):
- Buffer-passing for addressing chain ops (1.38s -> 1.0s)
- Shadow ConstOps for intermediate outputs (tape compaction)
- C-side pid filtering in walk_backward_ext (1.0s -> 0.66s)
- Bulk C shadow tag setting

Per-epoch breakdown (n=128, m=20, h=100, variable timesteps):

| Phase | Time | Notes |
|-------|------|-------|
| Forward: variable materialization + packVec | ~350ms | Endpoints still materialize Variables |
| Forward: C tensor compute | ~50ms | Actual math (Accelerate BLAS) |
| Forward: scalar ops + shadow ConstOps | ~60ms | Loss computation, sigmoid(rawAdd), tape appends |
| Backward: walk_backward_ext | ~35ms | Scanning ~3.5M tape entries (measured) |
| Backward: buildGradMap | ~50ms | ~67K results, SortedMap O(n log n) |
| Optimizer + applyDeltas | ~80ms | SortedMap with ~62K entries |
| syncBuffers | ~30ms | Writing values back to C WeightBufs |

## 2. Root Cause: Scalar vs Tensor Autograd

**PyTorch**: 1 autograd node per tensor op. ~7K nodes/epoch.
**Idris**: 1 tensor op + k output ConstOps/shadow ConstOps per tensor op. ~3.5M entries/epoch.

Buffer-passing eliminated Variable materialization for intermediate addressing outputs
(~3,840 Variables/timestep), but shadow ConstOps (tag=25) still occupy tape slots for
gradient routing. The backward pass skips them (`continue`), but they inflate tape_sz.

## 3. What PyTorch Does Differently

**Fundamental (architecture)**:
- Tensor-level autograd graph: O(ops) nodes, not O(ops x output_size)
- Tensor data stays in C memory -- no per-element Scheme<->C marshalling
- Compiled BLAS/LAPACK with SIMD

**NOT Python-specific**: Python is just orchestration over C++. Scheme could do the same.

## 4. Optimization Paths

### Path A: Complete Buffer-Passing (DONE)

Eliminated Variable materialization for intermediate addressing chain results.
Chain ops pass `(AnyPtr, Int)` buffers instead of `Vector n Variable`.
Shadow ConstOps provide gradient slots without Variable allocation.

### Path B: C-side Pid Filtering (DONE)

C-side `tape_pid_ids` integer array parallel to tape. `walk_backward_ext` only
collects ConstOps with `pid_id >= 0`. Eliminates ~200K+ unnecessary result iterations
in Scheme. Reduces to ~67K named parameter results.

### Path C: Tensor-level Variable Type

Replace `Vector n Variable` with a tensor-level Variable that holds an entire vector
in C memory. Would eliminate per-element Variable materialization entirely.
This is the only path to close the remaining ~3.3x gap.

### Actual Results

| Path | Epoch time | Speedup | vs PyTorch |
|------|-----------|---------|------------|
| Baseline (pre-optimization) | 1.38s | -- | ~7x |
| A: Buffer-passing + shadow ConstOps | ~1.0s | 1.4x | ~5x |
| B: + C-side pid filtering | ~0.66s | 2.1x | ~3.3x |
| C: Tensor-level Variable (estimated) | ~0.3s | -- | ~1.5x |

### Remaining bottleneck analysis

The forward pass dominates (~460ms of 660ms). The C backward pass is only ~35ms.
Key forward costs:
- Endpoint Variable materialization: `buildOutputScalars` for network state (addressing
  weights, hidden state, cell state, read output) still creates Scheme Variable records
- `packVec` for tensor op inputs from Variables: per-element `ensureOnTape` + `setDouble` + `setInt32`
- Loss computation: scalar ops for binary cross-entropy with logits

Irreducible gap (~1.5x) from: Scheme orchestration overhead, non-SIMD C kernels.
