# NTM Performance Analysis

## 1. Current Performance Profile

NTM copy task: **~0.46s/epoch** vs PyTorch **~0.2s/epoch** = ~2.3x gap.

Optimizations applied (from 1.38s baseline):
- Buffer-passing for addressing chain ops (1.38s -> 1.0s)
- Shadow ConstOps for intermediate outputs (tape compaction)
- C-side pid filtering in walk_backward_ext (1.0s -> 0.66s)
- Bulk C shadow tag setting
- Dense optimizer: C arrays replace SortedMap (0.66s -> 0.46s)

Per-epoch breakdown (n=128, m=20, h=100, variable timesteps):

| Phase | Time | Notes |
|-------|------|-------|
| Forward pass | ~195ms | Variable materialization + C tensor compute + loss |
| Backward: walk_backward_ext_dense | ~32ms | Dense accumulation into C array by pid_id |
| Optimizer: C in-place step | ~0ms | rmsprop_vc_step on dense array (negligible) |
| Apply deltas + sync buffers | ~16ms | O(1) hash lookup per param + WeightBuf sync |

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
| D: + Dense optimizer (C arrays) | ~0.46s | 3.0x | ~2.3x |
| C: Tensor-level Variable (estimated) | ~0.25s | -- | ~1.3x |

### Path D: Dense Optimizer (DONE)

Replaced `SortedMap String Double` in optimizer hot path with dense C arrays
indexed by integer `pid_id`. Three changes:

1. **Dense backward accumulation**: `walk_backward_ext_dense` accumulates gradients
   directly into a pre-allocated C array (`out_dense[pid_id] += grad`). Eliminates
   `buildGradMap` entirely — no per-result FFI calls, no O(log n) SortedMap inserts.

2. **C optimizer step**: `rmsprop_vc_step`/`sgd_step`/`adam_gc_step` operate in-place
   on the dense gradient array, transforming it to deltas. Eliminates `toList` + fold
   with O(log n) lookups/inserts per parameter.

3. **Hash lookup for deltas**: `applyDeltasDense` uses Scheme's `pid-to-id` hash table
   (O(1)) instead of SortedMap lookup (O(log n)) to find each parameter's delta.

Measured improvement: backward+optimizer went from ~130ms to ~32ms (saved ~100ms).
Total epoch from ~660ms to ~460ms.

### Remaining bottleneck analysis

The forward pass dominates (~195ms of ~243ms total non-forward). The backward pass
(~32ms) and apply-deltas (~16ms) are now well-optimized.
Key forward costs:
- Endpoint Variable materialization: `buildOutputScalars` for network state (addressing
  weights, hidden state, cell state, read output) still creates Scheme Variable records
- `packVec` for tensor op inputs from Variables: per-element `ensureOnTape` + `setDouble` + `setInt32`
- Loss computation: scalar ops for binary cross-entropy with logits

Irreducible gap (~1.3x) from: Scheme orchestration overhead, non-SIMD C kernels.
