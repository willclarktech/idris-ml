# NTM Performance Analysis

## 1. Current Performance Profile

NTM copy task: **~1.38s/epoch** vs PyTorch **~0.2s/epoch** = ~7x gap.

Per-epoch breakdown (n=128, m=20, h=100, 352 timesteps):

| Phase | Time | Dominant cost |
|-------|------|---------------|
| Forward: Variable materialization | ~350ms | 1.46M Variables x 3 Scheme heap allocs each = 4.4M allocs |
| Forward: packVec | ~250ms | ~830K element iterations (ensureOnTape + setDouble + setInt32) |
| Forward: C tensor compute | ~50ms | Actual math |
| Forward: scalar ops + overhead | ~80ms | fromDouble literals, tape appends |
| Backward: walk_backward_ext | ~100ms | Scanning 1.56M tape entries |
| Backward: result collection | ~100ms | Collecting ~200K+ non-zero-gradient ConstOps |
| Backward: buildGradMap | ~150ms | Iterating results in Scheme (pid lookup + string compare) |
| Optimizer + applyDeltas | ~150ms | SortedMap with ~62K entries, O(n log n) |
| syncBuffers | ~50ms | Writing values back to C WeightBufs |

## 2. Root Cause: Scalar vs Tensor Autograd

**PyTorch**: 1 autograd node per tensor op. ~7K nodes/epoch.
**Idris**: 1 tensor op + k output ConstOps per tensor op. ~1.56M entries/epoch.

Per-timestep tape entry accounting:

| Component | Tensor ops | Output ConstOps |
|-----------|-----------|-----------------|
| LSTM (buffer-passing) | 3 | 200 |
| Head FCs (3 MatVec+Bias) | 3 | 80 |
| Read head addressing (6 ops) | 6 | 643 |
| Read head readOp | 1 | 20 |
| Write head addressing (6 ops) | 6 | 643 |
| Write head interpWrite | 1 | 2560 |
| Write head sigmoid(rawAdd) | 0 (20 scalar) | 0 |
| **Per timestep** | **~20** | **~4,146** |

The 4,146 output ConstOps per timestep x 352 = 1.46M ConstOps dominate the tape. Each
creates a Scheme Variable record that is either immediately consumed by the next op (waste)
or stored as network state.

## 3. What PyTorch Does Differently

**Fundamental (architecture)**:
- Tensor-level autograd graph: O(ops) nodes, not O(ops x output_size)
- Tensor data stays in C memory -- no per-element Scheme<->C marshalling
- Compiled BLAS/LAPACK with SIMD

**NOT Python-specific**: Python is just orchestration over C++. Scheme could do the same.

## 4. Optimization Paths

### Path A: Complete Buffer-Passing

Eliminate Variable materialization for intermediate addressing chain results.
Chain ops pass `(AnyPtr, Int)` buffers instead of `Vector n Variable`.

**Savings per timestep**: ~3,840 Variables eliminated (read head: 640, write head: 640 addressing + 2560 interpWrite).
**Total per epoch**: 3,840 x 352 = ~1.35M fewer Variables.

### Path B: Tape Compaction + C-side Gradients

Eliminate intermediate ConstOps entirely. Buffer-chained ops reserve gradient regions
without creating tape entries. Reduce tape from ~1.56M to ~262K entries.

Add C-side pid filtering: `walk_backward_ext` filters by integer pid_id instead of
returning all ConstOps. Reduces Scheme-side iteration from ~200K to ~62K entries.

### Path C: Tensor-level Variable Type

Replace `Vector n Variable` with a tensor-level Variable that holds an entire vector
in C memory. Would eliminate per-element Variable materialization entirely.

### Expected Results

| Path | Epoch time | Speedup | vs PyTorch |
|------|-----------|---------|------------|
| Current | 1.38s | -- | ~7x |
| A: Buffer-passing | ~1.0s | 1.4x | ~5x |
| B: + Tape compaction + C-side grads | ~0.45s | 3.1x | ~2.3x |
| C: Tensor-level Variable | ~0.3s | 1.5x gap | ~1.5x |

Irreducible gap (~1.5x) from: Scheme orchestration, non-SIMD C, state Variable materialization.
