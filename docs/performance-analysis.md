# NTM Performance Analysis

## 1. Current Performance Profile

Apples-to-apples benchmark (`make bench-compare`): identical architecture, optimizer,
loss function, data format, and batch size on both sides.

| Model | Idris (ms) | PyTorch (ms) | Ratio | Notes |
|-------|-----------|-------------|-------|-------|
| Supervised (1000 ep) | 77 | 230 | 0.33x | Idris 3x faster |
| RNN (1000 ep) | 355 | 2076 | 0.17x | Idris 6x faster |
| NTM-small (100 ep) | 538 | 1242 | 0.43x | Idris 2.3x faster (w=3, n=10, m=5, h=20, batch=5) |
| NTM-copy (100 ep) | 25030 | 14044 | 1.78x | **PyTorch 1.8x faster** (w=8, n=128, m=20, h=100, batch=16) |

The small NTM hides the scalar-vs-tensor autograd gap. At production scale (NTM-copy),
PyTorch's tensor-level autograd gives it a ~1.8x advantage.

NTM-copy per-epoch: Idris **~234ms** vs PyTorch **~140ms**.

Optimizations applied (from 1.38s/epoch baseline):
- Buffer-passing for addressing chain ops (1.38s -> 1.0s)
- Shadow ConstOps for intermediate outputs (tape compaction)
- C-side pid filtering in walk_backward_ext (1.0s -> 0.66s)
- Bulk C shadow tag setting
- Dense optimizer: C arrays replace SortedMap (0.66s -> 0.46s)
- C-bulk ConstOp creation: memset+memcpy replaces per-element Scheme loops (0.247s -> 0.234s)

Per-epoch breakdown (n=128, m=20, h=100, variable timesteps):

| Phase | Time | Notes |
|-------|------|-------|
| Forward pass | ~176ms | Variable materialization + C tensor compute + loss |
| Backward: walk_backward_ext_dense | ~37ms | Dense accumulation into C array by pid_id |
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
This is the only path to close the remaining ~2x gap.

### Actual Results

| Path | Epoch time | Speedup | vs PyTorch |
|------|-----------|---------|------------|
| Baseline (pre-optimization) | 1.38s | -- | ~11x |
| A: Buffer-passing + shadow ConstOps | ~1.0s | 1.4x | ~8x |
| B: + C-side pid filtering | ~0.66s | 2.1x | ~5x |
| D: + Dense optimizer (C arrays) | ~0.247s | 5.6x | ~2.0x |
| E: + C-bulk ConstOps (memset+memcpy) | ~0.234s | 5.9x | ~1.8x |
| C: Tensor-level Variable (estimated) | ~0.15s | -- | ~1.2x |

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

Irreducible gap (~1.2x) from: Scheme orchestration overhead, non-SIMD C kernels.

## 5. Forward-Pass Sub-Phase Profile

### Baseline (2026-03-04, commit 937b4d2)

Profiler: `src/Example/Profile.idr` (N=128 M=20 H=100, Batch=16, seqLen=1-20)

```
Epoch   Enc(ms)   Out(ms)  Loss(ms)   Bwd(ms)   Opt(ms)  Sync(ms)  TapeSize    Loss
    1      97.8      82.3       4.7      35.8       0.0      18.6   3561699
    2      97.8      85.0       4.9      39.3       0.0      17.0   3561699
    3     103.9      82.3       5.5      38.0       0.0      15.6   3561699
   avg    ~101       ~84        ~5       ~38        ~0       ~18    3561699
```

Forward (Enc+Out) = **~185ms** (75% of ~246ms epoch).

Tape histogram:
```
  ConstOps:   1,288,190  (36%)
  ScalarOps:     33,955  (<1%)
  TensorOps:     12,078  (<1%)
  ShadowOps:  2,227,476  (63%)
  Total:      3,561,699
```

Tensor detail: MatVec=5490 Dot=0 Softmax=1464 LogSoftmax=0 BatchCosSim=732
               ReadOp=732 WriteOp=0 InterpWrite=366 Interpolate=732
               Shift=732 Focus=732 LstmCell=1098

Bench-compare:
```
Model             Idris (ms)   PyTorch (ms)    Ratio
Supervised              86.9          262.2    0.33x
RNN                    397.2         2372.2    0.17x
NTM-small              586.0         1449.9    0.40x
NTM-copy             27596.1        14223.9    1.94x
```

Root cause: `prim__appendOutputConst`/`prim__appendOutputConstOff` use per-element
Scheme `foreign-set!` loops (3 writes/element × 1.29M elements = ~3.87M FFI calls).
`prim__appendShadowConst` already uses efficient C bulk `tape_set_shadow_tags`.

### After C-bulk ConstOps (2026-03-04, commit 6efc2ae)

Replaced per-element Scheme loops with C `tape_bulk_set_const`/`tape_bulk_set_const_off`
(memset tags + memcpy values). Pid writes skipped (default "" from make-vector init).

```
Epoch   Enc(ms)   Out(ms)  Loss(ms)   Bwd(ms)   Opt(ms)  Sync(ms)  TapeSize    Loss
    1      95.8      80.2       4.8      36.2       0.0      15.8   3561699
   avg     ~97       ~79        ~5       ~37        ~0       ~16    3561699
```

Forward (Enc+Out) = **~176ms** (76% of ~234ms epoch). Saved **~9ms** forward.

Bench-compare:
```
Model             Idris (ms)   PyTorch (ms)    Ratio
Supervised              83.2          240.1    0.35x
RNN                    387.6         2332.7    0.17x
NTM-small              600.6         1390.2    0.43x
NTM-copy             25029.9        14044.1    1.78x
```

NTM-copy: 27596ms -> 25030ms (**1.94x -> 1.78x** vs PyTorch). ~10% improvement.

Note: Chez Scheme `foreign-set!` is a native operation (not a C FFI crossing), so
the per-element cost was lower than estimated. The remaining forward pass time is
dominated by Scheme Variable record allocation and `ensureOnTape`/`packVec` loops,
not by ConstOp tape writes.
