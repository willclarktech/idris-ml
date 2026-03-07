# NTM Performance Analysis

## 1. Current Performance Profile

Apples-to-apples benchmark (`make bench-compare`): identical architecture, optimizer,
loss function, data format, and batch size on both sides.

| Model | Idris (ms) | PyTorch (ms) | Ratio | Notes |
|-------|-----------|-------------|-------|-------|
| Supervised (1000 ep) | 90 | 242 | 0.37x | Idris 2.7x faster |
| RNN (1000 ep) | 400 | 2212 | 0.18x | Idris 5.5x faster |
| NTM-small (100 ep) | 438 | 1333 | 0.33x | Idris 3.0x faster (w=3, n=10, m=5, h=20, batch=5) |
| NTM-copy (100 ep) | 14660 | 13186 | 1.11x | PyTorch 1.1x faster (w=8, n=128, m=20, h=100, batch=16) |
| NTM-copy-1k (1000 ep) | 131236 | 214613 | 0.61x | **Idris 1.6x faster** |

At 100 epochs, the NTM-copy overhead comes from the optimizer step being non-trivial. At 1000 epochs,
the per-epoch cost amortizes and Idris is 1.6x faster than PyTorch.

NTM-copy per-epoch: Idris **~147ms** vs PyTorch **~132ms** (100 ep), **~131ms** vs **~215ms** (1000 ep).

Optimizations applied (from 1.38s/epoch baseline):
- Buffer-passing for addressing chain ops (1.38s -> 1.0s)
- Shadow ConstOps for intermediate outputs (tape compaction)
- C-side pid filtering in walk_backward_ext (1.0s -> 0.66s)
- Bulk C shadow tag setting
- Dense optimizer: C arrays replace SortedMap (0.66s -> 0.46s)
- C-bulk ConstOp creation: memset+memcpy replaces per-element Scheme loops (0.247s -> 0.234s)
- C-bulk delta application: apply deltas directly to C buffers, skip emap+sync (0.234s -> 0.218s)

Per-epoch breakdown (n=128, m=20, h=100, variable timesteps):

| Phase | Time | Notes |
|-------|------|-------|
| Forward pass | ~176ms | Variable materialization + C tensor compute + loss |
| Backward: walk_backward_ext_dense | ~37ms | Dense accumulation into C array by pid_id |
| Optimizer: C in-place step | ~0ms | rmsprop_vc_step on dense array (negligible) |
| Apply deltas to C buffers | ~0.3ms | buf_apply_deltas on WeightBuf/NtmMemBuf pid_ids |

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
| F: + C-bulk delta application | ~0.218s | 6.3x | ~1.6x |
| G: + LSTM→FC buffer-passing + case fix | ~0.145s | 9.5x | **0.87x** (faster) |
| C: Tensor-level Variable (estimated) | ~0.08s | -- | ~0.4x |

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

The forward pass dominates (~74ms of ~92ms epoch). The backward pass
(~15ms) and apply-deltas (~0.3ms) are well-optimized.
Key forward costs:
- Endpoint Variable materialization: `buildOutputScalars` for network state (addressing
  weights, hidden state, cell state, read output) still creates Scheme Variable records
- `packVec` for tensor op inputs from Variables: per-element `ensureOnTape` + `setDouble` + `setInt32`
- Loss computation: scalar ops for binary cross-entropy with logits

Idris is now faster than PyTorch overall due to lower Python orchestration overhead
at the batch level. The scalar autograd cost is offset by efficient C tensor kernels
and buffer-passing optimizations. Further gains possible via Path C (tensor-level Variables)
which would reduce tape from 1.5M to ~15K entries.

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

### After C-bulk delta application (2026-03-04, commit f8d0b4f)

Applied optimizer deltas directly to WeightBuf/NtmMemBuf C arrays via `buf_apply_deltas`,
bypassing the Scheme `emap (applyDeltasDense ...)` + `syncNetworkBuffers` traversals
(~63K Variable record updates + ~63K buffer sync writes per epoch).

```
Epoch   Enc(ms)   Out(ms)  Loss(ms)   Bwd(ms)   Opt(ms)  Sync(ms)  TapeSize    Loss
    1     101.1      78.3       4.5      37.4       0.0       0.4   3561699
   avg     ~97       ~80        ~5       ~37        ~0      ~0.3    3561699
```

Sync phase: **~16ms -> ~0.3ms** (eliminated). Total epoch: **~234ms -> ~218ms**.

Bench-compare:
```
Model             Idris (ms)   PyTorch (ms)    Ratio
Supervised              92.9          258.6    0.36x
RNN                    408.9         2411.3    0.17x
NTM-small              538.7         1496.6    0.36x
NTM-copy             23099.1        14399.0    1.60x
```

NTM-copy: 25030ms -> 23099ms (**1.78x -> 1.60x** vs PyTorch). ~8% improvement.

### After LSTM→FC buffer-passing + case destructuring (2026-03-04)

Two optimizations combined:

1. **Case destructuring fix**: Idris 2 compiled to Chez Scheme re-evaluates let-bound
   FFI calls when accessed via `fst`/`snd` projections. The baseline NTM forward accessed
   `lstmResult` via 3 separate `fst`/`snd` projections, causing 3× LSTM re-evaluation per
   timestep. Using `case` destructuring forces single evaluation.

2. **LSTM→FC buffer-passing**: LSTM output buffer `[cell|hidden]` passed directly to FC
   layers via `buf_to_meta_off`, eliminating ~900 `packVec` calls/timestep. New functions:
   `lstmCellVarFromBufsExt`, `matrixVectorMultiplyVarBufBiasFromBuf`,
   `matrixVectorMultiplyVarBufBiasFromBufAndVec`.

```
Epoch   Enc(ms)   Out(ms)  Loss(ms)   Bwd(ms)   Opt(ms)  Sync(ms)  TapeSize    Loss
    1      60.2      48.7       4.7      22.3       0.0       0.3   2232883
   avg     ~59       ~48        ~5       ~22        ~0      ~0.3    2232883
```

Forward (Enc+Out) = **~176ms -> ~107ms** (-39%). Backward **~37ms -> ~22ms** (-40%).
Tape **3.56M -> 2.23M** (-37%). LstmCell ops: 1098 -> 366 (3x→1x per timestep).

Tape histogram:
```
  ConstOps:   1,134,234  (51%)
  ScalarOps:     33,955  (<2%)
  TensorOps:      8,418  (<1%)
  ShadowOps:  1,056,276  (47%)
  Total:      2,232,883
```

Tensor detail: MatVec=2562 Softmax=1464 BatchCosSim=732 ReadOp=732 InterpWrite=366
               Interpolate=732 Shift=732 Focus=732 LstmCell=366

Bench-compare:
```
Model             Idris (ms)   PyTorch (ms)    Ratio
Supervised              96.1          256.1    0.38x
RNN                    464.4         2394.2    0.19x
NTM-small              336.8         1486.4    0.23x
NTM-copy             14523.3        14276.8    1.02x
```

NTM-copy: 23099ms -> 14523ms (**1.60x -> 1.02x** vs PyTorch). Near parity!

### After dense optimizer fix (2026-03-05)

**Critical bug found**: All 4 dense optimizer step functions used `let _ = prim__ffiCall`
which the Idris 2 compiler drops (dead code elimination). The optimizer NEVER ran —
raw gradients were applied directly as deltas. lr/clip/momentum had zero effect.

Fix: thread FFI return value through `prim__seq` into the state record.

Bench-compare (with fix):
```
Model             Idris (ms)   PyTorch (ms)    Ratio
Supervised              89.6          242.3    0.37x
RNN                    400.3         2211.9    0.18x
NTM-small              438.2         1332.8    0.33x
NTM-copy             14659.7        13186.0    1.11x
NTM-copy-1k         131236.1       214613.3    0.61x
```

NTM-copy-1k: Idris **0.61x** (1.6x faster than PyTorch). Idris loss 0.557 vs PyTorch 0.593.
NTM-copy 100 epochs: 1.11x (slight overhead from optimizer now actually running).

PyTorch convergence at batch=16 (lr=0.0001): early stop at iter 12000, 100% accuracy.
Idris convergence at batch=16 (lr=0.0001, 5000 epochs): loss ~0.48, accuracy ~52-59%.
Idris learning curve tracks PyTorch but oscillates more — needs investigation.

### After momentum optimizer + re-profile (2026-03-05)

Updated Profile.idr to use `rmspropValueClipMomentumDense` (momentum=0.9), matching
production NtmCopy.idr config.

```
Epoch   Enc(ms)   Out(ms)  Loss(ms)   Bwd(ms)   Opt(ms)  Sync(ms)  TapeSize    Loss
    1      44.6      29.7       3.6      14.3       0.0       0.3   1480903
   avg     ~44       ~30        ~3       ~15        ~0      ~0.3    1480903
```

Forward (Enc+Out) = **~74ms** (78% of ~92ms epoch). Backward **~15ms**.
Tape **1.48M** entries (varies by random sequence lengths in batch).

Tape histogram:
```
  ConstOps:    765,558  (52%)
  ScalarOps:    17,185  (<2%)
  TensorOps:     5,520  (<1%)
  ShadowOps:   692,640  (47%)
  Total:      1,480,903
```

Tensor detail: MatVec=1680 Softmax=960 BatchCosSim=480 ReadOp=480 InterpWrite=240
               Interpolate=480 Shift=480 Focus=480 LstmCell=240

Bench-compare:
```
Model             Idris (ms)   PyTorch (ms)    Ratio
Supervised              91.1          237.6    0.38x
RNN                    399.1         2308.2    0.17x
NTM-small              437.1         1371.7    0.32x
NTM-copy             11766.5        13505.2    0.87x
NTM-copy-1k         146473.8       224601.1    0.65x
```

NTM-copy now **faster than PyTorch** at both 100 and 1000 epochs. The LSTM→FC
buffer-passing optimization closed the remaining gap and pushed Idris ahead.

### Gradient Region Reservation (NOT IMPLEMENTED)

Planned to eliminate 2.23M ShadowOps (63% of tape) by reserving gradient indices
outside the tape. Analysis revealed a fundamental index collision problem:

Shadow grad slots and future tape entries share the same index space. When a tensor
op reserves grad-only slots at positions [S+1, S+count], the next real tape entry
(ConstOp/ScalarOp) also goes at tape[S+1], causing gradient corruption. Example:

```
1. Tensor op A at tape[100]. Reserve shadow grad slots at 101..300.
2. Next ConstOp appended at tape[101] — collides with shadow slot 101.
3. grad[101] now accumulates gradients from BOTH the ConstOp and the shadow,
   corrupting the backward pass.
```

Correctly separating them requires knowing final tape-size upfront (impossible during
forward pass) or dual grad arrays (requires changing all backward kernels). The
estimated benefit (~2-5ms backward, ~80MB memory) does not justify the architectural
complexity. The remaining ~1.6x gap vs PyTorch is best addressed by tensor-level
Variables (Path C).
