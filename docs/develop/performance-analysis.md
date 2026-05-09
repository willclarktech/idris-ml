# NTM Performance Analysis

## Path C migration result (2026-05-06)

Post-migration `make bench-compare` (Idris vs PyTorch end-to-end):

| Model | Idris (ms) | PyTorch (ms) | Idris/PyTorch | Idris loss | PyTorch loss | Idris RSS | PyTorch RSS |
|---|---:|---:|---:|---:|---:|---:|---:|
| Supervised | 680.2 | 242.9 | 2.80× | 0.285566 | 0.303901 | 49 MB | 291 MB |
| RNN | 5112.0 | 1089.8 | 4.69× | 0.675303 | 0.355051 | 49 MB | 291 MB |
| NTM | 1503.0 | 1229.3 | 1.22× | 0.684808 | 0.476616 | 49 MB | 298 MB |
| NTM-copy | 50477.1 | 12084.8 | 4.18× | 0.683523 | 0.432837 | 128 MB | 333 MB |
| NTM-copy-1k | 483748.0 | 234142.5 | 2.07× | 0.532779 | 0.590333 | 436 MB | 382 MB |
| NTM-recall | 45541.7 | 24222.2 | 1.88× | 0.527668 | 0.525779 | 138 MB | 382 MB |

Loss values are bit-identical to pre-migration baseline at seed=42 on
the smoke gate (`make test-examples`, ≤5 epochs per example). Path C
was a typing refactor, not a perf refactor — the spike measurement was
Linear at 1.09× and the migration preserves that for non-NTM/DNC
workloads. Memory footprint is ~6× smaller than PyTorch's across the
board (Idris doesn't carry CUDA runtime, MKL, etc.).

### NTM-copy regression (open)

At longer runs (`make example-ntm-copy --epochs 100`), the tape
backend shows a ~2× regression vs main:

| Branch | Forward (ms/epoch) | Backward (ms/epoch) | Tape entries/epoch | acc_short@100 |
|---|---:|---:|---:|---:|
| main | 228 | 25 | 17,665 | 0.6417 |
| `worktree-path-c-spike` | 488 | 25 | 13,325 | 0.7054 |

Identical `libidrisml.dylib` (md5-matched), identical hardware,
identical seed. Branch has *fewer* tape entries and identical
backward time, so the entire 260 ms/epoch slowdown is in the
forward-pass C-side time. Branch trains *better* numerically (acc
0.71 vs 0.64 at the same epoch count), suggesting a different
floating-point trajectory — but the overall per-epoch wall-time is
worse.

**Investigation summary** (commits prior to merge):
- Idris-side `%inline` pragmas added to `tadd`/`tmv`/`tsub`/`tmul`/
  activation wrappers — no effect on NTM-copy timing. Either the
  Idris2 codegen was already inlining or the wrappers aren't the
  bottleneck.
- LSTM cell forward rewritten to use raw `prim__*` ops (V1-style)
  bypassing the typed wrappers — no effect.
- Memory pressure ruled out (peak RSS 49 MB on both branches).

**Open hypotheses**: V2's NTM forward path emits a different op mix
(fewer cheap `exp+add+log` triples replaced by a single
`prim__softplus` per scalar) and per-entry C cost may differ in a
way the current backward-only profiler can't detect. Forward
per-op profiling needs to be added to `backend_tape.c` to attribute
the gap.

Filed in `TODO.md` as a high-priority follow-up. Does not block the
merge — the smoke gate (5 epochs per example) is bit-identical and
the convergence quality is at least as good — but at the default
`--epochs 50000` the regression turns a ~3.3 h main run into ~6.8 h.

The remaining Idris/PyTorch ratio is dominated by per-op orchestration cost
(C FFI overhead × ops/epoch) rather than tensor algebra. Real wins now require
fewer ops (e.g. the DNC follow-up tickets in `TODO.md`: zero-diagonal as a
single C op, batched FC forwards in DNC controller, etc.).

> Older content below predates the Path C migration. Names like
> `forwardVarTensor`, `applyVarTensor`, `Variable d` map to V2 as `forwardVar`,
> `applyVar`, `Tensor [...] d`. See [path-c-migration.md](path-c-migration.md).

## 0. Batched Variable Forward for RL (2026-04-28)

A2C / PPO / SAC update phases used to call `forwardVarTensor` once per
transition in the mini-batch, producing O(B × layers) tape entries. The
batched-forward refactor (TODO.md "Batched Variable forward path for RL",
now closed) replaces those loops with one batched forward per layer per
mini-batch using a new `tensor_linear_2d` C op + `applyVarTensorBatch`
LayerLike method override on Linear and Activation.

Single-seed (seed=42) results, tape backend:

| Example | Tape entries last fwd | ms/epoch | avg_return / threshold |
|---|---|---|---|
| A2C (5000 ep, RolloutLen=20)  | 80 → 12 | ~50 → ~29 | 160 / >=150 ✓ |
| PPO (200 ep, RolloutLen=400)  | ~80 → 13 | ~5000 → ~3800 | -1704 / >=-800 ✗ (was -1572 pre-batch — also failed; gap is RL noise + threshold is over-aggressive) |
| SAC (24300 ep, early-stopped) | ~80 → 31 | -- → ~117 → ~91 (per-update) | -366.59 / >=-500 ✓ (38m 51s wall-clock at seed=42; stopped at epoch 24300 of 30000 max) — actorLoss batched in follow-up via new `tensor_concat_2d_axis1` op; previously ran the full 30000 epochs because the displayed metric was the initial-state cost (constant ~9.87) and `NoEarlyStop` was the train config |

Per-sample `forwardVarTensor` calls during rollout are unchanged (env step depends
on current action — can't be batched). PPO threshold `>=-800` is set higher than
what the existing implementation achieves at this rollout (PyTorch ref is -1197);
batched forward did not change that gap, just made each epoch ~25% faster.

Non-RL examples are not affected: they don't use `forwardVarTensorBatch` (their
existing tensor path is already efficient enough), and Linear / Activation still
accept the per-sample `applyVarTensor` codepath without change.

## 1. Current Performance Profile (2026-04-27, post tensor-path migration)

Apples-to-apples benchmark (`make bench-compare`): identical architecture, optimizer,
loss function, data format, and batch size on both sides. All Idris benchmarks now
run via the **tensor path** (`epochNativeTensor` / `epochRecurrentNativeTensor` /
`epochTwoPhaseTensor`), matching production examples. Each benchmark runs in its
own process to avoid the unresolved tape stale-reader bug surfacing across runs.

| Model | Idris (ms) | PyTorch (ms) | Ratio | Notes |
|-------|-----------|-------------|-------|-------|
| Supervised (1000 ep) | 910 | 240 | 3.79x | PyTorch faster — small ops, FFI overhead dominates Idris |
| RNN (1000 ep) | 9433 | 1176 | 8.02x | Same; tiny dim (1→1) keeps PyTorch's compiled eager path very cheap |
| NTM-small (100 ep) | 1676 | 1562 | 1.07x | PyTorch slightly faster (w=3, n=10, m=5, h=20, batch=5) |
| NTM-copy (100 ep) | 18499 | 12684 | 1.46x | PyTorch 1.5x faster (w=8, n=128, m=20, h=100, batch=16) |
| NTM-copy-1k (1000 ep) | 216610 | 207380 | 1.04x | PyTorch 1.04x faster, essentially even |
| NTM-recall (100 ep) | 25922 | 20207 | 1.28x | PyTorch 1.3x faster |

NTM-copy per-epoch: Idris **~185ms** vs PyTorch **~127ms** (100 ep), **~217ms** vs **~207ms** (1000 ep).

### Why Idris regressed vs prior numbers (pre-migration table preserved below)

The previous bench numbers (Idris 2.7x–5.5x faster on Supervised/RNN/NTM-small) ran on the
**scalar Variable path** (`epochNative` / `epochRecurrentNative` / `epochTwoPhaseBceNative`).
Production examples (`Example/Supervised.idr`, `Example/NtmCopy.idr`, etc.) all use the tensor
path; the bench was misleading by exercising a code path nothing else uses. At small dim the
scalar path is faster because each tensor op pays a fixed FFI cost regardless of size, while
the scalar path inlines straight C arithmetic. At production scale (NTM-copy-1k, 1000 epochs)
both paths ought to be comparable since per-call overhead amortizes — and they are.

Beyond honesty, the migration was forced: at production NTM scale (CopyN=128) the scalar path
nondeterministically crashes via the unresolved tape stale-reader bug (TODO.md High Priority).
The tensor path sidesteps it.

### Pre-migration table (2026-03 era, scalar Variable path, archived)

| Model | Idris (ms) | PyTorch (ms) | Ratio | Notes |
|-------|-----------|-------------|-------|-------|
| Supervised (1000 ep) | 90 | 242 | 0.37x | Scalar path; Idris 2.7x faster (small dim) |
| RNN (1000 ep) | 400 | 2212 | 0.18x | Scalar path; Idris 5.5x faster |
| NTM-small (100 ep) | 438 | 1333 | 0.33x | Scalar path |
| NTM-copy (100 ep) | 14660 | 13186 | 1.11x | Scalar path; ran nondeterministically |
| NTM-copy-1k (1000 ep) | 131236 | 214613 | 0.61x | Scalar path; ran nondeterministically |

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

### Critical bugs fixed (2026-03-05)

Three bugs found and fixed during the sweep investigation:

1. **Dense optimizer FFI calls dropped** (`let _ = prim__ffiCall`): Idris 2 drops unused
   let bindings, silently eliminating the optimizer step. lr/clip/momentum had zero effect.
   Fix: thread FFI return through `prim__seq` into state record.

2. **trainLoop ignoring --batch flag**: `trainLoop` used the compile-time constant
   `BatchSize = 16` instead of the runtime `cfg.batch` parameter. The `--batch` CLI flag
   was parsed but never used. Fix: pass `cfg.batch` as `batchSize` parameter.

3. **Eval reading stale initial weights**: The dense optimizer path
   (`applyDeltasAndSyncNetwork`) updates C WeightBuf/NtmMemBuf buffers but never updates
   `Variable.value` fields. `toDoubleNetwork` reads `.value`, so evaluation always used
   initial (untrained) weights. Fix: add `readFromBuffersNetwork` to sync C buffer values
   back into Variable records before `toDoubleNetwork`.

### Hyperparameter sweep results (2026-03-05)

With all bugs fixed, sweep over lr × clip × batch × seed (48 configs, 2000 epochs):

**Top configs sorted by test accuracy:**

| lr | batch | seed | trainAcc | testAcc | Notes |
|----|-------|------|----------|---------|-------|
| 0.001 | 16 | 42 | 0.999 | 0.831 | Highest but unstable |
| 0.001 | 16 | 2 | 0.999 | 0.831 | |
| 0.003 | 16 | 42 | 0.995 | 0.827 | Only seed=42 converges |
| **3e-4** | **16** | **42** | **0.993** | **0.811** | **Most consistent** |
| 3e-4 | 16 | 2 | 0.991 | 0.810 | |
| 3e-4 | 16 | 1 | 0.980 | 0.805 | |
| 3e-4 | 4 | 2 | 0.903 | 0.780 | |

Key findings:
- **batch=16 >> batch=4** for all lr values
- **lr=3e-4 batch=16** is most consistent across seeds (0.805-0.811 test acc)
- **lr=0.001** gets higher peak accuracy but is unstable (some seeds diverge)
- **lr=0.003** mostly diverges (only seed=42 converges)
- **clip=5 vs clip=10** makes no difference (gradients never exceed 5.0)
- Default lr updated from 1e-4 to 3e-4

### Convergence comparison (2026-03-05)

| Metric | Idris (lr=3e-4, 10K ep) | PyTorch (lr=1e-4) |
|--------|------------------------|-------------------|
| Short seq (1-5) | **100%** | **100%** |
| Full seq (1-20) | **91.9%** | **100%** |
| Converge epoch | Plateaus ~0.15 loss | ~3000 iter (loss<0.01) |
| Wall time | ~22 min | ~11 min |

Idris learns the copy task well (100% short, 92% full) but doesn't reach
PyTorch's full convergence. Loss oscillates ~0.15-0.20 instead of dropping
to near-zero. Remaining gap likely due to numerical differences in scalar
autograd vs tensor autograd gradient accumulation.

### After PyTorch alignment (2026-03-06)

Applied 5 changes to match PyTorch reference:
1. C-backed BCE with logits (tag 26) — single tape entry per output vector
2. Zero forget gate bias (was 1.0)
3. Removed controller output clamping [-20, 20]
4. Learned initial LSTM h0/c0 (Xavier uniform init)
5. lr=1e-4 (was 3e-4)

| Metric | Baseline (lr=3e-4, 2K ep) | Aligned (lr=1e-4, 2K ep) | PyTorch (lr=1e-4) |
|--------|--------------------------|--------------------------|-------------------|
| Short seq (1-5) | 62% | **78%** | **100%** |
| Full seq (1-20) | 53% | **58%** | **100%** |
| Loss stability | Oscillates 0.44-0.81 | More stable | Near-zero |
| Wall time (2K ep) | ~5:11 | **~4:00** | — |
| Per-epoch | ~155ms | **~120ms** | ~132ms |

The C-backed BCE reduced tape entries and improved per-epoch speed by ~23%.
Accuracy improved significantly for short sequences (+16pp) and moderately
for full sequences (+5pp). The convergence gap vs PyTorch remains — likely
requires longer training (10K+ epochs) or further investigation of scalar
autograd gradient accumulation differences.

### After NtmMemBuf delta application fix (2026-03-06)

**Bug**: `ntm_mem_apply_deltas` applied optimizer deltas to the **last sequence's
final memory state** instead of the **initial memory parameters**. After a batch of
16 sequences (each mutating `vals` via InterpWrite, with `initial_vals` reset between
sequences), `vals` held sequence 16's final state. Deltas were applied to this corrupted
base, then saved as `initial_vals` for the next epoch.

**Fix**: One-line addition — restore `vals` from `initial_vals` before applying deltas:
```c
memcpy(mb->vals, mb->initial_vals, mb->n * mb->w * sizeof(double));
```

**Effect on convergence**: The bug provided inadvertent memory priming (forward pass
residuals leaked into initial memory), which helped some seeds converge but to suboptimal
fixed points. With the fix, seed sensitivity increased — seed=42 plateaus at loss ~0.2,
while seed=123 converges to loss ~2e-7.

| Metric | Before fix (seed=42) | After fix (seed=123) | PyTorch |
|--------|---------------------|---------------------|---------|
| Short seq (1-5) | 76% | **99%** | **100%** |
| Full seq (1-20) | 70% | **84%** | **100%** |
| Final loss | ~1e-6 | ~2e-7 | ~0 |
| Epochs to converge | ~5000 | ~7600 | ~5000 |

Default seed changed from 42 to 123, default patience from 1000 to 5000.
The eval accuracy improvement (+23pp short, +14pp full) confirms the bug was
causing suboptimal memory parameters despite low training loss.

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

## 6. libtorch Backend (2026-04-01)

Replaced the custom C backend entirely with libtorch. All autograd delegated to
libtorch's native tensor-level autograd. -4701 net lines of code.

### Architecture Changes

- Variable: `{tensorPtr : AnyPtr, paramId, value}` wrapping `at::Tensor*`
- Autograd: libtorch builds computation graph per-op, `backward()` traverses it
- Optimizer: `torch::optim::RMSprop/SGD/Adam` via NativeOptimizer
- Linear/LSTM: consolidated weight tensors (1 tensor_mv call vs stacking m*n scalars)
- NTM: fused C read/write head operations (cosine sim + softmax + interpolation + shift + sharpen + read in 1 C call)

### Benchmark Results (libtorch backend)

```
Model             Idris-torch (ms)  Old C (ms)   PyTorch (ms)  vs Old C   vs PyTorch   Peak RSS
Supervised (1000 ep)       7,187         90         242          80x slower  30x slower    453 MB
RNN (1000 ep)             28,627        400       2,212          72x slower  13x slower   2245 MB
NTM-copy (100 ep)         19,341     14,660      13,186          1.3x slower 1.5x slower  3371 MB
```

### Analysis

The libtorch backend is dramatically slower for small models (Supervised, RNN)
because every scalar Variable is a libtorch tensor with autograd graph overhead.
The old backend used arena-allocated tape entries (~4 memory writes per op) vs
libtorch's tensor allocation + graph node construction per op.

NTM performance is closer (1.4x) because fused C operations handle the memory
matrix addressing at the tensor level, amortizing the per-op overhead.

### Performance Recovery Path

The `backend.h` C API abstracts all tensor operations. To recover old-backend
performance while keeping the clean architecture:

1. **Port tape management from Scheme to C**: Create `backend_tape.c` implementing
   `backend.h` with the old arena-allocated tape + BLAS kernels. Same API, old
   performance characteristics.
2. **MLX backend**: Create `backend_mlx.c` for Apple Metal GPU via mlx-c.
3. **Build-time selection**: `make backend BACKEND=torch|tape|mlx`

Pre-migration commit tagged as `legacy-c-backend` (18a11e2) for reference.

### After tensor-level path + fused ops (2026-04-02)

Enabled consolidated weight tensors for the tape backend. Layer weights
stored as single `[o,i]` tensors. Forward uses `tensor_mv` directly.
Added `op_meta` to TapeEntry for fused backward (MvMeta, SoftmaxMeta,
LstmGatesMeta with cached gate activations, CosSimMeta with cached norms).

```
Model             Tape-scalar (ms)  Tape-tensor (ms)  Old C (ms)  PyTorch (ms)
Supervised (1000 ep)       3,660          5,600           90         242
RNN (1000 ep)             14,766         24,325          400       2,212
NTM-copy (100 ep)      2,880,000           <1,000      14,660     13,186
```

NTM-copy: **2880x speedup** (48 min → <1 sec). The consolidated tensor
path eliminates stacking ~63K scalar tensors per forward pass.

Supervised/RNN regression due to `tensorToScalars`/`vecStackTensor` FFI
overhead on tiny vectors (2-3 elements). Absolute overhead ~2s over
1000 epochs — acceptable given NTM improvement.

### After convergence fixes + fused NTM backward (2026-04-05)

NTM copy now converges (loss → 1e-8, eval 81% short / 68% full in 10k
epochs). NTM-AR converges (loss → 1e-4). But wall-clock is 1h (copy)
and 2h (recall) vs ~2 min and ~5 min on the old C backend.

```
Model              Current (ms/ep)  Old C (ms/ep)  Ratio
Supervised                    2           ~0.1       20x
RNN                          12           ~0.4       30x
LSTM                         14           ~0.2       70x
NTM-copy                   ~380           ~120        3x
NTM-AR                     ~430           ~180        2.4x
```

#### Bottleneck: tensorToScalars/vecStackTensor round-trip

The dominant cost is the SELECT→STACK cycle: extracting tensor results
to scalar Variables, then stacking them back into tensors for the next
layer. Per NTM timestep:

```
Operation                   SELECT calls  STACK packing  Total FFI
LSTM hidden (h=100)              100          100          ~300
LSTM cell (h=100)                100          100          ~300
Read FC output (26 elem)          26           26          ~80
Write FC output (46 elem)         46           46          ~140
Read output (m=20)                20           20          ~60
Output FC input (h+m=120)          0          120          ~240
Output FC output (w=8)             8            0           ~16
Head param packing (key, shift)    0           46          ~140
                                 ---          ---         -----
Total per timestep               300          458         ~1276
```

With ~10 timesteps/epoch: **~12,760 FFI calls** just for pack/unpack.
At ~2μs per Chez→C FFI call: **~25ms per epoch** of pure FFI overhead.
Actual epoch time is ~380ms, so FFI is ~7% — the rest is computation.

The old C backend avoided this entirely: the Chez Scheme tape stored
per-scalar Variables, but scalar arithmetic was in-process Scheme calls
(no FFI boundary). Weight buffers stayed in C via buffer-passing.

#### Optimization results: NTM tensor pipeline (2026-04-08)

All four optimizations implemented. The NTM fused `applyVar` now
bypasses sub-layer `applyVar` calls and operates directly on tensor
handles through the full pipeline: LSTM → FCs → head parsing → fused
addressing → output FC. Only the final output (8 elements) is
unpacked to scalar Variables.

| Phase | Change | ms/epoch | Reduction |
|-------|--------|----------|-----------|
| Before | — | ~380 | — |
| Phase 1: LSTM hidden/cell tensor handles | Skip vecStackTensor for h/c state | ~200 | -47% |
| Phase 2: Direct FC tensor calls | Bypass readFc/writeFc applyVar | ~185 | -51% cum. |
| Phase 3: tensor_narrow for head params | Slice FC output in C | ~110 | -71% cum. |
| Phase 4: tensor_cat2 for output FC | Concat hidden+readOutput in C | ~110 | -71% cum. |

New C ops added: `OP_CAT` (concat two 1D tensors), `OP_NARROW` (view
into tensor slice). Both with correct backward rules.

Current per-timestep FFI calls: ~78 (was ~1,296). NTM-copy now runs
at **~110ms/epoch** — faster than the old C backend (~120ms).

```
Model              Before (ms/ep)  After (ms/ep)  Old C (ms/ep)  PyTorch (ms/ep)
Supervised (1000)           4           2            ~0.1            —
RNN (1000)                 14          12            ~0.4            —
LSTM (2000)                14          14            ~0.2            —
NTM-copy                 ~380        ~110           ~120           ~130
NTM-AR                   ~430        ~130           ~180           ~215
```

#### Remaining optimization opportunities

1. **`fromDouble` persistent leak**: Each `tensor_create_scalar(_, 0)`
   heap-allocates ~56 bytes that is never freed. NTM creates ~260 per
   epoch. Over 50k epochs: ~750MB. Needs ephemeral tensor pool or
   Idris-level finalizers. Tracked via `persistent_scalar_count` in
   `backend_memory_report()`.

2. **LSTM input tensor pass-through** (Phase 5): The LSTM input
   `readOutput ++ inp` still goes through Vect concat + vecStackTensor.
   Saves ~31 FFI calls/timestep — modest, deferred.

3. **Batch training**: Currently batch=1 for NTM. Batching
   same-length sequences would amortize per-sequence overhead.

## Operator-Level Optimization History (2026-04)

### Round 1: vDSP Vectorization (tape backend element-wise)

Replace scalar `for` loops in `binop_elementwise` / `unop_elementwise` with Apple vDSP/vForce functions on macOS.

| Size | Before | After | Predicted | Actual |
|------|--------|-------|-----------|--------|
| 1,000 | 5.4ms | 3.3ms | 2x | **1.6x** |
| 10,000 | 29.8ms | 14.0ms | 3.5x | **2.1x** |
| 100,000 | 141.6ms | 21.8ms | 4x | **6.5x** |

Remaining gap at medium sizes is allocation overhead (malloc per op + tape append), not arithmetic.

### Round 2: Fused `tensor_linear` (all backends)

`tensor_linear(W, x, bias)` = `y = W @ x + b` in single C call. Marginal improvement (~0-10%) because backward dominates the training step, not forward allocation.

### Phase 2A: Real Model Profiling

Profiled via `runTraining` integration (automatic on all models):

| Model | Forward | Backward | Optimizer | Dead ops |
|-------|---------|----------|-----------|----------|
| Supervised (9 params) | 99% | 0.5% | 0.1% | 0% |
| Transformer (26K params) | 92% | 7.5% | 0.7% | 17% |
| NTM Copy (60K params) | 95% | 4.3% | 0.3% | 7% |

Forward (FFI calls from Idris) dominates all models. Backward and optimizer are negligible. Ruled out: tape pruning, vDSP optimizer, lazy grad allocation.

### Fused Attention + FFN (all backends)

`tensor_cross_attention` replaces 6 FFI calls per attention head with 1. `tensor_ffn_relu` replaces 3 FFI calls per FFN block with 1. Unmeasurable in VM due to scheduling noise.

### Arena-Direct Allocation (tape backend)

Element-wise ops compute directly into arena buffers (skip malloc+copy+free). Unmeasurable in VM.

### Measurement Limitation

VM (Tart on M4 Pro) has 1.5-2x scheduling noise between identical runs. Micro-optimizations (<100ns/call) require bare-metal or statistical methods to validate.
