# Gotchas Reference

Comprehensive reference for all known pitfalls in the idris-ml codebase. Organized into four categories. See also [design-decisions.md](design-decisions.md) for rationale behind key choices.

> **Note: Path C migration deleted several V1-era gotchas.**
> The V1 entries below referencing `Variable d` (shape-erased), `nameLayer`/`autoName`, `applyDeltas`,
> `toDoubleNetwork`, `Endofunctor.emap`, `DenseOptimizer`, `NtmMemBuf`, `WeightBuf`, scalar-tape
> internals, and the V1 `LayerLike` interface are **no longer applicable** post-migration. They are
> preserved as historical context — see [path-c-migration.md](path-c-migration.md) for what
> superseded them. Top-of-file sections (Idris 2 / Chez Scheme traps, Training & Numerics, NTM /
> DNC / MLX-specific gotchas) remain accurate for V2 code.

## Idris 2 / Chez Scheme Traps

These are compiler/runtime pitfalls that produce confusing errors or silent misbehavior.

### `total` is a keyword

Idris 2 reserves `total` as a totality annotation keyword. Never use it as a variable/parameter name — produces a cryptic "Couldn't parse declaration" error at the definition clause. Use `numEpochs`, `totalEpochs`, etc. instead.

### Build flags

Forgetting `--source-dir src` or `-p contrib` produces confusing import errors. Examples aren't in the package, so manual flags are needed:

```bash
idris2 --source-dir src -p contrib -o <name> src/Example/<Name>.idr
```

### Top-level `build/ttc/` cache goes stale on where-clause body changes

Idris 2's interface-hash dependency tracking invalidates downstream TTCs only when a module's public interface changes. When you edit the body of a where-clause local inside a public function (e.g. `logEpoch` inside `runTrainingIO`), the interface hash is unchanged and `build/ttc/<ver>/Example/*.ttc` are considered fresh — but they have the old inlined code baked into their Chez-compiled `.so`. Result: library changes install correctly (`~/.idris2/.../idris-ml-0/...`) but single-file `idris2 -o` example builds reuse stale code.

Symptom: you edit a library internal, `make install` succeeds, `make example-foo` succeeds, but the binary runs old behavior. `rm -rf build/ttc` makes the change take effect.

Mitigation: the Makefile has a `build/.library-cache-stamp` sentinel depending on every library `.idr` file. When any is newer than the stamp, the recipe wipes `build/ttc`. `install` depends on the stamp, so every `make example-<name>` / `make check-examples` / `make test-examples` path gets the fresh-cache guarantee transparently. If invoking `idris2` directly outside the Makefile, run `rm -rf build/ttc` after editing library internals.

### Temporary test files

Idris2 requires source files to be in `--source-dir`. Never put test files in `/tmp` — they won't compile. Instead, add temporary test files to `src/Example/` and remove them after debugging.

### Elementwise `(*)`

`Tensor`'s `Num` instance uses elementwise multiply. For matrix-vector products, use `matrixVectorMultiply` or `vectorMatrixMultiply` from Math.idr.

### Arena chunk size must exceed largest single allocation

The arena allocator uses chunked linked-list allocation. If a single `arena_alloc` request matches the chunk size exactly, subsequent allocations after `arena_reset` can hit chunk boundary corruption. Fixed by increasing `ARENA_INIT_SIZE` from 1MB to 4MB. The trigger was embedding output for batch=32 × seqLen=64 × dModel=64 = 131072 doubles = exactly 1MB.

### Large Nat type-level reduction hangs the compiler

Idris 2 represents `Nat` as Peano numbers at the type level — `2304` becomes `S (S (... (S Z)...))` with 2304 constructors. Type unification walks all of them. This causes the type-checker to **hang indefinitely** when:

- An identity layer (same input/output dim) is used at a large dimension. For example, `DropoutState 2304` or `BatchNormState 16 576` requires proving `2304 = 2304`, which means reducing `16 * (12 * 12)` to a chain of 2304 `S` constructors.
- The network chain (`~>`) gets long (10+ layers), compounding unification cost.

**Practical thresholds observed:**
- Dims ≤ 512: fine (dropout at `AfterPool2 = 32 * (4*4) = 512` compiles instantly)
- Dims ~ 2304: hangs (dropout at `AfterPool1 = 16 * (12*12) = 2304` never completes)
- Dims ~ 9216: hopeless (batch norm at `AfterConv1 = 16 * (24*24) = 9216`)

**Workarounds:**
- Place identity layers (dropout, batch norm) only at smaller dimensions (after pooling, before FC)
- Avoid identity layers at conv output dimensions (which can be thousands)
- For batch norm specifically, consider fusing it into the conv layer (conv-bn fusion) rather than making it a separate network layer

**Root cause:** Idris 2 lacks opaque/machine-backed type-level naturals (like GHC's `TypeLits`). This is the single largest practical limitation for type-safe tensor shapes at scale. See the Idris 2 issue tracker for discussion.

### Tensor Foldable reversal

The `foldr` instance for `Tensor` processes elements in reversed order (head into accumulator first). `toList` produces elements backwards. Use direct `Vect` traversal instead when element order matters (e.g., packing into C buffers, extracting prediction values for argmax).

Pattern for correct-order extraction:
```idris
tensorVals : {n : Nat} -> Vector n Variable -> List Double
tensorVals (VTensor xs) =
  let go : Vect k (Scalar Variable) -> List Double
      go [] = []
      go (STensor v :: rest) = prim__item v.tensorPtr :: go rest
  in go xs
```

This caused a subtle bug in the Transformer example where `toList` reversed prediction logits, making the loss function (which uses `vecStackTensor` in forward order) show near-zero loss while the argmax (which used `toList` in reversed order) gave wrong classes.

### Zero-arg FFI CSE trap

Idris 2 compiles zero-argument `%noinline` definitions as constants evaluated once at load time. `tapeGeneration` must take a dummy argument (the tape index) passed through to `prim__tapeGen` to prevent the Chez backend from caching the result. This also applies to any other FFI wrapper reading mutable state. Even making it `foo _ = expr` doesn't help — the argument must be passed THROUGH to the FFI call: `foo dummy = cast (prim__ffi (cast dummy))`.

### FFI side-effect threading

`let _ = ffiCall` is dropped by the compiler since the result is unused. FFI functions with side effects must return a value that is used in subsequent computation. `prim__gradAdd` returns the handle (`AnyPtr`), enabling handle threading through the backward pass. Dense optimizer steps use `prim__seq result st.v` to force evaluation: `let result = prim__rmspropVcStep ... in { v := prim__seq result st.v } st`. Without this, the optimizer call is silently eliminated and raw gradients are applied as deltas (lr/clip/momentum have zero effect).

### `fst`/`snd` re-evaluation trap

When a function with FFI side effects returns a tuple and the caller accesses fields via separate `fst`/`snd` projections (e.g., `fst result`, `snd result`, `fst result` again), Idris 2 compiled to Chez Scheme may re-evaluate the function call for each projection instead of sharing the result. This causes FFI side effects (tape appends, buffer allocations) to execute multiple times. Fix: use `case f args of (a, b, c) => ...` to destructure in a single pattern match. This was a 3x re-evaluation bug in the NTM forward pass — the LSTM controller was called 3 times per timestep instead of once.

### `prim__seq` for evaluation ordering

When two FFI side-effect chains must execute in order but have no data dependency, use `prim__seq a b` (Scheme `(lambda (a b) b)`) to force `a` to evaluate before `b` is used. Chez Scheme evaluates function arguments strictly.

### `foreign-set! 'void*` corrupts memory

Do NOT store C pointers in `foreign-alloc`'d arrays via `foreign-set! 'void*`. It corrupts memory in Chez Scheme — causes "invalid memory reference" crashes with large tape sizes. Use C helper functions (`ext_meta_set`) instead. Similarly, storing C `void*` pointers in Scheme vectors via `vector-set!` works initially but values silently become `#f` (possibly GC-related). Use C-side arrays for pointer storage.

### Chez Scheme output buffering

Stdout is fully buffered when redirected to file/pipe (e.g. background tasks). Use `stdbuf -oL ./build/exec/<name>` to force line-buffering for long-running training.

### C shared library required

`build/libidrisml.dylib` must exist before running any example. Build with `make build/libidrisml.dylib`. The library is loaded by the generated Chez Scheme code at startup. Idris 2 copies the dylib to `build/exec/<name>_app/` at compile time — the Makefile targets also copy it explicitly to ensure the latest version is used. When building manually (not via `make`), you must copy the dylib to the app dir after rebuilding: `cp build/libidrisml.dylib build/exec/<name>_app/`.

## Training & Numerics

Gradient flow, numerical stability, and training patterns.

### `paramId` is required for gradient flow

`Tensor`s without a `paramId` (i.e., `Nothing`) are invisible to the C-side optimizer and won't receive updates. Always pass a paramPrefix to `*LayerAny` constructors:

```idris
ll <- linearLayerAny {i=2} {o=3} "ll0"   -- registers "ll0_weights" + "ll0_bias"
```

For multi-network examples (A2C / PPO / SAC), pick distinct paramId prefixes per network (`"actor_"`, `"critic_"`, `"q1_"`, `"q1tgt_"`, ...) and create per-network optimizers via `nativeAdamGroup "actor_" ...`. The V1 "double `nameLayer` creates stale handles" bug class is structurally impossible in V2 since each layer is named exactly once, at construction.

### `logSoftmax` + `nllLoss`

Separate softmax + cross-entropy creates autograd intermediate gradients of 1/pp (up to 1e6) that destabilize recurrent training. Use `logSoftmaxLayer` + `nllLoss` instead. Note: the aligned NTM uses sigmoid + BCE instead, which doesn't have this issue.

### `pow` zero-base NaN

`pow(0, k)` backward for the exponent computes `0^k * log(0) = 0 * -Inf = NaN`. Fixed by returning 0 when base is 0.

### Detached max in `logSoftmax`

The max subtraction for numerical stability uses a detached constant (`fromDouble . cast`), not a reference to the max Variable. Otherwise the max element receives incorrect gradients.

### Gradient clipping

`adam` clips per-parameter; `adamGlobalClip` clips by global L2 norm (preserves gradient direction). Use `adamGlobalClip` for attention/recurrent models where parameters must coordinate — per-parameter clipping distorts direction and causes periodic loss spikes. Default maxNorm is 50.0 (Collier & Beel); 5.0 was too aggressive.

### Weight initialization

`linearLayer`/`rnnLayer` default to Xavier uniform. Biases are always zero. Init strategies compose a variance method with a distribution sampler: `xavier uniform` (default), `xavier normal`, `he normal`, `xavierGain 1.4 uniform`, etc. Use `linearLayerWith (fixedRange 1.0)` for the old `U(-1,1)` behavior. Use `linearLayerWithBias initFn biasStd` for custom bias init (normal with given std). NTM head FCs use `xavierGain 1.4 uniform` + `normal(0.01)` bias, output FC uses `he uniform` + `normal(0.01)` bias (matching PyTorch reference). NTM memory initialized to `sigmoid(xavier_random)` ≈ values in [0,1] (matching PyTorch's `sigmoid(FC_bias)`). Read output uses kaiming uniform. `Sampler.idr` provides `uniform` and `normal` (Box-Muller); `Init.idr` provides `xavier`, `xavierGain`, `he`, `lecun`, `fixedRange`.

### Hyperparameter tuning

Fix algorithmic issues first (bounded activations, correct clipping, efficient backward pass), then use `scripts/sweep.sh` for systematic grid search. Never manually loop over hyperparameters — see `design-decisions.md` for rationale.

### Periodic GC for long training

NTM training (50K+ epochs) OOMs without periodic forced GC. `forceGC` (exported from Variable.idr) calls Chez `(collect (collect-maximum-generation))` with `(heap-reserve-ratio 1.0)` every 10 epochs in NTM training loops. The `heap-reserve-ratio 1.0` minimizes retained heap (default ~2.0 retains 2x live data), and max-generation collection is more thorough. The FFI lambda must take 0 args — `%World` is erased in Chez Scheme's PrimIO calling convention.

### `getRssMB` peak RSS tracking

`getRssMB` (exported from Variable.idr) returns peak RSS in MB via C `get_rss_mb` (`getrusage(RUSAGE_SELF).ru_maxrss`). Takes a dummy `Nat` arg to prevent CSE (pass epoch number at call sites). Returns peak (high-water mark) RSS, not current — it only goes up. Division to MB done in C to avoid 64-bit return value issues. Used in training loop logs and bench output.

### `getCurrentRssMB` current RSS

`getCurrentRssMB` (exported from Variable.idr) returns current resident memory in MB via `mach_task_info` on macOS. Unlike `getRssMB` (peak), this reflects actual current usage and can decrease after GC. Returns -1 on non-macOS platforms.

### Curriculum learning

Available via the Curriculum module for staged training. The PyTorch-aligned NTM (LSTM controller + RMSprop) does not require curriculum — it converges directly with two-phase training. Curriculum was previously required for feedforward controllers (ajithcodesit finding).

## NTM-Specific

NTM architecture, training protocol, and convergence behavior.

### NTM dimension calculations

`ReadParamWidth m = (m + ShiftKernelSize) + 3` (key of width m + 3-element shift kernel + 3 dynamic params: β, g, γ). `WriteParamWidth m = ReadParamWidth m + m` (addressing params + add vector of width m). The LSTM controller input is `m + inputSize` (read output + input). The output FC input is `h + m` (hidden + read output). The `ntmLayer` constructor takes `{inputSize, outputSize, n, m, h}` as implicit args.

### NTM head parameters

β (key strength), g (interpolation gate), γ (sharpening) are dynamic — extracted from head FC outputs (fed by LSTM cell state). β uses softplus, g uses sigmoid, γ uses `1 + softplus(x)` (unbounded, [1, ∞)). Add vectors are raw linear (no activation). See `forwardReadHeadUnbounded`/`forwardWriteHeadInterp` in Memory.idr.

### NTM state flow

`readHeadOutput` from the previous timestep concatenates with current input to form LSTM input (width `m + inputSize`). LSTM cell state feeds head FCs, hidden state + read output feeds output FC. Memory, addressing weights, and read output all update each step.

### NTM batch size

Copy task converges well with batch=16 (uniform encode-then-decode structure, consistent gradient signal across sequences). Recall task requires batch=1 (online learning) — variable item counts (2-6), random query positions, and content-based retrieval create a complex optimization landscape with many local minima. Batch averaging dilutes the per-sequence addressing signal that the NTM needs to learn distinct write slots and query-triggered retrieval. All reference implementations (Graves 2014, Collier & Beel 2018, vlgiitr) use batch=1 for recall and train for 100K+ iterations. The snipsco/ntm-lasagne implementation found recall gets stuck in local minima even at 500K iterations with larger batches. Default: `NtmCopy.idr` uses batch=1 (seed=42 converges at ~9300 epochs; seed=123 does not converge — batch=1 is seed-sensitive), `NtmAssociativeRecall.idr` uses batch=1.

### NTM two-phase training

Copy/recall use `epochTwoPhaseDenseBce` — encoding inputs fed with outputs discarded, then zero inputs fed during output phase with loss on targets. The C-backed `bceWithLogitsVar` (tag 26) fuses sigmoid + BCE into a single tape entry per output vector, replacing ~7 scalar ops per element. No output activation layer needed.

### No tanh memory bounding

Interpolation write uses raw interpolation (no tanh) to match the PyTorch reference. The Collier & Beel tanh recommendation was for the original erase+add write mechanism, not interpolation write. Tanh caused cumulative degradation during output phase (near-zero write weights still applied tanh every timestep). `tanhBound` in Layer.idr is only used for LSTM gates, not NTM memory. The C kernel `interp_write_compute` supports both modes via `raw_mode` flag (1=raw, 0=tanh); Idris always sets raw_mode=1.

### NTM initial addressing

Read/write addressing weights are initialized to zeros and read output to Kaiming uniform (non-learnable, matching PyTorch reference). `syncLayerBuffers` projects addressing weights onto the probability simplex via `projectWeights` (clamp to [0, epsilon], renormalize) to prevent NaN from `pow(negative, non-integer)` in `focus`.

### NTM early stopping

NTM examples (copy/recall) use windowed-average convergence checking instead of best-loss patience. Parameters: `esThreshold` (default 0.01), `esWindow` (default 1000 epochs), `esPatience` (default 3 consecutive checks). Every 100 accumulated epochs, computes interval average loss, then averages the last `esWindow/100` intervals. Stops when this window average < threshold for `esPatience` consecutive checks. CLI flags: `--es-threshold`, `--es-window`, `--es-patience`. The LSTM example still uses the old best-loss patience mechanism.

### Controller output clipping (removed)

Previously `applyLayerVar` clamped raw NTM controller output to [-20, 20] via `clampVar`. Removed to match PyTorch reference which has no output clamping. The LSTM controller + RMSprop + value clip ±10 provide sufficient stability without artificial clamping.

### NTM-Copy convergence is highly seed-sensitive at 5K epochs

The aligned NTM-Copy model has high variance in convergence rate across seeds at moderate epoch counts. Both the PyTorch reference and the Idris tape backend show ~1/4 pass rate at 5K epochs (only specific seeds hit 99%+ accuracy at that budget). This is the model itself, not a backend bug.

Measurements at seed=42/7/99/123, batch=1, 5K epochs, threshold-disabled (`acc_short / acc_full`):

| Seed | tape         | PyTorch ref     |
|------|-------------:|----------------:|
| 42   | 75% / 59%    | **100% / 100%** |
| 7    | 82% / 74%    | 74% / 60%       |
| 99   | **99.8% / 99.8%** | 76% / 57%  |
| 123  | 75% / 62%    | 72% / 60%       |

Implication: don't read a single-seed under-budget run as a backend bug. Compare the same seed against PyTorch ref before concluding anything. Final convergence (e.g. 25K+ epochs with `WindowedPercentile` early-stop) is the right gate; 5K epoch snapshots are too noisy. The ≥4/5 multi-seed pass rate gate in the convergence plan should be applied at full convergence budgets, not at fixed-epoch checkpoints.

## Architecture & Infrastructure

C kernels, buffer systems, optimizer internals, and the layer system.

### Test suite

Run `make test` for Idris unit tests, `make test-c` for C tests. Tests live in `test/src/Test/*.idr` with `Harness.idr` providing assertion helpers.

### Interface-based layer system

The `LayerLike` interface + `AnyLayer` existential wrapper eliminates all mutual recursion. Each layer type lives in its own module. `AnyLayer` stores the type constructor as a non-erased parameter (`(l : Nat -> Nat -> Type -> Type)`) for interface dispatch after pattern matching. All interface methods need explicit `{i, o : Nat}` because Idris 2 QTT erases Nat type parameters by default. Instance heads for types with extra parameters (e.g., `NtmState n m h`) use `{n, m, h : Nat} -> LayerLike (NtmState n m h)` to make those Nats accessible. Adding a new layer type = one file implementing `LayerLike`, zero edits elsewhere.

### Hybrid tape architecture (legacy — old Chez Scheme tape, not current C tape backend)

Forward pass uses Scheme `foreign-set!` for scalar tape entries (tags/arg1/arg2/vals into `foreign-alloc` arrays — no FFI crossing) and C `ext_meta_set` for tensor op meta pointers (arena-allocated structs). Backward pass runs entirely in C via `walk_backward_ext`, reading meta from `ext_meta` array. PIDs stored in Scheme vector, looked up after C backward returns indices.

### Chunked arena allocator (legacy — current C tape backend has its own arena)

Meta structs AND tensor op output buffers are arena-allocated via `arena_alloc` (`prim__tensorAllocArena`). The arena uses a linked list of chunks (never `realloc`) to prevent invalidating previously allocated pointers when the arena grows mid-forward-pass. Reset frees old chunks and resets the head chunk. Output buffers are safe to arena-allocate because values are read into Variable records during `buildOutputScalars`/`buildVarsFromBuf` before `arena_reset`. `prim__tensorAlloc` (calloc) is still used for persistent WeightBuf allocations.

### Tape-based backward pass (legacy — see C Tape Backend section for current)

`collectGrads` allocates a mutable gradient array via FFI, seeds it with the initial gradient, then `walk_backward_ext` scans the tape in reverse in C. Scalar ops propagate inline; tensor ops dispatch to C backward kernels. ConstOps with non-zero gradient are collected as (index, grad) pairs. Scheme looks up PIDs and builds `SortedMap`. The tape is reset at the end of `collectGrads` (gen++).

### Scheme-native C memory access (legacy)

Use Chez Scheme's `foreign-ref`/`foreign-set!` for reading/writing C-allocated arrays instead of calling C functions per element. This avoids the Scheme->C boundary crossing overhead. See `prim__gradAdd`/`prim__gradGet` and `prim__setDouble`/`prim__setInt32` in Variable.idr.

### C-backed softmax/logSoftmax

`softmaxVar`/`logSoftmaxVar` in Variable.idr use C kernels and record a single SoftmaxOp/LogSoftmaxOp tape entry per vector instead of ~29 scalar entries. `applyLayerVar` dispatches NormalizationLayer "softmax"/"logSoftmax" to these.

### C-backed NTM memory ops

`batchCosineSimilarityVar`, `readOpVar`, `writeOpVar`, `interpolationWriteVar` in Variable.idr use C kernels (BatchCosSimOp/ReadOpOp/WriteOpOp/InterpolationWriteOp, tags 15-18) to reduce tape entries per NTM timestep. `forwardReadHeadUnboundedVar`/`forwardWriteHeadInterpVar` in Layer.idr wire these into the Variable-specialized NTM forward pass. Generic `forwardReadHeadUnbounded`/`forwardWriteHeadInterp` in Memory.idr remain parameterized on `NormalizationFunction ty` for the Double path.

### C-backed addressing ops

`interpolateVar`, `shiftVar`, `focusVar` in Variable.idr use C kernels (InterpolateOp/ShiftOp/FocusOp, tags 21-23) replacing ~1400 scalar tape entries per head with 3 tensor ops. `shiftVar` takes a pre-softmax'd kernel (apply `softmaxVar` first). Used in both `forwardReadHeadUnboundedVar` and `forwardReadHeadUnboundedVarBuf` in Layer.idr.

### C-backed LSTM cell op

`lstmCellVar` in Variable.idr uses a C kernel (LstmCellOp, tag 24) fusing bias add + gate activations (sigmoid/tanh) + cell/hidden update into a single tape entry. Replaces ~1700 scalar entries per LSTM timestep with 1. The two matmul ops (iW*x, rW*h) remain as separate MatVecOps. `applyLayerVar` in Layer.idr dispatches to `lstmCellVar` for the Variable-specialized LSTM path.

### C-backed BCE with logits

`bceWithLogitsVar` in Variable.idr uses a C kernel (BceWithLogitsOp, tag 26) fusing sigmoid + BCE loss into a single tape entry per output vector. Forward: `(1/n) * sum_i [max(p_i,0) - p_i*y_i + log(1+exp(-|p_i|))]`. Backward: `d_p_i = (1/n) * (sigmoid(p_i) - y_i) * d_loss` (gradients to predictions only, not targets). `epochTwoPhaseDenseBce` in Backprop.idr uses this directly instead of the scalar `binaryCrossEntropyWithLogits`. Meta stored via Scheme-side `ext_meta_set` (NOT C-side `tape_meta`) to match `walk_backward_ext` dispatch.

### Persistent NtmMemBuf

NTM memory matrix kept as persistent `NtmMemBuf` C struct across timesteps. Eliminates 4x per-timestep packMatrix (2560 elements each). Buffer initialized in `nameParams`, synced after `applyDeltas` via `syncLayerBuffers`, epoch-cached tape registration via `prim__ntmMemBufEnsure`. Buffer-aware ops: `batchCosineSimilarityVarBuf`, `readOpVarBuf`, `interpolationWriteVarBuf` in Variable.idr. **Per-sequence reset**: NtmMemBuf stores `initial_vals` (snapshotted at init and after optimizer deltas). `prim__ntmMemBufReset` restores `vals` from `initial_vals` and invalidates cache (forces tape re-registration). `resetNtmMemBufs` in Layer.idr reconstructs the Network with the reset buffer, called before each sequence in `calculateLossTwoPhaseVar`/`VarBce` to prevent cross-sequence mutation.

### Bias WeightBuf

LinearLayer and LstmLayer have bias WeightBuf fields (`bBuf : Maybe AnyPtr`) alongside weight WeightBufs. `nameParams` allocates them, `syncLayerBuffers` syncs after `applyDeltas`. LinearLayer fuses MatVec+Bias in a single C kernel (`matrixVectorMultiplyVarBufBias`). LstmLayer reads bias from WeightBuf in the C LSTM cell kernel (`lstmCellVarBuf`/`lstmCellVarFromBufs`). Eliminates per-timestep bias re-registration (~160K tape entries/epoch).

### Learned LSTM h0/c0

LstmLayer has `h0Buf : Maybe AnyPtr` and `c0Buf : Maybe AnyPtr` fields for learnable initial hidden/cell states. Initialized with Xavier uniform in `lstmLayerWith`. Named as `prefix_h0`/`prefix_c0` in `nameParams`, allocated as WeightBufs. Synced via `applyDeltasAndSyncLayer`/`readFromBuffersLayer`. Matches PyTorch reference's `nn.Parameter(torch.zeros(...))` learnable initial states.

### Buffer-passing MatVec to LstmCell

`matrixVectorMultiplyVarBufOut` returns raw `(AnyPtr, Int)` buffer+tapeStart instead of Variables. `lstmCellVarFromBufs` consumes these directly via `buf_to_meta` C helper, avoiding `buildOutputScalars`+`packVec` roundtrip for 2x4o intermediate elements per LSTM timestep.

### Bulk buildOutputScalars

`prim__appendOutputConstOff` bulk-appends ConstOps from a C buffer with offset in a single Scheme FFI call (internal loop), replacing per-element `tapeAppendConst`. `buildVarsFromBuf` reads values with sequential tape indices. Used by all tensor op output paths.

### Shadow ConstOps (tag=25) (legacy)

Buffer-passing ops (`*BufOut`, `*BufIO`) create shadow ConstOps instead of regular output ConstOps. These provide gradient slots without values/pids — skipped during backward collection (`if (tag == 25) continue`). Tags set via C bulk `tape_set_shadow_tags` instead of per-element Scheme `foreign-set!`. Shadow ConstOps still occupy tape entries; full elimination requires gradient region reservation (not yet implemented).

### C-side pid filtering (legacy)

`walk_backward_ext` filters ConstOps by integer `pid_id` (C-side `tape_pid_ids` array, parallel to tape). Only collects ConstOps with `pid_id >= 0` (named parameters). Dense pid_ids assigned via Scheme `pid-to-id` hash table in `prim__tapeSetParamId`. Set in three paths: `prim__tapeSetParamId` (initial naming), `prim__tapeAppendConst` (stale re-registration), `prim__tapeEnsureBulkConst`/`prim__ntmMemBufEnsure` (WeightBuf/NtmMemBuf). Reset via `tape_pid_ids_reset` after backward.

### `out_tape_start` semantics (legacy)

Tensor op meta structs store `out_tape_start = idx + 1` (first output gradient index, NOT the op entry index). Backward kernels read `meta->out_tape_start` directly without `+1`. Set by `tensor_op_set_out(tag, meta, idx+1)` during `prim__tapeAppendTensorOp`.

### Dense optimizer (legacy)

`DenseOptimizer`/`DenseOptimizerState` in Optimizer.idr use C arrays indexed by integer pid_id instead of `SortedMap String Double`. `collectGradsDense` accumulates gradients into a pre-allocated C array during backward (no per-result FFI calls, no SortedMap inserts). The gradient array is persistent across epochs via `grad_alloc_reuse` (calloc once, memset-zero on reuse). Optimizer step functions (`rmsprop_vc_step`, `sgd_step`, `adam_gc_step`) operate in-place on the array. Dense epoch functions use `applyDeltasAndSyncNetwork` which applies deltas directly to C buffers via `buf_apply_deltas` (bypassing `emap` + `syncLayerBuffers`). NTM examples use this path via `epochTwoPhaseDense`; supervised/LSTM examples still use the original `SortedMap` path. Must call `getNumPids 0` after `autoName` to get the parameter count for `initDenseState`.

### C-bulk delta application (legacy)

`applyDeltasAndSyncLayer`/`applyDeltasAndSyncNetwork` in Layer.idr apply optimizer deltas directly to WeightBuf/NtmMemBuf C arrays via `buf_apply_deltas(vals, pid_ids, count, deltas)`. Each buffer stores a parallel `int *pid_ids` array (populated during `nameParams`). This bypasses the Scheme `emap (applyDeltasDense ...)` + `syncNetworkBuffers` traversals (~63K Variable operations). WeightBuf pid_ids stored in Scheme 6-vector slot [4]; NtmMemBuf pid_ids stored in C struct field. Cache generations are reset to force tape re-registration next epoch. **Important**: Variable.value fields are NOT updated — call `readFromBuffersNetwork` before `toDoubleNetwork` to sync C buffer values back into Variable records for evaluation.

## C Tape Backend (backend_tape.c)

These gotchas apply to the C tape backend (`BACKEND=tape`), which implements `backend.h` with a flat Wengert list in C.

### `tensor_select` rank-0 identity

`binop_elementwise` produces scalars (rank 0, numel=1) when both inputs have numel==1. If `tensorToScalars` then calls `tensor_select` on the rank-0 result, it must return the tensor itself (identity) to preserve the tape entry. The fallback path (`make_scalar(t->data[index], t->requires_grad)`) creates a copy with NO tape entry, breaking the gradient chain. Affected: any layer with output size 1 (e.g., LSTM example's Linear<1:1>).

### Arena vs calloc for view tensors

`tensor_select` (rank-1 and rank-2) creates view Tensor structs. Use `arena_alloc` (freed on tape reset) not `calloc` (never freed). Each `tensorToScalars(n)` call creates n view tensors; over thousands of epochs this leaks GBs. Exception: `tensor_view_1d`/`tensor_view_2d` are called once in `nameLayer` and must persist — keep as `calloc`.

### Optimizer per-element buffers

RMSprop/Adam velocity and momentum buffers must be sized by total parameter ELEMENTS, not total parameter count. A [400,29] weight matrix needs 11,600 velocity slots, not 1. Index via `param_element_offset(i) + j`. SGD is unaffected (no buffers).

### Fused ops require backward rules — and prefer not to add them at all

Any fused C operation that sets `requires_grad=1` on its output MUST also append tape entries and implement backward cases. Without a backward rule, the gradient chain breaks silently — the op's result gets gradient but it's never propagated to inputs.

The corollary: **don't add architecture-specific fused C ops in the first place.** A `tensor_*` op should be something a PyTorch user would expect at the FFI surface (`F.cosine_similarity`, `nn.LSTMCell`, etc.). Per-paper fusions like NTM's read-head pipeline belong in Idris, composed from primitives. The previous `tensor_ntm_read_head` / `tensor_ntm_interp_write` fusion was rolled back; NTM now composes its addressing in `Layer/Ntm.idr` like DNC always did.

### NTM state is not a parameter

NTM memory, readAddr, writeAddr, readOutput are per-sequence state, NOT learned parameters. Do NOT register them with `prim__paramRegister` — the optimizer will corrupt them with gradient updates. Use `tensor_create_state_2d`/`tensor_create_state_1d` (persistent, `requires_grad=0`, no param registration). The decomposed addressing primitives still propagate gradients to the key, beta, g, gamma, shift inputs (which DO come from FC layers with `requires_grad=1`).

### `tensor_matmul` vector-matrix backward

`tensor_matmul` for [n]×[n,m] → [m] needs `OP_VECMAT` (not `OP_DOT`). The DOT backward only reads `grad[0]`, which is wrong for vector results. The VECMAT backward: `d_a[i] = Σ_j grad[j]*b[i,j]`, `d_b[i,j] = grad[j]*a[i]`.

### Arena never frees chunks

`arena_reset()` resets `.used` pointers but never frees chunks. Memory grows to accommodate the peak forward+backward pass, then stabilizes. For NTM with n=128, the peak is ~8MB (6 chunks). This is by design (avoids realloc invalidation) but means RSS never decreases within a run.

### `tape_reset` must call `arena_reset`

The arena holds all intermediate tensors from the forward+backward pass. `tape_reset()` must call `arena_reset()` to reclaim this memory. Without it, the arena grows by ~1.7MB/epoch indefinitely (the original bug that caused 8.5GB memory usage).

Additionally, `tape_reset` must free: (1) OP_STACK `inputs` arrays (heap-allocated `Tensor**`), and (2) grad arrays on non-persistent tensors (heap-allocated by `ensure_grad` during backward, leaked when arena tensors are reused).

### `fromDouble` persistent scalar leak

`tensor_create_scalar(value, 0)` must heap-allocate (persistent) because Idris may cache `fromDouble` results in Variables across epochs (e.g., `let data = map fromDouble ...` evaluated once, reused). Arena allocation would cause use-after-free when `arena_reset` runs between epochs. The tradeoff: ~56 bytes leaked per `fromDouble` call. For NTM training with fresh data each epoch, this is ~15KB/epoch. Over 50k epochs: ~750MB. A proper fix requires either Idris-level finalizers or an explicit ephemeral tensor pool.

### `toDoubleLayer` must use tensor handles for learned weights

After training with `NativeOptimizer`, the optimizer mutates param tensor data in-place. The scalar Variable `.value` fields are stale (from initial forward pass). `toDoubleLayer` must read from tensor handles (via `buildDoubleMatrix`/`buildDoubleVector` using `prim__item2d`/`prim__item1d`) for learned weights. Exception: non-learnable state (NTM memory, addressing) can use `map value` since those retain initial values.

## MLX Backend (backend_mlx.cpp)

### Tensor lifetime: tape vs non-tape

All `Tensor*` objects self-register in `all_tensors` via the constructor. `tape_reset()` frees non-persistent ones. Unlike the tape backend's arena (which bulk-frees by resetting a pointer), MLX individually `delete`s each tensor. This means:

- **Ephemeral data tensors** (from `tensor_create` with `requires_grad=0`): NOT on the tape, but still tracked in `all_tensors`. Freed at `tape_reset()`.
- **Persistent tensors** (params via `param_register`, state via `tensor_create_state_*`, views via `tensor_view_*`): Marked `persistent=1`, survive `tape_reset()`.
- **TensorPair structs**: Tracked in `all_pairs`, freed at `tape_reset()`.

Without this tracking, non-tape tensors (every `bulkToTensor` call, BCE constants, zero tensors) would leak ~250 objects per NTM epoch × 50K epochs = 2-4GB.

### State tensors must be persistent

`tensor_create_state_1d`/`_2d` must set `persistent=1`. Without it, NTM memory matrix, addressing weights, and read output tensors are freed at the first `tape_reset()`, causing use-after-free. The tape backend uses separate `calloc` with `persistent=1`; torch uses `from_tensor_persistent()`.

### Broadcasting gradient reduction

Binary op backwards (ADD, SUB, MUL, DIV, POW) must call `reduce_grad()` to sum gradients over broadcast-expanded dimensions. Without this, scalar × vector operations (e.g., `g * content_weights` in NTM interpolation) produce vector-shaped gradients for scalar parameters, corrupting the autograd chain.

### RMSprop optimizer must be implemented

`optimizer_step` must have a `case 1:` for RMSprop. Without it, `optimizer_step` falls through to `default: break;` (no-op) and weights are never updated. This silently affects any example using `nativeRmsprop` (NTM Copy, NTM Recall). SGD (case 0) and Adam (case 2) were implemented first; RMSprop was missing until the bug was caught.

### Conv1d circular d_kernel backward shift sign

`OP_CONV1D_CIRC` d_kernel backward must use `shift = j - half_k` (matching forward indexing), not `shift = half_k - j`. The forward computes `result[i] = sum_j(input[(i - half_k + j) % n] * kernel[j])`, so the backward needs `d_kernel[j] = sum_i(grad[i] * input[(i - half_k + j) % n])`. The inverted shift corrupts shift kernel gradients, preventing the NTM from learning memory addressing order.

### Fused OP_NORMALIZE for attention normalization

The attention weight normalization `focused = powered / sum(powered)` must use a fused `OP_NORMALIZE` op, not separate `div + sum + add` ops. The decomposed backward computes `d/d(numerator) = grad/denom` and `d/d(denominator) = -grad*numer/denom²` separately — these are huge values that nearly cancel. With peaked attention (near-converged NTM), catastrophic cancellation produces NaN. The fused formula `d_a[i] = (d_r[i] - dot(d_r, r)) / (sum(a) + eps)` avoids this.

### `mx::transpose` requires explicit axes

`mx::transpose(x)` with no axis argument reverses ALL dimensions — it does NOT swap the last two like PyTorch/NumPy. For a 2D matrix, this gives the wrong result. Use `mx::transpose(x, {1, 0})` for 2D, `mx::transpose(x, {0, 2, 1})` for batched 3D. This bug was the root cause of wrong MM backward gradients, broken NTM read head addressing, and incorrect transpose test values.

### `mx::array(double)` defaults to float32

`mx::array(3.0)` creates a float32 scalar, not float64. Reading it back with `item<double>()` returns 0.0 (reinterpreting float32 bits as float64). Always use `mx::array(value, mx::float64)` for double-precision scalars. This caused `tensor_create_scalar` to produce zero-valued tensors.

### Metal float32 transcendentals

MLX's Metal GPU computes `exp`, `sigmoid`, `tanh`, etc. in float32 even when the input array is float64. Expect ~1e-6 precision for these ops, not 1e-10. Test tolerances for transcendental functions should be 1e-5 or wider on MLX.

### Non-contiguous views and `data<T>()`

`mx::transpose` and similar ops return views with swapped strides. The raw `data<double>()` pointer still points to the original contiguous memory layout. Index arithmetic like `data[row * cols + col]` produces wrong results on transposed views. Use `mx::flatten` to force a contiguous copy first, or use MLX's indexing API.

### Lazy eval use-after-free in `tape_reset`

`tape_reset()` must call `mx::eval()` on ALL tensors before deleting any non-persistent ones. MLX array operations are lazy — `mx::add(a, b)` captures references to `a` and `b`, not copies. If `a` is deleted by `tape_reset` while a surviving tensor's lazy graph still references `a->data`, the next `mx::eval` hits a dangling pointer. The fix: batch-eval all tensor data and grads before the delete loop.

### NTM convergence comparison

MLX NTM (post-`tensor_linear` bias-on-tape fix and 2026-05-08 NTM model alignment) converges on `ntm-copy` to acc_short=0.994, acc_full=0.999 at epoch 8200 with the standard ES gate (seed=42, batch=1). Comparable to PyTorch ref's 100%/100% at ~4600 epochs. The aligned model is highly seed-sensitive at moderate budgets — see "NTM-Copy convergence is highly seed-sensitive at 5K epochs" in the NTM-Specific section.

### Replay-based VJP: every dependency must be on the tape

The MLX backend computes gradients by replaying the forward tape inside a closure passed to `mx::vjp`. The replay reconstructs each tensor's value from its tape op's `arg1`/`arg2`/meta — it does NOT use the result tensor's `data` field. Any forward op that mutates `data` after a sub-step (e.g. `result = mx::add(result, bias->data)` after the matmul) but doesn't record the dependency on the tape will produce a replay value that differs from the actual forward, and the missing input gets zero gradient (the VJP can't see a dependency that isn't in the closure).

**The bug this caught**: `tensor_linear(W, x, bias)` was recording only `OP_MV(W, x)` but adding the bias to `result.data` directly. When `tlinear` chained (one `tlinear`'s output passed as the next `tlinear`'s bias arg, e.g. `tlinear rwT h (tlinear iwT input bT)` in the LSTM combined-gates expression), the replay computed `pool[outer] = rw @ h` only — the entire inner branch (`iw`, `input`, `b`) had no path to the loss in the VJP. Gradients for every parameter on the inner branch (LSTM `iw`/`rw`/`b`, every NTM FC weight/bias in chained-FC settings) collapsed to exactly zero, and mlx training stalled at the random-baseline loss for the aligned NTM-Copy model.

**Fix**: decompose `tensor_linear` into `tensor_mv` + `tensor_add` when a bias is provided, so each dependency lands on the tape. Two tape entries instead of one is the small per-call cost; correctness requires every input read by the forward to be reachable from the tape.

**Diagnosing this class of bug**: the `DEBUG_PARAM_GRADS` env-var hook in `optimizer_step` (mirrors the one in `backend_tape.c`) dumps per-param grad L2 norm at the first optimizer step. Any `requires_grad=1` param with `grad_l2=0` is the smoking gun — that param is in the registry but has no path to the loss in the replay graph.

### Softplus must use the numerically stable form in float32

`softplus(x) = log(1 + exp(x))` overflows in float32 for x > ~88 (where `exp(x) > 3.4e38`). Use the stable form `softplus(x) = max(0, x) + log(1 + exp(-|x|))` instead — it gives the same answer everywhere, reduces to `x` for large positive x and to `exp(x)` for large negative x, and never overflows.

The bug this caught: NTM content addressing computes `betaT = softplus(scalar)` for the sharpening factor, then `softmax(betaT * cos_sim)`. With the naive softplus, once the controller drove `scalar` past ~88 (mid-training, model already at 94% accuracy), `betaT` jumped to `+inf`, the multiply produced `±inf` softmax inputs, and the whole content-addressing path NaN'd the loss in a single epoch. Tape uses a branch-on-magnitude form; torch uses `torch::softplus` (stable). Only mlx had the naive form.

**Diagnosing it**: `DEBUG_NAN_TRAP=1` in `tensor_backward` walks the forward tape on first appearance of NaN/Inf in any param grad, prints the first NaN-producing op and its args' value ranges. Found this one in one shot: `first NaN at tape[2165] op=SOFTMAX_2D` with arg1 (a MUL output) range `[-inf, +inf]`.

## Torch Backend (backend_torch.cpp)

### View tensors must be persistent

`tensor_view_2d`/`tensor_view_1d` must use `from_tensor_persistent()`, not `from_tensor()`. Views are created once at `nameLayer` time and referenced by scalar Variables for the lifetime of the model. If tracked as intermediates, `free_intermediates()` frees them after the first epoch, causing crash in `refreshValue` → `prim__item` on stale pointers.
