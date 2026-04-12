# Gotchas Reference

Comprehensive reference for all known pitfalls in the idris-ml codebase. Organized into four categories. See also [design-decisions.md](design-decisions.md) for rationale behind key choices.

## Idris 2 / Chez Scheme Traps

These are compiler/runtime pitfalls that produce confusing errors or silent misbehavior.

### `total` is a keyword

Idris 2 reserves `total` as a totality annotation keyword. Never use it as a variable/parameter name — produces a cryptic "Couldn't parse declaration" error at the definition clause. Use `numEpochs`, `totalEpochs`, etc. instead.

### Build flags

Forgetting `--source-dir src` or `-p contrib` produces confusing import errors. Examples aren't in the package, so manual flags are needed:

```bash
idris2 --source-dir src -p contrib -o <name> src/Example/<Name>.idr
```

### Temporary test files

Idris2 requires source files to be in `--source-dir`. Never put test files in `/tmp` — they won't compile. Instead, add temporary test files to `src/Example/` and remove them after debugging.

### Elementwise `(*)`

`Tensor`'s `Num` instance uses elementwise multiply. For matrix-vector products, use `matrixVectorMultiply` or `vectorMatrixMultiply` from Math.idr.

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

### `paramId` requirement / autoName

Variables without a `paramId` (i.e., `Nothing`) are invisible to gradient collection and won't receive updates. Use `autoName` (preferred) or `nameParams`/`nameNetworkParams` before training. `autoName` assigns type-based prefixes with per-type counters (`ll0`, `ll1`, `rnn0`, `lstm0`, `ntm0`, ...) and scopes NTM sub-layer names under their parent (`ntm0_lstm0_`, `ntm0_readFc_ll0_`), preventing the collision bug in `nameNetworkParams`. `setParamId` writes to both the Variable record and the tape's pid vector.

### Double `nameLayer` creates stale handles

Calling `nameLayer` on a layer state, then wrapping it in `autoName $ OutputLayer (MkAnyLayer ...)`, names the state TWICE. Each `nameLayer` call creates new consolidated parameter tensors via `prim__paramRegister`. The first set becomes stale — the optimizer only updates the second set (from `autoName`). If you hold a reference to the pre-`autoName` state (e.g., for a batched forward function), it reads stale weights and the model won't converge.

Fix: either use `autoName` alone (let it call `nameLayer` internally), or call `nameLayer` once and skip `autoName`. If you need both a `Network` and a direct state reference, name once and share:

```idris
let namedTfm = nameLayer "tfm0" tfm
    model = OutputLayer (MkAnyLayer ... namedTfm)  -- no autoName
```

### Tape generation staleness

After `collectGrads` resets the tape (gen++), Variables from the previous epoch are stale. `ensureOnTape` detects this via generation mismatch and re-registers with current `.value`. Same stale Variable used N times creates N Const entries — gradients accumulate correctly via `mergeWith (+)` on paramId.

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

Fix algorithmic issues first (bounded activations, correct clipping, efficient backward pass), then use `scripts/sweep.sh` for systematic grid search. Never manually loop over hyperparameters — see `docs/design-decisions.md` for rationale.

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

### Fused ops require backward rules

Any fused C operation that sets `requires_grad=1` on its output MUST also append tape entries and implement backward cases. Without a backward rule, the gradient chain breaks silently — the op's result gets gradient but it's never propagated to inputs. The NTM fused ops (`tensor_ntm_read_head`, `tensor_ntm_interp_write`) were originally forward-only; backward rules were added for OP_NTM_READ_HEAD, OP_NTM_READ_HEAD_READ, and OP_NTM_INTERP_WRITE.

### NTM state is not a parameter

NTM memory, readAddr, writeAddr, readOutput are per-sequence state, NOT learned parameters. Do NOT register them with `prim__paramRegister` — the optimizer will corrupt them with gradient updates. Use `tensor_create_state_2d`/`tensor_create_state_1d` (persistent, `requires_grad=0`, no param registration). The fused addressing ops still propagate gradients to the key, beta, g, gamma, shift inputs (which DO come from FC layers with `requires_grad=1`).

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
