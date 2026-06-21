||| Training-surface slices: Autograd, ParamRegistry, Optimizer,
||| Serialize, MemoryHygiene, Diagnostics, Profiling, TensorCreate.
module Executor.Core.Training

import Executor.Core.Compute
import Executor.Core.Kind

----------------------------------------------------------------------
-- Training surface: six cohesive sub-slices + an aggregate
--
-- What used to be `UserExecutorTraining` (~57 methods, the legacy
-- monolith covering autograd + param registry + optimizer + serialize
-- + profiling + dtype-streamed creators + 3 fused inference ops) is
-- split into six small interfaces that group by *responsibility*:
--
--   UserExecutorAutograd       — grad flag + no-grad bracket + backward
--   UserExecutorParamRegistry  — C-side param table + Polyak blend
--   UserExecutorOptimizer      — SGD/RMSProp/Adam/AdamW + native train step
--   UserExecutorSerialize      — SafeTensors param + optimizer round-trip
--   UserExecutorProfiling      — op counters + epoch hooks + live/peak
--   UserExecutorTensorCreate   — shape queries + item reads + dtype-
--                                streamed creators + fused param-init
--
-- `UserExecutorTraining ex` is now an aggregate constraint defined
-- below: every backend that wants the legacy surface declares trivial
-- empty instances of the six sub-interfaces (one per `primX = ...`
-- block) plus a one-liner `UserExecutorTraining FooExec where`. All
-- existing `UserExecutorTraining ex =>` callsites continue to
-- elaborate unchanged.
--
-- `UserExecutorInference ex` is a parallel aggregate documenting the
-- *minimum* surface a third-party inference-only backend needs:
-- `(Conv + TensorCreate + Transfer + Quant)`. Skipping the four
-- training-only sub-slices (Autograd / ParamRegistry / Optimizer /
-- Serialize) is a real reduction; an inference-only adapter sheds
-- 27 methods of stub work.
--
-- The 3 fused inference ops (`primSdpa2d`, `primRmsNorm2d`,
-- `primSwiGlu2d`) moved to `UserExecutorNN`. They are inference-side
-- compute kernels, not training-only machinery.
----------------------------------------------------------------------

||| Reverse-mode autodiff control surface.
public export
interface UserExecutorCore ex => UserExecutorAutograd (0 ex : Executor) where
  primRequiresGrad    : AnyPtr -> Int
  primSetRequiresGrad : AnyPtr -> Int -> PrimIO ()
  primNoGradBegin     : PrimIO ()
  primNoGradEnd       : PrimIO ()
  primDetach          : AnyPtr -> AnyPtr
  primWithGrad        : AnyPtr -> AnyPtr
  ||| Run reverse-mode autodiff from a scalar loss tensor.
  primBackward          : AnyPtr -> PrimIO ()

||| C-side parameter registry. Tracks named tensors so the optimizer
||| can step them, the serializer can save/load them, and the EMA can
||| blend them. Per-backend; the registry IS the canonical bind from
||| paramId → tensor handle.
public export
interface UserExecutorAutograd ex => UserExecutorParamRegistry (0 ex : Executor) where
  primParamRegister     : String -> AnyPtr -> AnyPtr
  ||| Register a non-learnable buffer (PyTorch register_buffer). Lands in
  ||| the same table as `primParamRegister` so save/load picks it up by
  ||| name, but `primParamIsBuffer` flags it so the optimizer skips it.
  primParamRegisterBuffer : String -> AnyPtr -> AnyPtr
  ||| 1 if the `i`th registered entry is a buffer (never stepped), else 0.
  primParamIsBuffer     : Int -> PrimIO Int
  ||| Number of params registered in this backend's registry.
  primParamCount        : PrimIO Int
  ||| paramId of the `i`th registered param.
  primParamName         : Int -> PrimIO String
  ||| Gradient element `j` of the `i`th registered param.
  primParamGradItemAt   : Int -> Int -> PrimIO Double
  ||| Zero every registered param's gradient.
  primParamZeroAll      : PrimIO ()
  ||| Drop every registered param whose paramId starts with `prefix`,
  ||| releasing its handle retain. Used by the activation-dump path
  ||| (TRACE level) to flush transient
  ||| `__act/<label>/<i>` entries before backward / step so they
  ||| don't pollute the optimizer's full-registry walk.
  primParamEraseByPrefix : String -> PrimIO ()

||| Optimizers built over the param registry. Each backend owns the
||| state buffers; `primNativeTrainStep` fuses zero_grad + backward +
||| clip + step into one FFI call per training step.
public export
interface UserExecutorParamRegistry ex => UserExecutorOptimizer (0 ex : Executor) where
  ||| Create a backend-specific optimizer over this backend's
  ||| registry. The returned handle is opaque and backend-bound; it
  ||| is consumed by `primNativeTrainStep` / the LR setters below.
  primOptimizerCreateSgd     : Double -> AnyPtr
  primOptimizerCreateRmsprop : Double -> Double -> Double -> Double -> Double -> AnyPtr
  primOptimizerCreateAdam    : Double -> Double -> Double -> Double -> AnyPtr
  primOptimizerCreateAdamW   : Double -> Double -> Double -> Double -> Double -> AnyPtr
  ||| Set the optimizer's base LR.
  primOptimizerSetLr      : AnyPtr -> Double -> PrimIO ()
  ||| Set a per-parameter LR override (matched by paramId).
  primOptimizerSetParamLr : AnyPtr -> String -> Double -> PrimIO ()
  ||| Add one exact param name to the optimizer's owned-set (empty = owns all).
  primOptimizerOwnParam   : AnyPtr -> String -> PrimIO ()
  ||| Fused step: zero_grad → backward → clip → step. Args:
  ||| (optimizer handle, clip mode, clip val, loss tensor, loss val).
  primNativeTrainStep     : AnyPtr -> Int -> Double -> AnyPtr -> Double -> Double
  ||| GradScaler-aware fused step (A3 of #410). Args same as
  ||| `primNativeTrainStep` plus a trailing `scale : Double`. The caller
  ||| pre-scaled the loss by `scale`; this op runs zero_grad → backward
  ||| → unscale grads (divide by `scale`, check non-finite) → clip →
  ||| step → return the *unscaled* loss. NaN return = non-finite grad
  ||| detected, step was skipped (caller halves the scale state).
  primNativeTrainStepScaled : AnyPtr -> Int -> Double -> AnyPtr -> Double -> Double -> Double

||| SafeTensors round-trip for the param registry + optimizer state.
||| Layered on Optimizer because the optimizer state buffers it
||| serializes belong to the optimizer instance.
|||
||| TODO audit: `primParamSaveByName` / `primParamSaveByNameRenamed`
||| are niche LoRA / subset-save paths; collapsing them with the full
||| `primParamSave` is a serialization-layer redesign, deferred to the
||| LoRA / PEFT follow-up rather than this audit.
public export
interface UserExecutorOptimizer ex => UserExecutorSerialize (0 ex : Executor) where
  ||| Save every registered param to a .safetensors file (rc 0 = ok).
  primParamSave           : String -> PrimIO Int
  ||| Save only the named subset to a .safetensors file. `namesNl` is
  ||| a newline-separated list of exact paramId names; `count` is the
  ||| number of names. On-disk order matches `namesNl` order. Used by
  ||| the LoRA adapter-only checkpoint path — see `Checkpoint.saveModelMatching`.
  primParamSaveByName     : String -> String -> Int -> PrimIO Int
  ||| Save the named subset under override on-disk names.
  ||| `lookupNamesNl` identifies which registry tensors to save;
  ||| `ondiskNamesNl` is what to write as their JSON-header keys.
  ||| Used by the LoRA/peft adapter export path to wrap idris-ml-side
  ||| names like `bert.[...].lora_A` in peft's on-disk decorations
  ||| `base_model.model.bert.[...].lora_A.default.weight` —
  ||| see `Checkpoint.saveModelMatchingRenamed`.
  primParamSaveByNameRenamed : String -> String -> String -> Int -> PrimIO Int
  ||| Load params from file into the registry, strict dtype.
  primParamLoad           : String -> PrimIO Int
  ||| Load params with a cast policy (`allowCast` = 0/1).
  primParamLoadWithPolicy : String -> Int -> PrimIO Int
  ||| Load only safetensors keys whose name starts with `prefix`.
  ||| `allowCast` semantics as `primParamLoadWithPolicy`; empty prefix
  ||| matches every key (degrades to `primParamLoadWithPolicy`).
  primParamLoadWithPrefix : String -> Int -> String -> PrimIO Int
  ||| Renamed load — symmetric inverse of `primParamSaveByNameRenamed`.
  ||| `registryNamesNl` identifies which registry params to fill;
  ||| `ondiskNamesNl` is the on-disk JSON key each one reads from (both
  ||| newline-joined, `count` long, in lockstep). A pair whose on-disk
  ||| key is absent from the file is skipped (warm-start semantics).
  ||| Used by the peft adapter load path to read keys like
  ||| `base_model.model.bert.[...].lora_A.default.weight` into registry
  ||| params named `bert.[...].lora_A` — see `Checkpoint.load`'s `remap`.
  primParamLoadRenamed : String -> Int -> String -> String -> Int -> PrimIO Int
  ||| Save optimizer state buffers to a file.
  primOptimizerSave       : AnyPtr -> String -> PrimIO Int
  ||| Load optimizer state buffers from a file.
  primOptimizerLoad       : AnyPtr -> String -> PrimIO Int

||| Opt-in memory-hygiene slice. Bracket calls around epoch boundaries
||| and persistent-pool flushes. MLX implements all four meaningfully;
||| tape and torch keep no-op (or cheap arena-reset) bodies — they
||| participate as superclass members of `UserExecutorTraining` /
||| `UserExecutorInference` so existing call sites resolve unchanged.
|||
||| BYO backends without managed-handle pools can omit the instance
||| entirely; `Train.idr`'s epoch loop and `Tensor.idr`'s inference
||| helpers (`releaseAllPersistent`, `resetForEval`) require the
||| constraint, which is the correct signal that those helpers are
||| not callable on backends without the underlying machinery.
public export
interface UserExecutorCore ex => UserExecutorMemoryHygiene (0 ex : Executor) where
  ||| Mark the start of a training epoch's tensor generation.
  primEpochBegin          : PrimIO ()
  ||| End the epoch generation: free wrap-only handles created since
  ||| `primEpochBegin` (grad-mode intermediates), sparing registry params
  ||| and pre-epoch state. mlx frees; tape/torch are no-ops.
  primEpochEnd            : PrimIO ()
  ||| Force the backend to release every persistent at::Tensor /
  ||| mx::array up front (before `main` returns). Inference programs
  ||| that complete with hundreds of MB of live handles hit a 14-22 min
  ||| post-main libtorch CPUAllocator destructor cascade on the CPU
  ||| lanes; calling this brings that work inside the timed region.
  ||| Cheap on tape (arena reset); meaningful on torch + mlx.
  primReleaseAllPersistent : PrimIO ()
  ||| Reset the backend's arena + autograd tape between inference
  ||| forward passes. Tape: drops the arena (drops every intermediate
  ||| from the previous forward — without this, multi-token decode
  ||| accumulates ~GB of arena and OOMs the 16 GB VM around 4-8 tokens
  ||| on Llama-1B). Torch + mlx: free_intermediates + zero param grads
  ||| (mild beneficial, no semantic change for `withNoGrad` callers).
  ||| Safe between forwards in pure-inference loops; UNSAFE in
  ||| training (clobbers param grads). Wraps the existing C
  ||| `backend_reset_for_eval` symbol.
  primResetForEval : PrimIO ()

||| Opt-in diagnostics slice. Handle-count reporting (live + peak) and
||| op-submission counters. All three in-tree backends implement it;
||| BYO backends opt in only if they expose the underlying counters.
||| Superclass of `UserExecutorTraining`, so `Train.idr`'s logging code
||| and `Tensor.idr`'s `perfOpCount` wrapper resolve unchanged for the
||| three in-tree backends.
public export
interface UserExecutorCore ex => UserExecutorDiagnostics (0 ex : Executor) where
  ||| Count of live backend tensor handles (mlx: all_tensors; torch:
  ||| intermediates; tape: tape entries). PrimIO sequencing forces a
  ||| fresh read at each call (Idris-Chez never folds across the IO
  ||| boundary).
  primLiveCount           : PrimIO Int
  ||| High-water mark of live handles since process start — the figure
  ||| that determines whether a backend hits its handle/buffer ceiling.
  primPeakLiveCount       : PrimIO Int
  ||| TODO #393 op-submission diagnostic. Bumped at every from_tensor()
  ||| wrap on torch (counts graph nodes per forward); no-op on tape
  ||| and mlx (their per-op submission story is different — tape is
  ||| eager-CPU, mlx is lazy-batched via mx::array). Use bracketed
  ||| `primPerfReset` (`UserExecutorProfiling`) + `primPerfOpCount` at
  ||| example sites to extract per-forward op counts without
  ||| instrumenting every kernel wrapper.
  |||
  ||| Returns `Int` (not `Bits64`) — `PrimIO Bits64` triggered a
  ||| cumulative-state crash on tape F32 HfLlama (#401). Idris-2's
  ||| chez codegen emits `unsigned-64` for Bits64 returns; something
  ||| about that path corrupts state across calls. `Int` (int64 on
  ||| 64-bit platforms) holds the same value and is the codepath used
  ||| by every other counter FFI in the codebase.
  primPerfOpCount         : PrimIO Int

||| Op-timing profile counters (cumulative timing breakdown across the
||| training surface). Orthogonal to autograd; observes, doesn't mutate.
public export
interface UserExecutorCore ex => UserExecutorProfiling (0 ex : Executor) where
  ||| Reset this backend's op-timing profile counters.
  primProfileReset        : PrimIO ()
  ||| Print this backend's profile breakdown to stderr.
  primProfileReport       : PrimIO ()
  ||| Reset op-submission counters. Bracket with `primPerfOpCount` in
  ||| `UserExecutorDiagnostics` to extract per-forward op counts.
  primPerfReset           : PrimIO ()

||| Tensor creation surface: shape queries + host item reads + dtype-
||| streamed creators + fused param-init. Inference-only adapters
||| implement this *without* the four training sub-slices above.
public export
interface UserExecutorCore ex => UserExecutorTensorCreate (0 ex : Executor) where
  -- Shape / info queries -------------------------------------------
  primTensorDim    : AnyPtr -> Int
  primTensorSizeAt : AnyPtr -> Int -> Int

  -- Scalar reads / data loading
  ||| Read element `(r, c)` from a 2-D tensor as a host Double.
  primItem2d            : AnyPtr -> Int -> Int -> Double
  ||| One-hot encode an int-index buffer into a [len, classes]
  ||| matrix in the dtype selected by the trailing `dtypeTag` (so the
  ||| produced tensor honestly matches the Idris `dt`; 0/1 is exact in
  ||| every dtype). Args: (index buffer, len, classes, dtypeTag).
  |||
  ||| Kept as a mandatory primitive (audit closure): replacing with a
  ||| pure-Idris path (`dtCreate2d` zeros + per-row `setItem`) would
  ||| require N+1 FFI hops per call on the LLM training hot path
  ||| (BERT mini batch_size=16 × seq_len=128 → 2049 hops vs 1). The
  ||| fused kernel is uniformly faster on all three backends; the
  ||| per-call FFI saving compounds across every training step.
  primOneHot            : AnyPtr -> Int -> Int -> Int -> AnyPtr

  -- dtype-streamed creation -----------------------------------------
  -- Each takes a trailing (streamTag, dtypeTag) pair; the backend's
  -- wrapper branches on dtypeTag (0=f32, 1=f64) to pick the right
  -- `_f32_streamed_<b>` / `_f64_streamed_<b>` C symbol. The `dtCreate*`
  -- free functions in `Tensor` source dtypeTag from `RuntimeDType`.
  primCreateScalarStreamed  : Double -> Int -> Int -> Int -> AnyPtr
  primCreateStreamed        : AnyPtr -> AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
  primCreate1dStreamed      : Int -> AnyPtr -> Int -> Int -> Int -> AnyPtr
  primCreate2dStreamed      : Int -> Int -> AnyPtr -> Int -> Int -> Int -> AnyPtr
  primCreateParam1dStreamed : Int -> AnyPtr -> Int -> Int -> AnyPtr
  primCreateParam2dStreamed : Int -> Int -> AnyPtr -> Int -> Int -> AnyPtr
  primCreateParam3dStreamed : Int -> Int -> Int -> AnyPtr -> Int -> Int -> AnyPtr
  primCreateParam4dStreamed : Int -> Int -> Int -> Int -> AnyPtr -> Int -> Int -> AnyPtr
  primCreateState1dStreamed : Int -> AnyPtr -> Int -> Int -> AnyPtr
  primCreateState2dStreamed : Int -> Int -> AnyPtr -> Int -> Int -> AnyPtr
  primCastStreamed          : AnyPtr -> Int -> Int -> AnyPtr

  -- Seed the backend's init RNG (torch::manual_seed equivalent). No-op
  -- on backends without a seedable init-RNG.
  primSetInitSeedStreamed : Bits64 -> Int -> PrimIO ()
