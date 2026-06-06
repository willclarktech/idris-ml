||| Pluggable-Executor interface. Phase 2.1 of the refactor; see
||| `docs/develop/design-decisions.md` "Pluggable Executor via sliced
||| `UserExecutor` interfaces" for the design.
|||
||| Phase 2.1 ships the **lifecycle + arithmetic** slice (~20 ops).
||| Later slices (2.2-2.5) extend with linear / NN / conv / tape ops.
|||
||| Users implementing their own backend declare an empty type and
||| an instance:
|||
|||   data MyDev : Type where MD : MyDev
|||
|||   UserExecutorCore MyDev where
|||     primAdd = prim__addMine
|||     ...
|||
||| The built-in `TapeExecutor` / `TorchExecutor` / `MlxExecutor` (in `Executor.Tape`,
||| `Executor.Torch`, `Executor.Mlx`) forward to the per-backend C
||| symbols emitted by Phase 1's rename headers
||| (`tensor_add_tape` / `tensor_add_torch` / `tensor_add_mlx`).
module Executor.Core


----------------------------------------------------------------------
-- `Executor` kind alias
--
-- `Executor` is a 0-quantity alias for `Type`. Tensor's `d` phantom is
-- declared as `(0 ex : Executor)`, which is exactly `(0 d : Type)`
-- underneath but reads as "d is a device tag" at every kind-binder
-- site. No type-system enforcement: nothing stops a caller writing
-- `Tensor [4] Bool`. But construction (`primCreate*`) and operations
-- (`tadd` etc.) both require `UserExecutorCore ex =>`, so non-device
-- `d`s can be declared but never inhabited or operated on.
--
-- See `docs/develop/design-decisions.md` "Open `d` kind: why
-- `Executor = Type` instead of a real sub-kind" for the alternatives
-- considered and why we kept it open.
----------------------------------------------------------------------

public export
0 Executor : Type
Executor = Type


----------------------------------------------------------------------
-- Linked — backend-linkage capability
--
-- Empty capability marker, sibling to `Compatible (device, dtype)`.
-- `Linked ex` declares "device `d`'s backend is compiled into this
-- `libidrisml`." Instances are NOT hardcoded here — they're emitted by
-- the generated `HwConfig` module from the build's `BACKEND` list, so a
-- torch-only build has no `Linked (MlxExecutor _)` instance and `MlxExecutor`
-- becomes unspellable at any constructor carrying the `Linked ex =>`
-- constraint. This is the compile-time *linkage* half of device
-- availability; the runtime *hardware-presence* half is EAFP (attempt
-- construction, catch the backend's exception). See
-- `docs/develop/device-availability-gating.md`.
--
-- Linkage is per-backend, not per-hardware-variant: a torch build admits
-- every `TorchExecutor hw` (TCpu / TMps / TCuda n) at the type level; whether
-- the MPS chip or `cuda:n` actually exists is the runtime question.
----------------------------------------------------------------------

public export
interface Linked (0 ex : Executor) where


----------------------------------------------------------------------
-- HardwareClass — physical-silicon classification (orthogonal to backend)
--
-- Backend-scoping (TorchExecutor TMps vs MlxExecutor MGpu) is correct: you can't
-- mix their tensor handles even though both live on the same Apple GPU.
-- But that scoping hides the hardware *commonality*. `HardwareClass`
-- recovers it as runtime data — for *reporting* / grouping during
-- discovery only. It never unifies tensor types: TMps and MGpu both map
-- to `AppleGpu`, yet their tensors still can't meet. See
-- `docs/develop/device-availability-gating.md`.
----------------------------------------------------------------------

public export
data HardwareClass = HostCpu | AppleGpu | Nvidia Nat | Other String

public export
Eq HardwareClass where
  HostCpu   == HostCpu   = True
  AppleGpu  == AppleGpu  = True
  Nvidia m  == Nvidia n  = m == n
  Other a   == Other b   = a == b
  _         == _         = False

public export
Show HardwareClass where
  show HostCpu    = "host-cpu"
  show AppleGpu   = "apple-gpu"
  show (Nvidia n) = "nvidia:" ++ show n
  show (Other s)  = s


----------------------------------------------------------------------
-- UserExecutorCore — lifecycle + arithmetic slice
----------------------------------------------------------------------

||| Phase 2.1 interface: the ~20 ops needed for tensor lifecycle and
||| elementwise arithmetic. Later phases add `UserExecutorLinear`,
||| `UserExecutorNN`, `UserExecutorConv`, `UserExecutorTraining` slices.
public export
interface UserExecutorCore (0 ex : Executor) where
  ||| Human-readable device tag: "tape", "torch", "mlx", "mybackend".
  ||| Used in logs and `Show Executor`-style stringification.
  deviceName : String

  ||| Per-device stream-selection tag. Threaded into every C-side
  ||| `_streamed` primitive so the type-level `d` drives stream
  ||| dispatch instead of `default_stream_tag()` (which reads the
  ||| `MLX_DEVICE` env var at process start).
  |||
  ||| MLX devices return their stream's tag (`MGpu = 1`, `MCpu = 0`);
  ||| every other backend returns 0 (the tape and torch backends
  ||| have no stream concept — their `_streamed` C entries are
  ||| no-op wrappers that ignore the arg).
  deviceStreamTag : Int

  -- Lifecycle ---------------------------------------------------------
  ||| Allocate a 0-rank tensor with the given value and grad flag.
  ||| `requires_grad` is 0 or 1.
  primCreateScalar : Double -> Int -> AnyPtr

  ||| Release a tensor handle. No-op on backends that GC.
  primFree : AnyPtr -> ()

  ||| Read a 0-rank tensor's value.
  primItem : AnyPtr -> Double

  ||| Read element `idx` from a 1-D tensor as a host Double.
  primItem1d : AnyPtr -> Int -> Double

  ||| Deep-copy a tensor (new handle, same shape and values, fresh
  ||| autograd node).
  primClone : AnyPtr -> AnyPtr

  -- Elementwise arithmetic --------------------------------------------
  primAdd     : AnyPtr -> AnyPtr -> AnyPtr
  primSub     : AnyPtr -> AnyPtr -> AnyPtr
  primMul     : AnyPtr -> AnyPtr -> AnyPtr
  primDiv     : AnyPtr -> AnyPtr -> AnyPtr
  primNeg     : AnyPtr -> AnyPtr
  primAbs     : AnyPtr -> AnyPtr
  primExp     : AnyPtr -> AnyPtr
  primLog     : AnyPtr -> AnyPtr
  primSqrt    : AnyPtr -> AnyPtr
  primPow     : AnyPtr -> AnyPtr -> AnyPtr
  primSigmoid : AnyPtr -> AnyPtr
  primTanh    : AnyPtr -> AnyPtr

  -- Scalar arithmetic -------------------------------------------------
  primAddScalar : AnyPtr -> Double -> AnyPtr
  primMulScalar : AnyPtr -> Double -> AnyPtr
  primClampMin  : AnyPtr -> Double -> AnyPtr
  primClamp     : AnyPtr -> Double -> Double -> AnyPtr
  primRound     : AnyPtr -> AnyPtr


----------------------------------------------------------------------
-- UserExecutorLinear — matmul + reductions + reshape + indexing slice
----------------------------------------------------------------------

||| The second slice. Covers linear algebra (mv, matmul, linear,
||| dot, outer, bmm), reductions (sum, mean, min, max, sumDim),
||| reshape / select (view, reshape, select, unsqueeze, squeeze,
||| stack, narrow, transpose), indexing (gather, scatter_add), and
||| sort/scan (argsort, cumprod). ~30 ops.
|||
||| Subclass of `UserExecutorCore`: an implementer also provides
||| lifecycle + arithmetic ops, so a single `UserExecutorLinear ex =>`
||| constraint in scope is enough to use both slices' methods. The
||| convention scales as later slices (`UserExecutorNN`, `Conv`,
||| `Tape`) layer on top.
||| Open per-device hardware classification. Built-ins map to the
||| obvious class (`TapeExecutor`/`*Cpu` → `HostCpu`, `TMps`/`MGpu` →
||| `AppleGpu`, `TCuda n` → `Nvidia n`). BYO backends map to
||| `Other "user/<name>"` — or a built-in class if they genuinely share
||| silicon. Opt-in (separate from `UserExecutorCore`) so adding it costs
||| no cascade on existing instances and BYO authors implement it only
||| if they want discovery/grouping.
public export
interface UserExecutorCore ex => HardwareClassed (0 ex : Executor) where
  hardwareClass : HardwareClass

public export
interface UserExecutorCore ex => UserExecutorLinear (0 ex : Executor) where
  -- Linear algebra ----------------------------------------------------
  primMv          : AnyPtr -> AnyPtr -> AnyPtr
  primMm          : AnyPtr -> AnyPtr -> AnyPtr
  primMatmul      : AnyPtr -> AnyPtr -> AnyPtr
  primLinear      : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr
  primDot         : AnyPtr -> AnyPtr -> AnyPtr
  primOuter       : AnyPtr -> AnyPtr -> AnyPtr
  primBmm         : AnyPtr -> AnyPtr -> AnyPtr
  primLinear2d    : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr

  -- Reductions --------------------------------------------------------
  primSum         : AnyPtr -> AnyPtr
  primMean        : AnyPtr -> AnyPtr
  primTensorMin   : AnyPtr -> AnyPtr
  primTensorMax   : AnyPtr -> AnyPtr
  primSumDim      : AnyPtr -> Int -> Int -> AnyPtr

  -- Shape / view / reshape -------------------------------------------
  primSelect      : AnyPtr -> Int -> Int -> AnyPtr
  primUnsqueeze   : AnyPtr -> Int -> AnyPtr
  primSqueeze     : AnyPtr -> Int -> AnyPtr
  primStack       : AnyPtr -> Int -> Int -> AnyPtr
  primView1d      : AnyPtr -> Int -> AnyPtr
  primView2d      : AnyPtr -> Int -> Int -> AnyPtr
  primReshape1d   : AnyPtr -> Int -> AnyPtr
  primReshape2d   : AnyPtr -> Int -> Int -> AnyPtr
  primReshape3d   : AnyPtr -> Int -> Int -> Int -> AnyPtr
  primReshape4d   : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
  primTile2d      : AnyPtr -> Int -> Int -> AnyPtr
  primNarrow      : AnyPtr -> Int -> Int -> Int -> AnyPtr
  primTransposeLast2 : AnyPtr -> AnyPtr
  primTranspose2d : AnyPtr -> AnyPtr

  -- Concatenation -----------------------------------------------------
  primCat         : AnyPtr -> Int -> Int -> AnyPtr
  primCat2        : AnyPtr -> AnyPtr -> AnyPtr
  primConcat2dAxis1 : AnyPtr -> AnyPtr -> AnyPtr

  -- Indexing ----------------------------------------------------------
  primGather      : AnyPtr -> AnyPtr -> Int -> AnyPtr
  primScatterAdd  : AnyPtr -> AnyPtr -> Int -> AnyPtr

  -- Sort / scan -------------------------------------------------------
  primArgsort     : AnyPtr -> Int -> Int -> AnyPtr
  primCumprod    : AnyPtr -> Int -> AnyPtr


----------------------------------------------------------------------
-- UserExecutorNN — activations + softmax + norms + losses + recurrent
-- cells + embedding + attention slice
----------------------------------------------------------------------

||| The third slice. Adds non-linearities, normalizations, recurrent
||| cells, embeddings, and the loss surfaces. Subclass of
||| `UserExecutorLinear` (transitively `UserExecutorCore`).
public export
interface UserExecutorLinear ex => UserExecutorNN (0 ex : Executor) where
  -- Activations -------------------------------------------------------
  primGelu        : AnyPtr -> AnyPtr
  primLeakyRelu   : AnyPtr -> Double -> AnyPtr
  primSilu        : AnyPtr -> AnyPtr
  primSoftplus    : AnyPtr -> AnyPtr

  -- Softmax family ----------------------------------------------------
  primSoftmax     : AnyPtr -> Int -> AnyPtr
  primLogSoftmax  : AnyPtr -> Int -> AnyPtr
  primSoftmax2d   : AnyPtr -> AnyPtr
  primLogSoftmax2d : AnyPtr -> AnyPtr
  primSoftmax3d   : AnyPtr -> AnyPtr

  -- Masking -----------------------------------------------------------
  primMaskedFill  : AnyPtr -> AnyPtr -> Double -> AnyPtr
  primExpandMask  : AnyPtr -> Int -> AnyPtr

  -- Norms / dropout ---------------------------------------------------
  primLayerNorm2d : AnyPtr -> AnyPtr -> AnyPtr -> Double -> AnyPtr
  primBatchNorm   : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr ->
                     Int -> Int -> Int -> Double -> Double -> AnyPtr
  primDropout     : AnyPtr -> Double -> Int -> Int -> AnyPtr

  -- Embedding / similarity / attention -------------------------------
  primEmbedding      : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr
  primEmbedding2d    : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr
  primCosineSimilarity : AnyPtr -> AnyPtr -> Int -> AnyPtr
  primCrossAttention : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> Double -> AnyPtr

  -- Loss --------------------------------------------------------------
  primBceWithLogits : AnyPtr -> AnyPtr -> AnyPtr

  -- Recurrent cells ---------------------------------------------------
  primGruCell        : AnyPtr -> AnyPtr -> AnyPtr -> Int -> AnyPtr
  primLstmGatesPair  : AnyPtr -> AnyPtr -> Int -> AnyPtr
  primPairFirst      : AnyPtr -> AnyPtr
  primPairSecond     : AnyPtr -> AnyPtr



----------------------------------------------------------------------
-- UserExecutorConv — convolution + pooling slice
----------------------------------------------------------------------

||| The fourth slice. Covers 1D and 2D convolution + pooling (~9
||| ops). Subclass of `UserExecutorNN` (transitively Linear + Core).
public export
interface UserExecutorNN ex => UserExecutorConv (0 ex : Executor) where
  -- 1D conv + pool
  primConv1d         : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr
  primConv1dCircular : AnyPtr -> AnyPtr -> AnyPtr
  primAvgPool1d      : AnyPtr -> Int -> Int -> AnyPtr
  primMaxPool1d      : AnyPtr -> Int -> Int -> AnyPtr
  -- 2D conv + pool
  primConv2d         : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
  primConv2dBatched  : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
  primAvgPool2d      : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
  primMaxPool2d      : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
  primMaxPool2dBatched : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr


----------------------------------------------------------------------
-- UserExecutorOptimizations — opt-in fused-op slice
--
-- Backends with native fused kernels for these ops declare an instance
-- and provide overrides. Backends without (BYO inference adapters,
-- hardware lacking the fused intrinsic) don't declare the instance at
-- all — any caller requiring a fused op carries a
-- `UserExecutorOptimizations ex =>` constraint, signalling which
-- backends it admits.
--
-- Categorical opt-in surface like `HardwareClassed` — no defaults
-- expressed via Core/Linear/NN primitives. The defaults would need
-- broadcasting / GQA-reshape / registry-iteration semantics that vary
-- subtly across backends; with no current BYO authors, investing in
-- fallback scaffolding fails the no-back-compat principle. If a BYO
-- author needs a method here, they implement it.
--
-- All three built-in backends (tape, torch, mlx) implement the entries
-- below natively today; the slice is also a superclass of the
-- `UserExecutorTraining` and `UserExecutorInference` aggregates, so
-- existing call sites using those aggregates continue to resolve.
----------------------------------------------------------------------

||| Opt-in fused-op surface. See module-level docs above.
public export
interface UserExecutorNN ex => UserExecutorOptimizations (0 ex : Executor) where
  ||| TODO #399 Commit B — fused scaled-dot-product attention.
  ||| Q : [seq, numHeads * headDim] (flat layout, axis-1 = nH*hd)
  ||| K : [seq, numKvHeads * headDim]
  ||| V : [seq, numKvHeads * headDim]
  ||| Result: [seq, numHeads * headDim].
  ||| Caller's responsibility: Q and K must already have RoPE applied
  ||| per-head before this call.
  primSdpa2d : AnyPtr -> AnyPtr -> AnyPtr
            -> Int -> Int -> Int  -- numHeads, numKvHeads, headDim
            -> Int                 -- isCausal (0/1)
            -> AnyPtr

  ||| Fused row-wise RMSNorm (HF LlamaRMSNorm formula).
  ||| input  : [seqLen, hidden]
  ||| weight : [hidden]
  ||| eps    : scalar
  ||| Per row i: rstd_i = 1 / sqrt(mean(input[i, :]^2) + eps);
  |||            out[i, j] = input[i, j] * rstd_i * weight[j].
  primRmsNorm2d : AnyPtr -> AnyPtr -> Double -> AnyPtr

  ||| Fused SwiGLU activation core: silu(gate) * up. Both inputs share
  ||| shape [seqLen, intermediate]; output is [seqLen, intermediate].
  primSwiGlu2d : AnyPtr -> AnyPtr -> AnyPtr

  ||| Polyak / EMA blend across the registry: for every pair of params
  ||| whose paramIds share the (onlineScope, targetScope) pair
  ||| (`<onlineScope><suffix>` and `<targetScope><suffix>`), set
  ||| target ← (1 - τ)·target + τ·online. Per-backend because the
  ||| registry is per-backend. Returns the count of pairs updated.
  primPolyakBlend : Double -> String -> String -> PrimIO Int

  -- Fused param create + in-place init. Replaces the per-element Idris-
  -- side sampler + per-element prim__setDouble FFI for model state
  -- construction. Each: (dims…, init-params, streamTag, dtypeTag) →
  -- AnyPtr; backends that don't implement these (currently tape and
  -- mlx) abort loudly at the FFI boundary via dtype_streamed.c.
  primCreateParam1dNormalStreamed : Int -> Double -> Double -> Int -> Int -> AnyPtr
  primCreateParam2dNormalStreamed : Int -> Int -> Double -> Double -> Int -> Int -> AnyPtr
  primCreateParam3dNormalStreamed : Int -> Int -> Int -> Double -> Double -> Int -> Int -> AnyPtr
  primCreateParam4dNormalStreamed : Int -> Int -> Int -> Int -> Double -> Double -> Int -> Int -> AnyPtr
  primCreateParam1dConstStreamed  : Int -> Double -> Int -> Int -> AnyPtr
  primCreateParam2dConstStreamed  : Int -> Int -> Double -> Int -> Int -> AnyPtr
  primCreateParam3dConstStreamed  : Int -> Int -> Int -> Double -> Int -> Int -> AnyPtr
  primCreateParam4dConstStreamed  : Int -> Int -> Int -> Int -> Double -> Int -> Int -> AnyPtr


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
  primRequiresGrad      : AnyPtr -> Int
  primSetRequiresGrad   : AnyPtr -> Int -> PrimIO ()
  primNoGradBegin       : PrimIO ()
  primNoGradEnd         : PrimIO ()
  primDetach            : AnyPtr -> AnyPtr
  primWithGrad          : AnyPtr -> AnyPtr
  ||| Run reverse-mode autodiff from a scalar loss tensor.
  primBackward          : AnyPtr -> PrimIO ()

||| C-side parameter registry. Tracks named tensors so the optimizer
||| can step them, the serializer can save/load them, and the EMA can
||| blend them. Per-backend; the registry IS the canonical bind from
||| paramId → tensor handle.
public export
interface UserExecutorAutograd ex => UserExecutorParamRegistry (0 ex : Executor) where
  primParamRegister     : String -> AnyPtr -> AnyPtr
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
  ||| in `forwardVarTraced` (TRACE level) to flush transient
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
  primOptimizerCreateSgd      : Double -> AnyPtr
  primOptimizerCreateRmsprop  : Double -> Double -> Double -> Double -> Double -> AnyPtr
  primOptimizerCreateAdam     : Double -> Double -> Double -> Double -> AnyPtr
  primOptimizerCreateAdamGroup : Double -> Double -> Double -> Double -> String -> AnyPtr
  primOptimizerCreateAdamW    : Double -> Double -> Double -> Double -> Double -> AnyPtr
  ||| Set the optimizer's base LR.
  primOptimizerSetLr      : AnyPtr -> Double -> PrimIO ()
  ||| Set a per-parameter LR override (matched by paramId).
  primOptimizerSetParamLr : AnyPtr -> String -> Double -> PrimIO ()
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
  ||| Save optimizer state buffers to a file.
  primOptimizerSave       : AnyPtr -> String -> PrimIO Int
  ||| Load optimizer state buffers from a file.
  primOptimizerLoad       : AnyPtr -> String -> PrimIO Int

||| Op-timing counters + epoch hooks + live/peak-handle reporting.
||| Orthogonal to autograd; sits next to `Core` because it observes,
||| doesn't mutate, the training surface.
public export
interface UserExecutorCore ex => UserExecutorProfiling (0 ex : Executor) where
  ||| Reset this backend's op-timing profile counters.
  primProfileReset        : PrimIO ()
  ||| Print this backend's profile breakdown to stderr.
  primProfileReport       : PrimIO ()
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
  ||| Count of live backend tensor handles (mlx: all_tensors; torch:
  ||| intermediates; tape: tape entries). The arg is ignored — it exists
  ||| only to defeat Idris-Chez constant-folding of the FFI call so the
  ||| count re-reads each epoch. Pass a varying value (e.g. the epoch).
  primLiveCount           : Int -> Int
  ||| High-water mark of live handles since process start — the figure
  ||| that determines whether a backend hits its handle/buffer ceiling.
  ||| Ignored arg defeats constant-folding; pass a varying value.
  primPeakLiveCount       : Int -> Int
  ||| TODO #393 op-submission diagnostic. Bumped at every from_tensor()
  ||| wrap on torch (counts graph nodes per forward); no-op on tape
  ||| and mlx (their per-op submission story is different — tape is
  ||| eager-CPU, mlx is lazy-batched via mx::array). Use bracketed
  ||| `primPerfReset` + `primPerfOpCount` at example sites to extract
  ||| per-forward op counts without instrumenting every kernel wrapper.
  primPerfReset           : PrimIO ()
  ||| Returns `Int` (not `Bits64`) — `PrimIO Bits64` triggered a
  ||| cumulative-state crash on tape F32 HfLlama (#401, 2026-05-31).
  ||| Idris-2's chez codegen emits `unsigned-64` for Bits64 returns;
  ||| something about that path corrupts state across calls. `Int`
  ||| (int64 on 64-bit platforms) holds the same value and is the
  ||| codepath used by every other counter FFI in the codebase.
  primPerfOpCount         : PrimIO Int

||| Tensor creation surface: shape queries + host item reads + dtype-
||| streamed creators + fused param-init. Inference-only adapters
||| implement this *without* the four training sub-slices above.
public export
interface UserExecutorCore ex => UserExecutorTensorCreate (0 ex : Executor) where
  -- Shape / info queries -------------------------------------------
  primTensorDim         : AnyPtr -> Int
  primTensorSizeAt      : AnyPtr -> Int -> Int

  -- Scalar reads / data loading
  ||| Read element `(r, c)` from a 2-D tensor as a host Double.
  primItem2d            : AnyPtr -> Int -> Int -> Double
  ||| One-hot encode an int-index buffer into a [len, classes]
  ||| matrix in the dtype selected by the trailing `dtypeTag` (so the
  ||| produced tensor honestly matches the Idris `dt`; 0/1 is exact in
  ||| every dtype). Args: (index buffer, len, classes, dtypeTag).
  primOneHot            : AnyPtr -> Int -> Int -> Int -> AnyPtr

  -- dtype-streamed creation -----------------------------------------
  -- Each takes a trailing (streamTag, dtypeTag) pair; the backend's
  -- wrapper branches on dtypeTag (0=f32, 1=f64) to pick the right
  -- `_f32_streamed_<b>` / `_f64_streamed_<b>` C symbol. The `dtCreate*`
  -- free functions in `Tensor` source dtypeTag from `RuntimeDType`.
  primCreateScalarStreamed : Double -> Int -> Int -> Int -> AnyPtr
  primCreateStreamed       : AnyPtr -> AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
  primCreate1dStreamed     : Int -> AnyPtr -> Int -> Int -> Int -> AnyPtr
  primCreate2dStreamed     : Int -> Int -> AnyPtr -> Int -> Int -> Int -> AnyPtr
  primCreateParam1dStreamed : Int -> AnyPtr -> Int -> Int -> AnyPtr
  primCreateParam2dStreamed : Int -> Int -> AnyPtr -> Int -> Int -> AnyPtr
  primCreateParam3dStreamed : Int -> Int -> Int -> AnyPtr -> Int -> Int -> AnyPtr
  primCreateParam4dStreamed : Int -> Int -> Int -> Int -> AnyPtr -> Int -> Int -> AnyPtr
  primCreateState1dStreamed : Int -> AnyPtr -> Int -> Int -> AnyPtr
  primCreateState2dStreamed : Int -> Int -> AnyPtr -> Int -> Int -> AnyPtr
  primCastStreamed         : AnyPtr -> Int -> Int -> AnyPtr

  -- Seed the backend's init RNG (torch::manual_seed equivalent). No-op
  -- on backends without a seedable init-RNG.
  primSetInitSeedStreamed : Bits64 -> Int -> PrimIO ()

||| Legacy training aggregate. Holds the full pre-split surface for
||| backwards compatibility with all `UserExecutorTraining ex =>`
||| callsites. Per-backend instances are one-liner `UserExecutorTraining
||| FooExec where` declarations — the actual prim* assignments live
||| in the seven sub-instance blocks above (six sub-slices + the
||| `UserExecutorOptimizations` opt-in slice). Resolving this
||| constraint transitively brings in everything an existing layer
||| needs.
public export
interface (UserExecutorConv ex,
           UserExecutorOptimizations ex,
           UserExecutorSerialize ex,
           UserExecutorProfiling ex,
           UserExecutorTensorCreate ex) =>
          UserExecutorTraining (0 ex : Executor) where

-- `UserExecutorInference` aggregate moved to end of file (after
-- `UserExecutorTransfer` + `UserExecutorQuant` are declared).


----------------------------------------------------------------------
-- UserExecutorTransfer — cross-backend tensor transfer surface
--
-- Backends that implement this can act as source or destination for
-- the generic `toExecutor` in `Tensor.idr`. The interface bundles
-- everything `toExecutor` needs to:
--   (a) recognise the backend at runtime (via `backendTag`);
--   (b) migrate a handle in place when source and dest share a
--       backend (via `primIntraMigrate`); and
--   (c) round-trip through host memory when they don't (via the
--       `primToHost` / `primCreateFromHost` pair plus the host
--       buffer-alloc helpers).
--
-- The five built-in backends today (tape, torch CPU/MPS/CUDA, mlx
-- CPU/GPU) all implement this. Users adding a BYO backend that
-- wants to plug into the generic `toExecutor` machinery declare their
-- own instance with a globally-unique `backendTag` (convention:
-- namespace as "user/<name>" to avoid colliding with built-ins).
----------------------------------------------------------------------

||| Cross-backend transfer surface. See module-level docs above.
public export
interface UserExecutorCore ex => UserExecutorTransfer (0 ex : Executor) where
  ||| Globally unique string identifying the backend (NOT the
  ||| hardware variant). Built-ins reserve "tape", "torch", "mlx".
  ||| BYO backends should namespace with "user/<name>". `toExecutor`
  ||| compares tags to decide intra-vs-cross-backend path; a
  ||| collision would route an intra fast-path through a foreign
  ||| backend's C symbols and crash on handle type mismatch.
  backendTag : String

  ||| Read a tensor's contents into a caller-allocated host double
  ||| buffer of `tensor_numel(handle)` slots. Returns the buffer so
  ||| Idris-Chez can't elide the FFI; threaded downstream into
  ||| `primCreateFromHost` on the destination backend.
  primToHost : AnyPtr -> AnyPtr -> AnyPtr

  ||| Allocate / free a host double buffer of `n` slots. Backend-
  ||| neutral host memory (calloc/free under the hood).
  primAllocHost : Int -> AnyPtr
  primFreeHost  : AnyPtr -> PrimIO ()

  ||| Allocate / write / free a host int buffer of `n` slots. Used
  ||| by `toExecutor` to build the shape array that
  ||| `primCreateFromHost` consumes.
  primAllocIntHost : Int -> AnyPtr
  primFreeIntHost  : AnyPtr -> PrimIO ()
  ||| Write `val` to `buf[idx]` and return `buf` (for threading).
  primSetIntHost   : AnyPtr -> Int -> Int -> AnyPtr

  ||| Create a tensor on this device from a host-allocated double
  ||| buffer + int shape buffer. The (data, shape, rank, rg) tuple
  ||| matches `tensor_create`'s ABI; the migration to this device's
  ||| hardware variant (e.g. TMps) happens internally so the
  ||| returned handle is on the right hw.
  primCreateFromHost : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr

  ||| Intra-backend hardware migration. Only sound when caller has
  ||| verified shared backend via `backendTag`. Mutates the
  ||| underlying tensor in place where the backend supports it;
  ||| preserves param-registry membership.
  primIntraMigrate : AnyPtr -> String -> AnyPtr


----------------------------------------------------------------------
-- UserExecutorQuant — quantization slice (BitNet b1.58 → #411)
----------------------------------------------------------------------

||| Opt-in slice for quantization ops. The three built-in backends
||| (tape, torch, mlx) implement it; BYO backends opt in only if they
||| want BitNet b1.58. Subclass of `UserExecutorCore` so a `UserExecutorQuant
||| d =>` constraint also brings the lifecycle + arithmetic surface.
|||
||| `primCreateTernaryPacked2d` takes (host-byte-buffer, byte_count, o,
||| i, requires_grad) and builds a `[o, i]` Ternary tensor with
||| dtype_tag = DT_TERNARY (25). Per-backend storage layout — packed
||| 2-bit on tape, unpacked int8 on torch/mlx — is hidden behind this
||| ABI; see design-decisions.md "Per-backend ternary storage".
|||
||| `primBitlinearFwd` runs y = (W_ternary .* scale[:, None]) @ x +
||| bias with W decoded inline (tape) or via int8-cast (torch/mlx).
||| Inference-only; STE-aware training is filed as a follow-up to #411.
|||
||| `primAbsmeanPerRow2d` returns the per-row absmean of a float [o, i]
||| weight: scale[j] = mean_k(|w[j, k]|), shape [o], same dtype as `w`.
||| `primTernaryQuantWithScale2d` takes the weight + that scale and
||| produces a Ternary tensor via per-row round-and-clamp. Together
||| they're the load-time recipe for converting an HF-stored F-dtype
||| BitNet checkpoint into our packed-ternary tag — see
||| `packages/pytorch/torch_ref/models/bitlinear.py`
||| `absmean_ternary_quant` for the reference implementation. Both
||| NoGrad; the pair runs once per linear at load.
|||
||| `primCreateTernaryFromHfPacked2d` reads HF's `[(o + 3) / 4, i]`
||| uint8 buffer (microsoft/bitnet-b1.58-2B-4T-style storage with
||| `{-1, 0, +1} -> {0, 1, 2}` codes packed along axis 0) and
||| produces a Ternary tensor in our layout. One-shot at safetensors
||| load.
public export
interface UserExecutorCore ex => UserExecutorQuant (0 ex : Executor) where
  primCreateTernaryPacked2d       : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
  primBitlinearFwd                : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr
  primAbsmeanPerRow2d             : AnyPtr -> AnyPtr
  primTernaryQuantWithScale2d     : AnyPtr -> AnyPtr -> AnyPtr
  primCreateTernaryFromHfPacked2d : AnyPtr -> Int -> Int -> AnyPtr
  primBitlinearFwdHfQuant         : AnyPtr -> Double -> AnyPtr -> AnyPtr -> Int -> AnyPtr -> Double -> AnyPtr


----------------------------------------------------------------------
-- UserExecutorInference — inference-only aggregate
----------------------------------------------------------------------

||| Inference-only aggregate. Documents the minimum surface a third-
||| party backend that ships only forward-pass + checkpoint-load (no
||| optimizer, no autograd) needs to implement: `Conv` (transitively
||| pulls in Core + Linear + NN), `Optimizations` (the fused inference
||| ops `primSdpa2d` / `primRmsNorm2d` / `primSwiGlu2d` + fused param-
||| init), `TensorCreate` (data loading + dtype-streamed creators),
||| `Transfer` (cross-backend handles), and `Quant` (BitNet ternary
||| surface). Skipping Autograd / ParamRegistry / Optimizer /
||| Serialize is a real reduction — those four sub-slices together
||| hold 27 of the 57 legacy Training methods.
public export
interface (UserExecutorConv ex,
           UserExecutorOptimizations ex,
           UserExecutorTensorCreate ex,
           UserExecutorTransfer ex,
           UserExecutorQuant ex) =>
          UserExecutorInference (0 ex : Executor) where
