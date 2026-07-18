||| Forward-compute interface slices: Core (lifecycle + arithmetic),
||| Streamed, HardwareClassed, Linear, NN, Conv, Optimizations.
module Ml.Executor.Core.Compute

import Ml.Executor.Core.Kind

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
-- UserExecutorStreamed — per-device stream-selection tag
----------------------------------------------------------------------

||| Opt-in stream-selection slice. Threaded into every C-side
||| `_streamed` primitive so the type-level `ex` drives stream
||| dispatch instead of `default_stream_tag()` (which would read the
||| `MLX_DEVICE` env var at process start).
|||
||| MLX devices return their stream's tag (`MGpu = 1`, `MCpu = 0`);
||| the tape and torch backends declare a `deviceStreamTag = 0`
||| instance so the streamed-FFI machinery threads a stable 0 through
||| their no-op `_streamed` C wrappers. BYO backends without a stream
||| concept can omit the instance entirely; the `_streamed` callers in
||| `Tensor.idr` will fail to elaborate, which is the correct signal
||| ("this backend doesn't model streams; use the non-streamed path").
|||
||| Superclass of both `UserExecutorTraining` and `UserExecutorInference`,
||| so existing callers of either aggregate resolve unchanged for the
||| three in-tree backends (all of which implement this slice).
public export
interface UserExecutorCore ex => UserExecutorStreamed (0 ex : Executor) where
  deviceStreamTag : Int

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
  primMv       : AnyPtr -> AnyPtr -> AnyPtr
  primMm       : AnyPtr -> AnyPtr -> AnyPtr
  primMatmul   : AnyPtr -> AnyPtr -> AnyPtr
  primLinear   : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr
  primDot      : AnyPtr -> AnyPtr -> AnyPtr
  primOuter    : AnyPtr -> AnyPtr -> AnyPtr
  primBmm      : AnyPtr -> AnyPtr -> AnyPtr
  primLinear2d : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr

  -- Reductions --------------------------------------------------------
  primSum       : AnyPtr -> AnyPtr
  primMean      : AnyPtr -> AnyPtr
  primTensorMin : AnyPtr -> AnyPtr
  primTensorMax : AnyPtr -> AnyPtr
  primSumDim    : AnyPtr -> Int -> Int -> AnyPtr

  -- Shape / view / reshape -------------------------------------------
  primSelect    : AnyPtr -> Int -> Int -> AnyPtr
  primUnsqueeze : AnyPtr -> Int -> AnyPtr
  primSqueeze   : AnyPtr -> Int -> AnyPtr
  primStack     : AnyPtr -> Int -> Int -> AnyPtr
  -- B × [...] -> [B, ...]: stack a ptr-array of `count` identically-shaped
  -- handles along a new leading axis (the single-FFI batch collation).
  primBatch          : AnyPtr -> Int -> AnyPtr
  primView1d         : AnyPtr -> Int -> AnyPtr
  primView2d         : AnyPtr -> Int -> Int -> AnyPtr
  primReshape1d      : AnyPtr -> Int -> AnyPtr
  primReshape2d      : AnyPtr -> Int -> Int -> AnyPtr
  primReshape3d      : AnyPtr -> Int -> Int -> Int -> AnyPtr
  primReshape4d      : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
  primNarrow         : AnyPtr -> Int -> Int -> Int -> AnyPtr
  primTransposeLast2 : AnyPtr -> AnyPtr
  primTranspose2d    : AnyPtr -> AnyPtr

  -- Concatenation -----------------------------------------------------
  primCat           : AnyPtr -> Int -> Int -> AnyPtr
  primCat2          : AnyPtr -> AnyPtr -> AnyPtr
  primConcat2dAxis1 : AnyPtr -> AnyPtr -> AnyPtr

  -- Indexing ----------------------------------------------------------
  primGather     : AnyPtr -> AnyPtr -> Int -> AnyPtr
  primGatherRows : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr
  primMaxRows    : AnyPtr -> Int -> Int -> AnyPtr
  primScatterAdd : AnyPtr -> AnyPtr -> Int -> AnyPtr

  -- Sort / scan -------------------------------------------------------
  primArgsort : AnyPtr -> Int -> Int -> AnyPtr
  primCumprod : AnyPtr -> Int -> AnyPtr

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
  primGelu      : AnyPtr -> AnyPtr
  primLeakyRelu : AnyPtr -> Double -> AnyPtr
  primSilu      : AnyPtr -> AnyPtr
  primSoftplus  : AnyPtr -> AnyPtr

  -- Softmax family ----------------------------------------------------
  primSoftmax      : AnyPtr -> Int -> AnyPtr
  primLogSoftmax   : AnyPtr -> Int -> AnyPtr
  primSoftmax2d    : AnyPtr -> AnyPtr
  primLogSoftmax2d : AnyPtr -> AnyPtr
  primSoftmax3d    : AnyPtr -> AnyPtr

  -- Masking -----------------------------------------------------------
  primMaskedFill : AnyPtr -> AnyPtr -> Double -> AnyPtr
  primExpandMask : AnyPtr -> Int -> AnyPtr

  -- Norms / dropout ---------------------------------------------------
  primLayerNorm2d : AnyPtr -> AnyPtr -> AnyPtr -> Double -> AnyPtr
  primBatchNorm   : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr ->
                     Int -> Int -> Int -> Double -> Double -> AnyPtr
  primDropout     : AnyPtr -> Double -> Int -> Int -> AnyPtr

  -- Embedding / similarity -------------------------------
  primEmbedding   : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr
  primEmbedding2d : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr
  ||| Kept as a mandatory `UserExecutorNN` method (audit closure): a
  ||| pure-Idris alternative (`sum(a * b, dim) / (sqrt(sum(a^2, dim))
  ||| * sqrt(sum(b^2, dim)))`) would emit 8 lazy-graph nodes per call
  ||| (4 reductions + 2 sqrts + mul + div). In NTM's per-timestep
  ||| content-addressing loop (3 callsites in `Nn/{Ntm,Dnc}.idr`)
  ||| under the `withNoGrad` bracket on mlx, that would compound
  ||| MTLBuffer pressure and risk hitting the existing handle-count
  ||| ceiling. The fused kernel is the right shape for this hot loop.
  primCosineSimilarity : AnyPtr -> AnyPtr -> Int -> AnyPtr

  -- Loss --------------------------------------------------------------
  primBceWithLogits : AnyPtr -> AnyPtr -> AnyPtr

  -- Recurrent cells ---------------------------------------------------
  primGruCell        : AnyPtr -> AnyPtr -> AnyPtr -> Int -> AnyPtr
  -- TODO audit: `primLstmGatesPair` + `primPairFirst` + `primPairSecond`
  -- are three interface methods (and nine `%foreign` bindings across
  -- the three backends) that together unpack ONE fused C return at one
  -- Idris call site (`Tensor.idr:tlstmGatesPair`). Collapse to a single
  -- backend-dispatchable method requires either (a) a new per-backend C
  -- entry-point with output-pointer semantics, or (b) a custom multi-
  -- return Scheme wrapper outside the `ffi_manifest.py` template's
  -- direct/streamed shape — both larger than the audit budget. Deferred
  -- to a future LSTM/recurrent-cell refactor that already needs to
  -- restructure the cell-state plumbing.
  primLstmGatesPair : AnyPtr -> AnyPtr -> Int -> AnyPtr
  primPairFirst     : AnyPtr -> AnyPtr
  primPairSecond    : AnyPtr -> AnyPtr

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
  primConv2d           : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
  primConv2dBatched    : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
  primAvgPool2d        : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
  primMaxPool2d        : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
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

  ||| Fused softmax cross-entropy with logits (soft/one-hot targets).
  ||| input  : [b, n] logits (rank-1 accepted as [1, n])
  ||| target : [b, n] targets, same shape/dtype
  ||| scale  : caller-chosen reduction scale (1/(b*n) for tnllLossMean)
  ||| out    : scalar -scale * sum(target * log_softmax(input, rows)).
  ||| One tape node replaces the decomposed logSoftmax->mul->sum->neg->
  ||| mulScalar chain; backward is scale*(softmax*rowsum(target)-target).
  primSoftmaxXent2d : AnyPtr -> AnyPtr -> Double -> AnyPtr

  ||| Fused cross-attention. Args: Q, K, V, mask (tensor), scale (scalar).
  ||| Runs `(Q·K^T / scale) + mask → softmax → ·V`. Caller provides the
  ||| precomputed Q/K/V + an additive mask tensor; this differs from
  ||| `primSdpa2d` (which takes head dims + isCausal as Int parameters).
  ||| Sibling fused op of `primSdpa2d`; placed in the same opt-in slice
  ||| for consistency. Inference + training-side multi-head attention.
  primCrossAttention : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> Double -> AnyPtr

  ||| Tile a 2D tensor by `(rep0, rep1)`: input `[m, n]` → output
  ||| `[m * rep0, n * rep1]`. Fused per-backend (`mx::tile` / `at::tile`
  ||| / manual memcpy). Used at the Transformer batch-embed site to
  ||| broadcast cached positional encodings across the batch dim;
  ||| the earlier reshape3d → add → reshape2d alternative regressed
  ||| mlx perf on small models (see `docs/develop/perf-changes.md`).
  primTile2d : AnyPtr -> Int -> Int -> AnyPtr

  ||| Polyak / EMA blend of one EXACTLY-named param pair: set
  ||| target ← (1 - τ)·target + τ·online for (onlineName, targetName).
  ||| Names are matched with strcmp (not prefix), so a name that is a
  ||| proper prefix of another can't over-match. Per-backend because the
  ||| registry is per-backend. Returns 1 if blended, 0 if a name is
  ||| absent or the shapes differ.
  primPolyakBlendPair : Double -> String -> String -> PrimIO Int

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
