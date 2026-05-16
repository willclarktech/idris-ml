||| Device tags for type-safe tensor placement.
|||
||| **Phase 2.1b change**: `Device` was a closed sum
||| (`CPU | CUDA Nat | MPS`); it's now a kind-level slot. `CPU`,
||| `CUDA n`, `MPS` are *types*, not values. `Tensor`'s phantom
||| parameter is now `0 d : Type` (was `0 d : Device`).
|||
||| Each of `CPU` / `CUDA n` / `MPS` has a `UserDeviceCore` instance
||| forwarding to the build's primary backend via unified-name C
||| symbols (`tensor_add` aliased to `tensor_add_<primary>` per
||| Phase 1's rename + alias mechanism). For backend-specific
||| dispatch, use `Device.Tape` / `Device.Torch` / `Device.Mlx`'s
||| `TapeDev` / `TorchDev` / `MlxDev` tags directly — those bind to
||| the suffixed symbols.
|||
||| Users implementing a custom backend declare their own type and
||| `UserDeviceCore` instance; see `docs/develop/design-decisions.md`
||| "Pluggable Device" for the recipe.
module Device

import public Device.Core


----------------------------------------------------------------------
-- Default device tags (host CPU / CUDA / MPS) — Phase 2.1b
--
-- These are *types*, not values. Existing `Tensor [..] CPU` keeps
-- compiling because Tensor's phantom is `0 d : Type`. The
-- `UserDeviceCore` instances forward to unified-name C symbols
-- (`tensor_add` — aliased by Phase 1's link step to the primary
-- backend's `tensor_add_<primary>`), so any `Tensor [..] CPU` op
-- transparently runs on whatever the build's primary backend is.
----------------------------------------------------------------------

||| Host CPU device tag. Forwards to the primary backend's CPU-side
||| operations.
public export
data CPU : Type where MkCPU : CPU

||| CUDA device tag, parameterised by device index. Untested (the
||| torch backend's CUDA path is wired but never exercised in CI as
||| of 2026-05-13).
public export
data CUDA : Nat -> Type where MkCUDA : (n : Nat) -> CUDA n

||| MPS (Apple Metal Performance Shaders) device tag. Untested.
public export
data MPS : Type where MkMPS : MPS


----------------------------------------------------------------------
-- HasDeviceIndex — runtime-observable parameter for parameterized
-- devices (e.g. `CUDA n`)
--
-- `UserDeviceCore` declares its `d` parameter at 0-quantity (it's a
-- pure type-level dispatch tag), so an instance method body cannot
-- observe the value of `d`'s type-level parameters. That makes
-- writing `deviceName = "cuda:" ++ show n` for `UserDeviceCore
-- (CUDA n)` impossible directly — `n` is erased.
--
-- `HasDeviceIndex` carries the runtime index separately: a
-- non-erased typeclass over the device. The method `deviceIndex`
-- returns the `Nat` parameter, so `UserDeviceCore (CUDA n)`'s
-- `deviceName` can call `deviceIndex` to recover it.
--
-- See `docs/grad-mode-and-device-typing.md` "Parameterized devices"
-- and `docs/develop/design-decisions.md` "Open `d` parameter".
----------------------------------------------------------------------

||| Devices whose type carries a runtime-observable Nat index (CUDA's
||| device number is the canonical example). Methods of
||| `UserDeviceCore` that need to see the parameter — most commonly
||| `deviceName` — call `deviceIndex` to recover it.
public export
interface HasDeviceIndex (d : Device) where
  deviceIndex : Nat

public export
{n : Nat} -> HasDeviceIndex (CUDA n) where
  deviceIndex = n


----------------------------------------------------------------------
-- Unified-name FFI bindings (Phase 1's primary-backend aliases)
----------------------------------------------------------------------

%foreign "C:tensor_create_scalar,libidrisml"
prim__createScalarUnified : Double -> Int -> AnyPtr

%foreign "C:tensor_create,libidrisml"
prim__createUnified : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr

%foreign "C:tensor_free,libidrisml"
prim__freeUnified : AnyPtr -> ()

%foreign "C:tensor_item,libidrisml"
prim__itemUnified : AnyPtr -> Double

%foreign "C:tensor_clone,libidrisml"
prim__cloneUnified : AnyPtr -> AnyPtr

%foreign "C:tensor_add,libidrisml"
prim__addUnified : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_sub,libidrisml"
prim__subUnified : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_mul,libidrisml"
prim__mulUnified : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_div,libidrisml"
prim__divUnified : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_neg,libidrisml"
prim__negUnified : AnyPtr -> AnyPtr

%foreign "C:tensor_abs,libidrisml"
prim__absUnified : AnyPtr -> AnyPtr

%foreign "C:tensor_exp,libidrisml"
prim__expUnified : AnyPtr -> AnyPtr

%foreign "C:tensor_log,libidrisml"
prim__logUnified : AnyPtr -> AnyPtr

%foreign "C:tensor_sqrt,libidrisml"
prim__sqrtUnified : AnyPtr -> AnyPtr

%foreign "C:tensor_pow,libidrisml"
prim__powUnified : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_sigmoid,libidrisml"
prim__sigmoidUnified : AnyPtr -> AnyPtr

%foreign "C:tensor_tanh,libidrisml"
prim__tanhUnified : AnyPtr -> AnyPtr

%foreign "C:tensor_add_scalar,libidrisml"
prim__addScalarUnified : AnyPtr -> Double -> AnyPtr

%foreign "C:tensor_mul_scalar,libidrisml"
prim__mulScalarUnified : AnyPtr -> Double -> AnyPtr

%foreign "C:tensor_clamp_min,libidrisml"
prim__clampMinUnified : AnyPtr -> Double -> AnyPtr


----------------------------------------------------------------------
-- UserDeviceCore instances — all three default tags forward through
-- the same unified-name FFI surface (Phase 1's primary-backend
-- aliases). This makes `Tensor [..] CPU` portable across builds:
-- whichever backend is primary handles the dispatch.
----------------------------------------------------------------------

public export
UserDeviceCore CPU where
  deviceName       = "cpu"
  primCreateScalar = prim__createScalarUnified
  primCreate       = prim__createUnified
  primFree         = prim__freeUnified
  primItem         = prim__itemUnified
  primClone        = prim__cloneUnified
  primAdd          = prim__addUnified
  primSub          = prim__subUnified
  primMul          = prim__mulUnified
  primDiv          = prim__divUnified
  primNeg          = prim__negUnified
  primAbs          = prim__absUnified
  primExp          = prim__expUnified
  primLog          = prim__logUnified
  primSqrt         = prim__sqrtUnified
  primPow          = prim__powUnified
  primSigmoid      = prim__sigmoidUnified
  primTanh         = prim__tanhUnified
  primAddScalar    = prim__addScalarUnified
  primMulScalar    = prim__mulScalarUnified
  primClampMin     = prim__clampMinUnified

public export
{n : Nat} -> UserDeviceCore (CUDA n) where
  deviceName       = "cuda:" ++ show n
  primCreateScalar = prim__createScalarUnified
  primCreate       = prim__createUnified
  primFree         = prim__freeUnified
  primItem         = prim__itemUnified
  primClone        = prim__cloneUnified
  primAdd          = prim__addUnified
  primSub          = prim__subUnified
  primMul          = prim__mulUnified
  primDiv          = prim__divUnified
  primNeg          = prim__negUnified
  primAbs          = prim__absUnified
  primExp          = prim__expUnified
  primLog          = prim__logUnified
  primSqrt         = prim__sqrtUnified
  primPow          = prim__powUnified
  primSigmoid      = prim__sigmoidUnified
  primTanh         = prim__tanhUnified
  primAddScalar    = prim__addScalarUnified
  primMulScalar    = prim__mulScalarUnified
  primClampMin     = prim__clampMinUnified

public export
UserDeviceCore MPS where
  deviceName       = "mps"
  primCreateScalar = prim__createScalarUnified
  primCreate       = prim__createUnified
  primFree         = prim__freeUnified
  primItem         = prim__itemUnified
  primClone        = prim__cloneUnified
  primAdd          = prim__addUnified
  primSub          = prim__subUnified
  primMul          = prim__mulUnified
  primDiv          = prim__divUnified
  primNeg          = prim__negUnified
  primAbs          = prim__absUnified
  primExp          = prim__expUnified
  primLog          = prim__logUnified
  primSqrt         = prim__sqrtUnified
  primPow          = prim__powUnified
  primSigmoid      = prim__sigmoidUnified
  primTanh         = prim__tanhUnified
  primAddScalar    = prim__addScalarUnified
  primMulScalar    = prim__mulScalarUnified
  primClampMin     = prim__clampMinUnified


----------------------------------------------------------------------
-- UserDeviceLinear instances (Phase 2.2). All three default tags
-- forward through unified-name FFI symbols, just like UserDeviceCore.
----------------------------------------------------------------------

%foreign "C:tensor_mv,libidrisml"
prim__mvUnified : AnyPtr -> AnyPtr -> AnyPtr
%foreign "C:tensor_matmul,libidrisml"
prim__matmulUnified : AnyPtr -> AnyPtr -> AnyPtr
%foreign "C:tensor_linear,libidrisml"
prim__linearUnified : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr
%foreign "C:tensor_dot,libidrisml"
prim__dotUnified : AnyPtr -> AnyPtr -> AnyPtr
%foreign "C:tensor_outer,libidrisml"
prim__outerUnified : AnyPtr -> AnyPtr -> AnyPtr
%foreign "C:tensor_bmm,libidrisml"
prim__bmmUnified : AnyPtr -> AnyPtr -> AnyPtr
%foreign "C:tensor_linear_2d,libidrisml"
prim__linear2dUnified : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr
%foreign "C:tensor_sum,libidrisml"
prim__sumUnified : AnyPtr -> AnyPtr
%foreign "C:tensor_mean,libidrisml"
prim__meanUnified : AnyPtr -> AnyPtr
%foreign "C:tensor_min,libidrisml"
prim__tensorMinUnified : AnyPtr -> AnyPtr
%foreign "C:tensor_max,libidrisml"
prim__tensorMaxUnified : AnyPtr -> AnyPtr
%foreign "C:tensor_sum_dim,libidrisml"
prim__sumDimUnified : AnyPtr -> Int -> Int -> AnyPtr
%foreign "C:tensor_select,libidrisml"
prim__selectUnified : AnyPtr -> Int -> Int -> AnyPtr
%foreign "C:tensor_unsqueeze,libidrisml"
prim__unsqueezeUnified : AnyPtr -> Int -> AnyPtr
%foreign "C:tensor_squeeze,libidrisml"
prim__squeezeUnified : AnyPtr -> Int -> AnyPtr
%foreign "C:tensor_stack,libidrisml"
prim__stackUnified : AnyPtr -> Int -> Int -> AnyPtr
%foreign "C:tensor_view_1d,libidrisml"
prim__view1dUnified : AnyPtr -> Int -> AnyPtr
%foreign "C:tensor_view_2d,libidrisml"
prim__view2dUnified : AnyPtr -> Int -> Int -> AnyPtr
%foreign "C:tensor_reshape_2d,libidrisml"
prim__reshape2dUnified : AnyPtr -> Int -> Int -> AnyPtr
%foreign "C:tensor_reshape_3d,libidrisml"
prim__reshape3dUnified : AnyPtr -> Int -> Int -> Int -> AnyPtr
%foreign "C:tensor_reshape_4d,libidrisml"
prim__reshape4dUnified : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
%foreign "C:tensor_narrow,libidrisml"
prim__narrowUnified : AnyPtr -> Int -> Int -> Int -> AnyPtr
%foreign "C:tensor_transpose_last2,libidrisml"
prim__transposeLast2Unified : AnyPtr -> AnyPtr
%foreign "C:tensor_transpose_2d,libidrisml"
prim__transpose2dUnified : AnyPtr -> AnyPtr
%foreign "C:tensor_cat,libidrisml"
prim__catUnified : AnyPtr -> Int -> Int -> AnyPtr
%foreign "C:tensor_cat2,libidrisml"
prim__cat2Unified : AnyPtr -> AnyPtr -> AnyPtr
%foreign "C:tensor_concat_2d_axis1,libidrisml"
prim__concat2dAxis1Unified : AnyPtr -> AnyPtr -> AnyPtr
%foreign "C:tensor_gather,libidrisml"
prim__gatherUnified : AnyPtr -> AnyPtr -> Int -> AnyPtr
%foreign "C:tensor_scatter_add,libidrisml"
prim__scatterAddUnified : AnyPtr -> AnyPtr -> Int -> AnyPtr
%foreign "C:tensor_argsort,libidrisml"
prim__argsortUnified : AnyPtr -> Int -> Int -> AnyPtr
%foreign "C:tensor_cumprod,libidrisml"
prim__cumprodUnified : AnyPtr -> Int -> AnyPtr


public export
UserDeviceLinear CPU where
  primMv = prim__mvUnified
  primMatmul = prim__matmulUnified
  primLinear = prim__linearUnified
  primDot = prim__dotUnified
  primOuter = prim__outerUnified
  primBmm = prim__bmmUnified
  primLinear2d = prim__linear2dUnified
  primSum = prim__sumUnified
  primMean = prim__meanUnified
  primTensorMin = prim__tensorMinUnified
  primTensorMax = prim__tensorMaxUnified
  primSumDim = prim__sumDimUnified
  primSelect = prim__selectUnified
  primUnsqueeze = prim__unsqueezeUnified
  primSqueeze = prim__squeezeUnified
  primStack = prim__stackUnified
  primView1d = prim__view1dUnified
  primView2d = prim__view2dUnified
  primReshape2d = prim__reshape2dUnified
  primReshape3d = prim__reshape3dUnified
  primReshape4d = prim__reshape4dUnified
  primNarrow = prim__narrowUnified
  primTransposeLast2 = prim__transposeLast2Unified
  primTranspose2d = prim__transpose2dUnified
  primCat = prim__catUnified
  primCat2 = prim__cat2Unified
  primConcat2dAxis1 = prim__concat2dAxis1Unified
  primGather = prim__gatherUnified
  primScatterAdd = prim__scatterAddUnified
  primArgsort = prim__argsortUnified
  primCumprod = prim__cumprodUnified

public export
{n : Nat} -> UserDeviceLinear (CUDA n) where
  primMv = prim__mvUnified
  primMatmul = prim__matmulUnified
  primLinear = prim__linearUnified
  primDot = prim__dotUnified
  primOuter = prim__outerUnified
  primBmm = prim__bmmUnified
  primLinear2d = prim__linear2dUnified
  primSum = prim__sumUnified
  primMean = prim__meanUnified
  primTensorMin = prim__tensorMinUnified
  primTensorMax = prim__tensorMaxUnified
  primSumDim = prim__sumDimUnified
  primSelect = prim__selectUnified
  primUnsqueeze = prim__unsqueezeUnified
  primSqueeze = prim__squeezeUnified
  primStack = prim__stackUnified
  primView1d = prim__view1dUnified
  primView2d = prim__view2dUnified
  primReshape2d = prim__reshape2dUnified
  primReshape3d = prim__reshape3dUnified
  primReshape4d = prim__reshape4dUnified
  primNarrow = prim__narrowUnified
  primTransposeLast2 = prim__transposeLast2Unified
  primTranspose2d = prim__transpose2dUnified
  primCat = prim__catUnified
  primCat2 = prim__cat2Unified
  primConcat2dAxis1 = prim__concat2dAxis1Unified
  primGather = prim__gatherUnified
  primScatterAdd = prim__scatterAddUnified
  primArgsort = prim__argsortUnified
  primCumprod = prim__cumprodUnified

public export
UserDeviceLinear MPS where
  primMv = prim__mvUnified
  primMatmul = prim__matmulUnified
  primLinear = prim__linearUnified
  primDot = prim__dotUnified
  primOuter = prim__outerUnified
  primBmm = prim__bmmUnified
  primLinear2d = prim__linear2dUnified
  primSum = prim__sumUnified
  primMean = prim__meanUnified
  primTensorMin = prim__tensorMinUnified
  primTensorMax = prim__tensorMaxUnified
  primSumDim = prim__sumDimUnified
  primSelect = prim__selectUnified
  primUnsqueeze = prim__unsqueezeUnified
  primSqueeze = prim__squeezeUnified
  primStack = prim__stackUnified
  primView1d = prim__view1dUnified
  primView2d = prim__view2dUnified
  primReshape2d = prim__reshape2dUnified
  primReshape3d = prim__reshape3dUnified
  primReshape4d = prim__reshape4dUnified
  primNarrow = prim__narrowUnified
  primTransposeLast2 = prim__transposeLast2Unified
  primTranspose2d = prim__transpose2dUnified
  primCat = prim__catUnified
  primCat2 = prim__cat2Unified
  primConcat2dAxis1 = prim__concat2dAxis1Unified
  primGather = prim__gatherUnified
  primScatterAdd = prim__scatterAddUnified
  primArgsort = prim__argsortUnified
  primCumprod = prim__cumprodUnified


----------------------------------------------------------------------
-- UserDeviceNN — unified-name FFI bindings (Phase 2.3) + 3 instances.
----------------------------------------------------------------------

%foreign "C:tensor_gelu,libidrisml"
prim__geluUnified : AnyPtr -> AnyPtr
%foreign "C:tensor_leaky_relu,libidrisml"
prim__leakyReluUnified : AnyPtr -> Double -> AnyPtr
%foreign "C:tensor_silu,libidrisml"
prim__siluUnified : AnyPtr -> AnyPtr
%foreign "C:tensor_softplus,libidrisml"
prim__softplusUnified : AnyPtr -> AnyPtr
%foreign "C:tensor_softmax,libidrisml"
prim__softmaxUnified : AnyPtr -> Int -> AnyPtr
%foreign "C:tensor_log_softmax,libidrisml"
prim__logSoftmaxUnified : AnyPtr -> Int -> AnyPtr
%foreign "C:tensor_softmax_2d,libidrisml"
prim__softmax2dUnified : AnyPtr -> AnyPtr
%foreign "C:tensor_log_softmax_2d,libidrisml"
prim__logSoftmax2dUnified : AnyPtr -> AnyPtr
%foreign "C:tensor_softmax_3d,libidrisml"
prim__softmax3dUnified : AnyPtr -> AnyPtr
%foreign "C:tensor_masked_fill,libidrisml"
prim__maskedFillUnified : AnyPtr -> AnyPtr -> Double -> AnyPtr
%foreign "C:tensor_expand_mask,libidrisml"
prim__expandMaskUnified : AnyPtr -> Int -> AnyPtr
%foreign "C:tensor_causal_mask,libidrisml"
prim__causalMaskUnified : Int -> AnyPtr
%foreign "C:tensor_layer_norm_2d,libidrisml"
prim__layerNorm2dUnified : AnyPtr -> AnyPtr -> AnyPtr -> Double -> AnyPtr
%foreign "C:tensor_batch_norm,libidrisml"
prim__batchNormUnified : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> Double -> Double -> AnyPtr
%foreign "C:tensor_dropout,libidrisml"
prim__dropoutUnified : AnyPtr -> Double -> Int -> Int -> AnyPtr
%foreign "C:tensor_embedding,libidrisml"
prim__embeddingUnified : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "C:tensor_cosine_similarity,libidrisml"
prim__cosineSimilarityUnified : AnyPtr -> AnyPtr -> Int -> AnyPtr
%foreign "C:tensor_cross_attention,libidrisml"
prim__crossAttentionUnified : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> Double -> AnyPtr
%foreign "C:tensor_bce_with_logits,libidrisml"
prim__bceWithLogitsUnified : AnyPtr -> AnyPtr -> AnyPtr
%foreign "C:tensor_gru_cell,libidrisml"
prim__gruCellUnified : AnyPtr -> AnyPtr -> AnyPtr -> Int -> AnyPtr
%foreign "C:tensor_lstm_gates_pair,libidrisml"
prim__lstmGatesPairUnified : AnyPtr -> AnyPtr -> Int -> AnyPtr
%foreign "C:tensor_pair_first,libidrisml"
prim__pairFirstUnified : AnyPtr -> AnyPtr
%foreign "C:tensor_pair_second,libidrisml"
prim__pairSecondUnified : AnyPtr -> AnyPtr

public export
UserDeviceNN CPU where
  primGelu             = prim__geluUnified
  primLeakyRelu        = prim__leakyReluUnified
  primSilu             = prim__siluUnified
  primSoftplus         = prim__softplusUnified
  primSoftmax          = prim__softmaxUnified
  primLogSoftmax       = prim__logSoftmaxUnified
  primSoftmax2d        = prim__softmax2dUnified
  primLogSoftmax2d     = prim__logSoftmax2dUnified
  primSoftmax3d        = prim__softmax3dUnified
  primMaskedFill       = prim__maskedFillUnified
  primExpandMask       = prim__expandMaskUnified
  primCausalMask       = prim__causalMaskUnified
  primLayerNorm2d      = prim__layerNorm2dUnified
  primBatchNorm        = prim__batchNormUnified
  primDropout          = prim__dropoutUnified
  primEmbedding        = prim__embeddingUnified
  primCosineSimilarity = prim__cosineSimilarityUnified
  primCrossAttention   = prim__crossAttentionUnified
  primBceWithLogits    = prim__bceWithLogitsUnified
  primGruCell          = prim__gruCellUnified
  primLstmGatesPair    = prim__lstmGatesPairUnified
  primPairFirst        = prim__pairFirstUnified
  primPairSecond       = prim__pairSecondUnified

public export
{n : Nat} -> UserDeviceNN (CUDA n) where
  primGelu             = prim__geluUnified
  primLeakyRelu        = prim__leakyReluUnified
  primSilu             = prim__siluUnified
  primSoftplus         = prim__softplusUnified
  primSoftmax          = prim__softmaxUnified
  primLogSoftmax       = prim__logSoftmaxUnified
  primSoftmax2d        = prim__softmax2dUnified
  primLogSoftmax2d     = prim__logSoftmax2dUnified
  primSoftmax3d        = prim__softmax3dUnified
  primMaskedFill       = prim__maskedFillUnified
  primExpandMask       = prim__expandMaskUnified
  primCausalMask       = prim__causalMaskUnified
  primLayerNorm2d      = prim__layerNorm2dUnified
  primBatchNorm        = prim__batchNormUnified
  primDropout          = prim__dropoutUnified
  primEmbedding        = prim__embeddingUnified
  primCosineSimilarity = prim__cosineSimilarityUnified
  primCrossAttention   = prim__crossAttentionUnified
  primBceWithLogits    = prim__bceWithLogitsUnified
  primGruCell          = prim__gruCellUnified
  primLstmGatesPair    = prim__lstmGatesPairUnified
  primPairFirst        = prim__pairFirstUnified
  primPairSecond       = prim__pairSecondUnified

public export
UserDeviceNN MPS where
  primGelu             = prim__geluUnified
  primLeakyRelu        = prim__leakyReluUnified
  primSilu             = prim__siluUnified
  primSoftplus         = prim__softplusUnified
  primSoftmax          = prim__softmaxUnified
  primLogSoftmax       = prim__logSoftmaxUnified
  primSoftmax2d        = prim__softmax2dUnified
  primLogSoftmax2d     = prim__logSoftmax2dUnified
  primSoftmax3d        = prim__softmax3dUnified
  primMaskedFill       = prim__maskedFillUnified
  primExpandMask       = prim__expandMaskUnified
  primCausalMask       = prim__causalMaskUnified
  primLayerNorm2d      = prim__layerNorm2dUnified
  primBatchNorm        = prim__batchNormUnified
  primDropout          = prim__dropoutUnified
  primEmbedding        = prim__embeddingUnified
  primCosineSimilarity = prim__cosineSimilarityUnified
  primCrossAttention   = prim__crossAttentionUnified
  primBceWithLogits    = prim__bceWithLogitsUnified
  primGruCell          = prim__gruCellUnified
  primLstmGatesPair    = prim__lstmGatesPairUnified
  primPairFirst        = prim__pairFirstUnified
  primPairSecond       = prim__pairSecondUnified
