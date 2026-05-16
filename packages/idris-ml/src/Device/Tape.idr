||| `TapeDev` — `UserDeviceCore` instance for the tape backend.
|||
||| Forwards to the tape-suffixed C symbols emitted under Phase 1's
||| `rename_tape.h` (e.g. `tensor_add_tape`). Only resolvable at
||| runtime if the build's BACKEND list includes `tape`.
module Device.Tape

import Device.Core


----------------------------------------------------------------------
-- Per-symbol bindings to the tape backend's suffixed C exports
----------------------------------------------------------------------

%foreign "C:tensor_create_scalar_tape,libidrisml"
prim__createScalarTape : Double -> Int -> AnyPtr

%foreign "C:tensor_create_tape,libidrisml"
prim__createTape : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr

%foreign "C:tensor_free_tape,libidrisml"
prim__freeTape : AnyPtr -> ()

%foreign "C:tensor_item_tape,libidrisml"
prim__itemTape : AnyPtr -> Double

%foreign "C:tensor_clone_tape,libidrisml"
prim__cloneTape : AnyPtr -> AnyPtr

%foreign "C:tensor_add_tape,libidrisml"
prim__addTape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_sub_tape,libidrisml"
prim__subTape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_mul_tape,libidrisml"
prim__mulTape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_div_tape,libidrisml"
prim__divTape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_neg_tape,libidrisml"
prim__negTape : AnyPtr -> AnyPtr

%foreign "C:tensor_abs_tape,libidrisml"
prim__absTape : AnyPtr -> AnyPtr

%foreign "C:tensor_exp_tape,libidrisml"
prim__expTape : AnyPtr -> AnyPtr

%foreign "C:tensor_log_tape,libidrisml"
prim__logTape : AnyPtr -> AnyPtr

%foreign "C:tensor_sqrt_tape,libidrisml"
prim__sqrtTape : AnyPtr -> AnyPtr

%foreign "C:tensor_pow_tape,libidrisml"
prim__powTape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_sigmoid_tape,libidrisml"
prim__sigmoidTape : AnyPtr -> AnyPtr

%foreign "C:tensor_tanh_tape,libidrisml"
prim__tanhTape : AnyPtr -> AnyPtr

%foreign "C:tensor_add_scalar_tape,libidrisml"
prim__addScalarTape : AnyPtr -> Double -> AnyPtr

%foreign "C:tensor_mul_scalar_tape,libidrisml"
prim__mulScalarTape : AnyPtr -> Double -> AnyPtr

%foreign "C:tensor_clamp_min_tape,libidrisml"
prim__clampMinTape : AnyPtr -> Double -> AnyPtr


----------------------------------------------------------------------
-- TapeDev type + UserDeviceCore instance
----------------------------------------------------------------------

||| The tape backend's `UserDeviceCore` instance head. An empty type
||| — it has no values; `Tensor [..] TapeDev` is just a typed tag for
||| "this tensor lives on the tape backend".
public export
data TapeDev : Type where MkTapeDev : TapeDev

public export
UserDeviceCore TapeDev where
  deviceName       = "tape"
  primCreateScalar = prim__createScalarTape
  primCreate       = prim__createTape
  primFree         = prim__freeTape
  primItem         = prim__itemTape
  primClone        = prim__cloneTape
  primAdd          = prim__addTape
  primSub          = prim__subTape
  primMul          = prim__mulTape
  primDiv          = prim__divTape
  primNeg          = prim__negTape
  primAbs          = prim__absTape
  primExp          = prim__expTape
  primLog          = prim__logTape
  primSqrt         = prim__sqrtTape
  primPow          = prim__powTape
  primSigmoid      = prim__sigmoidTape
  primTanh         = prim__tanhTape
  primAddScalar    = prim__addScalarTape
  primMulScalar    = prim__mulScalarTape
  primClampMin     = prim__clampMinTape

----------------------------------------------------------------------
-- Linear-slice FFI bindings (tape-suffixed)
----------------------------------------------------------------------

%foreign "C:tensor_mv_tape,libidrisml"
prim__mvTape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_matmul_tape,libidrisml"
prim__matmulTape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_linear_tape,libidrisml"
prim__linearTape : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_dot_tape,libidrisml"
prim__dotTape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_outer_tape,libidrisml"
prim__outerTape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_bmm_tape,libidrisml"
prim__bmmTape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_linear_2d_tape,libidrisml"
prim__linear2dTape : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_sum_tape,libidrisml"
prim__sumTape : AnyPtr -> AnyPtr

%foreign "C:tensor_mean_tape,libidrisml"
prim__meanTape : AnyPtr -> AnyPtr

%foreign "C:tensor_min_tape,libidrisml"
prim__tensorMinTape : AnyPtr -> AnyPtr

%foreign "C:tensor_max_tape,libidrisml"
prim__tensorMaxTape : AnyPtr -> AnyPtr

%foreign "C:tensor_sum_dim_tape,libidrisml"
prim__sumDimTape : AnyPtr -> Int -> Int -> AnyPtr

%foreign "C:tensor_select_tape,libidrisml"
prim__selectTape : AnyPtr -> Int -> Int -> AnyPtr

%foreign "C:tensor_unsqueeze_tape,libidrisml"
prim__unsqueezeTape : AnyPtr -> Int -> AnyPtr

%foreign "C:tensor_squeeze_tape,libidrisml"
prim__squeezeTape : AnyPtr -> Int -> AnyPtr

%foreign "C:tensor_stack_tape,libidrisml"
prim__stackTape : AnyPtr -> Int -> Int -> AnyPtr

%foreign "C:tensor_view_1d_tape,libidrisml"
prim__view1dTape : AnyPtr -> Int -> AnyPtr

%foreign "C:tensor_view_2d_tape,libidrisml"
prim__view2dTape : AnyPtr -> Int -> Int -> AnyPtr

%foreign "C:tensor_reshape_2d_tape,libidrisml"
prim__reshape2dTape : AnyPtr -> Int -> Int -> AnyPtr

%foreign "C:tensor_reshape_3d_tape,libidrisml"
prim__reshape3dTape : AnyPtr -> Int -> Int -> Int -> AnyPtr

%foreign "C:tensor_reshape_4d_tape,libidrisml"
prim__reshape4dTape : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr

%foreign "C:tensor_narrow_tape,libidrisml"
prim__narrowTape : AnyPtr -> Int -> Int -> Int -> AnyPtr

%foreign "C:tensor_transpose_last2_tape,libidrisml"
prim__transposeLast2Tape : AnyPtr -> AnyPtr

%foreign "C:tensor_transpose_2d_tape,libidrisml"
prim__transpose2dTape : AnyPtr -> AnyPtr

%foreign "C:tensor_cat_tape,libidrisml"
prim__catTape : AnyPtr -> Int -> Int -> AnyPtr

%foreign "C:tensor_cat2_tape,libidrisml"
prim__cat2Tape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_concat_2d_axis1_tape,libidrisml"
prim__concat2dAxis1Tape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_gather_tape,libidrisml"
prim__gatherTape : AnyPtr -> AnyPtr -> Int -> AnyPtr

%foreign "C:tensor_scatter_add_tape,libidrisml"
prim__scatterAddTape : AnyPtr -> AnyPtr -> Int -> AnyPtr

%foreign "C:tensor_argsort_tape,libidrisml"
prim__argsortTape : AnyPtr -> Int -> Int -> AnyPtr

%foreign "C:tensor_cumprod_tape,libidrisml"
prim__cumprodTape : AnyPtr -> Int -> AnyPtr


public export
UserDeviceLinear TapeDev where
  primMv             = prim__mvTape
  primMatmul         = prim__matmulTape
  primLinear         = prim__linearTape
  primDot            = prim__dotTape
  primOuter          = prim__outerTape
  primBmm            = prim__bmmTape
  primLinear2d       = prim__linear2dTape
  primSum            = prim__sumTape
  primMean           = prim__meanTape
  primTensorMin      = prim__tensorMinTape
  primTensorMax      = prim__tensorMaxTape
  primSumDim         = prim__sumDimTape
  primSelect         = prim__selectTape
  primUnsqueeze      = prim__unsqueezeTape
  primSqueeze        = prim__squeezeTape
  primStack          = prim__stackTape
  primView1d         = prim__view1dTape
  primView2d         = prim__view2dTape
  primReshape2d      = prim__reshape2dTape
  primReshape3d      = prim__reshape3dTape
  primReshape4d      = prim__reshape4dTape
  primNarrow         = prim__narrowTape
  primTransposeLast2 = prim__transposeLast2Tape
  primTranspose2d    = prim__transpose2dTape
  primCat            = prim__catTape
  primCat2           = prim__cat2Tape
  primConcat2dAxis1  = prim__concat2dAxis1Tape
  primGather         = prim__gatherTape
  primScatterAdd     = prim__scatterAddTape
  primArgsort        = prim__argsortTape
  primCumprod        = prim__cumprodTape


----------------------------------------------------------------------
-- NN-slice FFI bindings (tape-suffixed)
----------------------------------------------------------------------

%foreign "C:tensor_gelu_tape,libidrisml"
prim__geluTape : AnyPtr -> AnyPtr
%foreign "C:tensor_leaky_relu_tape,libidrisml"
prim__leakyReluTape : AnyPtr -> Double -> AnyPtr
%foreign "C:tensor_silu_tape,libidrisml"
prim__siluTape : AnyPtr -> AnyPtr
%foreign "C:tensor_softplus_tape,libidrisml"
prim__softplusTape : AnyPtr -> AnyPtr
%foreign "C:tensor_softmax_tape,libidrisml"
prim__softmaxTape : AnyPtr -> Int -> AnyPtr
%foreign "C:tensor_log_softmax_tape,libidrisml"
prim__logSoftmaxTape : AnyPtr -> Int -> AnyPtr
%foreign "C:tensor_softmax_2d_tape,libidrisml"
prim__softmax2dTape : AnyPtr -> AnyPtr
%foreign "C:tensor_log_softmax_2d_tape,libidrisml"
prim__logSoftmax2dTape : AnyPtr -> AnyPtr
%foreign "C:tensor_softmax_3d_tape,libidrisml"
prim__softmax3dTape : AnyPtr -> AnyPtr
%foreign "C:tensor_masked_fill_tape,libidrisml"
prim__maskedFillTape : AnyPtr -> AnyPtr -> Double -> AnyPtr
%foreign "C:tensor_expand_mask_tape,libidrisml"
prim__expandMaskTape : AnyPtr -> Int -> AnyPtr
%foreign "C:tensor_causal_mask_tape,libidrisml"
prim__causalMaskTape : Int -> AnyPtr
%foreign "C:tensor_layer_norm_2d_tape,libidrisml"
prim__layerNorm2dTape : AnyPtr -> AnyPtr -> AnyPtr -> Double -> AnyPtr
%foreign "C:tensor_batch_norm_tape,libidrisml"
prim__batchNormTape : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> Double -> Double -> AnyPtr
%foreign "C:tensor_dropout_tape,libidrisml"
prim__dropoutTape : AnyPtr -> Double -> Int -> Int -> AnyPtr
%foreign "C:tensor_embedding_tape,libidrisml"
prim__embeddingTape : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "C:tensor_cosine_similarity_tape,libidrisml"
prim__cosineSimilarityTape : AnyPtr -> AnyPtr -> Int -> AnyPtr
%foreign "C:tensor_cross_attention_tape,libidrisml"
prim__crossAttentionTape : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> Double -> AnyPtr
%foreign "C:tensor_bce_with_logits_tape,libidrisml"
prim__bceWithLogitsTape : AnyPtr -> AnyPtr -> AnyPtr
%foreign "C:tensor_gru_cell_tape,libidrisml"
prim__gruCellTape : AnyPtr -> AnyPtr -> AnyPtr -> Int -> AnyPtr
%foreign "C:tensor_lstm_gates_pair_tape,libidrisml"
prim__lstmGatesPairTape : AnyPtr -> AnyPtr -> Int -> AnyPtr
%foreign "C:tensor_pair_first_tape,libidrisml"
prim__pairFirstTape : AnyPtr -> AnyPtr
%foreign "C:tensor_pair_second_tape,libidrisml"
prim__pairSecondTape : AnyPtr -> AnyPtr


public export
UserDeviceNN TapeDev where
  primGelu             = prim__geluTape
  primLeakyRelu        = prim__leakyReluTape
  primSilu             = prim__siluTape
  primSoftplus         = prim__softplusTape
  primSoftmax          = prim__softmaxTape
  primLogSoftmax       = prim__logSoftmaxTape
  primSoftmax2d        = prim__softmax2dTape
  primLogSoftmax2d     = prim__logSoftmax2dTape
  primSoftmax3d        = prim__softmax3dTape
  primMaskedFill       = prim__maskedFillTape
  primExpandMask       = prim__expandMaskTape
  primCausalMask       = prim__causalMaskTape
  primLayerNorm2d      = prim__layerNorm2dTape
  primBatchNorm        = prim__batchNormTape
  primDropout          = prim__dropoutTape
  primEmbedding        = prim__embeddingTape
  primCosineSimilarity = prim__cosineSimilarityTape
  primCrossAttention   = prim__crossAttentionTape
  primBceWithLogits    = prim__bceWithLogitsTape
  primGruCell          = prim__gruCellTape
  primLstmGatesPair    = prim__lstmGatesPairTape
  primPairFirst        = prim__pairFirstTape
  primPairSecond       = prim__pairSecondTape
