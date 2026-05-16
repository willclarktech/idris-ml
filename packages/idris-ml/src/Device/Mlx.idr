||| `MlxDev` — `UserDeviceCore` instance for the mlx backend.
|||
||| Forwards to the mlx-suffixed C symbols emitted under Phase 1's
||| `rename_mlx.h` (e.g. `tensor_add_mlx`). Only resolvable at runtime
||| if the build's BACKEND list includes `mlx` (Apple-only).
module Device.Mlx

import Device.Core


----------------------------------------------------------------------
-- Per-symbol bindings to the mlx backend's suffixed C exports
----------------------------------------------------------------------

%foreign "C:tensor_create_scalar_mlx,libidrisml"
prim__createScalarMlx : Double -> Int -> AnyPtr

%foreign "C:tensor_create_mlx,libidrisml"
prim__createMlx : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr

%foreign "C:tensor_free_mlx,libidrisml"
prim__freeMlx : AnyPtr -> ()

%foreign "C:tensor_item_mlx,libidrisml"
prim__itemMlx : AnyPtr -> Double

%foreign "C:tensor_clone_mlx,libidrisml"
prim__cloneMlx : AnyPtr -> AnyPtr

%foreign "C:tensor_add_mlx,libidrisml"
prim__addMlx : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_sub_mlx,libidrisml"
prim__subMlx : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_mul_mlx,libidrisml"
prim__mulMlx : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_div_mlx,libidrisml"
prim__divMlx : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_neg_mlx,libidrisml"
prim__negMlx : AnyPtr -> AnyPtr

%foreign "C:tensor_abs_mlx,libidrisml"
prim__absMlx : AnyPtr -> AnyPtr

%foreign "C:tensor_exp_mlx,libidrisml"
prim__expMlx : AnyPtr -> AnyPtr

%foreign "C:tensor_log_mlx,libidrisml"
prim__logMlx : AnyPtr -> AnyPtr

%foreign "C:tensor_sqrt_mlx,libidrisml"
prim__sqrtMlx : AnyPtr -> AnyPtr

%foreign "C:tensor_pow_mlx,libidrisml"
prim__powMlx : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_sigmoid_mlx,libidrisml"
prim__sigmoidMlx : AnyPtr -> AnyPtr

%foreign "C:tensor_tanh_mlx,libidrisml"
prim__tanhMlx : AnyPtr -> AnyPtr

%foreign "C:tensor_add_scalar_mlx,libidrisml"
prim__addScalarMlx : AnyPtr -> Double -> AnyPtr

%foreign "C:tensor_mul_scalar_mlx,libidrisml"
prim__mulScalarMlx : AnyPtr -> Double -> AnyPtr

%foreign "C:tensor_clamp_min_mlx,libidrisml"
prim__clampMinMlx : AnyPtr -> Double -> AnyPtr


----------------------------------------------------------------------
-- MlxDev type + UserDeviceCore instance
----------------------------------------------------------------------

public export
data MlxDev : Type where MkMlxDev : MlxDev

public export
UserDeviceCore MlxDev where
  deviceName       = "mlx"
  primCreateScalar = prim__createScalarMlx
  primCreate       = prim__createMlx
  primFree         = prim__freeMlx
  primItem         = prim__itemMlx
  primClone        = prim__cloneMlx
  primAdd          = prim__addMlx
  primSub          = prim__subMlx
  primMul          = prim__mulMlx
  primDiv          = prim__divMlx
  primNeg          = prim__negMlx
  primAbs          = prim__absMlx
  primExp          = prim__expMlx
  primLog          = prim__logMlx
  primSqrt         = prim__sqrtMlx
  primPow          = prim__powMlx
  primSigmoid      = prim__sigmoidMlx
  primTanh         = prim__tanhMlx
  primAddScalar    = prim__addScalarMlx
  primMulScalar    = prim__mulScalarMlx
  primClampMin     = prim__clampMinMlx
----------------------------------------------------------------------
-- Linear-slice FFI bindings (mlx-suffixed)
----------------------------------------------------------------------

%foreign "C:tensor_mv_mlx,libidrisml"
prim__mvMlx : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_matmul_mlx,libidrisml"
prim__matmulMlx : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_linear_mlx,libidrisml"
prim__linearMlx : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_dot_mlx,libidrisml"
prim__dotMlx : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_outer_mlx,libidrisml"
prim__outerMlx : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_bmm_mlx,libidrisml"
prim__bmmMlx : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_linear_2d_mlx,libidrisml"
prim__linear2dMlx : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_sum_mlx,libidrisml"
prim__sumMlx : AnyPtr -> AnyPtr

%foreign "C:tensor_mean_mlx,libidrisml"
prim__meanMlx : AnyPtr -> AnyPtr

%foreign "C:tensor_min_mlx,libidrisml"
prim__tensorMinMlx : AnyPtr -> AnyPtr

%foreign "C:tensor_max_mlx,libidrisml"
prim__tensorMaxMlx : AnyPtr -> AnyPtr

%foreign "C:tensor_sum_dim_mlx,libidrisml"
prim__sumDimMlx : AnyPtr -> Int -> Int -> AnyPtr

%foreign "C:tensor_select_mlx,libidrisml"
prim__selectMlx : AnyPtr -> Int -> Int -> AnyPtr

%foreign "C:tensor_unsqueeze_mlx,libidrisml"
prim__unsqueezeMlx : AnyPtr -> Int -> AnyPtr

%foreign "C:tensor_squeeze_mlx,libidrisml"
prim__squeezeMlx : AnyPtr -> Int -> AnyPtr

%foreign "C:tensor_stack_mlx,libidrisml"
prim__stackMlx : AnyPtr -> Int -> Int -> AnyPtr

%foreign "C:tensor_view_1d_mlx,libidrisml"
prim__view1dMlx : AnyPtr -> Int -> AnyPtr

%foreign "C:tensor_view_2d_mlx,libidrisml"
prim__view2dMlx : AnyPtr -> Int -> Int -> AnyPtr

%foreign "C:tensor_reshape_2d_mlx,libidrisml"
prim__reshape2dMlx : AnyPtr -> Int -> Int -> AnyPtr

%foreign "C:tensor_reshape_3d_mlx,libidrisml"
prim__reshape3dMlx : AnyPtr -> Int -> Int -> Int -> AnyPtr

%foreign "C:tensor_reshape_4d_mlx,libidrisml"
prim__reshape4dMlx : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr

%foreign "C:tensor_narrow_mlx,libidrisml"
prim__narrowMlx : AnyPtr -> Int -> Int -> Int -> AnyPtr

%foreign "C:tensor_transpose_last2_mlx,libidrisml"
prim__transposeLast2Mlx : AnyPtr -> AnyPtr

%foreign "C:tensor_transpose_2d_mlx,libidrisml"
prim__transpose2dMlx : AnyPtr -> AnyPtr

%foreign "C:tensor_cat_mlx,libidrisml"
prim__catMlx : AnyPtr -> Int -> Int -> AnyPtr

%foreign "C:tensor_cat2_mlx,libidrisml"
prim__cat2Mlx : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_concat_2d_axis1_mlx,libidrisml"
prim__concat2dAxis1Mlx : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_gather_mlx,libidrisml"
prim__gatherMlx : AnyPtr -> AnyPtr -> Int -> AnyPtr

%foreign "C:tensor_scatter_add_mlx,libidrisml"
prim__scatterAddMlx : AnyPtr -> AnyPtr -> Int -> AnyPtr

%foreign "C:tensor_argsort_mlx,libidrisml"
prim__argsortMlx : AnyPtr -> Int -> Int -> AnyPtr

%foreign "C:tensor_cumprod_mlx,libidrisml"
prim__cumprodMlx : AnyPtr -> Int -> AnyPtr


public export
UserDeviceLinear MlxDev where
  primMv             = prim__mvMlx
  primMatmul         = prim__matmulMlx
  primLinear         = prim__linearMlx
  primDot            = prim__dotMlx
  primOuter          = prim__outerMlx
  primBmm            = prim__bmmMlx
  primLinear2d       = prim__linear2dMlx
  primSum            = prim__sumMlx
  primMean           = prim__meanMlx
  primTensorMin      = prim__tensorMinMlx
  primTensorMax      = prim__tensorMaxMlx
  primSumDim         = prim__sumDimMlx
  primSelect         = prim__selectMlx
  primUnsqueeze      = prim__unsqueezeMlx
  primSqueeze        = prim__squeezeMlx
  primStack          = prim__stackMlx
  primView1d         = prim__view1dMlx
  primView2d         = prim__view2dMlx
  primReshape2d      = prim__reshape2dMlx
  primReshape3d      = prim__reshape3dMlx
  primReshape4d      = prim__reshape4dMlx
  primNarrow         = prim__narrowMlx
  primTransposeLast2 = prim__transposeLast2Mlx
  primTranspose2d    = prim__transpose2dMlx
  primCat            = prim__catMlx
  primCat2           = prim__cat2Mlx
  primConcat2dAxis1  = prim__concat2dAxis1Mlx
  primGather         = prim__gatherMlx
  primScatterAdd     = prim__scatterAddMlx
  primArgsort        = prim__argsortMlx
  primCumprod        = prim__cumprodMlx


----------------------------------------------------------------------
-- NN-slice FFI bindings (mlx-suffixed)
----------------------------------------------------------------------

%foreign "C:tensor_gelu_mlx,libidrisml"
prim__geluMlx : AnyPtr -> AnyPtr
%foreign "C:tensor_leaky_relu_mlx,libidrisml"
prim__leakyReluMlx : AnyPtr -> Double -> AnyPtr
%foreign "C:tensor_silu_mlx,libidrisml"
prim__siluMlx : AnyPtr -> AnyPtr
%foreign "C:tensor_softplus_mlx,libidrisml"
prim__softplusMlx : AnyPtr -> AnyPtr
%foreign "C:tensor_softmax_mlx,libidrisml"
prim__softmaxMlx : AnyPtr -> Int -> AnyPtr
%foreign "C:tensor_log_softmax_mlx,libidrisml"
prim__logSoftmaxMlx : AnyPtr -> Int -> AnyPtr
%foreign "C:tensor_softmax_2d_mlx,libidrisml"
prim__softmax2dMlx : AnyPtr -> AnyPtr
%foreign "C:tensor_log_softmax_2d_mlx,libidrisml"
prim__logSoftmax2dMlx : AnyPtr -> AnyPtr
%foreign "C:tensor_softmax_3d_mlx,libidrisml"
prim__softmax3dMlx : AnyPtr -> AnyPtr
%foreign "C:tensor_masked_fill_mlx,libidrisml"
prim__maskedFillMlx : AnyPtr -> AnyPtr -> Double -> AnyPtr
%foreign "C:tensor_expand_mask_mlx,libidrisml"
prim__expandMaskMlx : AnyPtr -> Int -> AnyPtr
%foreign "C:tensor_causal_mask_mlx,libidrisml"
prim__causalMaskMlx : Int -> AnyPtr
%foreign "C:tensor_layer_norm_2d_mlx,libidrisml"
prim__layerNorm2dMlx : AnyPtr -> AnyPtr -> AnyPtr -> Double -> AnyPtr
%foreign "C:tensor_batch_norm_mlx,libidrisml"
prim__batchNormMlx : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> Double -> Double -> AnyPtr
%foreign "C:tensor_dropout_mlx,libidrisml"
prim__dropoutMlx : AnyPtr -> Double -> Int -> Int -> AnyPtr
%foreign "C:tensor_embedding_mlx,libidrisml"
prim__embeddingMlx : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "C:tensor_cosine_similarity_mlx,libidrisml"
prim__cosineSimilarityMlx : AnyPtr -> AnyPtr -> Int -> AnyPtr
%foreign "C:tensor_cross_attention_mlx,libidrisml"
prim__crossAttentionMlx : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> Double -> AnyPtr
%foreign "C:tensor_bce_with_logits_mlx,libidrisml"
prim__bceWithLogitsMlx : AnyPtr -> AnyPtr -> AnyPtr
%foreign "C:tensor_gru_cell_mlx,libidrisml"
prim__gruCellMlx : AnyPtr -> AnyPtr -> AnyPtr -> Int -> AnyPtr
%foreign "C:tensor_lstm_gates_pair_mlx,libidrisml"
prim__lstmGatesPairMlx : AnyPtr -> AnyPtr -> Int -> AnyPtr
%foreign "C:tensor_pair_first_mlx,libidrisml"
prim__pairFirstMlx : AnyPtr -> AnyPtr
%foreign "C:tensor_pair_second_mlx,libidrisml"
prim__pairSecondMlx : AnyPtr -> AnyPtr


public export
UserDeviceNN MlxDev where
  primGelu             = prim__geluMlx
  primLeakyRelu        = prim__leakyReluMlx
  primSilu             = prim__siluMlx
  primSoftplus         = prim__softplusMlx
  primSoftmax          = prim__softmaxMlx
  primLogSoftmax       = prim__logSoftmaxMlx
  primSoftmax2d        = prim__softmax2dMlx
  primLogSoftmax2d     = prim__logSoftmax2dMlx
  primSoftmax3d        = prim__softmax3dMlx
  primMaskedFill       = prim__maskedFillMlx
  primExpandMask       = prim__expandMaskMlx
  primCausalMask       = prim__causalMaskMlx
  primLayerNorm2d      = prim__layerNorm2dMlx
  primBatchNorm        = prim__batchNormMlx
  primDropout          = prim__dropoutMlx
  primEmbedding        = prim__embeddingMlx
  primCosineSimilarity = prim__cosineSimilarityMlx
  primCrossAttention   = prim__crossAttentionMlx
  primBceWithLogits    = prim__bceWithLogitsMlx
  primGruCell          = prim__gruCellMlx
  primLstmGatesPair    = prim__lstmGatesPairMlx
  primPairFirst        = prim__pairFirstMlx
  primPairSecond       = prim__pairSecondMlx
