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


----------------------------------------------------------------------
-- Conv-slice FFI bindings (mlx-suffixed)
----------------------------------------------------------------------

%foreign "C:tensor_conv1d_mlx,libidrisml"
prim__conv1dMlx : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "C:tensor_conv1d_circular_mlx,libidrisml"
prim__conv1dCircularMlx : AnyPtr -> AnyPtr -> AnyPtr
%foreign "C:tensor_avg_pool1d_mlx,libidrisml"
prim__avgPool1dMlx : AnyPtr -> Int -> Int -> AnyPtr
%foreign "C:tensor_max_pool1d_mlx,libidrisml"
prim__maxPool1dMlx : AnyPtr -> Int -> Int -> AnyPtr
%foreign "C:tensor_conv2d_mlx,libidrisml"
prim__conv2dMlx : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
%foreign "C:tensor_conv2d_batched_mlx,libidrisml"
prim__conv2dBatchedMlx : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
%foreign "C:tensor_avg_pool2d_mlx,libidrisml"
prim__avgPool2dMlx : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
%foreign "C:tensor_max_pool2d_mlx,libidrisml"
prim__maxPool2dMlx : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
%foreign "C:tensor_max_pool2d_batched_mlx,libidrisml"
prim__maxPool2dBatchedMlx : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr


public export
UserDeviceConv MlxDev where
  primConv1d           = prim__conv1dMlx
  primConv1dCircular   = prim__conv1dCircularMlx
  primAvgPool1d        = prim__avgPool1dMlx
  primMaxPool1d        = prim__maxPool1dMlx
  primConv2d           = prim__conv2dMlx
  primConv2dBatched    = prim__conv2dBatchedMlx
  primAvgPool2d        = prim__avgPool2dMlx
  primMaxPool2d        = prim__maxPool2dMlx
  primMaxPool2dBatched = prim__maxPool2dBatchedMlx


----------------------------------------------------------------------
-- Tape-slice FFI bindings (mlx-suffixed)
----------------------------------------------------------------------

%foreign "C:tensor_requires_grad_mlx,libidrisml"
prim__requiresGradMlx : AnyPtr -> Int
%foreign "C:tensor_set_requires_grad_mlx,libidrisml"
prim__setRequiresGradMlx : AnyPtr -> Int -> PrimIO ()
%foreign "C:tensor_no_grad_begin_mlx,libidrisml"
prim__noGradBeginMlx : PrimIO ()
%foreign "C:tensor_no_grad_end_mlx,libidrisml"
prim__noGradEndMlx : PrimIO ()
%foreign "C:tensor_detach_mlx,libidrisml"
prim__detachMlx : AnyPtr -> AnyPtr
%foreign "C:tensor_with_grad_mlx,libidrisml"
prim__withGradMlx : AnyPtr -> AnyPtr
%foreign "C:tensor_dim_mlx,libidrisml"
prim__tensorDimMlx : AnyPtr -> Int
%foreign "C:tensor_size_mlx,libidrisml"
prim__tensorSizeAtMlx : AnyPtr -> Int -> Int
%foreign "C:param_register_return_mlx,libidrisml"
prim__paramRegisterMlx : String -> AnyPtr -> AnyPtr
%foreign "C:param_clear_mlx,libidrisml"
prim__paramClearMlx : PrimIO ()
%foreign "C:param_count_mlx,libidrisml"
prim__paramCountMlx : Int
%foreign "C:param_name_mlx,libidrisml"
prim__paramNameMlx : Int -> String
%foreign "C:param_grad_item_mlx,libidrisml"
prim__paramGradItemMlx : Int -> Double
%foreign "C:param_grad_item_at_mlx,libidrisml"
prim__paramGradItemAtMlx : Int -> Int -> Double
%foreign "C:param_grad_item_and_zero_mlx,libidrisml"
prim__paramGradItemAndZeroMlx : Int -> Double
%foreign "C:param_zero_all_grads_return_mlx,libidrisml"
prim__paramZeroAllGradsMlx : Int -> Int
%foreign "C:param_subtract_delta_mlx,libidrisml"
prim__paramSubtractDeltaMlx : Int -> Double -> ()
%foreign "C:tensor_create_param_1d_mlx,libidrisml"
prim__createParam1dMlx : Int -> AnyPtr -> AnyPtr
%foreign "C:tensor_create_param_2d_mlx,libidrisml"
prim__createParam2dMlx : Int -> Int -> AnyPtr -> AnyPtr
%foreign "C:tensor_create_param_3d_mlx,libidrisml"
prim__createParam3dMlx : Int -> Int -> Int -> AnyPtr -> AnyPtr
%foreign "C:tensor_create_state_1d_mlx,libidrisml"
prim__createState1dMlx : Int -> AnyPtr -> AnyPtr
%foreign "C:tensor_create_state_2d_mlx,libidrisml"
prim__createState2dMlx : Int -> Int -> AnyPtr -> AnyPtr
%foreign "C:tensor_alloc_doubles_mlx,libidrisml"
prim__allocDoublesMlx : Int -> AnyPtr
%foreign "C:tensor_read_double_mlx,libidrisml"
prim__readDoubleMlx : AnyPtr -> Int -> Double
%foreign "C:tensor_write_double_mlx,libidrisml"
prim__writeDoubleMlx : AnyPtr -> Int -> Double -> ()
%foreign "C:tensor_print_mlx,libidrisml"
prim__printMlx : AnyPtr -> ()


public export
UserDeviceTape MlxDev where
  primRequiresGrad         = prim__requiresGradMlx
  primSetRequiresGrad      = prim__setRequiresGradMlx
  primNoGradBegin          = prim__noGradBeginMlx
  primNoGradEnd            = prim__noGradEndMlx
  primDetach               = prim__detachMlx
  primWithGrad             = prim__withGradMlx
  primTensorDim            = prim__tensorDimMlx
  primTensorSizeAt         = prim__tensorSizeAtMlx
  primParamRegister        = prim__paramRegisterMlx
  primParamClear        = prim__paramClearMlx

  primParamCount _         = prim__paramCountMlx
  primParamName            = prim__paramNameMlx
  primParamGradItem        = prim__paramGradItemMlx
  primParamGradItemAt      = prim__paramGradItemAtMlx
  primParamGradItemAndZero = prim__paramGradItemAndZeroMlx
  primParamZeroAllGrads    = prim__paramZeroAllGradsMlx
  primParamSubtractDelta   = prim__paramSubtractDeltaMlx
  primCreateParam1d        = prim__createParam1dMlx
  primCreateParam2d        = prim__createParam2dMlx
  primCreateParam3d        = prim__createParam3dMlx
  primCreateState1d        = prim__createState1dMlx
  primCreateState2d        = prim__createState2dMlx
  primAllocDoubles         = prim__allocDoublesMlx
  primReadDouble           = prim__readDoubleMlx
  primWriteDouble          = prim__writeDoubleMlx
  primPrint                = prim__printMlx
