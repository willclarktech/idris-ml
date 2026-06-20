||| Neural-network instance slices: NN (activations / softmax / norms /
||| losses / recurrent), Conv (conv + pooling), Optimizations (fused ops).
module Executor.Mlx.Nn

import BackendLib
import DType.Core
import Executor.Core
import public Executor.Mlx.Linear
import Hardware
import Preset

----------------------------------------------------------------------
-- NN-slice FFI bindings (mlx-suffixed)
----------------------------------------------------------------------

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_gelu_mlx_streamed\" (void* int) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__geluMlxStreamed : AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_leaky_relu_mlx_streamed\" (void* double int) void*) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__leakyReluMlxStreamed : AnyPtr -> Double -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_silu_mlx_streamed\" (void* int) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__siluMlxStreamed : AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_softplus_mlx_streamed\" (void* int) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__softplusMlxStreamed : AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_softmax_mlx_streamed\" (void* int int) void*) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__softmaxMlxStreamed : AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_log_softmax_mlx_streamed\" (void* int int) void*) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__logSoftmaxMlxStreamed : AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_softmax_2d_mlx_streamed\" (void* int) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__softmax2dMlxStreamed : AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_log_softmax_2d_mlx_streamed\" (void* int) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__logSoftmax2dMlxStreamed : AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_softmax_3d_mlx_streamed\" (void* int) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__softmax3dMlxStreamed : AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_masked_fill_mlx_streamed\" (void* void* double int) void*) (vector-ref a0 2) (vector-ref a1 2) a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__maskedFillMlxStreamed : AnyPtr -> AnyPtr -> Double -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_expand_mask_mlx_streamed\" (void* int int) void*) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__expandMaskMlxStreamed : AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (let ((raw_r ((foreign-procedure \"tensor_layer_norm_2d_mlx_streamed\" (void* void* void* double int) void*) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2) a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__layerNorm2dMlxStreamed : AnyPtr -> AnyPtr -> AnyPtr -> Double -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10)  (let ((raw_r ((foreign-procedure \"tensor_batch_norm_mlx_streamed\" (void* void* void* void* void* int int int double double int) void*) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2) (vector-ref a3 2) (vector-ref a4 2) a5 a6 a7 a8 a9 a10))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__batchNormMlxStreamed : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> Double -> Double -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (let ((raw_r ((foreign-procedure \"tensor_dropout_mlx_streamed\" (void* double int int int) void*) (vector-ref a0 2) a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__dropoutMlxStreamed : AnyPtr -> Double -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (let ((raw_r ((foreign-procedure \"tensor_embedding_mlx_streamed\" (void* void* int int int) void*) (vector-ref a0 2) (vector-ref a1 2) a2 a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__embeddingMlxStreamed : AnyPtr -> AnyPtr -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (let ((raw_r ((foreign-procedure \"tensor_embedding_2d_mlx_streamed\" (void* void* int int int) void*) (vector-ref a0 2) (vector-ref a1 2) a2 a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__embedding2dMlxStreamed : AnyPtr -> AnyPtr -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_cosine_similarity_mlx_streamed\" (void* void* int int) void*) (vector-ref a0 2) (vector-ref a1 2) a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__cosineSimilarityMlxStreamed : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5)  (let ((raw_r ((foreign-procedure \"tensor_cross_attention_mlx_streamed\" (void* void* void* void* double int) void*) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2) (vector-ref a3 2) a4 a5))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__crossAttentionMlxStreamed : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> Double -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_bce_with_logits_mlx_streamed\" (void* void* int) void*) (vector-ref a0 2) (vector-ref a1 2) a2))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__bceWithLogitsMlxStreamed : AnyPtr -> AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (let ((raw_r ((foreign-procedure \"tensor_gru_cell_mlx_streamed\" (void* void* void* int int) void*) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2) a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__gruCellMlxStreamed : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3)  ((foreign-procedure \"tensor_lstm_gates_pair_mlx_streamed\" (void* void* int int) void*) (vector-ref a0 2) (vector-ref a1 2) a2 a3))"
prim__lstmGatesPairMlxStreamed : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_pair_first_mlx_streamed\" (void* int) void*) a0 a1))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__pairFirstMlxStreamed : AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_pair_second_mlx_streamed\" (void* int) void*) a0 a1))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__pairSecondMlxStreamed : AnyPtr -> Int -> AnyPtr

-- Fused inference ops (used by `UserExecutorNN` below; FFI decls
-- moved up from the legacy `Training` slice region so they
-- precede their first use in the NN instance.)
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6)  (when (not (top-level-bound? 'idris-ffi-tensor-sdpa-2d-mlx)) (set-top-level-value! 'idris-ffi-tensor-sdpa-2d-mlx (foreign-procedure \"tensor_sdpa_2d_mlx\" (void* void* void* int int int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-sdpa-2d-mlx) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2) a3 a4 a5 a6))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
prim__sdpa2dMlx : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-rms-norm-2d-mlx)) (set-top-level-value! 'idris-ffi-tensor-rms-norm-2d-mlx (foreign-procedure \"tensor_rms_norm_2d_mlx\" (void* void* double) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-rms-norm-2d-mlx) (vector-ref a0 2) (vector-ref a1 2) a2))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
prim__rmsNorm2dMlx : AnyPtr -> AnyPtr -> Double -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-swiglu-2d-mlx)) (set-top-level-value! 'idris-ffi-tensor-swiglu-2d-mlx (foreign-procedure \"tensor_swiglu_2d_mlx\" (void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-swiglu-2d-mlx) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
prim__swiGlu2dMlx : AnyPtr -> AnyPtr -> AnyPtr

public export
{s : MlxStream} -> UserExecutorNN (MlxExecutor s) where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primBatchNorm a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 = prim__batchNormMlxStreamed a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 (streamTag s)
  primBceWithLogits a0 a1                     = prim__bceWithLogitsMlxStreamed a0 a1 (streamTag s)
  primCosineSimilarity a0 a1 a2               = prim__cosineSimilarityMlxStreamed a0 a1 a2 (streamTag s)
  primDropout a0 a1 a2 a3                     = prim__dropoutMlxStreamed a0 a1 a2 a3 (streamTag s)
  primEmbedding a0 a1 a2 a3                   = prim__embeddingMlxStreamed a0 a1 a2 a3 (streamTag s)
  primEmbedding2d a0 a1 a2 a3                 = prim__embedding2dMlxStreamed a0 a1 a2 a3 (streamTag s)
  primExpandMask a0 a1                        = prim__expandMaskMlxStreamed a0 a1 (streamTag s)
  primGelu a0                                 = prim__geluMlxStreamed a0 (streamTag s)
  primGruCell a0 a1 a2 a3                     = prim__gruCellMlxStreamed a0 a1 a2 a3 (streamTag s)
  primLayerNorm2d a0 a1 a2 a3                 = prim__layerNorm2dMlxStreamed a0 a1 a2 a3 (streamTag s)
  primLeakyRelu a0 a1                         = prim__leakyReluMlxStreamed a0 a1 (streamTag s)
  primLogSoftmax a0 a1                        = prim__logSoftmaxMlxStreamed a0 a1 (streamTag s)
  primLogSoftmax2d a0                         = prim__logSoftmax2dMlxStreamed a0 (streamTag s)
  primLstmGatesPair a0 a1 a2                  = prim__lstmGatesPairMlxStreamed a0 a1 a2 (streamTag s)
  primMaskedFill a0 a1 a2                     = prim__maskedFillMlxStreamed a0 a1 a2 (streamTag s)
  primPairFirst a0                            = prim__pairFirstMlxStreamed a0 (streamTag s)
  primPairSecond a0                           = prim__pairSecondMlxStreamed a0 (streamTag s)
  primSilu a0                                 = prim__siluMlxStreamed a0 (streamTag s)
  primSoftmax a0 a1                           = prim__softmaxMlxStreamed a0 a1 (streamTag s)
  primSoftmax2d a0                            = prim__softmax2dMlxStreamed a0 (streamTag s)
  primSoftmax3d a0                            = prim__softmax3dMlxStreamed a0 (streamTag s)
  primSoftplus a0                             = prim__softplusMlxStreamed a0 (streamTag s)
  -- <<< END GENERATED <<<

----------------------------------------------------------------------
-- Conv-slice FFI bindings (mlx-suffixed)
----------------------------------------------------------------------

%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5)  (let ((raw_r ((foreign-procedure \"tensor_conv1d_mlx_streamed\" (void* void* void* int int int) void*) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2) a3 a4 a5))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__conv1dMlxStreamed : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_conv1d_circular_mlx_streamed\" (void* void* int) void*) (vector-ref a0 2) (vector-ref a1 2) a2))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__conv1dCircularMlxStreamed : AnyPtr -> AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_avg_pool1d_mlx_streamed\" (void* int int int) void*) (vector-ref a0 2) a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__avgPool1dMlxStreamed : AnyPtr -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_max_pool1d_mlx_streamed\" (void* int int int) void*) (vector-ref a0 2) a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__maxPool1dMlxStreamed : AnyPtr -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6 a7)  (let ((raw_r ((foreign-procedure \"tensor_conv2d_mlx_streamed\" (void* void* void* int int int int int) void*) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2) a3 a4 a5 a6 a7))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__conv2dMlxStreamed : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6 a7)  (let ((raw_r ((foreign-procedure \"tensor_conv2d_batched_mlx_streamed\" (void* void* void* int int int int int) void*) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2) a3 a4 a5 a6 a7))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__conv2dBatchedMlxStreamed : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5)  (let ((raw_r ((foreign-procedure \"tensor_avg_pool2d_mlx_streamed\" (void* int int int int int) void*) (vector-ref a0 2) a1 a2 a3 a4 a5))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__avgPool2dMlxStreamed : AnyPtr -> Int -> Int -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5)  (let ((raw_r ((foreign-procedure \"tensor_max_pool2d_mlx_streamed\" (void* int int int int int) void*) (vector-ref a0 2) a1 a2 a3 a4 a5))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__maxPool2dMlxStreamed : AnyPtr -> Int -> Int -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5)  (let ((raw_r ((foreign-procedure \"tensor_max_pool2d_batched_mlx_streamed\" (void* int int int int int) void*) (vector-ref a0 2) a1 a2 a3 a4 a5))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
prim__maxPool2dBatchedMlxStreamed : AnyPtr -> Int -> Int -> Int -> Int -> Int -> AnyPtr

public export
{s : MlxStream} -> UserExecutorConv (MlxExecutor s) where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primAvgPool1d a0 a1 a2                 = prim__avgPool1dMlxStreamed a0 a1 a2 (streamTag s)
  primAvgPool2d a0 a1 a2 a3 a4           = prim__avgPool2dMlxStreamed a0 a1 a2 a3 a4 (streamTag s)
  primConv1d a0 a1 a2 a3 a4              = prim__conv1dMlxStreamed a0 a1 a2 a3 a4 (streamTag s)
  primConv1dCircular a0 a1               = prim__conv1dCircularMlxStreamed a0 a1 (streamTag s)
  primConv2d a0 a1 a2 a3 a4 a5 a6        = prim__conv2dMlxStreamed a0 a1 a2 a3 a4 a5 a6 (streamTag s)
  primConv2dBatched a0 a1 a2 a3 a4 a5 a6 = prim__conv2dBatchedMlxStreamed a0 a1 a2 a3 a4 a5 a6 (streamTag s)
  primMaxPool1d a0 a1 a2                 = prim__maxPool1dMlxStreamed a0 a1 a2 (streamTag s)
  primMaxPool2d a0 a1 a2 a3 a4           = prim__maxPool2dMlxStreamed a0 a1 a2 a3 a4 (streamTag s)
  primMaxPool2dBatched a0 a1 a2 a3 a4    = prim__maxPool2dBatchedMlxStreamed a0 a1 a2 a3 a4 (streamTag s)
  -- <<< END GENERATED <<<
----------------------------------------------------------------------
-- Tape-slice FFI bindings (mlx-suffixed)
----------------------------------------------------------------------

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-requires-grad-mlx)) (set-top-level-value! 'idris-ffi-tensor-requires-grad-mlx (foreign-procedure \"tensor_requires_grad_mlx\" (void*) int))) ((top-level-value 'idris-ffi-tensor-requires-grad-mlx) (vector-ref a0 2)))"
export
prim__requiresGradMlx : AnyPtr -> Int
%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-set-requires-grad-mlx)) (set-top-level-value! 'idris-ffi-tensor-set-requires-grad-mlx (foreign-procedure \"tensor_set_requires_grad_mlx\" (void* int) void))) ((top-level-value 'idris-ffi-tensor-set-requires-grad-mlx) (vector-ref a0 2) a1))"
export
prim__setRequiresGradMlx : AnyPtr -> Int -> PrimIO ()
%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-backward-mlx)) (set-top-level-value! 'idris-ffi-tensor-backward-mlx (foreign-procedure \"tensor_backward_mlx\" (void*) void))) ((top-level-value 'idris-ffi-tensor-backward-mlx) (vector-ref a0 2)))"
export
prim__backwardMlx : AnyPtr -> PrimIO ()
%foreign "C:tensor_no_grad_begin_mlx,libidrisml"
export
prim__noGradBeginMlx : PrimIO ()
%foreign "C:tensor_no_grad_end_mlx,libidrisml"
export
prim__noGradEndMlx : PrimIO ()
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_detach_mlx_streamed\" (void* int) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
export
prim__detachMlxStreamed : AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_with_grad_mlx_streamed\" (void* int) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_mlx\" (void*) void) raw_r) wr)))"
export
prim__withGradMlxStreamed : AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-dim-mlx)) (set-top-level-value! 'idris-ffi-tensor-dim-mlx (foreign-procedure \"tensor_dim_mlx\" (void*) int))) ((top-level-value 'idris-ffi-tensor-dim-mlx) (vector-ref a0 2)))"
export
prim__tensorDimMlx : AnyPtr -> Int
%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-size-mlx)) (set-top-level-value! 'idris-ffi-tensor-size-mlx (foreign-procedure \"tensor_size_mlx\" (void* int) int))) ((top-level-value 'idris-ffi-tensor-size-mlx) (vector-ref a0 2) a1))"
export
prim__tensorSizeAtMlx : AnyPtr -> Int -> Int
%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-param-register-return-mlx)) (set-top-level-value! 'idris-ffi-param-register-return-mlx (foreign-procedure \"param_register_return_mlx\" (string void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-param-register-return-mlx) a0 (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
export
prim__paramRegisterMlx : String -> AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-param-register-buffer-return-mlx)) (set-top-level-value! 'idris-ffi-param-register-buffer-return-mlx (foreign-procedure \"param_register_buffer_return_mlx\" (string void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-param-register-buffer-return-mlx) a0 (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
export
prim__paramRegisterBufferMlx : String -> AnyPtr -> AnyPtr
%foreign "C:param_is_buffer_mlx,libidrisml"
export
prim__paramIsBufferMlx : Int -> PrimIO Int
%foreign "C:polyak_blend_pair_mlx,libidrisml"
prim__polyakBlendPairMlx : Double -> String -> String -> PrimIO Int
%foreign "C:param_count_mlx,libidrisml"
export
prim__paramCountMlx : PrimIO Int
%foreign "C:param_name_mlx,libidrisml"
export
prim__paramNameMlx : Int -> PrimIO String
%foreign "C:param_grad_item_at_mlx,libidrisml"
export
prim__paramGradItemAtMlx : Int -> Int -> PrimIO Double
%foreign "C:param_zero_all_grads_mlx,libidrisml"
export
prim__paramZeroAllMlx : PrimIO ()
%foreign "C:param_erase_by_prefix_mlx,libidrisml"
export
prim__paramEraseByPrefixMlx : String -> PrimIO ()
%foreign "C:optimizer_create_sgd_mlx,libidrisml"
export
prim__optimizerCreateSgdMlx : Double -> AnyPtr
%foreign "C:optimizer_create_rmsprop_mlx,libidrisml"
export
prim__optimizerCreateRmspropMlx : Double -> Double -> Double -> Double -> Double -> AnyPtr
%foreign "C:optimizer_create_adam_mlx,libidrisml"
export
prim__optimizerCreateAdamMlx : Double -> Double -> Double -> Double -> AnyPtr
%foreign "C:optimizer_create_adamw_mlx,libidrisml"
export
prim__optimizerCreateAdamWMlx : Double -> Double -> Double -> Double -> Double -> AnyPtr
%foreign "C:optimizer_set_lr_mlx,libidrisml"
export
prim__optimizerSetLrMlx : AnyPtr -> Double -> PrimIO ()
%foreign "C:optimizer_set_param_lr_mlx,libidrisml"
export
prim__optimizerSetParamLrMlx : AnyPtr -> String -> Double -> PrimIO ()
%foreign "C:optimizer_own_param_mlx,libidrisml"
export
prim__optimizerOwnParamMlx : AnyPtr -> String -> PrimIO ()
%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (when (not (top-level-bound? 'idris-ffi-native-train-step-mlx)) (set-top-level-value! 'idris-ffi-native-train-step-mlx (foreign-procedure \"native_train_step_mlx\" (void* int double void* double) double))) ((top-level-value 'idris-ffi-native-train-step-mlx) a0 a1 a2 (vector-ref a3 2) a4))"
export
prim__nativeTrainStepMlx : AnyPtr -> Int -> Double -> AnyPtr -> Double -> Double
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5)  (when (not (top-level-bound? 'idris-ffi-native-train-step-scaled-mlx)) (set-top-level-value! 'idris-ffi-native-train-step-scaled-mlx (foreign-procedure \"native_train_step_scaled_mlx\" (void* int double void* double double) double))) ((top-level-value 'idris-ffi-native-train-step-scaled-mlx) a0 a1 a2 (vector-ref a3 2) a4 a5))"
export
prim__nativeTrainStepScaledMlx : AnyPtr -> Int -> Double -> AnyPtr -> Double -> Double -> Double
%foreign "C:param_save_mlx,libidrisml"
export
prim__paramSaveMlx : String -> PrimIO Int
%foreign "C:param_save_by_name_mlx,libidrisml"
export
prim__paramSaveByNameMlx : String -> String -> Int -> PrimIO Int
%foreign "C:param_save_by_name_renamed_mlx,libidrisml"
export
prim__paramSaveByNameRenamedMlx : String -> String -> String -> Int -> PrimIO Int
%foreign "C:param_load_mlx,libidrisml"
export
prim__paramLoadMlx : String -> PrimIO Int
%foreign "C:param_load_with_policy_mlx,libidrisml"
export
prim__paramLoadWithPolicyMlx : String -> Int -> PrimIO Int
%foreign "C:param_load_with_prefix_mlx,libidrisml"
export
prim__paramLoadWithPrefixMlx : String -> Int -> String -> PrimIO Int
%foreign "C:param_load_renamed_mlx,libidrisml"
export
prim__paramLoadRenamedMlx : String -> Int -> String -> String -> Int -> PrimIO Int
%foreign "C:optimizer_save_mlx,libidrisml"
export
prim__optimizerSaveMlx : AnyPtr -> String -> PrimIO Int
%foreign "C:optimizer_load_mlx,libidrisml"
export
prim__optimizerLoadMlx : AnyPtr -> String -> PrimIO Int
%foreign "C:backend_profile_reset_mlx,libidrisml"
export
prim__profileResetMlx : PrimIO ()
%foreign "C:backend_profile_report_mlx,libidrisml"
export
prim__profileReportMlx : PrimIO ()
%foreign "C:tensor_epoch_begin_mlx,libidrisml"
export
prim__epochBeginMlx : PrimIO ()
%foreign "C:tensor_epoch_end_mlx,libidrisml"
export
prim__epochEndMlx : PrimIO ()
%foreign "C:backend_release_all_persistent_mlx,libidrisml"
export
prim__releaseAllPersistentMlx : PrimIO ()
%foreign "C:backend_reset_for_eval_mlx,libidrisml"
export
prim__resetForEvalMlx : PrimIO ()
%foreign "C:tensor_live_count_mlx,libidrisml"
export
prim__liveCountMlx : PrimIO Int
%foreign "C:tensor_peak_live_count_mlx,libidrisml"
export
prim__peakLiveCountMlx : PrimIO Int
%foreign "C:tensor_perf_reset_mlx,libidrisml"
export
prim__perfResetMlx : PrimIO ()
%foreign "C:tensor_perf_op_count_mlx,libidrisml"
export
prim__perfOpCountMlx : PrimIO Int

%foreign "scheme:(lambda (a0 a1 a2 a3) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-scalar-streamed-mlx)) (set-top-level-value! 'idris-ffi-tensor-create-scalar-streamed-mlx (foreign-procedure \"tensor_create_scalar_streamed_mlx\" (double int int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-scalar-streamed-mlx) a0 a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
export
prim__createScalarStreamedMlx : Double -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-streamed-mlx)) (set-top-level-value! 'idris-ffi-tensor-create-streamed-mlx (foreign-procedure \"tensor_create_streamed_mlx\" (void* void* int int int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-streamed-mlx) a0 a1 a2 a3 a4 a5))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
export
prim__createStreamedMlx : AnyPtr -> AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-1d-streamed-mlx)) (set-top-level-value! 'idris-ffi-tensor-create-1d-streamed-mlx (foreign-procedure \"tensor_create_1d_streamed_mlx\" (int void* int int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-1d-streamed-mlx) a0 a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
export
prim__create1dStreamedMlx : Int -> AnyPtr -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-2d-streamed-mlx)) (set-top-level-value! 'idris-ffi-tensor-create-2d-streamed-mlx (foreign-procedure \"tensor_create_2d_streamed_mlx\" (int int void* int int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-2d-streamed-mlx) a0 a1 a2 a3 a4 a5))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
export
prim__create2dStreamedMlx : Int -> Int -> AnyPtr -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-param-1d-streamed-mlx)) (set-top-level-value! 'idris-ffi-tensor-create-param-1d-streamed-mlx (foreign-procedure \"tensor_create_param_1d_streamed_mlx\" (int void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-param-1d-streamed-mlx) a0 a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
export
prim__createParam1dStreamedMlx : Int -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-param-2d-streamed-mlx)) (set-top-level-value! 'idris-ffi-tensor-create-param-2d-streamed-mlx (foreign-procedure \"tensor_create_param_2d_streamed_mlx\" (int int void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-param-2d-streamed-mlx) a0 a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
export
prim__createParam2dStreamedMlx : Int -> Int -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-param-3d-streamed-mlx)) (set-top-level-value! 'idris-ffi-tensor-create-param-3d-streamed-mlx (foreign-procedure \"tensor_create_param_3d_streamed_mlx\" (int int int void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-param-3d-streamed-mlx) a0 a1 a2 a3 a4 a5))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
export
prim__createParam3dStreamedMlx : Int -> Int -> Int -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-param-4d-streamed-mlx)) (set-top-level-value! 'idris-ffi-tensor-create-param-4d-streamed-mlx (foreign-procedure \"tensor_create_param_4d_streamed_mlx\" (int int int int void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-param-4d-streamed-mlx) a0 a1 a2 a3 a4 a5 a6))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
export
prim__createParam4dStreamedMlx : Int -> Int -> Int -> Int -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-state-1d-streamed-mlx)) (set-top-level-value! 'idris-ffi-tensor-create-state-1d-streamed-mlx (foreign-procedure \"tensor_create_state_1d_streamed_mlx\" (int void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-state-1d-streamed-mlx) a0 a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
export
prim__createState1dStreamedMlx : Int -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-state-2d-streamed-mlx)) (set-top-level-value! 'idris-ffi-tensor-create-state-2d-streamed-mlx (foreign-procedure \"tensor_create_state_2d_streamed_mlx\" (int int void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-state-2d-streamed-mlx) a0 a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
export
prim__createState2dStreamedMlx : Int -> Int -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-cast-dtype-streamed-mlx)) (set-top-level-value! 'idris-ffi-tensor-cast-dtype-streamed-mlx (foreign-procedure \"tensor_cast_dtype_streamed_mlx\" (void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-cast-dtype-streamed-mlx) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
export
prim__castStreamedMlx : AnyPtr -> Int -> Int -> AnyPtr

-- Fused param create + in-place init. Mlx's C-side port slots stay
-- nullptr until Phase 7 lands the impl (mx::random::normal etc.); the
-- shared trampoline in `dtype_streamed.c` aborts loud if called. See
-- the matching block in Executor/Tape.idr for the rationale.
%foreign "scheme:(lambda (a0 a1 a2 a3 a4) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-param-1d-normal-streamed-mlx)) (set-top-level-value! 'idris-ffi-tensor-create-param-1d-normal-streamed-mlx (foreign-procedure \"tensor_create_param_1d_normal_streamed_mlx\" (int double double int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-param-1d-normal-streamed-mlx) a0 a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
prim__createParam1dNormalStreamedMlx : Int -> Double -> Double -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-param-2d-normal-streamed-mlx)) (set-top-level-value! 'idris-ffi-tensor-create-param-2d-normal-streamed-mlx (foreign-procedure \"tensor_create_param_2d_normal_streamed_mlx\" (int int double double int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-param-2d-normal-streamed-mlx) a0 a1 a2 a3 a4 a5))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
prim__createParam2dNormalStreamedMlx : Int -> Int -> Double -> Double -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-param-3d-normal-streamed-mlx)) (set-top-level-value! 'idris-ffi-tensor-create-param-3d-normal-streamed-mlx (foreign-procedure \"tensor_create_param_3d_normal_streamed_mlx\" (int int int double double int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-param-3d-normal-streamed-mlx) a0 a1 a2 a3 a4 a5 a6))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
prim__createParam3dNormalStreamedMlx : Int -> Int -> Int -> Double -> Double -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6 a7) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-param-4d-normal-streamed-mlx)) (set-top-level-value! 'idris-ffi-tensor-create-param-4d-normal-streamed-mlx (foreign-procedure \"tensor_create_param_4d_normal_streamed_mlx\" (int int int int double double int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-param-4d-normal-streamed-mlx) a0 a1 a2 a3 a4 a5 a6 a7))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
prim__createParam4dNormalStreamedMlx : Int -> Int -> Int -> Int -> Double -> Double -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-param-1d-const-streamed-mlx)) (set-top-level-value! 'idris-ffi-tensor-create-param-1d-const-streamed-mlx (foreign-procedure \"tensor_create_param_1d_const_streamed_mlx\" (int double int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-param-1d-const-streamed-mlx) a0 a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
prim__createParam1dConstStreamedMlx : Int -> Double -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-param-2d-const-streamed-mlx)) (set-top-level-value! 'idris-ffi-tensor-create-param-2d-const-streamed-mlx (foreign-procedure \"tensor_create_param_2d_const_streamed_mlx\" (int int double int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-param-2d-const-streamed-mlx) a0 a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
prim__createParam2dConstStreamedMlx : Int -> Int -> Double -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-param-3d-const-streamed-mlx)) (set-top-level-value! 'idris-ffi-tensor-create-param-3d-const-streamed-mlx (foreign-procedure \"tensor_create_param_3d_const_streamed_mlx\" (int int int double int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-param-3d-const-streamed-mlx) a0 a1 a2 a3 a4 a5))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
prim__createParam3dConstStreamedMlx : Int -> Int -> Int -> Double -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-param-4d-const-streamed-mlx)) (set-top-level-value! 'idris-ffi-tensor-create-param-4d-const-streamed-mlx (foreign-procedure \"tensor_create_param_4d_const_streamed_mlx\" (int int int int double int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-mlx)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-mlx (foreign-procedure \"tensor_retain_handle_mlx\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-param-4d-const-streamed-mlx) a0 a1 a2 a3 a4 a5 a6))) (let ((wr (vector 'tensor-handle-v2 \"mlx\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-mlx) raw_r) wr)))"
prim__createParam4dConstStreamedMlx : Int -> Int -> Int -> Int -> Double -> Int -> Int -> AnyPtr
%foreign "C:tensor_set_init_seed_streamed_mlx,libidrisml"
export
prim__setInitSeedStreamedMlx : Bits64 -> Int -> PrimIO ()

public export
{s : MlxStream} -> UserExecutorOptimizations (MlxExecutor s) where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primCreateParam1dConstStreamed    = prim__createParam1dConstStreamedMlx
  primCreateParam1dNormalStreamed   = prim__createParam1dNormalStreamedMlx
  primCreateParam2dConstStreamed    = prim__createParam2dConstStreamedMlx
  primCreateParam2dNormalStreamed   = prim__createParam2dNormalStreamedMlx
  primCreateParam3dConstStreamed    = prim__createParam3dConstStreamedMlx
  primCreateParam3dNormalStreamed   = prim__createParam3dNormalStreamedMlx
  primCreateParam4dConstStreamed    = prim__createParam4dConstStreamedMlx
  primCreateParam4dNormalStreamed   = prim__createParam4dNormalStreamedMlx
  primCrossAttention a0 a1 a2 a3 a4 = prim__crossAttentionMlxStreamed a0 a1 a2 a3 a4 (streamTag s)
  primPolyakBlendPair               = prim__polyakBlendPairMlx
  primRmsNorm2d                     = prim__rmsNorm2dMlx
  primSdpa2d                        = prim__sdpa2dMlx
  primSwiGlu2d                      = prim__swiGlu2dMlx
  primTile2d a0 a1 a2               = prim__tile2dMlxStreamed a0 a1 a2 (streamTag s)
  -- <<< END GENERATED <<<
