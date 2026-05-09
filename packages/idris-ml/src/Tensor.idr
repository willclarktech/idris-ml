module Tensor

import Data.List
import Data.Maybe
import Data.SortedMap
import Data.Vect
import Compat.Random

import DataPoint
import Device
import Floating
import Array
import Util


----------------------------------------------------------------------
-- Backend FFI (libtorch via libidrisml)
----------------------------------------------------------------------

-- Lifecycle
%foreign "C:tensor_create_scalar,libidrisml"
export prim__createScalar : Double -> Int -> AnyPtr

%foreign "C:tensor_free,libidrisml"
prim__free : AnyPtr -> ()

%foreign "C:tensor_item,libidrisml"
export prim__item : AnyPtr -> Double

-- Device transfer
%foreign "C:tensor_to_device,libidrisml"
export prim__toDevice : AnyPtr -> String -> AnyPtr

%foreign "C:tensor_device,libidrisml"
export prim__tensorDevice : AnyPtr -> String

-- Arithmetic (all return new tensors — libtorch builds autograd graph)
%foreign "C:tensor_add,libidrisml"
export prim__add : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_sub,libidrisml"
export prim__sub : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_mul,libidrisml"
export prim__mul : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_div,libidrisml"
export prim__div : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_neg,libidrisml"
export prim__neg : AnyPtr -> AnyPtr

%foreign "C:tensor_abs,libidrisml"
export prim__abs : AnyPtr -> AnyPtr

%foreign "C:tensor_exp,libidrisml"
export prim__exp : AnyPtr -> AnyPtr

%foreign "C:tensor_log,libidrisml"
export prim__log : AnyPtr -> AnyPtr

%foreign "C:tensor_sqrt,libidrisml"
prim__sqrt : AnyPtr -> AnyPtr

%foreign "C:tensor_pow,libidrisml"
export prim__pow : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_sigmoid,libidrisml"
export prim__sigmoid : AnyPtr -> AnyPtr

%foreign "C:tensor_tanh,libidrisml"
export prim__tanh : AnyPtr -> AnyPtr

-- Linear algebra
%foreign "C:tensor_mv,libidrisml"
export prim__mv : AnyPtr -> AnyPtr -> AnyPtr

-- Fused 1D linear: y = W @ x + bias. Eliminates the per-call FFI
-- overhead of separate prim__mv + prim__add.
%foreign "C:tensor_linear,libidrisml"
export prim__linear : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_dot,libidrisml"
prim__dot : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_outer,libidrisml"
export prim__outer : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_matmul,libidrisml"
export prim__matmul : AnyPtr -> AnyPtr -> AnyPtr

-- Activation
%foreign "C:tensor_softmax,libidrisml"
export prim__softmax : AnyPtr -> Int -> AnyPtr

%foreign "C:tensor_log_softmax,libidrisml"
export prim__logSoftmax : AnyPtr -> Int -> AnyPtr

-- Loss
%foreign "C:tensor_bce_with_logits,libidrisml"
prim__bceWithLogits : AnyPtr -> AnyPtr -> AnyPtr

-- Reduction
%foreign "C:tensor_sum,libidrisml"
export prim__sum : AnyPtr -> AnyPtr

%foreign "C:tensor_mean,libidrisml"
export prim__mean : AnyPtr -> AnyPtr

%foreign "C:tensor_min,libidrisml"
export prim__tensorMin : AnyPtr -> AnyPtr

%foreign "C:tensor_max,libidrisml"
export prim__tensorMax : AnyPtr -> AnyPtr

-- Array creation/accessors
%foreign "C:tensor_create,libidrisml"
prim__create : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr

%foreign "C:tensor_numel,libidrisml"
prim__numel : AnyPtr -> Int

%foreign "C:tensor_size,libidrisml"
prim__size : AnyPtr -> Int -> Int

%foreign "C:tensor_select,libidrisml"
export prim__select : AnyPtr -> Int -> Int -> AnyPtr

%foreign "C:tensor_unsqueeze,libidrisml"
export prim__unsqueeze : AnyPtr -> Int -> AnyPtr

%foreign "C:tensor_stack,libidrisml"
prim__stack : AnyPtr -> Int -> Int -> AnyPtr

%foreign "C:tensor_detach,libidrisml"
prim__detach : AnyPtr -> AnyPtr

%foreign "C:tensor_with_grad,libidrisml"
prim__withGrad : AnyPtr -> AnyPtr

%foreign "C:tensor_mul_scalar,libidrisml"
export prim__mulScalar : AnyPtr -> Double -> AnyPtr

%foreign "C:tensor_add_scalar,libidrisml"
export prim__addScalar : AnyPtr -> Double -> AnyPtr

%foreign "C:tensor_clamp_min,libidrisml"
export prim__clampMin : AnyPtr -> Double -> AnyPtr

-- NTM
%foreign "C:tensor_cosine_similarity,libidrisml"
export prim__cosineSimilarity : AnyPtr -> AnyPtr -> Int -> AnyPtr

%foreign "C:tensor_conv1d_circular,libidrisml"
export prim__conv1dCircular : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_gelu,libidrisml"
export prim__gelu : AnyPtr -> AnyPtr

%foreign "C:tensor_leaky_relu,libidrisml"
export prim__leakyRelu : AnyPtr -> Double -> AnyPtr

%foreign "C:tensor_silu,libidrisml"
export prim__silu : AnyPtr -> AnyPtr

%foreign "C:tensor_softplus,libidrisml"
export prim__softplus : AnyPtr -> AnyPtr

-- Cross-attention: Q @ K^T * scale [+ mask] -> softmax -> @ V
%foreign "C:tensor_cross_attention,libidrisml"
export
prim__crossAttention : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> Double -> AnyPtr

%foreign "C:tensor_gru_cell,libidrisml"
export prim__gruCell : AnyPtr -> AnyPtr -> Int -> AnyPtr

-- Embedding
%foreign "C:tensor_embedding,libidrisml"
export
prim__embedding : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr

-- Batch Norm
%foreign "C:tensor_batch_norm,libidrisml"
export
prim__batchNorm : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> Double -> Double -> AnyPtr

-- Dropout
%foreign "C:tensor_dropout,libidrisml"
export
prim__dropout : AnyPtr -> Double -> Int -> Int -> AnyPtr

-- Shape / info queries
%foreign "C:tensor_squeeze,libidrisml"
export prim__squeeze : AnyPtr -> Int -> AnyPtr

%foreign "C:tensor_clone,libidrisml"
export prim__clone : AnyPtr -> AnyPtr

%foreign "C:tensor_dim,libidrisml"
export prim__tensorDim : AnyPtr -> Int

%foreign "C:tensor_size,libidrisml"
export prim__tensorSizeAt : AnyPtr -> Int -> Int

%foreign "C:tensor_sum_dim,libidrisml"
export prim__sumDim : AnyPtr -> Int -> Int -> AnyPtr

%foreign "C:tensor_requires_grad,libidrisml"
export prim__requiresGrad : AnyPtr -> Int

-- Gather / Scatter
%foreign "C:tensor_gather,libidrisml"
export prim__gather : AnyPtr -> AnyPtr -> Int -> AnyPtr

%foreign "C:tensor_scatter_add,libidrisml"
export prim__scatterAdd : AnyPtr -> AnyPtr -> Int -> AnyPtr

-- Sort / Scan
%foreign "C:tensor_argsort,libidrisml"
export prim__argsort : AnyPtr -> Int -> Int -> AnyPtr

%foreign "C:tensor_cumprod,libidrisml"
export prim__cumprod : AnyPtr -> Int -> AnyPtr

-- Average Pooling
%foreign "C:tensor_avg_pool1d,libidrisml"
export
prim__avgPool1d : AnyPtr -> Int -> Int -> AnyPtr

%foreign "C:tensor_avg_pool2d,libidrisml"
export
prim__avgPool2d : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr

-- Conv1D / MaxPool1D
%foreign "C:tensor_conv1d,libidrisml"
export
prim__conv1d : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr

%foreign "C:tensor_max_pool1d,libidrisml"
export
prim__maxPool1d : AnyPtr -> Int -> Int -> AnyPtr

%foreign "C:tensor_create_param_3d,libidrisml"
export
prim__createParam3d : Int -> Int -> Int -> AnyPtr -> AnyPtr

-- Conv2D / MaxPool2D
%foreign "C:tensor_conv2d,libidrisml"
export
prim__conv2d : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr

%foreign "C:tensor_max_pool2d,libidrisml"
export
prim__maxPool2d : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr

-- MNIST data loading
%foreign "C:mnist_load,libidrisml"
export
prim__mnistLoad : String -> String -> AnyPtr

%foreign "C:mnist_count,libidrisml"
export
prim__mnistCount : AnyPtr -> Int

%foreign "C:mnist_get_image,libidrisml"
export
prim__mnistGetImage : AnyPtr -> Int -> AnyPtr

%foreign "C:mnist_get_label,libidrisml"
export
prim__mnistGetLabel : AnyPtr -> Int -> Int

-- Parameter registry
-- Registers a parameter: enables requires_grad and adds to the registry.
-- Returns the tensorPtr for threading (prevents dead code elimination).
%foreign "C:param_register_return,libidrisml"
export
prim__paramRegister : String -> AnyPtr -> AnyPtr

%foreign "C:param_clear,libidrisml"
prim__paramClear : ()

%foreign "C:param_count,libidrisml"
prim__paramCount : Int

%foreign "C:param_name,libidrisml"
prim__paramName : Int -> String

%foreign "C:param_grad_item,libidrisml"
prim__paramGradItem : Int -> Double

%foreign "C:param_grad_item_at,libidrisml"
export
prim__paramGradItemAt : Int -> Int -> Double

%foreign "C:param_grad_item_and_zero,libidrisml"
prim__paramGradItemAndZero : Int -> Double

%foreign "C:param_zero_all_grads_return,libidrisml"
prim__paramZeroAllGrads : Int -> Int

%foreign "C:param_subtract_delta,libidrisml"
prim__paramSubtractDelta : Int -> Double -> ()

-- In-place scalar subtract on a tensor (under no_grad). Returns tensor for threading.
%foreign "C:tensor_subtract_scalar_inplace,libidrisml"
export
prim__tensorSubScalarInplace : AnyPtr -> Double -> AnyPtr

-- Array-level parameter creation
%foreign "C:tensor_create_param_2d,libidrisml"
export
prim__createParam2d : Int -> Int -> AnyPtr -> AnyPtr

%foreign "C:tensor_create_param_1d,libidrisml"
export
prim__createParam1d : Int -> AnyPtr -> AnyPtr

-- Persistent non-param tensors (for non-learnable state like NTM memory)
%foreign "C:tensor_create_state_2d,libidrisml"
export
prim__createState2d : Int -> Int -> AnyPtr -> AnyPtr

%foreign "C:tensor_create_state_1d,libidrisml"
export
prim__createState1d : Int -> AnyPtr -> AnyPtr

%foreign "C:tensor_view_2d,libidrisml"
export
prim__view2d : AnyPtr -> Int -> Int -> AnyPtr

%foreign "C:tensor_view_1d,libidrisml"
export
prim__view1d : AnyPtr -> Int -> AnyPtr

%foreign "C:tensor_item_2d,libidrisml"
export
prim__item2d : AnyPtr -> Int -> Int -> Double

%foreign "C:tensor_item_1d,libidrisml"
export
prim__item1d : AnyPtr -> Int -> Double

-- Fused LSTM gates: takes combined [4*o] tensor + prev_cell [o], returns pair handle
%foreign "C:tensor_lstm_gates_pair,libidrisml"
export
prim__lstmGatesPair : AnyPtr -> AnyPtr -> Int -> AnyPtr

%foreign "C:tensor_pair_first,libidrisml"
export
prim__pairFirst : AnyPtr -> AnyPtr

%foreign "C:tensor_pair_second,libidrisml"
export
prim__pairSecond : AnyPtr -> AnyPtr

-- Array-level forward ops (used by layers with consolidated weight tensors)
||| Matrix-vector multiply on raw tensor pointers.
export
tensorMv : AnyPtr -> AnyPtr -> AnyPtr
tensorMv = prim__mv

||| Add two raw tensor pointers.
export
tensorAdd : AnyPtr -> AnyPtr -> AnyPtr
tensorAdd = prim__add

-- No-grad scope
%foreign "C:tensor_no_grad_begin,libidrisml"
prim__noGradBegin : ()

%foreign "C:tensor_no_grad_end,libidrisml"
prim__noGradEnd : ()

-- LSTM
%foreign "C:tensor_lstm_cell,libidrisml"
prim__lstmCell : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> ()

-- Debug
%foreign "C:tensor_print,libidrisml"
prim__print : AnyPtr -> ()


----------------------------------------------------------------------
-- Sequencing helper
----------------------------------------------------------------------

-- Force evaluation of first arg, return second.
-- Must use concrete AnyPtr types (not polymorphic) to avoid
-- argument count issues at the FFI boundary.
%foreign "C:idrisml_seq,libidrisml"
export
prim__seq : AnyPtr -> AnyPtr -> AnyPtr


----------------------------------------------------------------------
-- C-side allocation + bulk-load helpers
----------------------------------------------------------------------

%foreign "C:tensor_alloc_doubles,libidrisml"
export prim__allocDoubles : Int -> AnyPtr

%foreign "C:tensor_write_double,libidrisml"
export prim__writeDouble : AnyPtr -> Int -> Double -> ()

%foreign "C:tensor_read_double,libidrisml"
prim__readDouble : AnyPtr -> Int -> Double

-- Wrapper that returns the buffer pointer for threading through let chains
%foreign "C:tensor_write_double_return,libidrisml"
export
prim__setDouble : AnyPtr -> Int -> Double -> AnyPtr

%foreign "C:tensor_one_hot,libidrisml"
export
prim__oneHot : AnyPtr -> Int -> Int -> AnyPtr

%foreign "C:tensor_bmm,libidrisml"
export
prim__bmm : AnyPtr -> AnyPtr -> AnyPtr

-- 3D batched attention ops
%foreign "C:tensor_bmm_3x3,libidrisml"
export
prim__bmm3x3 : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_softmax_3d,libidrisml"
export
prim__softmax3d : AnyPtr -> AnyPtr

%foreign "C:tensor_transpose_last2,libidrisml"
export
prim__transposeLast2 : AnyPtr -> AnyPtr

%foreign "C:tensor_reshape_3d,libidrisml"
export
prim__reshape3d : AnyPtr -> Int -> Int -> Int -> AnyPtr

%foreign "C:tensor_expand_mask,libidrisml"
export
prim__expandMask : AnyPtr -> Int -> AnyPtr

%foreign "C:tensor_create_1d,libidrisml"
export
prim__create1d : Int -> AnyPtr -> Int -> AnyPtr

%foreign "C:tensor_create_2d,libidrisml"
export prim__create2d : Int -> Int -> AnyPtr -> Int -> AnyPtr

-- Array pointer array: stack scalar Tensor tensorPtrs to create
-- a 1D/2D tensor that preserves the autograd graph.
%foreign "C:tensor_ptr_array_alloc,libidrisml"
prim__ptrArrayAlloc : Int -> AnyPtr

-- Returns the array for threading
%foreign "C:tensor_ptr_array_set_return,libidrisml"
prim__ptrArraySet : AnyPtr -> Int -> AnyPtr -> AnyPtr

%foreign "C:tensor_stack_from_array,libidrisml"
prim__stackFromArray : AnyPtr -> Int -> Int -> AnyPtr

%foreign "C:tensor_cat_from_array,libidrisml"
export
prim__catFromArray : AnyPtr -> Int -> Int -> AnyPtr

%foreign "C:tensor_cat2,libidrisml"
export
prim__cat2 : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_concat_2d_axis1,libidrisml"
export
prim__concat2dAxis1 : AnyPtr -> AnyPtr -> AnyPtr

-- N-ary cat: caller retains ownership of the handle array.
-- See tensor_cat in backend.h.
%foreign "C:tensor_cat,libidrisml"
export
prim__cat : AnyPtr -> Int -> Int -> AnyPtr

-- Batch [...] tensors into [count, ...]. Equivalent to stack at dim=0.
%foreign "C:tensor_batch,libidrisml"
export
prim__batch : AnyPtr -> Int -> AnyPtr

%foreign "C:tensor_narrow,libidrisml"
export
prim__narrow : AnyPtr -> Int -> Int -> Int -> AnyPtr

%foreign "C:tensor_mm,libidrisml"
export
prim__mm : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_linear_2d,libidrisml"
export
prim__linear2d : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_transpose_2d,libidrisml"
export
prim__transpose2d : AnyPtr -> AnyPtr

%foreign "C:tensor_softmax_2d,libidrisml"
export
prim__softmax2d : AnyPtr -> AnyPtr

%foreign "C:tensor_masked_fill,libidrisml"
export
prim__maskedFill : AnyPtr -> AnyPtr -> Double -> AnyPtr

%foreign "C:tensor_causal_mask,libidrisml"
export
prim__causalMask : Int -> AnyPtr

%foreign "C:tensor_log_softmax_2d,libidrisml"
export
prim__logSoftmax2d : AnyPtr -> AnyPtr

%foreign "C:tensor_layer_norm_2d,libidrisml"
export
prim__layerNorm2d : AnyPtr -> AnyPtr -> AnyPtr -> Double -> AnyPtr

%foreign "C:tensor_reshape,libidrisml"
prim__reshape : AnyPtr -> AnyPtr -> Int -> AnyPtr

%foreign "C:tensor_alloc_ints,libidrisml"
export
prim__allocInts : Int -> AnyPtr

%foreign "C:tensor_write_int_return,libidrisml"
export
prim__setInt : AnyPtr -> Int -> Int -> AnyPtr

%foreign "C:tensor_reshape_2d,libidrisml"
export prim__reshape2d : AnyPtr -> Int -> Int -> AnyPtr

-- Reshape to 1D: flatten any tensor to [n]
export
prim__reshape1d : AnyPtr -> Int -> AnyPtr
prim__reshape1d t n =
  let shape = prim__allocInts 1
      shape' = prim__setInt shape 0 n
  in prim__reshape t shape' 1

%foreign "C:tensor_create_param_4d,libidrisml"
export
prim__createParam4d : Int -> Int -> Int -> Int -> AnyPtr -> AnyPtr


----------------------------------------------------------------------
-- Backpropagation: prims for native optimizer
----------------------------------------------------------------------

%foreign "C:tensor_backward_conditional,libidrisml"
prim__backwardAndCount : AnyPtr -> Int

%foreign "C:param_zero_all_grads_return,libidrisml"
prim__zeroAllGrads : Int -> Int


----------------------------------------------------------------------
-- Native Optimizer
----------------------------------------------------------------------

%foreign "C:optimizer_create_sgd,libidrisml"
prim__optimizerCreateSgd : Double -> AnyPtr

%foreign "C:optimizer_create_rmsprop,libidrisml"
prim__optimizerCreateRmsprop : Double -> Double -> Double -> Double -> Double -> AnyPtr

%foreign "C:optimizer_create_adam,libidrisml"
prim__optimizerCreateAdam : Double -> Double -> Double -> Double -> AnyPtr

%foreign "C:optimizer_create_adam_group,libidrisml"
export
prim__optimizerCreateAdamGroup : Double -> Double -> Double -> Double -> String -> AnyPtr

%foreign "C:polyak_blend,libidrisml"
export
prim__polyakBlend : Double -> String -> String -> Int

||| Polyak soft update for twin-network param groups registered under
||| `onlineScope` vs `targetScope`: for each online param, finds the
||| matching target param (same suffix after scope prefix) and blends
|||   target_data ← (1 − tau) · target_data + tau · online_data
||| in-place. Returns the number of param pairs blended. Used by SAC to
||| track target-Q networks.
export
polyakUpdate : (tau : Double) -> (onlineScope : String) -> (targetScope : String) -> IO Int
polyakUpdate tau onlineScope targetScope =
  pure (prim__polyakBlend tau onlineScope targetScope)

%foreign "C:optimizer_clip_grad_norm,libidrisml"
prim__clipGradNorm : Double -> Double

public export
data ClipMode = NoClip | ValueClip Double | NormClip Double

||| Native libtorch optimizer. Single step() call updates all parameters.
public export
record NativeOptimizer where
  constructor MkNativeOptimizer
  handle : AnyPtr
  clipMode : ClipMode

||| Create a native SGD optimizer.
export
nativeSgd : Double -> NativeOptimizer
nativeSgd lr = MkNativeOptimizer (prim__optimizerCreateSgd lr) NoClip

||| Create a native RMSprop optimizer (matches PyTorch defaults).
export
nativeRmsprop : (lr : Double) -> (alpha : Double) -> (eps : Double) ->
                (clipVal : Double) -> (momentum : Double) -> NativeOptimizer
nativeRmsprop lr alpha eps clipVal momentum =
  MkNativeOptimizer
    (prim__optimizerCreateRmsprop lr alpha eps 0.0 momentum)
    (ValueClip clipVal)

||| Create a native Adam optimizer with global norm clipping.
export
nativeAdamGlobalClip : (lr : Double) -> (beta1 : Double) -> (beta2 : Double) ->
                       (eps : Double) -> (maxNorm : Double) -> NativeOptimizer
nativeAdamGlobalClip lr beta1 beta2 eps maxNorm =
  MkNativeOptimizer
    (prim__optimizerCreateAdam lr beta1 beta2 eps)
    (NormClip maxNorm)

||| Create a native Adam optimizer that only manages params whose registry
||| paramId starts with `scope`. Empty scope behaves like
||| `nativeAdamGlobalClip`. Used for multi-network setups where each
||| network (e.g. SAC actor / q1 / q2) needs its own optimizer so that
||| gradient leakage from one network's loss doesn't update another
||| network's weights (matches PyTorch's one-optimizer-per-net pattern).
export
nativeAdamGroup : (scope : String) ->
                  (lr : Double) -> (beta1 : Double) -> (beta2 : Double) ->
                  (eps : Double) -> (maxNorm : Double) -> NativeOptimizer
nativeAdamGroup scope lr beta1 beta2 eps maxNorm =
  MkNativeOptimizer
    (prim__optimizerCreateAdamGroup lr beta1 beta2 eps scope)
    (NormClip maxNorm)

%foreign "C:optimizer_create_adamw,libidrisml"
prim__optimizerCreateAdamW : Double -> Double -> Double -> Double -> Double -> AnyPtr

||| Create a native AdamW optimizer (decoupled weight decay) with global norm clipping.
export
nativeAdamW : (lr : Double) -> (beta1 : Double) -> (beta2 : Double) ->
              (eps : Double) -> (weightDecay : Double) -> (maxNorm : Double) -> NativeOptimizer
nativeAdamW lr beta1 beta2 eps wd maxNorm =
  MkNativeOptimizer
    (prim__optimizerCreateAdamW lr beta1 beta2 eps wd)
    (NormClip maxNorm)

%foreign "C:optimizer_set_param_lr,libidrisml"
prim__optimizerSetParamLR : AnyPtr -> String -> Double -> ()

||| Set a per-parameter learning rate override. Parameters matching the given
||| name will use this LR instead of the optimizer's base LR.
||| Use LR=0 to freeze a parameter. Set LR<0 to revert to base LR.
export
setParamLR : NativeOptimizer -> String -> Double -> ()
setParamLR opt name lr = prim__optimizerSetParamLR opt.handle name lr

%foreign "C:optimizer_set_lr,libidrisml"
prim__optimizerSetLrC : AnyPtr -> Double -> PrimIO ()

||| Update the optimizer's base (global) learning rate. Per-parameter
||| overrides set via `setParamLR` remain in effect; only un-overridden
||| params pick up the new base LR. Used to apply LR schedules per epoch.
export
setLearningRate : NativeOptimizer -> Double -> IO ()
setLearningRate opt lr = primIO (prim__optimizerSetLrC opt.handle lr)

-- Fused native train step: zero_grad → backward → clip → step.
-- Fused: zero_grad → backward → clip → step in single C call.
-- Returns loss value (read before step, so not stale).
%foreign "C:native_train_step,libidrisml"
prim__nativeTrainStep : AnyPtr -> Int -> Double -> AnyPtr -> Double -> Double

----------------------------------------------------------------------
-- Parameter Count
----------------------------------------------------------------------

export
getNumPids : Int -> Int
getNumPids _ = prim__paramCount


----------------------------------------------------------------------
-- GC / RSS
----------------------------------------------------------------------

%foreign "C:backend_supports_tensor_params,libidrisml"
export
prim__backendSupportsTensorParams : Int

export
forceGC : IO ()
forceGC = pure ()

%foreign "C:get_rss_mb,libidrisml"
prim__getRssMB : Int

%foreign "C:get_current_rss_mb,libidrisml"
prim__getCurrentRssMB : Int

export
getRssMB : Nat -> Int
getRssMB _ = prim__getRssMB

export
getCurrentRssMB : Nat -> Int
getCurrentRssMB _ = prim__getCurrentRssMB

%foreign "C:backend_memory_report_return,libidrisml"
prim__memoryReport : Int -> Int

%foreign "C:backend_reset_for_eval_return,libidrisml"
prim__resetForEval : Int -> Int

||| Bulk-convert a Vector of Doubles to a C tensor handle.
||| The C tensor_create_1d function frees the input buffer after copying.
export
bulkToTensor : {n : Nat} -> Vector n Double -> AnyPtr
bulkToTensor {n} (VArray elems) =
  let nI = cast {to=Int} n
      buf = prim__allocDoubles nI
      buf' = packDoubleBuf buf 0 elems
  in prim__create1d nI buf' 0
  where
    packDoubleBuf : AnyPtr -> Int -> Vect k (Scalar Double) -> AnyPtr
    packDoubleBuf buf _ [] = buf
    packDoubleBuf buf off (SArray v :: rest) =
      let buf' = prim__setDouble buf off v
      in packDoubleBuf buf' (off + 1) rest

||| Bulk-convert a Vect of Vectors of Doubles to a [b, i] C tensor handle.
||| The C tensor_create_2d function frees the input buffer after copying.
||| Use to stack a per-sample input batch into a single batched tensor.
export
bulkToTensor2d : {b, i : Nat} -> Vect b (Vector i Double) -> AnyPtr
bulkToTensor2d {b} {i} rows =
  let bI = cast {to=Int} b
      iI = cast {to=Int} i
      buf = prim__allocDoubles (bI * iI)
      buf' = packRows buf 0 rows
  in prim__create2d bI iI buf' 0
  where
    packRow : AnyPtr -> Int -> Vect k (Scalar Double) -> AnyPtr
    packRow buf _ [] = buf
    packRow buf off (SArray v :: rest) =
      let buf' = prim__setDouble buf off v
      in packRow buf' (off + 1) rest
    packRows : AnyPtr -> Int -> Vect k (Vector i Double) -> AnyPtr
    packRows buf _ [] = buf
    packRows buf off (VArray row :: rest) =
      let buf' = packRow buf off row
      in packRows buf' (off + cast {to=Int} i) rest

||| Bulk-convert a Vector of Doubles to a persistent C tensor handle.
||| Persistent tensors survive tape resets — use when data is created once
||| and reused across training epochs.
export
vectorToTensorPersistent : {n : Nat} -> Vector n Double -> AnyPtr
vectorToTensorPersistent {n} (VArray elems) =
  let nI = cast {to=Int} n
      buf = prim__allocDoubles nI
      buf' = packBuf buf 0 elems
  in prim__createState1d nI buf'
  where
    packBuf : AnyPtr -> Int -> Vect k (Scalar Double) -> AnyPtr
    packBuf buf _ [] = buf
    packBuf buf off (SArray v :: rest) = packBuf (prim__setDouble buf off v) (off + 1) rest

||| Convert a DataPoint with Doubles to a TensorDataPoint with persistent C tensors.
export
toTDP : {i, o : Nat} -> DataPoint i o Double -> TensorDataPoint i o
toTDP dp = MkTensorDataPoint (vectorToTensorPersistent (x dp)) (vectorToTensorPersistent (y dp))

||| Reset tape + arena for a clean eval forward pass.
||| Returns a dummy value that should be threaded into subsequent computation.
export
resetForEval : Int -> Int
resetForEval dummy = prim__resetForEval dummy

||| Print detailed memory breakdown to stderr.
export
memoryReport : IO ()
memoryReport = let _ = prim__memoryReport 0 in pure ()

%foreign "C:backend_profile_reset_return,libidrisml"
prim__profileReset : Int -> Int

%foreign "C:backend_profile_report_return,libidrisml"
prim__profileReport : Int -> Int

%foreign "C:tensor_backward,libidrisml"
prim__backwardC : AnyPtr -> PrimIO ()

%foreign "C:param_zero_all_grads,libidrisml"
prim__zeroAllGradsC : PrimIO ()

||| Run backward on a loss tensor.
export
runBackward : AnyPtr -> IO ()
runBackward ptr = primIO (prim__backwardC ptr)

%foreign "C:param_count,libidrisml"
prim__paramCountC : PrimIO Int

%foreign "C:param_name,libidrisml"
prim__paramNameC : Int -> PrimIO String

%foreign "C:param_grad_item_at,libidrisml"
prim__paramGradItemAtC : Int -> Int -> PrimIO Double

||| Get parameter count (for gradient inspection).
export
getParamCount : IO Int
getParamCount = primIO prim__paramCountC

||| Get parameter name by index.
export
getParamName : Int -> IO String
getParamName i = primIO (prim__paramNameC i)

||| Get gradient element for param i, element j.
export
getParamGradAt : Int -> Int -> IO Double
getParamGradAt i j = primIO (prim__paramGradItemAtC i j)

||| Zero all parameter gradients.
export
zeroAllGrads : IO ()
zeroAllGrads = primIO prim__zeroAllGradsC

%foreign "C:backend_name,libidrisml"
prim__backendName : String

||| Get the name of the active backend ("tape", "mlx", "torch").
export
backendName : String
backendName = prim__backendName

%foreign "C:backend_profile_reset,libidrisml"
prim__profileResetC : PrimIO ()

%foreign "C:backend_profile_report,libidrisml"
prim__profileReportC : PrimIO ()

||| Reset profiling counters.
export
profileReset : IO ()
profileReset = primIO prim__profileResetC

||| Print profiling breakdown to stderr.
export
profileReport : IO ()
profileReport = primIO prim__profileReportC

----------------------------------------------------------------------
-- Path C P3-1 spike: rank-aware Tensor
----------------------------------------------------------------------
--
-- Today's `Tensor d` is shape-erased and packed into the outer
-- `Array dims (Tensor d)` via Vect-of-Vect, scalarising at every
-- op. `Tensor dims d` lifts shape onto the Tensor itself: one tensor
-- handle per typed shape, no per-element packing.
--
-- `Tensor []` is the scalar — distinguished from `Tensor [n]` etc. by
-- type. Loss naturally types as `Tensor [] d`.
--
-- Keep `paramId`: the C-side optimizer registry is keyed on it.
-- Drop the cached `value : Double` — read at the boundary via
-- `tensorItem`.
--
-- Spike-only; lives in a parallel layer/example axis.

public export
record Tensor (dims : Vect rank Nat) (0 d : Device) where
  constructor MkTensor
  tensorPtr : AnyPtr
  paramId   : Maybe String

||| Transfer a tensor to a different device. The one place where
||| device types intentionally change. Wraps `prim__toDevice`; the
||| resulting tensor handle is on `d2`. `paramId` is preserved (the
||| C-side parameter registry tracks the moved handle).
export
toDevice : (d2 : Device) -> Tensor dims d1 -> IO (Tensor dims d2)
toDevice d2 t =
  pure (MkTensor (prim__toDevice t.tensorPtr (deviceToString d2))
                 t.paramId)

||| Type-level aliases for common Tensor shapes. Aliases route shape
||| arithmetic (e.g. `4 * o`) through a Nat-argument slot rather than
||| inlining inside a Vect literal — the latter triggers an Idris 2
||| type-checker hang on multiplicative Nat expressions.
||| (`Tensor [4 * o, i] d` hangs; `TMat (4 * o) i d` works.)
public export
0 TVec : Nat -> Device -> Type
TVec n d = Tensor [n] d

public export
0 TMat : Nat -> Nat -> Device -> Type
TMat m n d = Tensor [m, n] d

-- Smart constructors --------------------------------------------------

||| Create a registered learnable [o, i] parameter from a flat (row-major)
||| double buffer. Mirrors Linear.nameLayer's tensor path.
export
tparam2d : {o, i : Nat} -> (paramId : String) -> AnyPtr -> Tensor [o, i] d
tparam2d {o} {i} pid buf =
  let oI = cast {to=Int} o
      iI = cast {to=Int} i
      reg = prim__paramRegister pid (prim__createParam2d oI iI buf)
  in MkTensor reg (Just pid)

||| Create a registered learnable [n] parameter from a double buffer.
export
tparam1d : {n : Nat} -> (paramId : String) -> AnyPtr -> Tensor [n] d
tparam1d {n} pid buf =
  let nI = cast {to=Int} n
      reg = prim__paramRegister pid (prim__createParam1d nI buf)
  in MkTensor reg (Just pid)

||| Wrap an existing 1D tensor handle as a non-parameter input.
export
tinput1d : {n : Nat} -> AnyPtr -> Tensor [n] d
tinput1d t = MkTensor t Nothing

||| Wrap an existing 2D tensor handle as a non-parameter input.
export
tinput2d : {m, n : Nat} -> AnyPtr -> Tensor [m, n] d
tinput2d t = MkTensor t Nothing

-- Arithmetic / linear algebra (autograd-tracked) ----------------------

||| Elementwise addition. Both operands share shape.
||| `%inline`: inlines to a direct `prim__add` + `MkTensor` allocation
||| at every call site. Critical for hot-path layers (LSTM/NTM/DNC
||| call this many times per timestep); without inlining, Idris2's
||| Chez codegen wraps each invocation in a closure dispatch that
||| adds ~20µs of Scheme-side overhead per call, accumulating to a
||| 2× regression on recurrent models.
export %inline
tadd : Tensor dims d -> Tensor dims d -> Tensor dims d
tadd a b = MkTensor (prim__add a.tensorPtr b.tensorPtr) Nothing

||| Matrix-vector multiply: [m, n] · [n] -> [m]. `%inline` for the
||| same reason as `tadd` (hot path in recurrent forward passes).
export %inline
tmv : Tensor [m, n] d -> Tensor [n] d -> Tensor [m] d
tmv w x = MkTensor (prim__mv w.tensorPtr x.tensorPtr) Nothing

||| Fused batched linear: W[o,i] · X^T[b,i] + bias[o] -> [b, o].
export %inline
tlinear2d : Tensor [o, i] d -> Tensor [b, i] d -> Tensor [o] d -> Tensor [b, o] d
tlinear2d w x bias =
  MkTensor (prim__linear2d w.tensorPtr x.tensorPtr bias.tensorPtr) Nothing

-- Per-sample extraction + scalar arithmetic (used by batched RL loss
-- builders: pluck a row from a [b, o] result, then a scalar from the
-- row, then build (q - target)^2 etc.) ---------------------------------

||| Select row `k` from a [b, n] Tensor, returning the n-vector slice.
||| Wraps `prim__select` on dim 0; preserves the autograd graph.
export
trowSelect : {0 d : Device} -> {b, n : Nat} ->
             Tensor [b, n] d -> Int -> Tensor [n] d
trowSelect t k = MkTensor (prim__select t.tensorPtr 0 k) Nothing

||| Select element `i` from an n-vector, returning a scalar Tensor.
export
telemSelect : {0 d : Device} -> {n : Nat} ->
              Tensor [n] d -> Int -> Tensor [] d
telemSelect t i = MkTensor (prim__select t.tensorPtr 0 i) Nothing

||| Scalar Tensor from a Double. Takes the value as a runtime argument
||| so Idris/Chez does NOT memoise the FFI result as a module-level
||| constant — same defence as `freshZeroLossT`. Non-grad: the C
||| backend creates a non-persistent scalar that is freed by the next
||| `tape_reset` (i.e. fine to call inside an epoch's loss builder).
export
tconstScalar : {0 d : Device} -> Double -> Tensor [] d
tconstScalar v = MkTensor (prim__createScalar v 0) Nothing

||| Subtract two equally-shaped Tensors (autograd-tracked).
export %inline
tsub : Tensor dims d -> Tensor dims d -> Tensor dims d
tsub a b = MkTensor (prim__sub a.tensorPtr b.tensorPtr) Nothing

||| Elementwise multiply two equally-shaped Tensors (autograd-tracked).
export %inline
tmul : Tensor dims d -> Tensor dims d -> Tensor dims d
tmul a b = MkTensor (prim__mul a.tensorPtr b.tensorPtr) Nothing

||| Negate a Tensor (autograd-tracked).
export %inline
tneg : Tensor dims d -> Tensor dims d
tneg a = MkTensor (prim__neg a.tensorPtr) Nothing

||| Scale a Tensor by a Double (broadcasts the scalar; autograd-tracked).
||| Useful for mean-reduction (`tmulScalar loss (1.0 / cast n)`) and for
||| building per-sample loss expressions where one side of a product is
||| a runtime Double (e.g. DQN target value).
export %inline
tmulScalar : Tensor dims d -> Double -> Tensor dims d
tmulScalar v s = MkTensor (prim__mulScalar v.tensorPtr s) Nothing

||| Elementwise exponential (autograd-tracked).
export %inline
texp : Tensor dims d -> Tensor dims d
texp v = MkTensor (prim__exp v.tensorPtr) Nothing

||| Elementwise natural log (autograd-tracked).
export %inline
tlog : Tensor dims d -> Tensor dims d
tlog v = MkTensor (prim__log v.tensorPtr) Nothing

||| Create a registered learnable scalar parameter (e.g. SAC's
||| state-independent log_std). Mirrors V1's `param`. The optimizer
||| picks it up automatically by paramId scope.
export
tparamScalar : {0 d : Device} -> (paramId : String) -> (val : Double) -> Tensor [] d
tparamScalar pid val =
  let ptr = prim__createScalar val 1                  -- requires_grad=true
      reg = prim__paramRegister pid ptr
  in MkTensor reg (Just pid)

||| Concatenate two [b, m] / [b, n] TVars along axis 1, producing
||| [b, m + n]. Wraps `prim__concat2dAxis1`. Used by SAC's actor loss
||| to build a [B, ObsDim + ActDim] Q-input from obs + reparametrized
||| action while preserving the autograd path through the action.
export
tconcat2dAxis1 : {b, m, n : Nat} -> Tensor [b, m] d -> Tensor [b, n] d ->
                 Tensor [b, m + n] d
tconcat2dAxis1 a b = MkTensor (prim__concat2dAxis1 a.tensorPtr b.tensorPtr) Nothing

-- Activations (shape-preserving, pass-through autograd) ---------------
-- All `%inline` for hot-path performance — see `tadd` rationale.

export %inline
ttanh : Tensor dims d -> Tensor dims d
ttanh v = MkTensor (prim__tanh v.tensorPtr) Nothing

export %inline
tsigmoid : Tensor dims d -> Tensor dims d
tsigmoid v = MkTensor (prim__sigmoid v.tensorPtr) Nothing

export %inline
trelu : Tensor dims d -> Tensor dims d
trelu v = MkTensor (prim__clampMin v.tensorPtr 0.0) Nothing

export %inline
tgelu : Tensor dims d -> Tensor dims d
tgelu v = MkTensor (prim__gelu v.tensorPtr) Nothing

export %inline
tsilu : Tensor dims d -> Tensor dims d
tsilu v = MkTensor (prim__silu v.tensorPtr) Nothing

export %inline
tleakyRelu : Double -> Tensor dims d -> Tensor dims d
tleakyRelu slope v = MkTensor (prim__leakyRelu v.tensorPtr slope) Nothing

||| Softmax along axis 0 (1D vector).
export %inline
tsoftmax1d : {n : Nat} -> Tensor [n] d -> Tensor [n] d
tsoftmax1d v = MkTensor (prim__softmax v.tensorPtr 0) Nothing

||| Log-softmax along axis 0 (1D vector).
export %inline
tlogSoftmax1d : {n : Nat} -> Tensor [n] d -> Tensor [n] d
tlogSoftmax1d v = MkTensor (prim__logSoftmax v.tensorPtr 0) Nothing

||| Fused LSTM gate computation: combined gates [4 * n] + previous cell [n]
||| → (new hidden [n], new cell [n]). Wraps `prim__lstmGatesPair`.
|||
||| The gate-vector size is encoded statically as `TVec (4 * n) d`
||| (alias for `Tensor [4 * n] d`). Routing the `4 * n` through the
||| `TVec` alias avoids the type-checker hang that direct
||| `Tensor [4 * n] d` triggers.
export
tlstmGatesPair : {n : Nat} -> TVec (4 * n) d -> TVec n d ->
                 (TVec n d, TVec n d)
tlstmGatesPair {n} combined prevCell =
  let nI = cast {to=Int} n
      pair = prim__lstmGatesPair combined.tensorPtr prevCell.tensorPtr nI
  in (MkTensor (prim__pairFirst pair) Nothing, MkTensor (prim__pairSecond pair) Nothing)

||| Allocate a zero-initialised persistent state Tensor of size [n].
||| Use for LSTM/RNN/GRU initial hidden + cell state. Persistent =
||| survives tape reset.
export
tzeroState1d : {n : Nat} -> Tensor [n] d
tzeroState1d {n} =
  let nI = cast {to=Int} n
      buf = prim__allocDoubles nI
  in MkTensor (prim__createState1d nI buf) Nothing

||| Fused GRU cell: combined gates [3 * n] + previous hidden [n] →
||| new hidden [n]. Wraps `prim__gruCell`. The gate-vector size is
||| encoded statically as `TVec (3 * n) d` via the alias.
export
tgruCell : {n : Nat} -> TVec (3 * n) d -> TVec n d -> TVec n d
tgruCell {n} combined prevH =
  let nI = cast {to=Int} n
  in MkTensor (prim__gruCell combined.tensorPtr prevH.tensorPtr nI) Nothing

-- Scalar boundary --------------------------------------------------

||| Read the scalar value out of a `Tensor [] d`.
export
tensorItem : Tensor [] d -> Double
tensorItem v = prim__item v.tensorPtr

-- Loss (vector targets → scalar loss) ---------------------------------

||| MSE loss over a 1D prediction/target pair. Sum-reduced.
export
tmseLoss : {n : Nat} -> Tensor [n] d -> Tensor [n] d -> Tensor [] d
tmseLoss p t =
  let diff = prim__sub p.tensorPtr t.tensorPtr in
  let sqDiff = prim__mul diff diff in
  MkTensor (prim__sum sqDiff) Nothing

||| NLL loss against a one-hot target. Mirrors
||| `Example.Supervised.nllLossTensor` (divide by n to match the
||| reference's mean reduction).
export
tnllLoss : {n : Nat} -> Tensor [n] d -> Tensor [n] d -> Tensor [] d
tnllLoss {n} p t =
  let logP = prim__logSoftmax p.tensorPtr 0 in
  let prod = prim__mul logP t.tensorPtr in
  let neg = prim__neg (prim__sum prod) in
  MkTensor (prim__mulScalar neg (1.0 / cast n)) Nothing

||| Binary cross-entropy with logits, mean-reduced. Numerically stable
||| (wraps `prim__bceWithLogits`). For multi-element predictions/targets
||| use `tbceLoss : TVec n d -> TVec n d -> Tensor [] d`; the C op
||| internally averages.
export
tbceLoss : {n : Nat} -> TVec n d -> TVec n d -> Tensor [] d
tbceLoss p t =
  MkTensor (prim__bceWithLogits p.tensorPtr t.tensorPtr) Nothing

-- Optimizer shim ------------------------------------------------------

||| Fused native train step on a Tensor loss: zero_grad → backward →
||| clip → step. Reads `prim__item` BEFORE the step so the returned
||| scalar is not stale. Mirrors `nativeTrainStep`.
export
nativeTrainStep : {d : Device} -> NativeOptimizer -> Tensor [] d -> Double
nativeTrainStep opt loss =
  let clipMode : Int
      clipMode = case opt.clipMode of NoClip => 0; ValueClip _ => 1; NormClip _ => 2
      clipVal  : Double
      clipVal  = case opt.clipMode of NoClip => 0.0; ValueClip v => v; NormClip v => v
      lossVal  = prim__item loss.tensorPtr
  in prim__nativeTrainStep opt.handle clipMode clipVal loss.tensorPtr lossVal
