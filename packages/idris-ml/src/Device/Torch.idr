||| `TorchDev` — `UserDeviceCore` instance for the libtorch backend.
|||
||| Forwards to the torch-suffixed C symbols emitted under Phase 1's
||| `rename_torch.h` (e.g. `tensor_add_torch`). Only resolvable at
||| runtime if the build's BACKEND list includes `torch`.
module Device.Torch

import Device.Core
import DType.Core


----------------------------------------------------------------------
-- Per-symbol bindings to the torch backend's suffixed C exports
----------------------------------------------------------------------

%foreign "scheme:(lambda (a0 a1) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_create_scalar_torch\" (double int) void*) a0 a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__createScalarTorch : Double -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_create_torch\" (void* void* int int) void*) a0 a1 a2 a3))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__createTorch : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0)  ((foreign-procedure \"tensor_free_torch\" (void*) void) (vector-ref a0 1)))"
prim__freeTorch : AnyPtr -> ()

%foreign "scheme:(lambda (a0)  ((foreign-procedure \"tensor_item_torch\" (void*) double) (vector-ref a0 1)))"
prim__itemTorch : AnyPtr -> Double

%foreign "scheme:(lambda (a0 a1)  ((foreign-procedure \"tensor_item_1d_torch\" (void* int) double) (vector-ref a0 1) a1))"
prim__item1dTorch : AnyPtr -> Int -> Double

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_clone_torch\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__cloneTorch : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_add_torch\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__addTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_sub_torch\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__subTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_mul_torch\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__mulTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_div_torch\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__divTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_neg_torch\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__negTorch : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_abs_torch\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__absTorch : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_exp_torch\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__expTorch : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_log_torch\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__logTorch : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_sqrt_torch\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__sqrtTorch : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_pow_torch\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__powTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_sigmoid_torch\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__sigmoidTorch : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_tanh_torch\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__tanhTorch : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_add_scalar_torch\" (void* double) void*) (vector-ref a0 1) a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__addScalarTorch : AnyPtr -> Double -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_mul_scalar_torch\" (void* double) void*) (vector-ref a0 1) a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__mulScalarTorch : AnyPtr -> Double -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_clamp_min_torch\" (void* double) void*) (vector-ref a0 1) a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__clampMinTorch : AnyPtr -> Double -> AnyPtr


----------------------------------------------------------------------
-- TorchDev type + UserDeviceCore instance
----------------------------------------------------------------------

public export
data TorchDev : Type where MkTorchDev : TorchDev

public export
UserDeviceCore TorchDev where
  deviceName       = "torch"
  primCreateScalar = prim__createScalarTorch
  primCreate       = prim__createTorch
  primFree         = prim__freeTorch
  primItem         = prim__itemTorch
  primItem1d       = prim__item1dTorch
  primClone        = prim__cloneTorch
  primAdd          = prim__addTorch
  primSub          = prim__subTorch
  primMul          = prim__mulTorch
  primDiv          = prim__divTorch
  primNeg          = prim__negTorch
  primAbs          = prim__absTorch
  primExp          = prim__expTorch
  primLog          = prim__logTorch
  primSqrt         = prim__sqrtTorch
  primPow          = prim__powTorch
  primSigmoid      = prim__sigmoidTorch
  primTanh         = prim__tanhTorch
  primAddScalar    = prim__addScalarTorch
  primMulScalar    = prim__mulScalarTorch
  primClampMin     = prim__clampMinTorch
----------------------------------------------------------------------
-- Linear-slice FFI bindings (torch-suffixed)
----------------------------------------------------------------------

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_mv_torch\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__mvTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_mm_torch\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__mmTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_matmul_torch\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__matmulTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_linear_torch\" (void* void* void*) void*) (vector-ref a0 1) (vector-ref a1 1) (vector-ref a2 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__linearTorch : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_dot_torch\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__dotTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_outer_torch\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__outerTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_bmm_torch\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__bmmTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_linear_2d_torch\" (void* void* void*) void*) (vector-ref a0 1) (vector-ref a1 1) (vector-ref a2 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__linear2dTorch : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_sum_torch\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__sumTorch : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_mean_torch\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__meanTorch : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_min_torch\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__tensorMinTorch : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_max_torch\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__tensorMaxTorch : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_sum_dim_torch\" (void* int int) void*) (vector-ref a0 1) a1 a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__sumDimTorch : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_select_torch\" (void* int int) void*) (vector-ref a0 1) a1 a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__selectTorch : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_unsqueeze_torch\" (void* int) void*) (vector-ref a0 1) a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__unsqueezeTorch : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_squeeze_torch\" (void* int) void*) (vector-ref a0 1) a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__squeezeTorch : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_stack_torch\" (void* int int) void*) a0 a1 a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__stackTorch : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_view_1d_torch\" (void* int) void*) (vector-ref a0 1) a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__view1dTorch : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_view_2d_torch\" (void* int int) void*) (vector-ref a0 1) a1 a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__view2dTorch : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_reshape_1d_torch\" (void* int) void*) (vector-ref a0 1) a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__reshape1dTorch : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_reshape_2d_torch\" (void* int int) void*) (vector-ref a0 1) a1 a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__reshape2dTorch : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_reshape_3d_torch\" (void* int int int) void*) (vector-ref a0 1) a1 a2 a3))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__reshape3dTorch : AnyPtr -> Int -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (let ((raw_r ((foreign-procedure \"tensor_reshape_4d_torch\" (void* int int int int) void*) (vector-ref a0 1) a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__reshape4dTorch : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_tile_2d_torch\" (void* int int) void*) (vector-ref a0 1) a1 a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__tile2dTorch : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_narrow_torch\" (void* int int int) void*) (vector-ref a0 1) a1 a2 a3))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__narrowTorch : AnyPtr -> Int -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_transpose_last2_torch\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__transposeLast2Torch : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_transpose_2d_torch\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__transpose2dTorch : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_cat_torch\" (void* int int) void*) a0 a1 a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__catTorch : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_cat2_torch\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__cat2Torch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_concat_2d_axis1_torch\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__concat2dAxis1Torch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_gather_torch\" (void* void* int) void*) (vector-ref a0 1) (vector-ref a1 1) a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__gatherTorch : AnyPtr -> AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_scatter_add_torch\" (void* void* int) void*) (vector-ref a0 1) (vector-ref a1 1) a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__scatterAddTorch : AnyPtr -> AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_argsort_torch\" (void* int int) void*) (vector-ref a0 1) a1 a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__argsortTorch : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_cumprod_torch\" (void* int) void*) (vector-ref a0 1) a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__cumprodTorch : AnyPtr -> Int -> AnyPtr


public export
UserDeviceLinear TorchDev where
  primMv             = prim__mvTorch
  primMm             = prim__mmTorch
  primMatmul         = prim__matmulTorch
  primLinear         = prim__linearTorch
  primDot            = prim__dotTorch
  primOuter          = prim__outerTorch
  primBmm            = prim__bmmTorch
  primLinear2d       = prim__linear2dTorch
  primSum            = prim__sumTorch
  primMean           = prim__meanTorch
  primTensorMin      = prim__tensorMinTorch
  primTensorMax      = prim__tensorMaxTorch
  primSumDim         = prim__sumDimTorch
  primSelect         = prim__selectTorch
  primUnsqueeze      = prim__unsqueezeTorch
  primSqueeze        = prim__squeezeTorch
  primStack          = prim__stackTorch
  primView1d         = prim__view1dTorch
  primView2d         = prim__view2dTorch
  primReshape1d      = prim__reshape1dTorch
  primReshape2d      = prim__reshape2dTorch
  primReshape3d      = prim__reshape3dTorch
  primReshape4d      = prim__reshape4dTorch
  primTile2d         = prim__tile2dTorch
  primNarrow         = prim__narrowTorch
  primTransposeLast2 = prim__transposeLast2Torch
  primTranspose2d    = prim__transpose2dTorch
  primCat            = prim__catTorch
  primCat2           = prim__cat2Torch
  primConcat2dAxis1  = prim__concat2dAxis1Torch
  primGather         = prim__gatherTorch
  primScatterAdd     = prim__scatterAddTorch
  primArgsort        = prim__argsortTorch
  primCumprod        = prim__cumprodTorch


----------------------------------------------------------------------
-- NN-slice FFI bindings (torch-suffixed)
----------------------------------------------------------------------

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_gelu_torch\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__geluTorch : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_leaky_relu_torch\" (void* double) void*) (vector-ref a0 1) a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__leakyReluTorch : AnyPtr -> Double -> AnyPtr
%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_silu_torch\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__siluTorch : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_softplus_torch\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__softplusTorch : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_softmax_torch\" (void* int) void*) (vector-ref a0 1) a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__softmaxTorch : AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_log_softmax_torch\" (void* int) void*) (vector-ref a0 1) a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__logSoftmaxTorch : AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_softmax_2d_torch\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__softmax2dTorch : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_log_softmax_2d_torch\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__logSoftmax2dTorch : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_softmax_3d_torch\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__softmax3dTorch : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_masked_fill_torch\" (void* void* double) void*) (vector-ref a0 1) (vector-ref a1 1) a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__maskedFillTorch : AnyPtr -> AnyPtr -> Double -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_expand_mask_torch\" (void* int) void*) (vector-ref a0 1) a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__expandMaskTorch : AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_causal_mask_torch\" (int) void*) a0))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__causalMaskTorch : Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_layer_norm_2d_torch\" (void* void* void* double) void*) (vector-ref a0 1) (vector-ref a1 1) (vector-ref a2 1) a3))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__layerNorm2dTorch : AnyPtr -> AnyPtr -> AnyPtr -> Double -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6 a7 a8 a9)  (let ((raw_r ((foreign-procedure \"tensor_batch_norm_torch\" (void* void* void* void* void* int int int double double) void*) (vector-ref a0 1) (vector-ref a1 1) (vector-ref a2 1) (vector-ref a3 1) (vector-ref a4 1) a5 a6 a7 a8 a9))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__batchNormTorch : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> Double -> Double -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_dropout_torch\" (void* double int int) void*) (vector-ref a0 1) a1 a2 a3))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__dropoutTorch : AnyPtr -> Double -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_embedding_torch\" (void* void* int int) void*) (vector-ref a0 1) (vector-ref a1 1) a2 a3))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__embeddingTorch : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_cosine_similarity_torch\" (void* void* int) void*) (vector-ref a0 1) (vector-ref a1 1) a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__cosineSimilarityTorch : AnyPtr -> AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (let ((raw_r ((foreign-procedure \"tensor_cross_attention_torch\" (void* void* void* void* double) void*) (vector-ref a0 1) (vector-ref a1 1) (vector-ref a2 1) (vector-ref a3 1) a4))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__crossAttentionTorch : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> Double -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_bce_with_logits_torch\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__bceWithLogitsTorch : AnyPtr -> AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_gru_cell_torch\" (void* void* void* int) void*) (vector-ref a0 1) (vector-ref a1 1) (vector-ref a2 1) a3))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__gruCellTorch : AnyPtr -> AnyPtr -> AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  ((foreign-procedure \"tensor_lstm_gates_pair_torch\" (void* void* int) void*) (vector-ref a0 1) (vector-ref a1 1) a2))"
prim__lstmGatesPairTorch : AnyPtr -> AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_pair_first_torch\" (void*) void*) a0))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__pairFirstTorch : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_pair_second_torch\" (void*) void*) a0))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__pairSecondTorch : AnyPtr -> AnyPtr


public export
UserDeviceNN TorchDev where
  primGelu             = prim__geluTorch
  primLeakyRelu        = prim__leakyReluTorch
  primSilu             = prim__siluTorch
  primSoftplus         = prim__softplusTorch
  primSoftmax          = prim__softmaxTorch
  primLogSoftmax       = prim__logSoftmaxTorch
  primSoftmax2d        = prim__softmax2dTorch
  primLogSoftmax2d     = prim__logSoftmax2dTorch
  primSoftmax3d        = prim__softmax3dTorch
  primMaskedFill       = prim__maskedFillTorch
  primExpandMask       = prim__expandMaskTorch
  primCausalMask       = prim__causalMaskTorch
  primLayerNorm2d      = prim__layerNorm2dTorch
  primBatchNorm        = prim__batchNormTorch
  primDropout          = prim__dropoutTorch
  primEmbedding        = prim__embeddingTorch
  primCosineSimilarity = prim__cosineSimilarityTorch
  primCrossAttention   = prim__crossAttentionTorch
  primBceWithLogits    = prim__bceWithLogitsTorch
  primGruCell          = prim__gruCellTorch
  primLstmGatesPair    = prim__lstmGatesPairTorch
  primPairFirst        = prim__pairFirstTorch
  primPairSecond       = prim__pairSecondTorch


----------------------------------------------------------------------
-- Conv-slice FFI bindings (torch-suffixed)
----------------------------------------------------------------------

%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (let ((raw_r ((foreign-procedure \"tensor_conv1d_torch\" (void* void* void* int int) void*) (vector-ref a0 1) (vector-ref a1 1) (vector-ref a2 1) a3 a4))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__conv1dTorch : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_conv1d_circular_torch\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__conv1dCircularTorch : AnyPtr -> AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_avg_pool1d_torch\" (void* int int) void*) (vector-ref a0 1) a1 a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__avgPool1dTorch : AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_max_pool1d_torch\" (void* int int) void*) (vector-ref a0 1) a1 a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__maxPool1dTorch : AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6)  (let ((raw_r ((foreign-procedure \"tensor_conv2d_torch\" (void* void* void* int int int int) void*) (vector-ref a0 1) (vector-ref a1 1) (vector-ref a2 1) a3 a4 a5 a6))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__conv2dTorch : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6)  (let ((raw_r ((foreign-procedure \"tensor_conv2d_batched_torch\" (void* void* void* int int int int) void*) (vector-ref a0 1) (vector-ref a1 1) (vector-ref a2 1) a3 a4 a5 a6))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__conv2dBatchedTorch : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (let ((raw_r ((foreign-procedure \"tensor_avg_pool2d_torch\" (void* int int int int) void*) (vector-ref a0 1) a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__avgPool2dTorch : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (let ((raw_r ((foreign-procedure \"tensor_max_pool2d_torch\" (void* int int int int) void*) (vector-ref a0 1) a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__maxPool2dTorch : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (let ((raw_r ((foreign-procedure \"tensor_max_pool2d_batched_torch\" (void* int int int int) void*) (vector-ref a0 1) a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__maxPool2dBatchedTorch : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr


public export
UserDeviceConv TorchDev where
  primConv1d           = prim__conv1dTorch
  primConv1dCircular   = prim__conv1dCircularTorch
  primAvgPool1d        = prim__avgPool1dTorch
  primMaxPool1d        = prim__maxPool1dTorch
  primConv2d           = prim__conv2dTorch
  primConv2dBatched    = prim__conv2dBatchedTorch
  primAvgPool2d        = prim__avgPool2dTorch
  primMaxPool2d        = prim__maxPool2dTorch
  primMaxPool2dBatched = prim__maxPool2dBatchedTorch


----------------------------------------------------------------------
-- Tape-slice FFI bindings (torch-suffixed)
----------------------------------------------------------------------

%foreign "scheme:(lambda (a0)  ((foreign-procedure \"tensor_requires_grad_torch\" (void*) int) (vector-ref a0 1)))"
prim__requiresGradTorch : AnyPtr -> Int
%foreign "scheme:(lambda (a0 a1)  ((foreign-procedure \"tensor_set_requires_grad_torch\" (void* int) void) (vector-ref a0 1) a1))"
prim__setRequiresGradTorch : AnyPtr -> Int -> PrimIO ()
%foreign "C:tensor_no_grad_begin_torch,libidrisml"
prim__noGradBeginTorch : PrimIO ()
%foreign "C:tensor_no_grad_end_torch,libidrisml"
prim__noGradEndTorch : PrimIO ()
%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_detach_torch\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__detachTorch : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_with_grad_torch\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__withGradTorch : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0)  ((foreign-procedure \"tensor_dim_torch\" (void*) int) (vector-ref a0 1)))"
prim__tensorDimTorch : AnyPtr -> Int
%foreign "scheme:(lambda (a0 a1)  ((foreign-procedure \"tensor_size_torch\" (void* int) int) (vector-ref a0 1) a1))"
prim__tensorSizeAtTorch : AnyPtr -> Int -> Int
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"param_register_return_torch\" (string void*) void*) a0 (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__paramRegisterTorch : String -> AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_create_param_1d_torch\" (int void*) void*) a0 a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__createParam1dTorch : Int -> AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_create_param_2d_torch\" (int int void*) void*) a0 a1 a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__createParam2dTorch : Int -> Int -> AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_create_param_3d_torch\" (int int int void*) void*) a0 a1 a2 a3))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__createParam3dTorch : Int -> Int -> Int -> AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_create_state_1d_torch\" (int void*) void*) a0 a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__createState1dTorch : Int -> AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_create_state_2d_torch\" (int int void*) void*) a0 a1 a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__createState2dTorch : Int -> Int -> AnyPtr -> AnyPtr
%foreign "C:tensor_alloc_doubles_torch,libidrisml"
prim__allocDoublesTorch : Int -> AnyPtr
%foreign "C:tensor_read_double_torch,libidrisml"
prim__readDoubleTorch : AnyPtr -> Int -> Double


public export
UserDeviceTape TorchDev where
  primRequiresGrad         = prim__requiresGradTorch
  primSetRequiresGrad      = prim__setRequiresGradTorch
  primNoGradBegin          = prim__noGradBeginTorch
  primNoGradEnd            = prim__noGradEndTorch
  primDetach               = prim__detachTorch
  primWithGrad             = prim__withGradTorch
  primTensorDim            = prim__tensorDimTorch
  primTensorSizeAt         = prim__tensorSizeAtTorch
  primParamRegister        = prim__paramRegisterTorch
  primCreateParam1d        = prim__createParam1dTorch
  primCreateParam2d        = prim__createParam2dTorch
  primCreateParam3d        = prim__createParam3dTorch
  primCreateState1d        = prim__createState1dTorch
  primCreateState2d        = prim__createState2dTorch
  primAllocDoubles         = prim__allocDoublesTorch
  primReadDouble           = prim__readDoubleTorch


----------------------------------------------------------------------
-- Compatible (TorchDev, dt). The torch backend hardcodes
-- `torch::kFloat64` today; threading a runtime dtype through
-- `tensor_create*` is the F32 unlock (deferred).
----------------------------------------------------------------------

public export
Compatible TorchDev F64 where
