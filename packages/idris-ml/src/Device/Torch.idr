||| `TorchDev` — `UserDeviceCore` instance for the libtorch backend.
|||
||| Forwards to the torch-suffixed C symbols emitted under Phase 1's
||| `rename_torch.h` (e.g. `tensor_add_torch`). Only resolvable at
||| runtime if the build's BACKEND list includes `torch`.
module Device.Torch

import Device.Core


----------------------------------------------------------------------
-- Per-symbol bindings to the torch backend's suffixed C exports
----------------------------------------------------------------------

%foreign "C:tensor_create_scalar_torch,libidrisml"
prim__createScalarTorch : Double -> Int -> AnyPtr

%foreign "C:tensor_create_torch,libidrisml"
prim__createTorch : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr

%foreign "C:tensor_free_torch,libidrisml"
prim__freeTorch : AnyPtr -> ()

%foreign "C:tensor_item_torch,libidrisml"
prim__itemTorch : AnyPtr -> Double

%foreign "C:tensor_clone_torch,libidrisml"
prim__cloneTorch : AnyPtr -> AnyPtr

%foreign "C:tensor_add_torch,libidrisml"
prim__addTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_sub_torch,libidrisml"
prim__subTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_mul_torch,libidrisml"
prim__mulTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_div_torch,libidrisml"
prim__divTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_neg_torch,libidrisml"
prim__negTorch : AnyPtr -> AnyPtr

%foreign "C:tensor_abs_torch,libidrisml"
prim__absTorch : AnyPtr -> AnyPtr

%foreign "C:tensor_exp_torch,libidrisml"
prim__expTorch : AnyPtr -> AnyPtr

%foreign "C:tensor_log_torch,libidrisml"
prim__logTorch : AnyPtr -> AnyPtr

%foreign "C:tensor_sqrt_torch,libidrisml"
prim__sqrtTorch : AnyPtr -> AnyPtr

%foreign "C:tensor_pow_torch,libidrisml"
prim__powTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_sigmoid_torch,libidrisml"
prim__sigmoidTorch : AnyPtr -> AnyPtr

%foreign "C:tensor_tanh_torch,libidrisml"
prim__tanhTorch : AnyPtr -> AnyPtr

%foreign "C:tensor_add_scalar_torch,libidrisml"
prim__addScalarTorch : AnyPtr -> Double -> AnyPtr

%foreign "C:tensor_mul_scalar_torch,libidrisml"
prim__mulScalarTorch : AnyPtr -> Double -> AnyPtr

%foreign "C:tensor_clamp_min_torch,libidrisml"
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

%foreign "C:tensor_mv_torch,libidrisml"
prim__mvTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_matmul_torch,libidrisml"
prim__matmulTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_linear_torch,libidrisml"
prim__linearTorch : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_dot_torch,libidrisml"
prim__dotTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_outer_torch,libidrisml"
prim__outerTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_bmm_torch,libidrisml"
prim__bmmTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_linear_2d_torch,libidrisml"
prim__linear2dTorch : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_sum_torch,libidrisml"
prim__sumTorch : AnyPtr -> AnyPtr

%foreign "C:tensor_mean_torch,libidrisml"
prim__meanTorch : AnyPtr -> AnyPtr

%foreign "C:tensor_min_torch,libidrisml"
prim__tensorMinTorch : AnyPtr -> AnyPtr

%foreign "C:tensor_max_torch,libidrisml"
prim__tensorMaxTorch : AnyPtr -> AnyPtr

%foreign "C:tensor_sum_dim_torch,libidrisml"
prim__sumDimTorch : AnyPtr -> Int -> Int -> AnyPtr

%foreign "C:tensor_select_torch,libidrisml"
prim__selectTorch : AnyPtr -> Int -> Int -> AnyPtr

%foreign "C:tensor_unsqueeze_torch,libidrisml"
prim__unsqueezeTorch : AnyPtr -> Int -> AnyPtr

%foreign "C:tensor_squeeze_torch,libidrisml"
prim__squeezeTorch : AnyPtr -> Int -> AnyPtr

%foreign "C:tensor_stack_torch,libidrisml"
prim__stackTorch : AnyPtr -> Int -> Int -> AnyPtr

%foreign "C:tensor_view_1d_torch,libidrisml"
prim__view1dTorch : AnyPtr -> Int -> AnyPtr

%foreign "C:tensor_view_2d_torch,libidrisml"
prim__view2dTorch : AnyPtr -> Int -> Int -> AnyPtr

%foreign "C:tensor_reshape_2d_torch,libidrisml"
prim__reshape2dTorch : AnyPtr -> Int -> Int -> AnyPtr

%foreign "C:tensor_reshape_3d_torch,libidrisml"
prim__reshape3dTorch : AnyPtr -> Int -> Int -> Int -> AnyPtr

%foreign "C:tensor_reshape_4d_torch,libidrisml"
prim__reshape4dTorch : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr

%foreign "C:tensor_narrow_torch,libidrisml"
prim__narrowTorch : AnyPtr -> Int -> Int -> Int -> AnyPtr

%foreign "C:tensor_transpose_last2_torch,libidrisml"
prim__transposeLast2Torch : AnyPtr -> AnyPtr

%foreign "C:tensor_transpose_2d_torch,libidrisml"
prim__transpose2dTorch : AnyPtr -> AnyPtr

%foreign "C:tensor_cat_torch,libidrisml"
prim__catTorch : AnyPtr -> Int -> Int -> AnyPtr

%foreign "C:tensor_cat2_torch,libidrisml"
prim__cat2Torch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_concat_2d_axis1_torch,libidrisml"
prim__concat2dAxis1Torch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_gather_torch,libidrisml"
prim__gatherTorch : AnyPtr -> AnyPtr -> Int -> AnyPtr

%foreign "C:tensor_scatter_add_torch,libidrisml"
prim__scatterAddTorch : AnyPtr -> AnyPtr -> Int -> AnyPtr

%foreign "C:tensor_argsort_torch,libidrisml"
prim__argsortTorch : AnyPtr -> Int -> Int -> AnyPtr

%foreign "C:tensor_cumprod_torch,libidrisml"
prim__cumprodTorch : AnyPtr -> Int -> AnyPtr


public export
UserDeviceLinear TorchDev where
  primMv             = prim__mvTorch
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
  primReshape2d      = prim__reshape2dTorch
  primReshape3d      = prim__reshape3dTorch
  primReshape4d      = prim__reshape4dTorch
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
