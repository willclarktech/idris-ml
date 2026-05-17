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

%foreign "scheme:(lambda (a0 a1) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_create_scalar_mlx\" (double int) void*) a0 a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__createScalarMlx : Double -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_create_mlx\" (void* void* int int) void*) a0 a1 a2 a3))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__createMlx : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0)  ((foreign-procedure \"tensor_free_mlx\" (void*) void) (vector-ref a0 1)))"
prim__freeMlx : AnyPtr -> ()

%foreign "scheme:(lambda (a0)  ((foreign-procedure \"tensor_item_mlx\" (void*) double) (vector-ref a0 1)))"
prim__itemMlx : AnyPtr -> Double

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_clone_mlx\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__cloneMlx : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_add_mlx\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__addMlx : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_sub_mlx\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__subMlx : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_mul_mlx\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__mulMlx : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_div_mlx\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__divMlx : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_neg_mlx\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__negMlx : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_abs_mlx\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__absMlx : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_exp_mlx\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__expMlx : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_log_mlx\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__logMlx : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_sqrt_mlx\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__sqrtMlx : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_pow_mlx\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__powMlx : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_sigmoid_mlx\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__sigmoidMlx : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_tanh_mlx\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__tanhMlx : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_add_scalar_mlx\" (void* double) void*) (vector-ref a0 1) a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__addScalarMlx : AnyPtr -> Double -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_mul_scalar_mlx\" (void* double) void*) (vector-ref a0 1) a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__mulScalarMlx : AnyPtr -> Double -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_clamp_min_mlx\" (void* double) void*) (vector-ref a0 1) a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
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

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_mv_mlx\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__mvMlx : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_matmul_mlx\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__matmulMlx : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_linear_mlx\" (void* void* void*) void*) (vector-ref a0 1) (vector-ref a1 1) (vector-ref a2 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__linearMlx : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_dot_mlx\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__dotMlx : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_outer_mlx\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__outerMlx : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_bmm_mlx\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__bmmMlx : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_linear_2d_mlx\" (void* void* void*) void*) (vector-ref a0 1) (vector-ref a1 1) (vector-ref a2 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__linear2dMlx : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_sum_mlx\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__sumMlx : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_mean_mlx\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__meanMlx : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_min_mlx\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__tensorMinMlx : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_max_mlx\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__tensorMaxMlx : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_sum_dim_mlx\" (void* int int) void*) (vector-ref a0 1) a1 a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__sumDimMlx : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_select_mlx\" (void* int int) void*) (vector-ref a0 1) a1 a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__selectMlx : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_unsqueeze_mlx\" (void* int) void*) (vector-ref a0 1) a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__unsqueezeMlx : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_squeeze_mlx\" (void* int) void*) (vector-ref a0 1) a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__squeezeMlx : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_stack_mlx\" (void* int int) void*) a0 a1 a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__stackMlx : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_view_1d_mlx\" (void* int) void*) (vector-ref a0 1) a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__view1dMlx : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_view_2d_mlx\" (void* int int) void*) (vector-ref a0 1) a1 a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__view2dMlx : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_reshape_2d_mlx\" (void* int int) void*) (vector-ref a0 1) a1 a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__reshape2dMlx : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_reshape_3d_mlx\" (void* int int int) void*) (vector-ref a0 1) a1 a2 a3))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__reshape3dMlx : AnyPtr -> Int -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (let ((raw_r ((foreign-procedure \"tensor_reshape_4d_mlx\" (void* int int int int) void*) (vector-ref a0 1) a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__reshape4dMlx : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_narrow_mlx\" (void* int int int) void*) (vector-ref a0 1) a1 a2 a3))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__narrowMlx : AnyPtr -> Int -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_transpose_last2_mlx\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__transposeLast2Mlx : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_transpose_2d_mlx\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__transpose2dMlx : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_cat_mlx\" (void* int int) void*) a0 a1 a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__catMlx : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_cat2_mlx\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__cat2Mlx : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_concat_2d_axis1_mlx\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__concat2dAxis1Mlx : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_gather_mlx\" (void* void* int) void*) (vector-ref a0 1) (vector-ref a1 1) a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__gatherMlx : AnyPtr -> AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_scatter_add_mlx\" (void* void* int) void*) (vector-ref a0 1) (vector-ref a1 1) a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__scatterAddMlx : AnyPtr -> AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_argsort_mlx\" (void* int int) void*) (vector-ref a0 1) a1 a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__argsortMlx : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_cumprod_mlx\" (void* int) void*) (vector-ref a0 1) a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
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

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_gelu_mlx\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__geluMlx : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_leaky_relu_mlx\" (void* double) void*) (vector-ref a0 1) a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__leakyReluMlx : AnyPtr -> Double -> AnyPtr
%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_silu_mlx\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__siluMlx : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_softplus_mlx\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__softplusMlx : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_softmax_mlx\" (void* int) void*) (vector-ref a0 1) a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__softmaxMlx : AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_log_softmax_mlx\" (void* int) void*) (vector-ref a0 1) a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__logSoftmaxMlx : AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_softmax_2d_mlx\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__softmax2dMlx : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_log_softmax_2d_mlx\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__logSoftmax2dMlx : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_softmax_3d_mlx\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__softmax3dMlx : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_masked_fill_mlx\" (void* void* double) void*) (vector-ref a0 1) (vector-ref a1 1) a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__maskedFillMlx : AnyPtr -> AnyPtr -> Double -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_expand_mask_mlx\" (void* int) void*) (vector-ref a0 1) a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__expandMaskMlx : AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_causal_mask_mlx\" (int) void*) a0))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__causalMaskMlx : Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_layer_norm_2d_mlx\" (void* void* void* double) void*) (vector-ref a0 1) (vector-ref a1 1) (vector-ref a2 1) a3))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__layerNorm2dMlx : AnyPtr -> AnyPtr -> AnyPtr -> Double -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6 a7 a8 a9)  (let ((raw_r ((foreign-procedure \"tensor_batch_norm_mlx\" (void* void* void* void* void* int int int double double) void*) (vector-ref a0 1) (vector-ref a1 1) (vector-ref a2 1) (vector-ref a3 1) (vector-ref a4 1) a5 a6 a7 a8 a9))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__batchNormMlx : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> Double -> Double -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_dropout_mlx\" (void* double int int) void*) (vector-ref a0 1) a1 a2 a3))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__dropoutMlx : AnyPtr -> Double -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_embedding_mlx\" (void* void* int int) void*) (vector-ref a0 1) (vector-ref a1 1) a2 a3))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__embeddingMlx : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_cosine_similarity_mlx\" (void* void* int) void*) (vector-ref a0 1) (vector-ref a1 1) a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__cosineSimilarityMlx : AnyPtr -> AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (let ((raw_r ((foreign-procedure \"tensor_cross_attention_mlx\" (void* void* void* void* double) void*) (vector-ref a0 1) (vector-ref a1 1) (vector-ref a2 1) (vector-ref a3 1) a4))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__crossAttentionMlx : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> Double -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_bce_with_logits_mlx\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__bceWithLogitsMlx : AnyPtr -> AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_gru_cell_mlx\" (void* void* void* int) void*) (vector-ref a0 1) (vector-ref a1 1) (vector-ref a2 1) a3))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__gruCellMlx : AnyPtr -> AnyPtr -> AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  ((foreign-procedure \"tensor_lstm_gates_pair_mlx\" (void* void* int) void*) (vector-ref a0 1) (vector-ref a1 1) a2))"
prim__lstmGatesPairMlx : AnyPtr -> AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_pair_first_mlx\" (void*) void*) a0))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__pairFirstMlx : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_pair_second_mlx\" (void*) void*) a0))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
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

%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (let ((raw_r ((foreign-procedure \"tensor_conv1d_mlx\" (void* void* void* int int) void*) (vector-ref a0 1) (vector-ref a1 1) (vector-ref a2 1) a3 a4))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__conv1dMlx : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_conv1d_circular_mlx\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__conv1dCircularMlx : AnyPtr -> AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_avg_pool1d_mlx\" (void* int int) void*) (vector-ref a0 1) a1 a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__avgPool1dMlx : AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_max_pool1d_mlx\" (void* int int) void*) (vector-ref a0 1) a1 a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__maxPool1dMlx : AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6)  (let ((raw_r ((foreign-procedure \"tensor_conv2d_mlx\" (void* void* void* int int int int) void*) (vector-ref a0 1) (vector-ref a1 1) (vector-ref a2 1) a3 a4 a5 a6))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__conv2dMlx : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6)  (let ((raw_r ((foreign-procedure \"tensor_conv2d_batched_mlx\" (void* void* void* int int int int) void*) (vector-ref a0 1) (vector-ref a1 1) (vector-ref a2 1) a3 a4 a5 a6))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__conv2dBatchedMlx : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (let ((raw_r ((foreign-procedure \"tensor_avg_pool2d_mlx\" (void* int int int int) void*) (vector-ref a0 1) a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__avgPool2dMlx : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (let ((raw_r ((foreign-procedure \"tensor_max_pool2d_mlx\" (void* int int int int) void*) (vector-ref a0 1) a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__maxPool2dMlx : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (let ((raw_r ((foreign-procedure \"tensor_max_pool2d_batched_mlx\" (void* int int int int) void*) (vector-ref a0 1) a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
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

%foreign "scheme:(lambda (a0)  ((foreign-procedure \"tensor_requires_grad_mlx\" (void*) int) (vector-ref a0 1)))"
prim__requiresGradMlx : AnyPtr -> Int
%foreign "scheme:(lambda (a0 a1)  ((foreign-procedure \"tensor_set_requires_grad_mlx\" (void* int) void) (vector-ref a0 1) a1))"
prim__setRequiresGradMlx : AnyPtr -> Int -> PrimIO ()
%foreign "C:tensor_no_grad_begin_mlx,libidrisml"
prim__noGradBeginMlx : PrimIO ()
%foreign "C:tensor_no_grad_end_mlx,libidrisml"
prim__noGradEndMlx : PrimIO ()
%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_detach_mlx\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__detachMlx : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_with_grad_mlx\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__withGradMlx : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0)  ((foreign-procedure \"tensor_dim_mlx\" (void*) int) (vector-ref a0 1)))"
prim__tensorDimMlx : AnyPtr -> Int
%foreign "scheme:(lambda (a0 a1)  ((foreign-procedure \"tensor_size_mlx\" (void* int) int) (vector-ref a0 1) a1))"
prim__tensorSizeAtMlx : AnyPtr -> Int -> Int
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"param_register_return_mlx\" (string void*) void*) a0 (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__paramRegisterMlx : String -> AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_create_param_1d_mlx\" (int void*) void*) a0 a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__createParam1dMlx : Int -> AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_create_param_2d_mlx\" (int int void*) void*) a0 a1 a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__createParam2dMlx : Int -> Int -> AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_create_param_3d_mlx\" (int int int void*) void*) a0 a1 a2 a3))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__createParam3dMlx : Int -> Int -> Int -> AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_create_state_1d_mlx\" (int void*) void*) a0 a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__createState1dMlx : Int -> AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_create_state_2d_mlx\" (int int void*) void*) a0 a1 a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__createState2dMlx : Int -> Int -> AnyPtr -> AnyPtr
%foreign "C:tensor_alloc_doubles_mlx,libidrisml"
prim__allocDoublesMlx : Int -> AnyPtr
%foreign "C:tensor_read_double_mlx,libidrisml"
prim__readDoubleMlx : AnyPtr -> Int -> Double


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
  primCreateParam1d        = prim__createParam1dMlx
  primCreateParam2d        = prim__createParam2dMlx
  primCreateParam3d        = prim__createParam3dMlx
  primCreateState1d        = prim__createState1dMlx
  primCreateState2d        = prim__createState2dMlx
  primAllocDoubles         = prim__allocDoublesMlx
  primReadDouble           = prim__readDoubleMlx
