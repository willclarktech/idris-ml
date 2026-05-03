||| `TapeDev` — `UserDeviceCore` instance for the tape backend.
|||
||| Forwards to the tape-suffixed C symbols emitted under Phase 1's
||| `rename_tape.h` (e.g. `tensor_add_tape`). Only resolvable at
||| runtime if the build's BACKEND list includes `tape`.
module Device.Tape

import Device.Core
import DType.Core


----------------------------------------------------------------------
-- Per-symbol bindings to the tape backend's suffixed C exports
----------------------------------------------------------------------

%foreign "scheme:(lambda (a0 a1) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_create_scalar_tape\" (double int) void*) a0 a1))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__createScalarTape : Double -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_create_tape\" (void* void* int int) void*) a0 a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__createTape : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0)  ((foreign-procedure \"tensor_free_tape\" (void*) void) (vector-ref a0 2)))"
prim__freeTape : AnyPtr -> ()

%foreign "scheme:(lambda (a0)  ((foreign-procedure \"tensor_item_tape\" (void*) double) (vector-ref a0 2)))"
prim__itemTape : AnyPtr -> Double

%foreign "scheme:(lambda (a0 a1)  ((foreign-procedure \"tensor_item_1d_tape\" (void* int) double) (vector-ref a0 2) a1))"
prim__item1dTape : AnyPtr -> Int -> Double

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_clone_tape\" (void*) void*) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__cloneTape : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_add_tape\" (void* void*) void*) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__addTape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_sub_tape\" (void* void*) void*) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__subTape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_mul_tape\" (void* void*) void*) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__mulTape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_div_tape\" (void* void*) void*) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__divTape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_neg_tape\" (void*) void*) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__negTape : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_abs_tape\" (void*) void*) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__absTape : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_exp_tape\" (void*) void*) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__expTape : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_log_tape\" (void*) void*) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__logTape : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_sqrt_tape\" (void*) void*) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__sqrtTape : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_pow_tape\" (void* void*) void*) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__powTape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_sigmoid_tape\" (void*) void*) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__sigmoidTape : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_tanh_tape\" (void*) void*) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__tanhTape : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_add_scalar_tape\" (void* double) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__addScalarTape : AnyPtr -> Double -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_mul_scalar_tape\" (void* double) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__mulScalarTape : AnyPtr -> Double -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_clamp_min_tape\" (void* double) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__clampMinTape : AnyPtr -> Double -> AnyPtr


----------------------------------------------------------------------
-- TapeDev type + UserDeviceCore instance
----------------------------------------------------------------------

||| The tape backend's `UserDeviceCore` instance head. An empty type
||| — it has no values; `Tensor [..] TapeDev` is just a typed tag for
||| "this tensor lives on the tape backend".
public export
data TapeDev : Type where MkTapeDev : TapeDev

%foreign "scheme:(lambda (a0 a1 a2)  ((foreign-procedure \"tensor_item_2d_tape\" (void* int int) double) (vector-ref a0 2) a1 a2))"
prim__item2dTape : AnyPtr -> Int -> Int -> Double
%foreign "scheme:(lambda (a0 a1 a2) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_create_1d_tape\" (int void* int) void*) a0 a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__create1dTape : Int -> AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"mnist_get_image_tape\" (void* int) void*) a0 a1))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__mnistGetImageTape : AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_one_hot_tape\" (void* int int int) void*) a0 a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__oneHotTape : AnyPtr -> Int -> Int -> Int -> AnyPtr

public export
UserDeviceCore TapeDev where
  deviceName       = "tape"
  deviceStreamTag  = 0
  primCreateScalar = prim__createScalarTape
  primCreate       = prim__createTape
  primFree         = prim__freeTape
  primItem         = prim__itemTape
  primItem1d       = prim__item1dTape
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

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_mv_tape\" (void* void*) void*) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__mvTape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_mm_tape\" (void* void*) void*) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__mmTape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_matmul_tape\" (void* void*) void*) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__matmulTape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_linear_tape\" (void* void* void*) void*) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__linearTape : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_dot_tape\" (void* void*) void*) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__dotTape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_outer_tape\" (void* void*) void*) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__outerTape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_bmm_tape\" (void* void*) void*) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__bmmTape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_linear_2d_tape\" (void* void* void*) void*) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__linear2dTape : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_sum_tape\" (void*) void*) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__sumTape : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_mean_tape\" (void*) void*) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__meanTape : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_min_tape\" (void*) void*) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__tensorMinTape : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_max_tape\" (void*) void*) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__tensorMaxTape : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_sum_dim_tape\" (void* int int) void*) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__sumDimTape : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_select_tape\" (void* int int) void*) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__selectTape : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_unsqueeze_tape\" (void* int) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__unsqueezeTape : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_squeeze_tape\" (void* int) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__squeezeTape : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_stack_tape\" (void* int int) void*) a0 a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__stackTape : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_view_1d_tape\" (void* int) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__view1dTape : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_view_2d_tape\" (void* int int) void*) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__view2dTape : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_reshape_1d_tape\" (void* int) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__reshape1dTape : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_reshape_2d_tape\" (void* int int) void*) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__reshape2dTape : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_reshape_3d_tape\" (void* int int int) void*) (vector-ref a0 2) a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__reshape3dTape : AnyPtr -> Int -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (let ((raw_r ((foreign-procedure \"tensor_reshape_4d_tape\" (void* int int int int) void*) (vector-ref a0 2) a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__reshape4dTape : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_tile_2d_tape\" (void* int int) void*) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__tile2dTape : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_narrow_tape\" (void* int int int) void*) (vector-ref a0 2) a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__narrowTape : AnyPtr -> Int -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_transpose_last2_tape\" (void*) void*) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__transposeLast2Tape : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_transpose_2d_tape\" (void*) void*) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__transpose2dTape : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_cat_tape\" (void* int int) void*) a0 a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__catTape : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_cat2_tape\" (void* void*) void*) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__cat2Tape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_concat_2d_axis1_tape\" (void* void*) void*) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__concat2dAxis1Tape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_gather_tape\" (void* void* int) void*) (vector-ref a0 2) (vector-ref a1 2) a2))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__gatherTape : AnyPtr -> AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_scatter_add_tape\" (void* void* int) void*) (vector-ref a0 2) (vector-ref a1 2) a2))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__scatterAddTape : AnyPtr -> AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_argsort_tape\" (void* int int) void*) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__argsortTape : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_cumprod_tape\" (void* int) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__cumprodTape : AnyPtr -> Int -> AnyPtr


public export
UserDeviceLinear TapeDev where
  primMv             = prim__mvTape
  primMm             = prim__mmTape
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
  primReshape1d      = prim__reshape1dTape
  primReshape2d      = prim__reshape2dTape
  primReshape3d      = prim__reshape3dTape
  primReshape4d      = prim__reshape4dTape
  primTile2d         = prim__tile2dTape
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

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_gelu_tape\" (void*) void*) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__geluTape : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_leaky_relu_tape\" (void* double) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__leakyReluTape : AnyPtr -> Double -> AnyPtr
%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_silu_tape\" (void*) void*) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__siluTape : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_softplus_tape\" (void*) void*) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__softplusTape : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_softmax_tape\" (void* int) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__softmaxTape : AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_log_softmax_tape\" (void* int) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__logSoftmaxTape : AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_softmax_2d_tape\" (void*) void*) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__softmax2dTape : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_log_softmax_2d_tape\" (void*) void*) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__logSoftmax2dTape : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_softmax_3d_tape\" (void*) void*) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__softmax3dTape : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_masked_fill_tape\" (void* void* double) void*) (vector-ref a0 2) (vector-ref a1 2) a2))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__maskedFillTape : AnyPtr -> AnyPtr -> Double -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_expand_mask_tape\" (void* int) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__expandMaskTape : AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_layer_norm_2d_tape\" (void* void* void* double) void*) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2) a3))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__layerNorm2dTape : AnyPtr -> AnyPtr -> AnyPtr -> Double -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6 a7 a8 a9)  (let ((raw_r ((foreign-procedure \"tensor_batch_norm_tape\" (void* void* void* void* void* int int int double double) void*) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2) (vector-ref a3 2) (vector-ref a4 2) a5 a6 a7 a8 a9))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__batchNormTape : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> Double -> Double -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_dropout_tape\" (void* double int int) void*) (vector-ref a0 2) a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__dropoutTape : AnyPtr -> Double -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_embedding_tape\" (void* void* int int) void*) (vector-ref a0 2) (vector-ref a1 2) a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__embeddingTape : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_cosine_similarity_tape\" (void* void* int) void*) (vector-ref a0 2) (vector-ref a1 2) a2))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__cosineSimilarityTape : AnyPtr -> AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (let ((raw_r ((foreign-procedure \"tensor_cross_attention_tape\" (void* void* void* void* double) void*) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2) (vector-ref a3 2) a4))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__crossAttentionTape : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> Double -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_bce_with_logits_tape\" (void* void*) void*) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__bceWithLogitsTape : AnyPtr -> AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_gru_cell_tape\" (void* void* void* int) void*) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2) a3))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__gruCellTape : AnyPtr -> AnyPtr -> AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  ((foreign-procedure \"tensor_lstm_gates_pair_tape\" (void* void* int) void*) (vector-ref a0 2) (vector-ref a1 2) a2))"
prim__lstmGatesPairTape : AnyPtr -> AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_pair_first_tape\" (void*) void*) a0))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__pairFirstTape : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_pair_second_tape\" (void*) void*) a0))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
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


----------------------------------------------------------------------
-- Conv-slice FFI bindings (tape-suffixed)
----------------------------------------------------------------------

%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (let ((raw_r ((foreign-procedure \"tensor_conv1d_tape\" (void* void* void* int int) void*) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2) a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__conv1dTape : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_conv1d_circular_tape\" (void* void*) void*) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__conv1dCircularTape : AnyPtr -> AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_avg_pool1d_tape\" (void* int int) void*) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__avgPool1dTape : AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_max_pool1d_tape\" (void* int int) void*) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__maxPool1dTape : AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6)  (let ((raw_r ((foreign-procedure \"tensor_conv2d_tape\" (void* void* void* int int int int) void*) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2) a3 a4 a5 a6))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__conv2dTape : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6)  (let ((raw_r ((foreign-procedure \"tensor_conv2d_batched_tape\" (void* void* void* int int int int) void*) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2) a3 a4 a5 a6))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__conv2dBatchedTape : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (let ((raw_r ((foreign-procedure \"tensor_avg_pool2d_tape\" (void* int int int int) void*) (vector-ref a0 2) a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__avgPool2dTape : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (let ((raw_r ((foreign-procedure \"tensor_max_pool2d_tape\" (void* int int int int) void*) (vector-ref a0 2) a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__maxPool2dTape : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (let ((raw_r ((foreign-procedure \"tensor_max_pool2d_batched_tape\" (void* int int int int) void*) (vector-ref a0 2) a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__maxPool2dBatchedTape : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr


public export
UserDeviceConv TapeDev where
  primConv1d           = prim__conv1dTape
  primConv1dCircular   = prim__conv1dCircularTape
  primAvgPool1d        = prim__avgPool1dTape
  primMaxPool1d        = prim__maxPool1dTape
  primConv2d           = prim__conv2dTape
  primConv2dBatched    = prim__conv2dBatchedTape
  primAvgPool2d        = prim__avgPool2dTape
  primMaxPool2d        = prim__maxPool2dTape
  primMaxPool2dBatched = prim__maxPool2dBatchedTape


----------------------------------------------------------------------
-- Tape-slice FFI bindings (tape-suffixed)
----------------------------------------------------------------------

%foreign "scheme:(lambda (a0)  ((foreign-procedure \"tensor_requires_grad_tape\" (void*) int) (vector-ref a0 2)))"
prim__requiresGradTape : AnyPtr -> Int
%foreign "scheme:(lambda (a0 a1)  ((foreign-procedure \"tensor_set_requires_grad_tape\" (void* int) void) (vector-ref a0 2) a1))"
prim__setRequiresGradTape : AnyPtr -> Int -> PrimIO ()
%foreign "scheme:(lambda (a0)  ((foreign-procedure \"tensor_backward_tape\" (void*) void) (vector-ref a0 2)))"
prim__backwardTape : AnyPtr -> PrimIO ()
%foreign "C:tensor_no_grad_begin_tape,libidrisml"
prim__noGradBeginTape : PrimIO ()
%foreign "C:tensor_no_grad_end_tape,libidrisml"
prim__noGradEndTape : PrimIO ()
%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_detach_tape\" (void*) void*) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__detachTape : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_with_grad_tape\" (void*) void*) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__withGradTape : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0)  ((foreign-procedure \"tensor_dim_tape\" (void*) int) (vector-ref a0 2)))"
prim__tensorDimTape : AnyPtr -> Int
%foreign "scheme:(lambda (a0 a1)  ((foreign-procedure \"tensor_size_tape\" (void* int) int) (vector-ref a0 2) a1))"
prim__tensorSizeAtTape : AnyPtr -> Int -> Int
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"param_register_return_tape\" (string void*) void*) a0 (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__paramRegisterTape : String -> AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_create_param_1d_tape\" (int void*) void*) a0 a1))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__createParam1dTape : Int -> AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_create_param_2d_tape\" (int int void*) void*) a0 a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__createParam2dTape : Int -> Int -> AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_create_param_3d_tape\" (int int int void*) void*) a0 a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__createParam3dTape : Int -> Int -> Int -> AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_create_state_1d_tape\" (int void*) void*) a0 a1))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__createState1dTape : Int -> AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_create_state_2d_tape\" (int int void*) void*) a0 a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__createState2dTape : Int -> Int -> AnyPtr -> AnyPtr
%foreign "C:polyak_blend_tape,libidrisml"
prim__polyakBlendTape : Double -> String -> String -> PrimIO Int
%foreign "C:param_count_tape,libidrisml"
prim__paramCountTape : PrimIO Int
%foreign "C:param_name_tape,libidrisml"
prim__paramNameTape : Int -> PrimIO String
%foreign "C:param_grad_item_at_tape,libidrisml"
prim__paramGradItemAtTape : Int -> Int -> PrimIO Double
%foreign "C:param_zero_all_grads_tape,libidrisml"
prim__paramZeroAllTape : PrimIO ()
%foreign "C:optimizer_create_sgd_tape,libidrisml"
prim__optimizerCreateSgdTape : Double -> AnyPtr
%foreign "C:optimizer_create_rmsprop_tape,libidrisml"
prim__optimizerCreateRmspropTape : Double -> Double -> Double -> Double -> Double -> AnyPtr
%foreign "C:optimizer_create_adam_tape,libidrisml"
prim__optimizerCreateAdamTape : Double -> Double -> Double -> Double -> AnyPtr
%foreign "C:optimizer_create_adam_group_tape,libidrisml"
prim__optimizerCreateAdamGroupTape : Double -> Double -> Double -> Double -> String -> AnyPtr
%foreign "C:optimizer_create_adamw_tape,libidrisml"
prim__optimizerCreateAdamWTape : Double -> Double -> Double -> Double -> Double -> AnyPtr
%foreign "C:optimizer_set_lr_tape,libidrisml"
prim__optimizerSetLrTape : AnyPtr -> Double -> PrimIO ()
%foreign "C:optimizer_set_param_lr_tape,libidrisml"
prim__optimizerSetParamLrTape : AnyPtr -> String -> Double -> PrimIO ()
%foreign "scheme:(lambda (a0 a1 a2 a3 a4) (let ((result ((foreign-procedure \"native_train_step_tape\" (void* int double void* double) double) a0 a1 a2 (vector-ref a3 2) a4))) (collect 0) (when (top-level-bound? 'idris-drain-once) (let loop () (when ((top-level-value 'idris-drain-once)) (loop)))) result))"
prim__nativeTrainStepTape : AnyPtr -> Int -> Double -> AnyPtr -> Double -> Double
%foreign "C:param_save_tape,libidrisml"
prim__paramSaveTape : String -> PrimIO Int
%foreign "C:param_load_tape,libidrisml"
prim__paramLoadTape : String -> PrimIO Int
%foreign "C:param_load_with_policy_tape,libidrisml"
prim__paramLoadWithPolicyTape : String -> Int -> PrimIO Int
%foreign "C:optimizer_save_tape,libidrisml"
prim__optimizerSaveTape : AnyPtr -> String -> PrimIO Int
%foreign "C:optimizer_load_tape,libidrisml"
prim__optimizerLoadTape : AnyPtr -> String -> PrimIO Int
%foreign "C:backend_profile_reset_tape,libidrisml"
prim__profileResetTape : PrimIO ()
%foreign "C:backend_profile_report_tape,libidrisml"
prim__profileReportTape : PrimIO ()
%foreign "C:tensor_epoch_begin_tape,libidrisml"
prim__epochBeginTape : PrimIO ()
%foreign "C:tensor_epoch_end_tape,libidrisml"
prim__epochEndTape : PrimIO ()
%foreign "C:tensor_live_count_tape,libidrisml"
prim__liveCountTape : Int -> Int
%foreign "C:tensor_peak_live_count_tape,libidrisml"
prim__peakLiveCountTape : Int -> Int


%foreign "scheme:(lambda (val rg stream dtag) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r (if (= dtag 0) ((foreign-procedure \"tensor_create_scalar_f32_streamed_tape\" (double int int) void*) val rg stream) ((foreign-procedure \"tensor_create_scalar_f64_streamed_tape\" (double int int) void*) val rg stream)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__createScalarStreamedTape : Double -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (data shape rank rg stream dtag) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r (if (= dtag 0) ((foreign-procedure \"tensor_create_f32_streamed_tape\" (void* void* int int int) void*) data shape rank rg stream) ((foreign-procedure \"tensor_create_f64_streamed_tape\" (void* void* int int int) void*) data shape rank rg stream)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__createStreamedTape : AnyPtr -> AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (n data rg stream dtag) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r (if (= dtag 0) ((foreign-procedure \"tensor_create_1d_f32_streamed_tape\" (int void* int int) void*) n data rg stream) ((foreign-procedure \"tensor_create_1d_f64_streamed_tape\" (int void* int int) void*) n data rg stream)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__create1dStreamedTape : Int -> AnyPtr -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (rows cols data rg stream dtag) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r (if (= dtag 0) ((foreign-procedure \"tensor_create_2d_f32_streamed_tape\" (int int void* int int) void*) rows cols data rg stream) ((foreign-procedure \"tensor_create_2d_f64_streamed_tape\" (int int void* int int) void*) rows cols data rg stream)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__create2dStreamedTape : Int -> Int -> AnyPtr -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (n data stream dtag) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r (if (= dtag 0) ((foreign-procedure \"tensor_create_param_1d_f32_streamed_tape\" (int void* int) void*) n data stream) ((foreign-procedure \"tensor_create_param_1d_f64_streamed_tape\" (int void* int) void*) n data stream)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__createParam1dStreamedTape : Int -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (rows cols data stream dtag) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r (if (= dtag 0) ((foreign-procedure \"tensor_create_param_2d_f32_streamed_tape\" (int int void* int) void*) rows cols data stream) ((foreign-procedure \"tensor_create_param_2d_f64_streamed_tape\" (int int void* int) void*) rows cols data stream)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__createParam2dStreamedTape : Int -> Int -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (d0 d1 d2 data stream dtag) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r (if (= dtag 0) ((foreign-procedure \"tensor_create_param_3d_f32_streamed_tape\" (int int int void* int) void*) d0 d1 d2 data stream) ((foreign-procedure \"tensor_create_param_3d_f64_streamed_tape\" (int int int void* int) void*) d0 d1 d2 data stream)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__createParam3dStreamedTape : Int -> Int -> Int -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (d0 d1 d2 d3 data stream dtag) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r (if (= dtag 0) ((foreign-procedure \"tensor_create_param_4d_f32_streamed_tape\" (int int int int void* int) void*) d0 d1 d2 d3 data stream) ((foreign-procedure \"tensor_create_param_4d_f64_streamed_tape\" (int int int int void* int) void*) d0 d1 d2 d3 data stream)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__createParam4dStreamedTape : Int -> Int -> Int -> Int -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (n data stream dtag) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r (if (= dtag 0) ((foreign-procedure \"tensor_create_state_1d_f32_streamed_tape\" (int void* int) void*) n data stream) ((foreign-procedure \"tensor_create_state_1d_f64_streamed_tape\" (int void* int) void*) n data stream)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__createState1dStreamedTape : Int -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (rows cols data stream dtag) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r (if (= dtag 0) ((foreign-procedure \"tensor_create_state_2d_f32_streamed_tape\" (int int void* int) void*) rows cols data stream) ((foreign-procedure \"tensor_create_state_2d_f64_streamed_tape\" (int int void* int) void*) rows cols data stream)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__createState2dStreamedTape : Int -> Int -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 stream dtag) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r (if (= dtag 0) ((foreign-procedure \"tensor_cast_dtype_f32_streamed_tape\" (void* int) void*) (vector-ref a0 2) stream) ((foreign-procedure \"tensor_cast_dtype_f64_streamed_tape\" (void* int) void*) (vector-ref a0 2) stream)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__castStreamedTape : AnyPtr -> Int -> Int -> AnyPtr

public export
UserDeviceTape TapeDev where
  primCreateScalarStreamed        = prim__createScalarStreamedTape
  primCreateStreamed              = prim__createStreamedTape
  primCreate1dStreamed            = prim__create1dStreamedTape
  primCreate2dStreamed            = prim__create2dStreamedTape
  primCreateParam1dStreamed       = prim__createParam1dStreamedTape
  primCreateParam2dStreamed       = prim__createParam2dStreamedTape
  primCreateParam3dStreamed       = prim__createParam3dStreamedTape
  primCreateParam4dStreamed       = prim__createParam4dStreamedTape
  primCreateState1dStreamed       = prim__createState1dStreamedTape
  primCreateState2dStreamed       = prim__createState2dStreamedTape
  primCastStreamed                = prim__castStreamedTape
  primRequiresGrad         = prim__requiresGradTape
  primSetRequiresGrad      = prim__setRequiresGradTape
  primBackward             = prim__backwardTape
  primNoGradBegin          = prim__noGradBeginTape
  primNoGradEnd            = prim__noGradEndTape
  primDetach               = prim__detachTape
  primWithGrad             = prim__withGradTape
  primTensorDim            = prim__tensorDimTape
  primTensorSizeAt         = prim__tensorSizeAtTape
  primParamRegister        = prim__paramRegisterTape
  primItem2d               = prim__item2dTape
  primCreate1d             = prim__create1dTape
  primMnistGetImage        = prim__mnistGetImageTape
  primOneHot               = prim__oneHotTape
  primCreateParam1d        = prim__createParam1dTape
  primCreateParam2d        = prim__createParam2dTape
  primCreateParam3d        = prim__createParam3dTape
  primCreateState1d        = prim__createState1dTape
  primCreateState2d        = prim__createState2dTape
  primPolyakBlend          = prim__polyakBlendTape
  primParamCount           = prim__paramCountTape
  primParamName            = prim__paramNameTape
  primParamGradItemAt      = prim__paramGradItemAtTape
  primParamZeroAll         = prim__paramZeroAllTape
  primOptimizerCreateSgd       = prim__optimizerCreateSgdTape
  primOptimizerCreateRmsprop   = prim__optimizerCreateRmspropTape
  primOptimizerCreateAdam      = prim__optimizerCreateAdamTape
  primOptimizerCreateAdamGroup = prim__optimizerCreateAdamGroupTape
  primOptimizerCreateAdamW     = prim__optimizerCreateAdamWTape
  primOptimizerSetLr           = prim__optimizerSetLrTape
  primOptimizerSetParamLr      = prim__optimizerSetParamLrTape
  primNativeTrainStep          = prim__nativeTrainStepTape
  primParamSave                = prim__paramSaveTape
  primParamLoad                = prim__paramLoadTape
  primParamLoadWithPolicy      = prim__paramLoadWithPolicyTape
  primOptimizerSave            = prim__optimizerSaveTape
  primOptimizerLoad            = prim__optimizerLoadTape
  primProfileReset             = prim__profileResetTape
  primProfileReport            = prim__profileReportTape
  primEpochBegin               = prim__epochBeginTape
  primEpochEnd                 = prim__epochEndTape
  primLiveCount                = prim__liveCountTape
  primPeakLiveCount            = prim__peakLiveCountTape


----------------------------------------------------------------------
-- UserDeviceTransfer instance (cross-backend transfer surface)
--
-- Tape lives entirely on host CPU; there are no hardware variants
-- to switch between, so `primIntraMigrate` is a literal no-op (the
-- C-side `tensor_to_device_tape` returns the input handle as-is).
----------------------------------------------------------------------

%foreign "scheme:(lambda (a0 a1)  ((foreign-procedure \"tensor_to_doubles_tape\" (void* void*) void) (vector-ref a0 2) a1) a1)"
prim__toHostTape : AnyPtr -> AnyPtr -> AnyPtr

-- The host buffer helpers (alloc / free / write-return for doubles
-- and ints) are byte-identical across all three backends, so they
-- live as unified definitions in `packages/backends/shared_utils.c`.
-- All three `UserDeviceTransfer` instances bind through the same
-- unified C symbols here.
%foreign "C:tensor_alloc_doubles,libidrisml"
prim__allocHostTape : Int -> AnyPtr

%foreign "C:tensor_free_doubles,libidrisml"
prim__freeHostTape : AnyPtr -> PrimIO ()

%foreign "C:tensor_alloc_ints,libidrisml"
prim__allocIntHostTape : Int -> AnyPtr

%foreign "C:tensor_free_ints,libidrisml"
prim__freeIntHostTape : AnyPtr -> PrimIO ()

%foreign "C:tensor_write_int_return,libidrisml"
prim__setIntHostTape : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_create_tape\" (void* void* int int) void*) a0 a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__createFromHostTape : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_to_device_tape\" (void* string) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__intraMigrateTape : AnyPtr -> String -> AnyPtr

public export
UserDeviceTransfer TapeDev where
  backendTag         = "tape"
  primToHost         = prim__toHostTape
  primAllocHost      = prim__allocHostTape
  primFreeHost       = prim__freeHostTape
  primAllocIntHost   = prim__allocIntHostTape
  primFreeIntHost    = prim__freeIntHostTape
  primSetIntHost     = prim__setIntHostTape
  primCreateFromHost = prim__createFromHostTape
  primIntraMigrate   = prim__intraMigrateTape


----------------------------------------------------------------------
-- Compatible (TapeDev, dt). Tape backend stores doubles only; F32
-- would require a parallel `float*` arena (deferred).
----------------------------------------------------------------------

public export
Compatible TapeDev F64 where


----------------------------------------------------------------------
-- HardwareClass: the tape backend runs on the host CPU.
----------------------------------------------------------------------

public export
HardwareClassed TapeDev where
  hardwareClass = HostCpu
