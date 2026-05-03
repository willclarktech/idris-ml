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

%foreign "scheme:(lambda (a0 a1) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_create_scalar_torch\" (double int) void*) a0 a1))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__createScalarTorch : Double -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_create_torch\" (void* void* int int) void*) a0 a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__createTorch : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0)  ((foreign-procedure \"tensor_free_torch\" (void*) void) (vector-ref a0 2)))"
prim__freeTorch : AnyPtr -> ()

%foreign "scheme:(lambda (a0)  ((foreign-procedure \"tensor_item_torch\" (void*) double) (vector-ref a0 2)))"
prim__itemTorch : AnyPtr -> Double

%foreign "scheme:(lambda (a0 a1)  ((foreign-procedure \"tensor_item_1d_torch\" (void* int) double) (vector-ref a0 2) a1))"
prim__item1dTorch : AnyPtr -> Int -> Double

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_clone_torch\" (void*) void*) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__cloneTorch : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_add_torch\" (void* void*) void*) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__addTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_sub_torch\" (void* void*) void*) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__subTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_mul_torch\" (void* void*) void*) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__mulTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_div_torch\" (void* void*) void*) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__divTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_neg_torch\" (void*) void*) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__negTorch : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_abs_torch\" (void*) void*) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__absTorch : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_exp_torch\" (void*) void*) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__expTorch : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_log_torch\" (void*) void*) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__logTorch : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_sqrt_torch\" (void*) void*) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__sqrtTorch : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_pow_torch\" (void* void*) void*) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__powTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_sigmoid_torch\" (void*) void*) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__sigmoidTorch : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_tanh_torch\" (void*) void*) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__tanhTorch : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_add_scalar_torch\" (void* double) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__addScalarTorch : AnyPtr -> Double -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_mul_scalar_torch\" (void* double) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__mulScalarTorch : AnyPtr -> Double -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_clamp_min_torch\" (void* double) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__clampMinTorch : AnyPtr -> Double -> AnyPtr


----------------------------------------------------------------------
-- TorchHwDev + TorchDev type + UserDeviceCore instance
--
-- `TorchHwDev` enumerates the hardware variants the torch backend
-- supports: CPU (the historical default), MPS (Apple Metal), and
-- CUDA n (NVIDIA, indexed). Every torch-backed `Tensor` carries one
-- of these via `TorchDev d`, so the type-system can prevent
-- cross-device op attempts at compile time while libtorch's
-- auto-dispatch handles intra-device routing at run time.
----------------------------------------------------------------------

public export
data TorchHwDev : Type where
  TCpu  : TorchHwDev
  TMps  : TorchHwDev
  TCuda : Nat -> TorchHwDev

||| Maps a `TorchHwDev` to the device string libtorch's `at::Device`
||| accepts: "cpu", "mps", or "cuda:<n>". This is what gets passed to
||| `tensor_to_device(handle, str)` after every fresh tensor
||| construction so the new tensor lands on the right hardware.
public export
torchHwDevName : TorchHwDev -> String
torchHwDevName TCpu      = "cpu"
torchHwDevName TMps      = "mps"
torchHwDevName (TCuda n) = "cuda:" ++ show n

public export
data TorchDev : TorchHwDev -> Type where MkTorchDev : TorchDev d

||| FFI binding for libtorch's `tensor.to(device_str)`. Used by every
||| `UserDeviceCore (TorchDev d)` create method to migrate fresh
||| (CPU-allocated) tensors to the target hardware. On `TCpu` the
||| migration is a self-move (`.to("cpu")` is a no-op for CPU tensors).
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_to_device_torch\" (void* string) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__toDeviceTorch : AnyPtr -> String -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  ((foreign-procedure \"tensor_item_2d_torch\" (void* int int) double) (vector-ref a0 2) a1 a2))"
prim__item2dTorch : AnyPtr -> Int -> Int -> Double
%foreign "scheme:(lambda (a0 a1 a2) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_create_1d_torch\" (int void* int) void*) a0 a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__create1dTorch : Int -> AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"mnist_get_image_torch\" (void* int) void*) a0 a1))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__mnistGetImageTorch : AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_one_hot_torch\" (void* int int) void*) a0 a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__oneHotTorch : AnyPtr -> Int -> Int -> AnyPtr

public export
{d : TorchHwDev} -> UserDeviceCore (TorchDev d) where
  deviceName       = torchHwDevName d
  deviceStreamTag  = 0
  -- Create primitives go through libtorch's CPU-bound construction
  -- path (`torch::from_blob().clone()`), then migrate to the target
  -- hardware via `tensor_to_device`. Self-move on `TCpu`.
  primCreateScalar val rg =
    prim__toDeviceTorch (prim__createScalarTorch val rg) (torchHwDevName d)
  primCreate dat sh rank rg =
    prim__toDeviceTorch (prim__createTorch dat sh rank rg) (torchHwDevName d)
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

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_mv_torch\" (void* void*) void*) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__mvTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_mm_torch\" (void* void*) void*) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__mmTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_matmul_torch\" (void* void*) void*) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__matmulTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_linear_torch\" (void* void* void*) void*) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__linearTorch : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_dot_torch\" (void* void*) void*) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__dotTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_outer_torch\" (void* void*) void*) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__outerTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_bmm_torch\" (void* void*) void*) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__bmmTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_linear_2d_torch\" (void* void* void*) void*) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__linear2dTorch : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_sum_torch\" (void*) void*) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__sumTorch : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_mean_torch\" (void*) void*) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__meanTorch : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_min_torch\" (void*) void*) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__tensorMinTorch : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_max_torch\" (void*) void*) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__tensorMaxTorch : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_sum_dim_torch\" (void* int int) void*) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__sumDimTorch : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_select_torch\" (void* int int) void*) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__selectTorch : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_unsqueeze_torch\" (void* int) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__unsqueezeTorch : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_squeeze_torch\" (void* int) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__squeezeTorch : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_stack_torch\" (void* int int) void*) a0 a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__stackTorch : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_view_1d_torch\" (void* int) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__view1dTorch : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_view_2d_torch\" (void* int int) void*) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__view2dTorch : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_reshape_1d_torch\" (void* int) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__reshape1dTorch : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_reshape_2d_torch\" (void* int int) void*) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__reshape2dTorch : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_reshape_3d_torch\" (void* int int int) void*) (vector-ref a0 2) a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__reshape3dTorch : AnyPtr -> Int -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (let ((raw_r ((foreign-procedure \"tensor_reshape_4d_torch\" (void* int int int int) void*) (vector-ref a0 2) a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__reshape4dTorch : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_tile_2d_torch\" (void* int int) void*) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__tile2dTorch : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_narrow_torch\" (void* int int int) void*) (vector-ref a0 2) a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__narrowTorch : AnyPtr -> Int -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_transpose_last2_torch\" (void*) void*) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__transposeLast2Torch : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_transpose_2d_torch\" (void*) void*) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__transpose2dTorch : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_cat_torch\" (void* int int) void*) a0 a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__catTorch : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_cat2_torch\" (void* void*) void*) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__cat2Torch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_concat_2d_axis1_torch\" (void* void*) void*) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__concat2dAxis1Torch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_gather_torch\" (void* void* int) void*) (vector-ref a0 2) (vector-ref a1 2) a2))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__gatherTorch : AnyPtr -> AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_scatter_add_torch\" (void* void* int) void*) (vector-ref a0 2) (vector-ref a1 2) a2))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__scatterAddTorch : AnyPtr -> AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_argsort_torch\" (void* int int) void*) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__argsortTorch : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_cumprod_torch\" (void* int) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__cumprodTorch : AnyPtr -> Int -> AnyPtr


public export
{d : TorchHwDev} -> UserDeviceLinear (TorchDev d) where
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

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_gelu_torch\" (void*) void*) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__geluTorch : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_leaky_relu_torch\" (void* double) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__leakyReluTorch : AnyPtr -> Double -> AnyPtr
%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_silu_torch\" (void*) void*) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__siluTorch : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_softplus_torch\" (void*) void*) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__softplusTorch : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_softmax_torch\" (void* int) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__softmaxTorch : AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_log_softmax_torch\" (void* int) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__logSoftmaxTorch : AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_softmax_2d_torch\" (void*) void*) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__softmax2dTorch : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_log_softmax_2d_torch\" (void*) void*) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__logSoftmax2dTorch : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_softmax_3d_torch\" (void*) void*) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__softmax3dTorch : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_masked_fill_torch\" (void* void* double) void*) (vector-ref a0 2) (vector-ref a1 2) a2))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__maskedFillTorch : AnyPtr -> AnyPtr -> Double -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_expand_mask_torch\" (void* int) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__expandMaskTorch : AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_causal_mask_torch\" (int) void*) a0))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__causalMaskTorch : Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_layer_norm_2d_torch\" (void* void* void* double) void*) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2) a3))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__layerNorm2dTorch : AnyPtr -> AnyPtr -> AnyPtr -> Double -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6 a7 a8 a9)  (let ((raw_r ((foreign-procedure \"tensor_batch_norm_torch\" (void* void* void* void* void* int int int double double) void*) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2) (vector-ref a3 2) (vector-ref a4 2) a5 a6 a7 a8 a9))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__batchNormTorch : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> Double -> Double -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_dropout_torch\" (void* double int int) void*) (vector-ref a0 2) a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__dropoutTorch : AnyPtr -> Double -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_embedding_torch\" (void* void* int int) void*) (vector-ref a0 2) (vector-ref a1 2) a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__embeddingTorch : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_cosine_similarity_torch\" (void* void* int) void*) (vector-ref a0 2) (vector-ref a1 2) a2))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__cosineSimilarityTorch : AnyPtr -> AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (let ((raw_r ((foreign-procedure \"tensor_cross_attention_torch\" (void* void* void* void* double) void*) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2) (vector-ref a3 2) a4))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__crossAttentionTorch : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> Double -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_bce_with_logits_torch\" (void* void*) void*) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__bceWithLogitsTorch : AnyPtr -> AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_gru_cell_torch\" (void* void* void* int) void*) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2) a3))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__gruCellTorch : AnyPtr -> AnyPtr -> AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  ((foreign-procedure \"tensor_lstm_gates_pair_torch\" (void* void* int) void*) (vector-ref a0 2) (vector-ref a1 2) a2))"
prim__lstmGatesPairTorch : AnyPtr -> AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_pair_first_torch\" (void*) void*) a0))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__pairFirstTorch : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_pair_second_torch\" (void*) void*) a0))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__pairSecondTorch : AnyPtr -> AnyPtr


public export
{d : TorchHwDev} -> UserDeviceNN (TorchDev d) where
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

%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (let ((raw_r ((foreign-procedure \"tensor_conv1d_torch\" (void* void* void* int int) void*) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2) a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__conv1dTorch : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_conv1d_circular_torch\" (void* void*) void*) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__conv1dCircularTorch : AnyPtr -> AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_avg_pool1d_torch\" (void* int int) void*) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__avgPool1dTorch : AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_max_pool1d_torch\" (void* int int) void*) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__maxPool1dTorch : AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6)  (let ((raw_r ((foreign-procedure \"tensor_conv2d_torch\" (void* void* void* int int int int) void*) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2) a3 a4 a5 a6))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__conv2dTorch : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6)  (let ((raw_r ((foreign-procedure \"tensor_conv2d_batched_torch\" (void* void* void* int int int int) void*) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2) a3 a4 a5 a6))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__conv2dBatchedTorch : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (let ((raw_r ((foreign-procedure \"tensor_avg_pool2d_torch\" (void* int int int int) void*) (vector-ref a0 2) a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__avgPool2dTorch : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (let ((raw_r ((foreign-procedure \"tensor_max_pool2d_torch\" (void* int int int int) void*) (vector-ref a0 2) a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__maxPool2dTorch : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (let ((raw_r ((foreign-procedure \"tensor_max_pool2d_batched_torch\" (void* int int int int) void*) (vector-ref a0 2) a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__maxPool2dBatchedTorch : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr


public export
{d : TorchHwDev} -> UserDeviceConv (TorchDev d) where
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

%foreign "scheme:(lambda (a0)  ((foreign-procedure \"tensor_requires_grad_torch\" (void*) int) (vector-ref a0 2)))"
prim__requiresGradTorch : AnyPtr -> Int
%foreign "scheme:(lambda (a0 a1)  ((foreign-procedure \"tensor_set_requires_grad_torch\" (void* int) void) (vector-ref a0 2) a1))"
prim__setRequiresGradTorch : AnyPtr -> Int -> PrimIO ()
%foreign "C:tensor_backward_torch,libidrisml"
prim__backwardTorch : AnyPtr -> PrimIO ()
%foreign "C:tensor_no_grad_begin_torch,libidrisml"
prim__noGradBeginTorch : PrimIO ()
%foreign "C:tensor_no_grad_end_torch,libidrisml"
prim__noGradEndTorch : PrimIO ()
%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_detach_torch\" (void*) void*) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__detachTorch : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_with_grad_torch\" (void*) void*) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__withGradTorch : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0)  ((foreign-procedure \"tensor_dim_torch\" (void*) int) (vector-ref a0 2)))"
prim__tensorDimTorch : AnyPtr -> Int
%foreign "scheme:(lambda (a0 a1)  ((foreign-procedure \"tensor_size_torch\" (void* int) int) (vector-ref a0 2) a1))"
prim__tensorSizeAtTorch : AnyPtr -> Int -> Int
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"param_register_return_torch\" (string void*) void*) a0 (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__paramRegisterTorch : String -> AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_create_param_1d_torch\" (int void*) void*) a0 a1))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__createParam1dTorch : Int -> AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_create_param_2d_torch\" (int int void*) void*) a0 a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__createParam2dTorch : Int -> Int -> AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_create_param_3d_torch\" (int int int void*) void*) a0 a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__createParam3dTorch : Int -> Int -> Int -> AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_create_state_1d_torch\" (int void*) void*) a0 a1))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__createState1dTorch : Int -> AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_create_state_2d_torch\" (int int void*) void*) a0 a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__createState2dTorch : Int -> Int -> AnyPtr -> AnyPtr
%foreign "C:polyak_blend_torch,libidrisml"
prim__polyakBlendTorch : Double -> String -> String -> PrimIO Int
%foreign "C:param_count_torch,libidrisml"
prim__paramCountTorch : PrimIO Int
%foreign "C:param_name_torch,libidrisml"
prim__paramNameTorch : Int -> PrimIO String
%foreign "C:param_grad_item_at_torch,libidrisml"
prim__paramGradItemAtTorch : Int -> Int -> PrimIO Double
%foreign "C:param_zero_all_grads_torch,libidrisml"
prim__paramZeroAllTorch : PrimIO ()
%foreign "C:optimizer_create_sgd_torch,libidrisml"
prim__optimizerCreateSgdTorch : Double -> AnyPtr
%foreign "C:optimizer_create_rmsprop_torch,libidrisml"
prim__optimizerCreateRmspropTorch : Double -> Double -> Double -> Double -> Double -> AnyPtr
%foreign "C:optimizer_create_adam_torch,libidrisml"
prim__optimizerCreateAdamTorch : Double -> Double -> Double -> Double -> AnyPtr
%foreign "C:optimizer_create_adam_group_torch,libidrisml"
prim__optimizerCreateAdamGroupTorch : Double -> Double -> Double -> Double -> String -> AnyPtr
%foreign "C:optimizer_create_adamw_torch,libidrisml"
prim__optimizerCreateAdamWTorch : Double -> Double -> Double -> Double -> Double -> AnyPtr
%foreign "C:optimizer_set_lr_torch,libidrisml"
prim__optimizerSetLrTorch : AnyPtr -> Double -> PrimIO ()
%foreign "C:optimizer_set_param_lr_torch,libidrisml"
prim__optimizerSetParamLrTorch : AnyPtr -> String -> Double -> PrimIO ()
%foreign "scheme:(lambda (a0 a1 a2 a3 a4) (let ((result ((foreign-procedure \"native_train_step_torch\" (void* int double void* double) double) a0 a1 a2 (vector-ref a3 2) a4))) (collect 0) (when (top-level-bound? 'idris-drain-once) (let loop () (when ((top-level-value 'idris-drain-once)) (loop)))) result))"
prim__nativeTrainStepTorch : AnyPtr -> Int -> Double -> AnyPtr -> Double -> Double
%foreign "C:param_save_torch,libidrisml"
prim__paramSaveTorch : String -> PrimIO Int
%foreign "C:param_load_torch,libidrisml"
prim__paramLoadTorch : String -> PrimIO Int
%foreign "C:param_load_with_policy_torch,libidrisml"
prim__paramLoadWithPolicyTorch : String -> Int -> PrimIO Int
%foreign "C:optimizer_save_torch,libidrisml"
prim__optimizerSaveTorch : AnyPtr -> String -> PrimIO Int
%foreign "C:optimizer_load_torch,libidrisml"
prim__optimizerLoadTorch : AnyPtr -> String -> PrimIO Int
%foreign "C:backend_profile_reset_torch,libidrisml"
prim__profileResetTorch : PrimIO ()
%foreign "C:backend_profile_report_torch,libidrisml"
prim__profileReportTorch : PrimIO ()
%foreign "C:tensor_epoch_begin_torch,libidrisml"
prim__epochBeginTorch : PrimIO ()
%foreign "C:tensor_epoch_end_torch,libidrisml"
prim__epochEndTorch : PrimIO ()
%foreign "C:tensor_live_count_torch,libidrisml"
prim__liveCountTorch : Int -> Int
%foreign "C:tensor_peak_live_count_torch,libidrisml"
prim__peakLiveCountTorch : Int -> Int


%foreign "scheme:(lambda (val rg stream dtag) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r (if (= dtag 0) ((foreign-procedure \"tensor_create_scalar_f32_streamed_torch\" (double int int) void*) val rg stream) ((foreign-procedure \"tensor_create_scalar_f64_streamed_torch\" (double int int) void*) val rg stream)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__createScalarStreamedTorch : Double -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (data shape rank rg stream dtag) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r (if (= dtag 0) ((foreign-procedure \"tensor_create_f32_streamed_torch\" (void* void* int int int) void*) data shape rank rg stream) ((foreign-procedure \"tensor_create_f64_streamed_torch\" (void* void* int int int) void*) data shape rank rg stream)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__createStreamedTorch : AnyPtr -> AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (n data rg stream dtag) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r (if (= dtag 0) ((foreign-procedure \"tensor_create_1d_f32_streamed_torch\" (int void* int int) void*) n data rg stream) ((foreign-procedure \"tensor_create_1d_f64_streamed_torch\" (int void* int int) void*) n data rg stream)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__create1dStreamedTorch : Int -> AnyPtr -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (rows cols data rg stream dtag) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r (if (= dtag 0) ((foreign-procedure \"tensor_create_2d_f32_streamed_torch\" (int int void* int int) void*) rows cols data rg stream) ((foreign-procedure \"tensor_create_2d_f64_streamed_torch\" (int int void* int int) void*) rows cols data rg stream)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__create2dStreamedTorch : Int -> Int -> AnyPtr -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (n data stream dtag) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r (if (= dtag 0) ((foreign-procedure \"tensor_create_param_1d_f32_streamed_torch\" (int void* int) void*) n data stream) ((foreign-procedure \"tensor_create_param_1d_f64_streamed_torch\" (int void* int) void*) n data stream)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__createParam1dStreamedTorch : Int -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (rows cols data stream dtag) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r (if (= dtag 0) ((foreign-procedure \"tensor_create_param_2d_f32_streamed_torch\" (int int void* int) void*) rows cols data stream) ((foreign-procedure \"tensor_create_param_2d_f64_streamed_torch\" (int int void* int) void*) rows cols data stream)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__createParam2dStreamedTorch : Int -> Int -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (d0 d1 d2 data stream dtag) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r (if (= dtag 0) ((foreign-procedure \"tensor_create_param_3d_f32_streamed_torch\" (int int int void* int) void*) d0 d1 d2 data stream) ((foreign-procedure \"tensor_create_param_3d_f64_streamed_torch\" (int int int void* int) void*) d0 d1 d2 data stream)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__createParam3dStreamedTorch : Int -> Int -> Int -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (d0 d1 d2 d3 data stream dtag) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r (if (= dtag 0) ((foreign-procedure \"tensor_create_param_4d_f32_streamed_torch\" (int int int int void* int) void*) d0 d1 d2 d3 data stream) ((foreign-procedure \"tensor_create_param_4d_f64_streamed_torch\" (int int int int void* int) void*) d0 d1 d2 d3 data stream)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__createParam4dStreamedTorch : Int -> Int -> Int -> Int -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (n data stream dtag) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r (if (= dtag 0) ((foreign-procedure \"tensor_create_state_1d_f32_streamed_torch\" (int void* int) void*) n data stream) ((foreign-procedure \"tensor_create_state_1d_f64_streamed_torch\" (int void* int) void*) n data stream)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__createState1dStreamedTorch : Int -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (rows cols data stream dtag) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r (if (= dtag 0) ((foreign-procedure \"tensor_create_state_2d_f32_streamed_torch\" (int int void* int) void*) rows cols data stream) ((foreign-procedure \"tensor_create_state_2d_f64_streamed_torch\" (int int void* int) void*) rows cols data stream)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__createState2dStreamedTorch : Int -> Int -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 stream dtag) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r (if (= dtag 0) ((foreign-procedure \"tensor_cast_dtype_f32_streamed_torch\" (void* int) void*) (vector-ref a0 2) stream) ((foreign-procedure \"tensor_cast_dtype_f64_streamed_torch\" (void* int) void*) (vector-ref a0 2) stream)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__castStreamedTorch : AnyPtr -> Int -> Int -> AnyPtr

public export
{d : TorchHwDev} -> UserDeviceTape (TorchDev d) where
  primCreateScalarStreamed        = prim__createScalarStreamedTorch
  primCreateStreamed              = prim__createStreamedTorch
  primCreate1dStreamed            = prim__create1dStreamedTorch
  primCreate2dStreamed            = prim__create2dStreamedTorch
  primCreateParam1dStreamed       = prim__createParam1dStreamedTorch
  primCreateParam2dStreamed       = prim__createParam2dStreamedTorch
  primCreateParam3dStreamed       = prim__createParam3dStreamedTorch
  primCreateParam4dStreamed       = prim__createParam4dStreamedTorch
  primCreateState1dStreamed       = prim__createState1dStreamedTorch
  primCreateState2dStreamed       = prim__createState2dStreamedTorch
  primCastStreamed                = prim__castStreamedTorch
  primRequiresGrad         = prim__requiresGradTorch
  primSetRequiresGrad      = prim__setRequiresGradTorch
  primBackward             = prim__backwardTorch
  primNoGradBegin          = prim__noGradBeginTorch
  primNoGradEnd            = prim__noGradEndTorch
  primDetach               = prim__detachTorch
  primWithGrad             = prim__withGradTorch
  primTensorDim            = prim__tensorDimTorch
  primTensorSizeAt         = prim__tensorSizeAtTorch
  primParamRegister        = prim__paramRegisterTorch
  primItem2d               = prim__item2dTorch
  primCreate1d             = prim__create1dTorch
  primMnistGetImage        = prim__mnistGetImageTorch
  primOneHot               = prim__oneHotTorch
  -- Param / state creation primitives go through libtorch's
  -- CPU-bound `torch::from_blob().clone()`, so we migrate to the
  -- target hardware before the tensor escapes the create method.
  primCreateParam1d n dat       =
    prim__toDeviceTorch (prim__createParam1dTorch n dat) (torchHwDevName d)
  primCreateParam2d r c dat     =
    prim__toDeviceTorch (prim__createParam2dTorch r c dat) (torchHwDevName d)
  primCreateParam3d d0 d1 d2 dat =
    prim__toDeviceTorch (prim__createParam3dTorch d0 d1 d2 dat) (torchHwDevName d)
  primCreateState1d n dat       =
    prim__toDeviceTorch (prim__createState1dTorch n dat) (torchHwDevName d)
  primCreateState2d r c dat     =
    prim__toDeviceTorch (prim__createState2dTorch r c dat) (torchHwDevName d)
  primPolyakBlend          = prim__polyakBlendTorch
  primParamCount           = prim__paramCountTorch
  primParamName            = prim__paramNameTorch
  primParamGradItemAt      = prim__paramGradItemAtTorch
  primParamZeroAll         = prim__paramZeroAllTorch
  primOptimizerCreateSgd       = prim__optimizerCreateSgdTorch
  primOptimizerCreateRmsprop   = prim__optimizerCreateRmspropTorch
  primOptimizerCreateAdam      = prim__optimizerCreateAdamTorch
  primOptimizerCreateAdamGroup = prim__optimizerCreateAdamGroupTorch
  primOptimizerCreateAdamW     = prim__optimizerCreateAdamWTorch
  primOptimizerSetLr           = prim__optimizerSetLrTorch
  primOptimizerSetParamLr      = prim__optimizerSetParamLrTorch
  primNativeTrainStep          = prim__nativeTrainStepTorch
  primParamSave                = prim__paramSaveTorch
  primParamLoad                = prim__paramLoadTorch
  primParamLoadWithPolicy      = prim__paramLoadWithPolicyTorch
  primOptimizerSave            = prim__optimizerSaveTorch
  primOptimizerLoad            = prim__optimizerLoadTorch
  primProfileReset             = prim__profileResetTorch
  primProfileReport            = prim__profileReportTorch
  primEpochBegin               = prim__epochBeginTorch
  primEpochEnd                 = prim__epochEndTorch
  primLiveCount                = prim__liveCountTorch
  primPeakLiveCount            = prim__peakLiveCountTorch


----------------------------------------------------------------------
-- Compatible (TorchDev, dt).
--
-- F32 is admitted on every hardware variant (CPU / MPS / CUDA), F64
-- on CPU and CUDA. **MPS + F64 is deliberately NOT compatible**:
-- libtorch's MPS backend rejects F64 tensor *construction* outright
-- (`Cannot convert a MPS Tensor to float64 dtype`), not just at op
-- dispatch — so admitting the combination would let the type
-- system mint a value the runtime can't represent. Users wanting
-- F64-precision on MPS hardware should pin to `(TorchDev TCpu) F64`
-- or `(TorchDev (TCuda n)) F64`. Mirrors the
-- `Compatible (MlxDev MGpu) F64`-rejection demo for mlx.
----------------------------------------------------------------------

public export
{d : TorchHwDev} -> Compatible (TorchDev d) F32 where

public export
Compatible (TorchDev TCpu) F64 where

public export
{n : Nat} -> Compatible (TorchDev (TCuda n)) F64 where


----------------------------------------------------------------------
-- UserDeviceTransfer instance (cross-backend transfer surface)
--
-- The torch hardware-migrate path is the only one that does real
-- work: `tensor_to_device_torch(handle, "mps"|"cuda:n")` migrates a
-- libtorch tensor in place between CPU, MPS, and CUDA without
-- allocating a fresh handle, preserving param-registry membership.
----------------------------------------------------------------------

%foreign "scheme:(lambda (a0 a1)  ((foreign-procedure \"tensor_to_doubles_torch\" (void* void*) void) (vector-ref a0 2) a1) a1)"
prim__toHostTorch : AnyPtr -> AnyPtr -> AnyPtr

-- Host buffer helpers — unified across backends, see Device/Tape.idr.
%foreign "C:tensor_alloc_doubles,libidrisml"
prim__allocHostTorch : Int -> AnyPtr

%foreign "C:tensor_free_doubles,libidrisml"
prim__freeHostTorch : AnyPtr -> ()

%foreign "C:tensor_alloc_ints,libidrisml"
prim__allocIntHostTorch : Int -> AnyPtr

%foreign "C:tensor_free_ints,libidrisml"
prim__freeIntHostTorch : AnyPtr -> ()

%foreign "C:tensor_write_int_return,libidrisml"
prim__setIntHostTorch : AnyPtr -> Int -> Int -> AnyPtr

||| Create from host data + auto-migrate to the target torch hw.
||| The closure here calls the rank-generic `tensor_create_torch`
||| (which lands on CPU by default in libtorch) then
||| `tensor_to_device_torch(handle, "mps"|"cuda:n")` so the returned
||| tensor is on the right hardware variant. Matches the post-create
||| migration the existing `primCreate` does in `UserDeviceCore
||| (TorchDev d)`.
prim__createFromHostTorch : (d : TorchHwDev) -> AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr
prim__createFromHostTorch d dat sh rank rg =
  prim__toDeviceTorch (prim__createTorch dat sh rank rg) (torchHwDevName d)

public export
{d : TorchHwDev} -> UserDeviceTransfer (TorchDev d) where
  backendTag         = "torch"
  primToHost         = prim__toHostTorch
  primAllocHost      = prim__allocHostTorch
  primFreeHost       = prim__freeHostTorch
  primAllocIntHost   = prim__allocIntHostTorch
  primFreeIntHost    = prim__freeIntHostTorch
  primSetIntHost     = prim__setIntHostTorch
  primCreateFromHost = prim__createFromHostTorch d
  primIntraMigrate h hwName =
    prim__toDeviceTorch h hwName


----------------------------------------------------------------------
-- HardwareClass: map each torch hw variant to its physical silicon.
----------------------------------------------------------------------

public export
{d : TorchHwDev} -> HardwareClassed (TorchDev d) where
  hardwareClass = case d of
    TCpu    => HostCpu
    TMps    => AppleGpu
    TCuda n => Nvidia n
