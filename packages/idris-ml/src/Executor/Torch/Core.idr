||| Executor type + Core / Streamed / HardwareClassed instance
||| slices (lifecycle, elementwise arithmetic, stream tag, hw class).
module Executor.Torch.Core

import BackendLib
import DType.Core
import Executor.Core
import Hardware
import Preset

----------------------------------------------------------------------
-- Per-symbol bindings to the torch backend's suffixed C exports
----------------------------------------------------------------------

%foreign "scheme:(lambda (a0 a1) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-scalar-torch)) (set-top-level-value! 'idris-ffi-tensor-create-scalar-torch (foreign-procedure \"tensor_create_scalar_torch\" (double int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-scalar-torch) a0 a1))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__createScalarTorch : Double -> Int -> AnyPtr

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-free-torch)) (set-top-level-value! 'idris-ffi-tensor-free-torch (foreign-procedure \"tensor_free_torch\" (void*) void))) ((top-level-value 'idris-ffi-tensor-free-torch) (vector-ref a0 2)))"
prim__freeTorch : AnyPtr -> ()

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-item-torch)) (set-top-level-value! 'idris-ffi-tensor-item-torch (foreign-procedure \"tensor_item_torch\" (void*) double))) ((top-level-value 'idris-ffi-tensor-item-torch) (vector-ref a0 2)))"
prim__itemTorch : AnyPtr -> Double

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-item-1d-torch)) (set-top-level-value! 'idris-ffi-tensor-item-1d-torch (foreign-procedure \"tensor_item_1d_torch\" (void* int) double))) ((top-level-value 'idris-ffi-tensor-item-1d-torch) (vector-ref a0 2) a1))"
prim__item1dTorch : AnyPtr -> Int -> Double

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-clone-torch)) (set-top-level-value! 'idris-ffi-tensor-clone-torch (foreign-procedure \"tensor_clone_torch\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-clone-torch) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__cloneTorch : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-add-torch)) (set-top-level-value! 'idris-ffi-tensor-add-torch (foreign-procedure \"tensor_add_torch\" (void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-add-torch) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__addTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-sub-torch)) (set-top-level-value! 'idris-ffi-tensor-sub-torch (foreign-procedure \"tensor_sub_torch\" (void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-sub-torch) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__subTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-mul-torch)) (set-top-level-value! 'idris-ffi-tensor-mul-torch (foreign-procedure \"tensor_mul_torch\" (void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-mul-torch) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__mulTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-div-torch)) (set-top-level-value! 'idris-ffi-tensor-div-torch (foreign-procedure \"tensor_div_torch\" (void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-div-torch) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__divTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-neg-torch)) (set-top-level-value! 'idris-ffi-tensor-neg-torch (foreign-procedure \"tensor_neg_torch\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-neg-torch) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__negTorch : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-abs-torch)) (set-top-level-value! 'idris-ffi-tensor-abs-torch (foreign-procedure \"tensor_abs_torch\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-abs-torch) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__absTorch : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-exp-torch)) (set-top-level-value! 'idris-ffi-tensor-exp-torch (foreign-procedure \"tensor_exp_torch\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-exp-torch) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__expTorch : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-log-torch)) (set-top-level-value! 'idris-ffi-tensor-log-torch (foreign-procedure \"tensor_log_torch\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-log-torch) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__logTorch : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-sqrt-torch)) (set-top-level-value! 'idris-ffi-tensor-sqrt-torch (foreign-procedure \"tensor_sqrt_torch\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-sqrt-torch) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__sqrtTorch : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-pow-torch)) (set-top-level-value! 'idris-ffi-tensor-pow-torch (foreign-procedure \"tensor_pow_torch\" (void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-pow-torch) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__powTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-sigmoid-torch)) (set-top-level-value! 'idris-ffi-tensor-sigmoid-torch (foreign-procedure \"tensor_sigmoid_torch\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-sigmoid-torch) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__sigmoidTorch : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-tanh-torch)) (set-top-level-value! 'idris-ffi-tensor-tanh-torch (foreign-procedure \"tensor_tanh_torch\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-tanh-torch) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__tanhTorch : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-add-scalar-torch)) (set-top-level-value! 'idris-ffi-tensor-add-scalar-torch (foreign-procedure \"tensor_add_scalar_torch\" (void* double) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-add-scalar-torch) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__addScalarTorch : AnyPtr -> Double -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-mul-scalar-torch)) (set-top-level-value! 'idris-ffi-tensor-mul-scalar-torch (foreign-procedure \"tensor_mul_scalar_torch\" (void* double) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-mul-scalar-torch) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__mulScalarTorch : AnyPtr -> Double -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-clamp-min-torch)) (set-top-level-value! 'idris-ffi-tensor-clamp-min-torch (foreign-procedure \"tensor_clamp_min_torch\" (void* double) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-clamp-min-torch) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__clampMinTorch : AnyPtr -> Double -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-clamp-torch)) (set-top-level-value! 'idris-ffi-tensor-clamp-torch (foreign-procedure \"tensor_clamp_torch\" (void* double double) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-clamp-torch) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__clampTorch : AnyPtr -> Double -> Double -> AnyPtr

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-round-torch)) (set-top-level-value! 'idris-ffi-tensor-round-torch (foreign-procedure \"tensor_round_torch\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-round-torch) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__roundTorch : AnyPtr -> AnyPtr

----------------------------------------------------------------------
-- TorchHwDev + TorchExecutor type + UserExecutorCore instance
--
-- `TorchHwDev` enumerates the hardware variants the torch backend
-- supports: CPU (the historical default), MPS (Apple Metal), and
-- CUDA n (NVIDIA, indexed). Every torch-backed `Tensor` carries one
-- of these via `TorchExecutor d`, so the type-system can prevent
-- cross-device op attempts at compile time while libtorch's
-- auto-dispatch handles intra-device routing at run time.
----------------------------------------------------------------------

public export
data TorchHwDev : Type where
  TCpu  : TorchHwDev
  TMps  : TorchHwDev
  TCuda : Nat -> TorchHwDev

||| Maps a `TorchHwDev` to the device string libtorch's `at::Executor`
||| accepts: "cpu", "mps", or "cuda:<n>". This is what gets passed to
||| `tensor_to_device(handle, str)` after every fresh tensor
||| construction so the new tensor lands on the right hardware.
public export
torchHwDevName : TorchHwDev -> String
torchHwDevName TCpu      = "cpu"
torchHwDevName TMps      = "mps"
torchHwDevName (TCuda n) = "cuda:" ++ show n

public export
data TorchExecutor : TorchHwDev -> Type where MkTorchExecutor : TorchExecutor d

||| FFI binding for the PARAM-LIFETIME `tensor.to(device_str)` variant
||| (`tensor_to_device_persistent`): the result is exempt from
||| optimizer-step intermediates cleanup. Used by every
||| `UserExecutorCore (TorchExecutor d)` create method to migrate fresh
||| (CPU-allocated) tensors to the target hardware, and by
||| `primIntraMigrate` — both produce params or user-held tensors that
||| must survive `optimizer_step`. Binding the TRACKED `tensor_to_device`
||| here made every such tensor a use-after-free once the first step's
||| `free_intermediates` ran (Hpo.LrFinder SIGABRT class; root-caused
||| 2026-06-12). On `TCpu` the migration is a self-move (`.to("cpu")`
||| is a no-op for CPU tensors).
%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-to-device-torch)) (set-top-level-value! 'idris-ffi-tensor-to-device-torch (foreign-procedure \"tensor_to_device_persistent_torch\" (void* string) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-to-device-torch) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
export
prim__toDeviceTorch : AnyPtr -> String -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-item-2d-torch)) (set-top-level-value! 'idris-ffi-tensor-item-2d-torch (foreign-procedure \"tensor_item_2d_torch\" (void* int int) double))) ((top-level-value 'idris-ffi-tensor-item-2d-torch) (vector-ref a0 2) a1 a2))"
export
prim__item2dTorch : AnyPtr -> Int -> Int -> Double
%foreign "scheme:(lambda (a0 a1 a2 a3) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-one-hot-torch)) (set-top-level-value! 'idris-ffi-tensor-one-hot-torch (foreign-procedure \"tensor_one_hot_torch\" (void* int int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-one-hot-torch) a0 a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
export
prim__oneHotTorch : AnyPtr -> Int -> Int -> Int -> AnyPtr

public export
{d : TorchHwDev} -> UserExecutorCore (TorchExecutor d) where
  deviceName       = torchHwDevName d
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primAbs       = prim__absTorch
  primAdd       = prim__addTorch
  primAddScalar = prim__addScalarTorch
  primClamp     = prim__clampTorch
  primClampMin  = prim__clampMinTorch
  primClone     = prim__cloneTorch
  primDiv       = prim__divTorch
  primExp       = prim__expTorch
  primFree      = prim__freeTorch
  primItem      = prim__itemTorch
  primItem1d    = prim__item1dTorch
  primLog       = prim__logTorch
  primMul       = prim__mulTorch
  primMulScalar = prim__mulScalarTorch
  primNeg       = prim__negTorch
  primPow       = prim__powTorch
  primRound     = prim__roundTorch
  primSigmoid   = prim__sigmoidTorch
  primSqrt      = prim__sqrtTorch
  primSub       = prim__subTorch
  primTanh      = prim__tanhTorch
  -- <<< END GENERATED <<<
  -- Hand-written overrides:
  -- Create primitives go through libtorch's CPU-bound construction
  -- path (`torch::from_blob().clone()`), then migrate to the target
  -- hardware via `tensor_to_device`. Self-move on `TCpu`.
  primCreateScalar val rg =
    prim__toDeviceTorch (prim__createScalarTorch val rg) (torchHwDevName d)

public export
{d : TorchHwDev} -> UserExecutorStreamed (TorchExecutor d) where
  deviceStreamTag = 0

----------------------------------------------------------------------
-- HardwareClass: map each torch hw variant to its physical silicon.
----------------------------------------------------------------------

public export
{d : TorchHwDev} -> HardwareClassed (TorchExecutor d) where
  hardwareClass = case d of
    TCpu    => HostCpu
    TMps    => AppleGpu
    TCuda n => Nvidia n

----------------------------------------------------------------------
-- Hardware (type-level): map each torch hw variant to its kind tag.
----------------------------------------------------------------------

public export RunsOn (TorchExecutor TCpu)        Cpu       where
public export RunsOn (TorchExecutor TMps)        AppleGpu  where
public export {n : Nat} -> RunsOn (TorchExecutor (TCuda n)) (Cuda n) where

----------------------------------------------------------------------
-- Backend (type-level): every torch hardware variant is provided by
-- TorchBackend.
----------------------------------------------------------------------

public export
{hw : TorchHwDev} -> RunsVia (TorchExecutor hw) TorchBackend where

----------------------------------------------------------------------
-- Preset: per-Hardware defaults for libtorch.
--   * Cpu      → TorchExecutor TCpu        + F64
--   * AppleGpu → TorchExecutor TMps        + F32   (libtorch Metal is F32-only)
--   * Cuda n   → TorchExecutor (TCuda n)   + F64
----------------------------------------------------------------------

public export
Preset TorchBackend Cpu where
  presetExecutor = TorchExecutor TCpu
  presetDType    = F64

public export
Preset TorchBackend AppleGpu where
  presetExecutor = TorchExecutor TMps
  presetDType    = F32

public export
{n : Nat} -> Preset TorchBackend (Cuda n) where
  presetExecutor = TorchExecutor (TCuda n)
  presetDType    = F64
