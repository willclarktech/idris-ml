||| `TorchExecutor` — `UserExecutorCore` instance for the libtorch backend.
|||
||| Forwards to the torch-suffixed C symbols emitted under Phase 1's
||| `rename_torch.h` (e.g. `tensor_add_torch`). Only resolvable at
||| runtime if the build's BACKEND list includes `torch`.
module Executor.Torch

import Executor.Core
import DType.Core
import Backend
import Hardware
import Preset


----------------------------------------------------------------------
-- Per-symbol bindings to the torch backend's suffixed C exports
----------------------------------------------------------------------

%foreign "scheme:(lambda (a0 a1) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-scalar-torch)) (set-top-level-value! 'idris-ffi-tensor-create-scalar-torch (foreign-procedure \"tensor_create_scalar_torch\" (double int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-scalar-torch) a0 a1))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__createScalarTorch : Double -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-torch)) (set-top-level-value! 'idris-ffi-tensor-create-torch (foreign-procedure \"tensor_create_torch\" (void* void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-torch) a0 a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__createTorch : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr

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

||| FFI binding for libtorch's `tensor.to(device_str)`. Used by every
||| `UserExecutorCore (TorchExecutor d)` create method to migrate fresh
||| (CPU-allocated) tensors to the target hardware. On `TCpu` the
||| migration is a self-move (`.to("cpu")` is a no-op for CPU tensors).
%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-to-device-torch)) (set-top-level-value! 'idris-ffi-tensor-to-device-torch (foreign-procedure \"tensor_to_device_torch\" (void* string) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-to-device-torch) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__toDeviceTorch : AnyPtr -> String -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-item-2d-torch)) (set-top-level-value! 'idris-ffi-tensor-item-2d-torch (foreign-procedure \"tensor_item_2d_torch\" (void* int int) double))) ((top-level-value 'idris-ffi-tensor-item-2d-torch) (vector-ref a0 2) a1 a2))"
prim__item2dTorch : AnyPtr -> Int -> Int -> Double
%foreign "scheme:(lambda (a0 a1 a2 a3) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-one-hot-torch)) (set-top-level-value! 'idris-ffi-tensor-one-hot-torch (foreign-procedure \"tensor_one_hot_torch\" (void* int int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-one-hot-torch) a0 a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__oneHotTorch : AnyPtr -> Int -> Int -> Int -> AnyPtr

public export
{d : TorchHwDev} -> UserExecutorCore (TorchExecutor d) where
  deviceName       = torchHwDevName d
  deviceStreamTag  = 0
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primAbs = prim__absTorch
  primAdd = prim__addTorch
  primAddScalar = prim__addScalarTorch
  primClamp = prim__clampTorch
  primClampMin = prim__clampMinTorch
  primClone = prim__cloneTorch
  primDiv = prim__divTorch
  primExp = prim__expTorch
  primFree = prim__freeTorch
  primItem = prim__itemTorch
  primItem1d = prim__item1dTorch
  primLog = prim__logTorch
  primMul = prim__mulTorch
  primMulScalar = prim__mulScalarTorch
  primNeg = prim__negTorch
  primPow = prim__powTorch
  primRound = prim__roundTorch
  primSigmoid = prim__sigmoidTorch
  primSqrt = prim__sqrtTorch
  primSub = prim__subTorch
  primTanh = prim__tanhTorch
  -- <<< END GENERATED <<<
  -- Hand-written overrides:
  -- Create primitives go through libtorch's CPU-bound construction
  -- path (`torch::from_blob().clone()`), then migrate to the target
  -- hardware via `tensor_to_device`. Self-move on `TCpu`.
  primCreateScalar val rg =
    prim__toDeviceTorch (prim__createScalarTorch val rg) (torchHwDevName d)
----------------------------------------------------------------------
-- Linear-slice FFI bindings (torch-suffixed)
----------------------------------------------------------------------

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-mv-torch)) (set-top-level-value! 'idris-ffi-tensor-mv-torch (foreign-procedure \"tensor_mv_torch\" (void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-mv-torch) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__mvTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-mm-torch)) (set-top-level-value! 'idris-ffi-tensor-mm-torch (foreign-procedure \"tensor_mm_torch\" (void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-mm-torch) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__mmTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-matmul-torch)) (set-top-level-value! 'idris-ffi-tensor-matmul-torch (foreign-procedure \"tensor_matmul_torch\" (void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-matmul-torch) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__matmulTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-linear-torch)) (set-top-level-value! 'idris-ffi-tensor-linear-torch (foreign-procedure \"tensor_linear_torch\" (void* void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-linear-torch) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__linearTorch : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-dot-torch)) (set-top-level-value! 'idris-ffi-tensor-dot-torch (foreign-procedure \"tensor_dot_torch\" (void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-dot-torch) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__dotTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-outer-torch)) (set-top-level-value! 'idris-ffi-tensor-outer-torch (foreign-procedure \"tensor_outer_torch\" (void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-outer-torch) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__outerTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-bmm-torch)) (set-top-level-value! 'idris-ffi-tensor-bmm-torch (foreign-procedure \"tensor_bmm_torch\" (void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-bmm-torch) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__bmmTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-linear-2d-torch)) (set-top-level-value! 'idris-ffi-tensor-linear-2d-torch (foreign-procedure \"tensor_linear_2d_torch\" (void* void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-linear-2d-torch) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__linear2dTorch : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-sum-torch)) (set-top-level-value! 'idris-ffi-tensor-sum-torch (foreign-procedure \"tensor_sum_torch\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-sum-torch) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__sumTorch : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-mean-torch)) (set-top-level-value! 'idris-ffi-tensor-mean-torch (foreign-procedure \"tensor_mean_torch\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-mean-torch) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__meanTorch : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-min-torch)) (set-top-level-value! 'idris-ffi-tensor-min-torch (foreign-procedure \"tensor_min_torch\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-min-torch) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__tensorMinTorch : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-max-torch)) (set-top-level-value! 'idris-ffi-tensor-max-torch (foreign-procedure \"tensor_max_torch\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-max-torch) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__tensorMaxTorch : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-sum-dim-torch)) (set-top-level-value! 'idris-ffi-tensor-sum-dim-torch (foreign-procedure \"tensor_sum_dim_torch\" (void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-sum-dim-torch) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__sumDimTorch : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-select-torch)) (set-top-level-value! 'idris-ffi-tensor-select-torch (foreign-procedure \"tensor_select_torch\" (void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-select-torch) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__selectTorch : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-unsqueeze-torch)) (set-top-level-value! 'idris-ffi-tensor-unsqueeze-torch (foreign-procedure \"tensor_unsqueeze_torch\" (void* int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-unsqueeze-torch) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__unsqueezeTorch : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-squeeze-torch)) (set-top-level-value! 'idris-ffi-tensor-squeeze-torch (foreign-procedure \"tensor_squeeze_torch\" (void* int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-squeeze-torch) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__squeezeTorch : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-stack-torch)) (set-top-level-value! 'idris-ffi-tensor-stack-torch (foreign-procedure \"tensor_stack_torch\" (void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-stack-torch) a0 a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__stackTorch : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-view-1d-torch)) (set-top-level-value! 'idris-ffi-tensor-view-1d-torch (foreign-procedure \"tensor_view_1d_torch\" (void* int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-view-1d-torch) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__view1dTorch : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-view-2d-torch)) (set-top-level-value! 'idris-ffi-tensor-view-2d-torch (foreign-procedure \"tensor_view_2d_torch\" (void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-view-2d-torch) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__view2dTorch : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_reshape_1d_torch\" (void* int) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_torch\" (void*) void) raw_r) wr)))"
prim__reshape1dTorch : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-reshape-2d-torch)) (set-top-level-value! 'idris-ffi-tensor-reshape-2d-torch (foreign-procedure \"tensor_reshape_2d_torch\" (void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-reshape-2d-torch) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__reshape2dTorch : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3)  (when (not (top-level-bound? 'idris-ffi-tensor-reshape-3d-torch)) (set-top-level-value! 'idris-ffi-tensor-reshape-3d-torch (foreign-procedure \"tensor_reshape_3d_torch\" (void* int int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-reshape-3d-torch) (vector-ref a0 2) a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__reshape3dTorch : AnyPtr -> Int -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (when (not (top-level-bound? 'idris-ffi-tensor-reshape-4d-torch)) (set-top-level-value! 'idris-ffi-tensor-reshape-4d-torch (foreign-procedure \"tensor_reshape_4d_torch\" (void* int int int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-reshape-4d-torch) (vector-ref a0 2) a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__reshape4dTorch : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-tile-2d-torch)) (set-top-level-value! 'idris-ffi-tensor-tile-2d-torch (foreign-procedure \"tensor_tile_2d_torch\" (void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-tile-2d-torch) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__tile2dTorch : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3)  (when (not (top-level-bound? 'idris-ffi-tensor-narrow-torch)) (set-top-level-value! 'idris-ffi-tensor-narrow-torch (foreign-procedure \"tensor_narrow_torch\" (void* int int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-narrow-torch) (vector-ref a0 2) a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__narrowTorch : AnyPtr -> Int -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-transpose-last2-torch)) (set-top-level-value! 'idris-ffi-tensor-transpose-last2-torch (foreign-procedure \"tensor_transpose_last2_torch\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-transpose-last2-torch) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__transposeLast2Torch : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-transpose-2d-torch)) (set-top-level-value! 'idris-ffi-tensor-transpose-2d-torch (foreign-procedure \"tensor_transpose_2d_torch\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-transpose-2d-torch) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__transpose2dTorch : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-cat-torch)) (set-top-level-value! 'idris-ffi-tensor-cat-torch (foreign-procedure \"tensor_cat_torch\" (void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-cat-torch) a0 a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__catTorch : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-cat2-torch)) (set-top-level-value! 'idris-ffi-tensor-cat2-torch (foreign-procedure \"tensor_cat2_torch\" (void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-cat2-torch) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__cat2Torch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-concat-2d-axis1-torch)) (set-top-level-value! 'idris-ffi-tensor-concat-2d-axis1-torch (foreign-procedure \"tensor_concat_2d_axis1_torch\" (void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-concat-2d-axis1-torch) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__concat2dAxis1Torch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-gather-torch)) (set-top-level-value! 'idris-ffi-tensor-gather-torch (foreign-procedure \"tensor_gather_torch\" (void* void* int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-gather-torch) (vector-ref a0 2) (vector-ref a1 2) a2))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__gatherTorch : AnyPtr -> AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-scatter-add-torch)) (set-top-level-value! 'idris-ffi-tensor-scatter-add-torch (foreign-procedure \"tensor_scatter_add_torch\" (void* void* int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-scatter-add-torch) (vector-ref a0 2) (vector-ref a1 2) a2))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__scatterAddTorch : AnyPtr -> AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-argsort-torch)) (set-top-level-value! 'idris-ffi-tensor-argsort-torch (foreign-procedure \"tensor_argsort_torch\" (void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-argsort-torch) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__argsortTorch : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-cumprod-torch)) (set-top-level-value! 'idris-ffi-tensor-cumprod-torch (foreign-procedure \"tensor_cumprod_torch\" (void* int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-cumprod-torch) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__cumprodTorch : AnyPtr -> Int -> AnyPtr


public export
{d : TorchHwDev} -> UserExecutorLinear (TorchExecutor d) where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primArgsort = prim__argsortTorch
  primBmm = prim__bmmTorch
  primCat = prim__catTorch
  primCat2 = prim__cat2Torch
  primConcat2dAxis1 = prim__concat2dAxis1Torch
  primCumprod = prim__cumprodTorch
  primDot = prim__dotTorch
  primGather = prim__gatherTorch
  primLinear = prim__linearTorch
  primLinear2d = prim__linear2dTorch
  primMatmul = prim__matmulTorch
  primMean = prim__meanTorch
  primMm = prim__mmTorch
  primMv = prim__mvTorch
  primNarrow = prim__narrowTorch
  primOuter = prim__outerTorch
  primReshape1d = prim__reshape1dTorch
  primReshape2d = prim__reshape2dTorch
  primReshape3d = prim__reshape3dTorch
  primReshape4d = prim__reshape4dTorch
  primScatterAdd = prim__scatterAddTorch
  primSelect = prim__selectTorch
  primSqueeze = prim__squeezeTorch
  primStack = prim__stackTorch
  primSum = prim__sumTorch
  primSumDim = prim__sumDimTorch
  primTensorMax = prim__tensorMaxTorch
  primTensorMin = prim__tensorMinTorch
  primTile2d = prim__tile2dTorch
  primTranspose2d = prim__transpose2dTorch
  primTransposeLast2 = prim__transposeLast2Torch
  primUnsqueeze = prim__unsqueezeTorch
  primView1d = prim__view1dTorch
  primView2d = prim__view2dTorch
  -- <<< END GENERATED <<<


----------------------------------------------------------------------
-- NN-slice FFI bindings (torch-suffixed)
----------------------------------------------------------------------

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-gelu-torch)) (set-top-level-value! 'idris-ffi-tensor-gelu-torch (foreign-procedure \"tensor_gelu_torch\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-gelu-torch) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__geluTorch : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-leaky-relu-torch)) (set-top-level-value! 'idris-ffi-tensor-leaky-relu-torch (foreign-procedure \"tensor_leaky_relu_torch\" (void* double) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-leaky-relu-torch) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__leakyReluTorch : AnyPtr -> Double -> AnyPtr
%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-silu-torch)) (set-top-level-value! 'idris-ffi-tensor-silu-torch (foreign-procedure \"tensor_silu_torch\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-silu-torch) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__siluTorch : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-softplus-torch)) (set-top-level-value! 'idris-ffi-tensor-softplus-torch (foreign-procedure \"tensor_softplus_torch\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-softplus-torch) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__softplusTorch : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-softmax-torch)) (set-top-level-value! 'idris-ffi-tensor-softmax-torch (foreign-procedure \"tensor_softmax_torch\" (void* int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-softmax-torch) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__softmaxTorch : AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-log-softmax-torch)) (set-top-level-value! 'idris-ffi-tensor-log-softmax-torch (foreign-procedure \"tensor_log_softmax_torch\" (void* int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-log-softmax-torch) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__logSoftmaxTorch : AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-softmax-2d-torch)) (set-top-level-value! 'idris-ffi-tensor-softmax-2d-torch (foreign-procedure \"tensor_softmax_2d_torch\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-softmax-2d-torch) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__softmax2dTorch : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-log-softmax-2d-torch)) (set-top-level-value! 'idris-ffi-tensor-log-softmax-2d-torch (foreign-procedure \"tensor_log_softmax_2d_torch\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-log-softmax-2d-torch) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__logSoftmax2dTorch : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-softmax-3d-torch)) (set-top-level-value! 'idris-ffi-tensor-softmax-3d-torch (foreign-procedure \"tensor_softmax_3d_torch\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-softmax-3d-torch) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__softmax3dTorch : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-masked-fill-torch)) (set-top-level-value! 'idris-ffi-tensor-masked-fill-torch (foreign-procedure \"tensor_masked_fill_torch\" (void* void* double) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-masked-fill-torch) (vector-ref a0 2) (vector-ref a1 2) a2))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__maskedFillTorch : AnyPtr -> AnyPtr -> Double -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-expand-mask-torch)) (set-top-level-value! 'idris-ffi-tensor-expand-mask-torch (foreign-procedure \"tensor_expand_mask_torch\" (void* int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-expand-mask-torch) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__expandMaskTorch : AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3)  (when (not (top-level-bound? 'idris-ffi-tensor-layer-norm-2d-torch)) (set-top-level-value! 'idris-ffi-tensor-layer-norm-2d-torch (foreign-procedure \"tensor_layer_norm_2d_torch\" (void* void* void* double) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-layer-norm-2d-torch) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2) a3))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__layerNorm2dTorch : AnyPtr -> AnyPtr -> AnyPtr -> Double -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6 a7 a8 a9)  (when (not (top-level-bound? 'idris-ffi-tensor-batch-norm-torch)) (set-top-level-value! 'idris-ffi-tensor-batch-norm-torch (foreign-procedure \"tensor_batch_norm_torch\" (void* void* void* void* void* int int int double double) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-batch-norm-torch) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2) (vector-ref a3 2) (vector-ref a4 2) a5 a6 a7 a8 a9))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__batchNormTorch : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> Double -> Double -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3)  (when (not (top-level-bound? 'idris-ffi-tensor-dropout-torch)) (set-top-level-value! 'idris-ffi-tensor-dropout-torch (foreign-procedure \"tensor_dropout_torch\" (void* double int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-dropout-torch) (vector-ref a0 2) a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__dropoutTorch : AnyPtr -> Double -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3)  (when (not (top-level-bound? 'idris-ffi-tensor-embedding-torch)) (set-top-level-value! 'idris-ffi-tensor-embedding-torch (foreign-procedure \"tensor_embedding_torch\" (void* void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-embedding-torch) (vector-ref a0 2) (vector-ref a1 2) a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__embeddingTorch : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3)  (when (not (top-level-bound? 'idris-ffi-tensor-embedding-2d-torch)) (set-top-level-value! 'idris-ffi-tensor-embedding-2d-torch (foreign-procedure \"tensor_embedding_2d_torch\" (void* void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-embedding-2d-torch) (vector-ref a0 2) (vector-ref a1 2) a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__embedding2dTorch : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-cosine-similarity-torch)) (set-top-level-value! 'idris-ffi-tensor-cosine-similarity-torch (foreign-procedure \"tensor_cosine_similarity_torch\" (void* void* int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-cosine-similarity-torch) (vector-ref a0 2) (vector-ref a1 2) a2))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__cosineSimilarityTorch : AnyPtr -> AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (when (not (top-level-bound? 'idris-ffi-tensor-cross-attention-torch)) (set-top-level-value! 'idris-ffi-tensor-cross-attention-torch (foreign-procedure \"tensor_cross_attention_torch\" (void* void* void* void* double) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-cross-attention-torch) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2) (vector-ref a3 2) a4))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__crossAttentionTorch : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> Double -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-bce-with-logits-torch)) (set-top-level-value! 'idris-ffi-tensor-bce-with-logits-torch (foreign-procedure \"tensor_bce_with_logits_torch\" (void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-bce-with-logits-torch) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__bceWithLogitsTorch : AnyPtr -> AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3)  (when (not (top-level-bound? 'idris-ffi-tensor-gru-cell-torch)) (set-top-level-value! 'idris-ffi-tensor-gru-cell-torch (foreign-procedure \"tensor_gru_cell_torch\" (void* void* void* int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-gru-cell-torch) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2) a3))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__gruCellTorch : AnyPtr -> AnyPtr -> AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-lstm-gates-pair-torch)) (set-top-level-value! 'idris-ffi-tensor-lstm-gates-pair-torch (foreign-procedure \"tensor_lstm_gates_pair_torch\" (void* void* int) void*))) ((top-level-value 'idris-ffi-tensor-lstm-gates-pair-torch) (vector-ref a0 2) (vector-ref a1 2) a2))"
prim__lstmGatesPairTorch : AnyPtr -> AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-pair-first-torch)) (set-top-level-value! 'idris-ffi-tensor-pair-first-torch (foreign-procedure \"tensor_pair_first_torch\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-pair-first-torch) a0))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__pairFirstTorch : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-pair-second-torch)) (set-top-level-value! 'idris-ffi-tensor-pair-second-torch (foreign-procedure \"tensor_pair_second_torch\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-pair-second-torch) a0))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__pairSecondTorch : AnyPtr -> AnyPtr


-- Fused inference ops (used by `UserExecutorNN` below; FFI decls
-- moved up from the legacy `Training` slice region so they
-- precede their first use in the NN instance.)
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6)  (when (not (top-level-bound? 'idris-ffi-tensor-sdpa-2d-torch)) (set-top-level-value! 'idris-ffi-tensor-sdpa-2d-torch (foreign-procedure \"tensor_sdpa_2d_torch\" (void* void* void* int int int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-sdpa-2d-torch) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2) a3 a4 a5 a6))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__sdpa2dTorch : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-rms-norm-2d-torch)) (set-top-level-value! 'idris-ffi-tensor-rms-norm-2d-torch (foreign-procedure \"tensor_rms_norm_2d_torch\" (void* void* double) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-rms-norm-2d-torch) (vector-ref a0 2) (vector-ref a1 2) a2))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__rmsNorm2dTorch : AnyPtr -> AnyPtr -> Double -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-swiglu-2d-torch)) (set-top-level-value! 'idris-ffi-tensor-swiglu-2d-torch (foreign-procedure \"tensor_swiglu_2d_torch\" (void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-swiglu-2d-torch) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__swiGlu2dTorch : AnyPtr -> AnyPtr -> AnyPtr

public export
{d : TorchHwDev} -> UserExecutorNN (TorchExecutor d) where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primBatchNorm = prim__batchNormTorch
  primBceWithLogits = prim__bceWithLogitsTorch
  primCosineSimilarity = prim__cosineSimilarityTorch
  primCrossAttention = prim__crossAttentionTorch
  primDropout = prim__dropoutTorch
  primEmbedding = prim__embeddingTorch
  primEmbedding2d = prim__embedding2dTorch
  primExpandMask = prim__expandMaskTorch
  primGelu = prim__geluTorch
  primGruCell = prim__gruCellTorch
  primLayerNorm2d = prim__layerNorm2dTorch
  primLeakyRelu = prim__leakyReluTorch
  primLogSoftmax = prim__logSoftmaxTorch
  primLogSoftmax2d = prim__logSoftmax2dTorch
  primLstmGatesPair = prim__lstmGatesPairTorch
  primMaskedFill = prim__maskedFillTorch
  primPairFirst = prim__pairFirstTorch
  primPairSecond = prim__pairSecondTorch
  primRmsNorm2d = prim__rmsNorm2dTorch
  primSdpa2d = prim__sdpa2dTorch
  primSilu = prim__siluTorch
  primSoftmax = prim__softmaxTorch
  primSoftmax2d = prim__softmax2dTorch
  primSoftmax3d = prim__softmax3dTorch
  primSoftplus = prim__softplusTorch
  primSwiGlu2d = prim__swiGlu2dTorch
  -- <<< END GENERATED <<<


----------------------------------------------------------------------
-- Conv-slice FFI bindings (torch-suffixed)
----------------------------------------------------------------------

%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (when (not (top-level-bound? 'idris-ffi-tensor-conv1d-torch)) (set-top-level-value! 'idris-ffi-tensor-conv1d-torch (foreign-procedure \"tensor_conv1d_torch\" (void* void* void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-conv1d-torch) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2) a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__conv1dTorch : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-conv1d-circular-torch)) (set-top-level-value! 'idris-ffi-tensor-conv1d-circular-torch (foreign-procedure \"tensor_conv1d_circular_torch\" (void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-conv1d-circular-torch) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__conv1dCircularTorch : AnyPtr -> AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-avg-pool1d-torch)) (set-top-level-value! 'idris-ffi-tensor-avg-pool1d-torch (foreign-procedure \"tensor_avg_pool1d_torch\" (void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-avg-pool1d-torch) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__avgPool1dTorch : AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-max-pool1d-torch)) (set-top-level-value! 'idris-ffi-tensor-max-pool1d-torch (foreign-procedure \"tensor_max_pool1d_torch\" (void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-max-pool1d-torch) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__maxPool1dTorch : AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6)  (when (not (top-level-bound? 'idris-ffi-tensor-conv2d-torch)) (set-top-level-value! 'idris-ffi-tensor-conv2d-torch (foreign-procedure \"tensor_conv2d_torch\" (void* void* void* int int int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-conv2d-torch) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2) a3 a4 a5 a6))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__conv2dTorch : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6)  (when (not (top-level-bound? 'idris-ffi-tensor-conv2d-batched-torch)) (set-top-level-value! 'idris-ffi-tensor-conv2d-batched-torch (foreign-procedure \"tensor_conv2d_batched_torch\" (void* void* void* int int int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-conv2d-batched-torch) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2) a3 a4 a5 a6))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__conv2dBatchedTorch : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (when (not (top-level-bound? 'idris-ffi-tensor-avg-pool2d-torch)) (set-top-level-value! 'idris-ffi-tensor-avg-pool2d-torch (foreign-procedure \"tensor_avg_pool2d_torch\" (void* int int int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-avg-pool2d-torch) (vector-ref a0 2) a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__avgPool2dTorch : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (when (not (top-level-bound? 'idris-ffi-tensor-max-pool2d-torch)) (set-top-level-value! 'idris-ffi-tensor-max-pool2d-torch (foreign-procedure \"tensor_max_pool2d_torch\" (void* int int int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-max-pool2d-torch) (vector-ref a0 2) a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__maxPool2dTorch : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (when (not (top-level-bound? 'idris-ffi-tensor-max-pool2d-batched-torch)) (set-top-level-value! 'idris-ffi-tensor-max-pool2d-batched-torch (foreign-procedure \"tensor_max_pool2d_batched_torch\" (void* int int int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-max-pool2d-batched-torch) (vector-ref a0 2) a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__maxPool2dBatchedTorch : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr


public export
{d : TorchHwDev} -> UserExecutorConv (TorchExecutor d) where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primAvgPool1d = prim__avgPool1dTorch
  primAvgPool2d = prim__avgPool2dTorch
  primConv1d = prim__conv1dTorch
  primConv1dCircular = prim__conv1dCircularTorch
  primConv2d = prim__conv2dTorch
  primConv2dBatched = prim__conv2dBatchedTorch
  primMaxPool1d = prim__maxPool1dTorch
  primMaxPool2d = prim__maxPool2dTorch
  primMaxPool2dBatched = prim__maxPool2dBatchedTorch
  -- <<< END GENERATED <<<


----------------------------------------------------------------------
-- Tape-slice FFI bindings (torch-suffixed)
----------------------------------------------------------------------

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-requires-grad-torch)) (set-top-level-value! 'idris-ffi-tensor-requires-grad-torch (foreign-procedure \"tensor_requires_grad_torch\" (void*) int))) ((top-level-value 'idris-ffi-tensor-requires-grad-torch) (vector-ref a0 2)))"
prim__requiresGradTorch : AnyPtr -> Int
%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-set-requires-grad-torch)) (set-top-level-value! 'idris-ffi-tensor-set-requires-grad-torch (foreign-procedure \"tensor_set_requires_grad_torch\" (void* int) void))) ((top-level-value 'idris-ffi-tensor-set-requires-grad-torch) (vector-ref a0 2) a1))"
prim__setRequiresGradTorch : AnyPtr -> Int -> PrimIO ()
%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-backward-torch)) (set-top-level-value! 'idris-ffi-tensor-backward-torch (foreign-procedure \"tensor_backward_torch\" (void*) void))) ((top-level-value 'idris-ffi-tensor-backward-torch) (vector-ref a0 2)))"
prim__backwardTorch : AnyPtr -> PrimIO ()
%foreign "C:tensor_no_grad_begin_torch,libidrisml"
prim__noGradBeginTorch : PrimIO ()
%foreign "C:tensor_no_grad_end_torch,libidrisml"
prim__noGradEndTorch : PrimIO ()
%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-detach-torch)) (set-top-level-value! 'idris-ffi-tensor-detach-torch (foreign-procedure \"tensor_detach_torch\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-detach-torch) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__detachTorch : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-with-grad-torch)) (set-top-level-value! 'idris-ffi-tensor-with-grad-torch (foreign-procedure \"tensor_with_grad_torch\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-with-grad-torch) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__withGradTorch : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-dim-torch)) (set-top-level-value! 'idris-ffi-tensor-dim-torch (foreign-procedure \"tensor_dim_torch\" (void*) int))) ((top-level-value 'idris-ffi-tensor-dim-torch) (vector-ref a0 2)))"
prim__tensorDimTorch : AnyPtr -> Int
%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-size-torch)) (set-top-level-value! 'idris-ffi-tensor-size-torch (foreign-procedure \"tensor_size_torch\" (void* int) int))) ((top-level-value 'idris-ffi-tensor-size-torch) (vector-ref a0 2) a1))"
prim__tensorSizeAtTorch : AnyPtr -> Int -> Int
%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-param-register-return-torch)) (set-top-level-value! 'idris-ffi-param-register-return-torch (foreign-procedure \"param_register_return_torch\" (string void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-param-register-return-torch) a0 (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__paramRegisterTorch : String -> AnyPtr -> AnyPtr
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
%foreign "C:param_erase_by_prefix_torch,libidrisml"
prim__paramEraseByPrefixTorch : String -> PrimIO ()
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
%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (when (not (top-level-bound? 'idris-ffi-native-train-step-torch)) (set-top-level-value! 'idris-ffi-native-train-step-torch (foreign-procedure \"native_train_step_torch\" (void* int double void* double) double))) ((top-level-value 'idris-ffi-native-train-step-torch) a0 a1 a2 (vector-ref a3 2) a4))"
prim__nativeTrainStepTorch : AnyPtr -> Int -> Double -> AnyPtr -> Double -> Double
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5)  (when (not (top-level-bound? 'idris-ffi-native-train-step-scaled-torch)) (set-top-level-value! 'idris-ffi-native-train-step-scaled-torch (foreign-procedure \"native_train_step_scaled_torch\" (void* int double void* double double) double))) ((top-level-value 'idris-ffi-native-train-step-scaled-torch) a0 a1 a2 (vector-ref a3 2) a4 a5))"
prim__nativeTrainStepScaledTorch : AnyPtr -> Int -> Double -> AnyPtr -> Double -> Double -> Double
%foreign "C:param_save_torch,libidrisml"
prim__paramSaveTorch : String -> PrimIO Int
%foreign "C:param_save_by_name_torch,libidrisml"
prim__paramSaveByNameTorch : String -> String -> Int -> PrimIO Int
%foreign "C:param_save_by_name_renamed_torch,libidrisml"
prim__paramSaveByNameRenamedTorch : String -> String -> String -> Int -> PrimIO Int
%foreign "C:param_load_torch,libidrisml"
prim__paramLoadTorch : String -> PrimIO Int
%foreign "C:param_load_with_policy_torch,libidrisml"
prim__paramLoadWithPolicyTorch : String -> Int -> PrimIO Int
%foreign "C:param_load_with_prefix_torch,libidrisml"
prim__paramLoadWithPrefixTorch : String -> Int -> String -> PrimIO Int
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
%foreign "C:backend_release_all_persistent_torch,libidrisml"
prim__releaseAllPersistentTorch : PrimIO ()
%foreign "C:backend_reset_for_eval_torch,libidrisml"
prim__resetForEvalTorch : PrimIO ()
%foreign "C:tensor_live_count_torch,libidrisml"
prim__liveCountTorch : Int -> Int
%foreign "C:tensor_peak_live_count_torch,libidrisml"
prim__peakLiveCountTorch : Int -> Int
%foreign "C:tensor_perf_reset_torch,libidrisml"
prim__perfResetTorch : PrimIO ()
%foreign "C:tensor_perf_op_count_torch,libidrisml"
prim__perfOpCountTorch : PrimIO Int




%foreign "scheme:(lambda (a0 a1 a2 a3) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-scalar-streamed-torch)) (set-top-level-value! 'idris-ffi-tensor-create-scalar-streamed-torch (foreign-procedure \"tensor_create_scalar_streamed_torch\" (double int int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-scalar-streamed-torch) a0 a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__createScalarStreamedTorch : Double -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-streamed-torch)) (set-top-level-value! 'idris-ffi-tensor-create-streamed-torch (foreign-procedure \"tensor_create_streamed_torch\" (void* void* int int int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-streamed-torch) a0 a1 a2 a3 a4 a5))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__createStreamedTorch : AnyPtr -> AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-1d-streamed-torch)) (set-top-level-value! 'idris-ffi-tensor-create-1d-streamed-torch (foreign-procedure \"tensor_create_1d_streamed_torch\" (int void* int int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-1d-streamed-torch) a0 a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__create1dStreamedTorch : Int -> AnyPtr -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-2d-streamed-torch)) (set-top-level-value! 'idris-ffi-tensor-create-2d-streamed-torch (foreign-procedure \"tensor_create_2d_streamed_torch\" (int int void* int int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-2d-streamed-torch) a0 a1 a2 a3 a4 a5))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__create2dStreamedTorch : Int -> Int -> AnyPtr -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-param-1d-streamed-torch)) (set-top-level-value! 'idris-ffi-tensor-create-param-1d-streamed-torch (foreign-procedure \"tensor_create_param_1d_streamed_torch\" (int void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-param-1d-streamed-torch) a0 a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__createParam1dStreamedTorch : Int -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-param-2d-streamed-torch)) (set-top-level-value! 'idris-ffi-tensor-create-param-2d-streamed-torch (foreign-procedure \"tensor_create_param_2d_streamed_torch\" (int int void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-param-2d-streamed-torch) a0 a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__createParam2dStreamedTorch : Int -> Int -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-param-3d-streamed-torch)) (set-top-level-value! 'idris-ffi-tensor-create-param-3d-streamed-torch (foreign-procedure \"tensor_create_param_3d_streamed_torch\" (int int int void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-param-3d-streamed-torch) a0 a1 a2 a3 a4 a5))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__createParam3dStreamedTorch : Int -> Int -> Int -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-param-4d-streamed-torch)) (set-top-level-value! 'idris-ffi-tensor-create-param-4d-streamed-torch (foreign-procedure \"tensor_create_param_4d_streamed_torch\" (int int int int void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-param-4d-streamed-torch) a0 a1 a2 a3 a4 a5 a6))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__createParam4dStreamedTorch : Int -> Int -> Int -> Int -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-state-1d-streamed-torch)) (set-top-level-value! 'idris-ffi-tensor-create-state-1d-streamed-torch (foreign-procedure \"tensor_create_state_1d_streamed_torch\" (int void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-state-1d-streamed-torch) a0 a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__createState1dStreamedTorch : Int -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-state-2d-streamed-torch)) (set-top-level-value! 'idris-ffi-tensor-create-state-2d-streamed-torch (foreign-procedure \"tensor_create_state_2d_streamed_torch\" (int int void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-state-2d-streamed-torch) a0 a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__createState2dStreamedTorch : Int -> Int -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-cast-dtype-streamed-torch)) (set-top-level-value! 'idris-ffi-tensor-cast-dtype-streamed-torch (foreign-procedure \"tensor_cast_dtype_streamed_torch\" (void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-cast-dtype-streamed-torch) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__castStreamedTorch : AnyPtr -> Int -> Int -> AnyPtr

-- Fused param create + in-place init. C-side does both allocation
-- (torch::empty on host then migrate) AND the init kernel
-- (torch::nn::init::normal_ / t.fill_), bypassing the per-element
-- Idris sampler + per-element prim__setDouble FFI that dominated
-- model state construction. See `packages/backends/backend_torch/
-- training/dtype_init.cpp`.
%foreign "scheme:(lambda (a0 a1 a2 a3 a4) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-param-1d-normal-streamed-torch)) (set-top-level-value! 'idris-ffi-tensor-create-param-1d-normal-streamed-torch (foreign-procedure \"tensor_create_param_1d_normal_streamed_torch\" (int double double int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-param-1d-normal-streamed-torch) a0 a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__createParam1dNormalStreamedTorch : Int -> Double -> Double -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-param-2d-normal-streamed-torch)) (set-top-level-value! 'idris-ffi-tensor-create-param-2d-normal-streamed-torch (foreign-procedure \"tensor_create_param_2d_normal_streamed_torch\" (int int double double int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-param-2d-normal-streamed-torch) a0 a1 a2 a3 a4 a5))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__createParam2dNormalStreamedTorch : Int -> Int -> Double -> Double -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-param-3d-normal-streamed-torch)) (set-top-level-value! 'idris-ffi-tensor-create-param-3d-normal-streamed-torch (foreign-procedure \"tensor_create_param_3d_normal_streamed_torch\" (int int int double double int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-param-3d-normal-streamed-torch) a0 a1 a2 a3 a4 a5 a6))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__createParam3dNormalStreamedTorch : Int -> Int -> Int -> Double -> Double -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6 a7) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-param-4d-normal-streamed-torch)) (set-top-level-value! 'idris-ffi-tensor-create-param-4d-normal-streamed-torch (foreign-procedure \"tensor_create_param_4d_normal_streamed_torch\" (int int int int double double int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-param-4d-normal-streamed-torch) a0 a1 a2 a3 a4 a5 a6 a7))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__createParam4dNormalStreamedTorch : Int -> Int -> Int -> Int -> Double -> Double -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-param-1d-const-streamed-torch)) (set-top-level-value! 'idris-ffi-tensor-create-param-1d-const-streamed-torch (foreign-procedure \"tensor_create_param_1d_const_streamed_torch\" (int double int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-param-1d-const-streamed-torch) a0 a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__createParam1dConstStreamedTorch : Int -> Double -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-param-2d-const-streamed-torch)) (set-top-level-value! 'idris-ffi-tensor-create-param-2d-const-streamed-torch (foreign-procedure \"tensor_create_param_2d_const_streamed_torch\" (int int double int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-param-2d-const-streamed-torch) a0 a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__createParam2dConstStreamedTorch : Int -> Int -> Double -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-param-3d-const-streamed-torch)) (set-top-level-value! 'idris-ffi-tensor-create-param-3d-const-streamed-torch (foreign-procedure \"tensor_create_param_3d_const_streamed_torch\" (int int int double int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-param-3d-const-streamed-torch) a0 a1 a2 a3 a4 a5))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__createParam3dConstStreamedTorch : Int -> Int -> Int -> Double -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-param-4d-const-streamed-torch)) (set-top-level-value! 'idris-ffi-tensor-create-param-4d-const-streamed-torch (foreign-procedure \"tensor_create_param_4d_const_streamed_torch\" (int int int int double int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-param-4d-const-streamed-torch) a0 a1 a2 a3 a4 a5 a6))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__createParam4dConstStreamedTorch : Int -> Int -> Int -> Int -> Double -> Int -> Int -> AnyPtr
%foreign "C:tensor_set_init_seed_streamed_torch,libidrisml"
prim__setInitSeedStreamedTorch : Bits64 -> Int -> PrimIO ()

public export
{d : TorchHwDev} -> UserExecutorAutograd (TorchExecutor d) where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primBackward = prim__backwardTorch
  primDetach = prim__detachTorch
  primNoGradBegin = prim__noGradBeginTorch
  primNoGradEnd = prim__noGradEndTorch
  primRequiresGrad = prim__requiresGradTorch
  primSetRequiresGrad = prim__setRequiresGradTorch
  primWithGrad = prim__withGradTorch
  -- <<< END GENERATED <<<

public export
{d : TorchHwDev} -> UserExecutorParamRegistry (TorchExecutor d) where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primParamCount = prim__paramCountTorch
  primParamEraseByPrefix = prim__paramEraseByPrefixTorch
  primParamGradItemAt = prim__paramGradItemAtTorch
  primParamName = prim__paramNameTorch
  primParamRegister = prim__paramRegisterTorch
  primParamZeroAll = prim__paramZeroAllTorch
  primPolyakBlend = prim__polyakBlendTorch
  -- <<< END GENERATED <<<

public export
{d : TorchHwDev} -> UserExecutorOptimizer (TorchExecutor d) where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primNativeTrainStep = prim__nativeTrainStepTorch
  primNativeTrainStepScaled = prim__nativeTrainStepScaledTorch
  primOptimizerCreateAdam = prim__optimizerCreateAdamTorch
  primOptimizerCreateAdamGroup = prim__optimizerCreateAdamGroupTorch
  primOptimizerCreateAdamW = prim__optimizerCreateAdamWTorch
  primOptimizerCreateRmsprop = prim__optimizerCreateRmspropTorch
  primOptimizerCreateSgd = prim__optimizerCreateSgdTorch
  primOptimizerSetLr = prim__optimizerSetLrTorch
  primOptimizerSetParamLr = prim__optimizerSetParamLrTorch
  -- <<< END GENERATED <<<

public export
{d : TorchHwDev} -> UserExecutorSerialize (TorchExecutor d) where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primOptimizerLoad = prim__optimizerLoadTorch
  primOptimizerSave = prim__optimizerSaveTorch
  primParamLoad = prim__paramLoadTorch
  primParamLoadWithPolicy = prim__paramLoadWithPolicyTorch
  primParamLoadWithPrefix = prim__paramLoadWithPrefixTorch
  primParamSave = prim__paramSaveTorch
  primParamSaveByName = prim__paramSaveByNameTorch
  primParamSaveByNameRenamed = prim__paramSaveByNameRenamedTorch
  -- <<< END GENERATED <<<

public export
{d : TorchHwDev} -> UserExecutorProfiling (TorchExecutor d) where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primEpochBegin = prim__epochBeginTorch
  primEpochEnd = prim__epochEndTorch
  primLiveCount = prim__liveCountTorch
  primPeakLiveCount = prim__peakLiveCountTorch
  primPerfOpCount = prim__perfOpCountTorch
  primPerfReset = prim__perfResetTorch
  primProfileReport = prim__profileReportTorch
  primProfileReset = prim__profileResetTorch
  primReleaseAllPersistent = prim__releaseAllPersistentTorch
  primResetForEval = prim__resetForEvalTorch
  -- <<< END GENERATED <<<

public export
{d : TorchHwDev} -> UserExecutorTensorCreate (TorchExecutor d) where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primCastStreamed = prim__castStreamedTorch
  primCreate1dStreamed = prim__create1dStreamedTorch
  primCreate2dStreamed = prim__create2dStreamedTorch
  primCreateParam1dConstStreamed = prim__createParam1dConstStreamedTorch
  primCreateParam1dNormalStreamed = prim__createParam1dNormalStreamedTorch
  primCreateParam1dStreamed = prim__createParam1dStreamedTorch
  primCreateParam2dConstStreamed = prim__createParam2dConstStreamedTorch
  primCreateParam2dNormalStreamed = prim__createParam2dNormalStreamedTorch
  primCreateParam2dStreamed = prim__createParam2dStreamedTorch
  primCreateParam3dConstStreamed = prim__createParam3dConstStreamedTorch
  primCreateParam3dNormalStreamed = prim__createParam3dNormalStreamedTorch
  primCreateParam3dStreamed = prim__createParam3dStreamedTorch
  primCreateParam4dConstStreamed = prim__createParam4dConstStreamedTorch
  primCreateParam4dNormalStreamed = prim__createParam4dNormalStreamedTorch
  primCreateParam4dStreamed = prim__createParam4dStreamedTorch
  primCreateScalarStreamed = prim__createScalarStreamedTorch
  primCreateState1dStreamed = prim__createState1dStreamedTorch
  primCreateState2dStreamed = prim__createState2dStreamedTorch
  primCreateStreamed = prim__createStreamedTorch
  primItem2d = prim__item2dTorch
  primOneHot = prim__oneHotTorch
  primSetInitSeedStreamed = prim__setInitSeedStreamedTorch
  primTensorDim = prim__tensorDimTorch
  primTensorSizeAt = prim__tensorSizeAtTorch
  -- <<< END GENERATED <<<

public export
{d : TorchHwDev} -> UserExecutorTraining (TorchExecutor d) where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  -- <<< END GENERATED <<<
----------------------------------------------------------------------
-- Compatible (TorchExecutor, dt).
--
-- F32 is admitted on every hardware variant (CPU / MPS / CUDA), F64
-- on CPU and CUDA. **MPS + F64 is deliberately NOT compatible**:
-- libtorch's MPS backend rejects F64 tensor *construction* outright
-- (`Cannot convert a MPS Tensor to float64 dtype`), not just at op
-- dispatch — so admitting the combination would let the type
-- system mint a value the runtime can't represent. Users wanting
-- F64-precision on MPS hardware should pin to `(TorchExecutor TCpu) F64`
-- or `(TorchExecutor (TCuda n)) F64`. Mirrors the
-- `Compatible (MlxExecutor MGpu) F64`-rejection demo for mlx.
----------------------------------------------------------------------

public export
{d : TorchHwDev} -> Compatible (TorchExecutor d) F32 where

public export
Compatible (TorchExecutor TCpu) F64 where

public export
{n : Nat} -> Compatible (TorchExecutor (TCuda n)) F64 where

-- Inference-only dtypes (2026-05-22): BF16/F16/Int*/Bool on TCpu + TCuda.
-- MPS BF16 added 2026-05-28 (opt-in via TORCH_DTYPE=BF16 BuildConfig
-- cell). The earlier "MPS deliberately excluded" exclusion was retired
-- after the Llama-3.2-1B perf push showed BF16 storage halves the
-- memory footprint (5 GB → 2.5 GB) and the libtorch MPS backend has
-- shipped BF16 kernel coverage for the ops the HF forward exercises.
-- F16 + Int* / Bool on MPS stay excluded — F16's reduced-precision
-- training support is unproven, and Int* + Bool on MPS run into the
-- same construction-time rejection as F64 (Metal storage support is
-- per-version). Wiring is torch-only; tape/mlx have no instances.
public export
Compatible (TorchExecutor TCpu) BF16 where
public export
{n : Nat} -> Compatible (TorchExecutor (TCuda n)) BF16 where
public export
Compatible (TorchExecutor TMps) BF16 where
public export
Compatible (TorchExecutor TCpu) F16 where
public export
{n : Nat} -> Compatible (TorchExecutor (TCuda n)) F16 where
public export
Compatible (TorchExecutor TCpu) I8 where
public export
{n : Nat} -> Compatible (TorchExecutor (TCuda n)) I8 where
public export
Compatible (TorchExecutor TCpu) I16 where
public export
{n : Nat} -> Compatible (TorchExecutor (TCuda n)) I16 where
public export
Compatible (TorchExecutor TCpu) I32 where
public export
{n : Nat} -> Compatible (TorchExecutor (TCuda n)) I32 where
public export
Compatible (TorchExecutor TCpu) I64 where
public export
{n : Nat} -> Compatible (TorchExecutor (TCuda n)) I64 where
public export
Compatible (TorchExecutor TCpu) U8 where
public export
{n : Nat} -> Compatible (TorchExecutor (TCuda n)) U8 where
public export
Compatible (TorchExecutor TCpu) Bool where
public export
{n : Nat} -> Compatible (TorchExecutor (TCuda n)) Bool where

-- Sub-byte quantization dtypes (#411 BitNet b1.58). CPU + CUDA only —
-- libtorch MPS lacks the construction-side sub-byte storage routing
-- (mirrors the Int* / Bool MPS exclusion). The Idris-side Compatible
-- gate is the structural prereq; per-backend kernels arrive in B3.
public export
Compatible (TorchExecutor TCpu) Ternary where
public export
Compatible (TorchExecutor TMps) Ternary where
public export
{n : Nat} -> Compatible (TorchExecutor (TCuda n)) Ternary where
public export
Compatible (TorchExecutor TCpu) Binary where
public export
Compatible (TorchExecutor TMps) Binary where
public export
{n : Nat} -> Compatible (TorchExecutor (TCuda n)) Binary where


----------------------------------------------------------------------
-- UserExecutorTransfer instance (cross-backend transfer surface)
--
-- The torch hardware-migrate path is the only one that does real
-- work: `tensor_to_device_torch(handle, "mps"|"cuda:n")` migrates a
-- libtorch tensor in place between CPU, MPS, and CUDA without
-- allocating a fresh handle, preserving param-registry membership.
----------------------------------------------------------------------

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-to-doubles-torch)) (set-top-level-value! 'idris-ffi-tensor-to-doubles-torch (foreign-procedure \"tensor_to_doubles_torch\" (void* void*) void))) ((top-level-value 'idris-ffi-tensor-to-doubles-torch) (vector-ref a0 2) a1))"
prim__toHostTorch : AnyPtr -> AnyPtr -> AnyPtr

-- Host buffer helpers — unified across backends, see Executor/Tape.idr.
%foreign "C:tensor_alloc_doubles,libidrisml"
prim__allocHostTorch : Int -> AnyPtr

%foreign "C:tensor_free_doubles,libidrisml"
prim__freeHostTorch : AnyPtr -> PrimIO ()

%foreign "C:tensor_alloc_ints,libidrisml"
prim__allocIntHostTorch : Int -> AnyPtr

%foreign "C:tensor_free_ints,libidrisml"
prim__freeIntHostTorch : AnyPtr -> PrimIO ()

%foreign "C:tensor_write_int_return,libidrisml"
prim__setIntHostTorch : AnyPtr -> Int -> Int -> AnyPtr

||| Create from host data + auto-migrate to the target torch hw.
||| The closure here calls the rank-generic `tensor_create_torch`
||| (which lands on CPU by default in libtorch) then
||| `tensor_to_device_torch(handle, "mps"|"cuda:n")` so the returned
||| tensor is on the right hardware variant. The non-streamed
||| `prim__createTorch` FFI binding it wraps stays alive for this
||| purpose (the typeclass method that used to drive it from
||| `UserExecutorCore` was deleted as dead code).
prim__createFromHostTorch : (d : TorchHwDev) -> AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr
prim__createFromHostTorch d dat sh rank rg =
  prim__toDeviceTorch (prim__createTorch dat sh rank rg) (torchHwDevName d)

public export
{d : TorchHwDev} -> UserExecutorTransfer (TorchExecutor d) where
  backendTag         = "torch"
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primAllocHost = prim__allocHostTorch
  primAllocIntHost = prim__allocIntHostTorch
  primFreeHost = prim__freeHostTorch
  primFreeIntHost = prim__freeIntHostTorch
  primSetIntHost = prim__setIntHostTorch
  primToHost = prim__toHostTorch
  -- <<< END GENERATED <<<
  -- Hand-written overrides:
  primCreateFromHost = prim__createFromHostTorch d
  primIntraMigrate h hwName =
    prim__toDeviceTorch h hwName


----------------------------------------------------------------------
-- UserExecutorQuant instance (#411 BitNet b1.58)
----------------------------------------------------------------------
--
-- Torch unpacks the 2-bit codes to int8 at construction (storage is
-- `at::Tensor` with `at::ScalarType::Char`); the forward dequants
-- via `.to(scale.dtype())` then runs `at::matmul`. See
-- design-decisions.md "Per-backend ternary storage" + backend_torch/
-- nn/quantization/bitlinear.cpp.

%foreign "scheme:(lambda (a0 a1 a2 a3 a4) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-ternary-packed-2d-torch)) (set-top-level-value! 'idris-ffi-tensor-create-ternary-packed-2d-torch (foreign-procedure \"tensor_create_ternary_packed_2d_torch\" (void* int int int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-ternary-packed-2d-torch) a0 a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__createTernaryPacked2dTorch : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3)  (when (not (top-level-bound? 'idris-ffi-tensor-bitlinear-fwd-torch)) (set-top-level-value! 'idris-ffi-tensor-bitlinear-fwd-torch (foreign-procedure \"tensor_bitlinear_fwd_torch\" (void* void* void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-bitlinear-fwd-torch) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2) (vector-ref a3 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__bitlinearFwdTorch : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-absmean-per-row-2d-torch)) (set-top-level-value! 'idris-ffi-tensor-absmean-per-row-2d-torch (foreign-procedure \"tensor_absmean_per_row_2d_torch\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-absmean-per-row-2d-torch) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__absmeanPerRow2dTorch : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-ternary-quant-with-scale-2d-torch)) (set-top-level-value! 'idris-ffi-tensor-ternary-quant-with-scale-2d-torch (foreign-procedure \"tensor_ternary_quant_with_scale_2d_torch\" (void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-ternary-quant-with-scale-2d-torch) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__ternaryQuantWithScale2dTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-ternary-from-hf-packed-2d-torch)) (set-top-level-value! 'idris-ffi-tensor-create-ternary-from-hf-packed-2d-torch (foreign-procedure \"tensor_create_ternary_from_hf_packed_2d_torch\" (void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-ternary-from-hf-packed-2d-torch) a0 a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__createTernaryFromHfPacked2dTorch : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6)  (when (not (top-level-bound? 'idris-ffi-tensor-bitlinear-fwd-hf-quant-torch)) (set-top-level-value! 'idris-ffi-tensor-bitlinear-fwd-hf-quant-torch (foreign-procedure \"tensor_bitlinear_fwd_hf_quant_torch\" (void* double void* void* int void* double) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-torch)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-torch (foreign-procedure \"tensor_retain_handle_torch\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-bitlinear-fwd-hf-quant-torch) (vector-ref a0 2) a1 (vector-ref a2 2) (vector-ref a3 2) a4 (vector-ref a5 2) a6))) (let ((wr (vector 'tensor-handle-v2 \"torch\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-torch) raw_r) wr)))"
prim__bitlinearFwdHfQuantTorch : AnyPtr -> Double -> AnyPtr -> AnyPtr -> Int -> AnyPtr -> Double -> AnyPtr

public export
{d : TorchHwDev} -> UserExecutorQuant (TorchExecutor d) where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primAbsmeanPerRow2d = prim__absmeanPerRow2dTorch
  primBitlinearFwd = prim__bitlinearFwdTorch
  primBitlinearFwdHfQuant = prim__bitlinearFwdHfQuantTorch
  primCreateTernaryFromHfPacked2d = prim__createTernaryFromHfPacked2dTorch
  primCreateTernaryPacked2d = prim__createTernaryPacked2dTorch
  primTernaryQuantWithScale2d = prim__ternaryQuantWithScale2dTorch
  -- <<< END GENERATED <<<


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
