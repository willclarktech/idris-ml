||| `TapeExecutor` — `UserExecutorCore` instance for the tape backend.
|||
||| Forwards to the tape-suffixed C symbols emitted under Phase 1's
||| `rename_tape.h` (e.g. `tensor_add_tape`). Only resolvable at
||| runtime if the build's BACKEND list includes `tape`.
module Executor.Tape

import Executor.Core
import DType.Core
import Backend
import Hardware
import Preset


----------------------------------------------------------------------
-- Per-symbol bindings to the tape backend's suffixed C exports
----------------------------------------------------------------------

%foreign "scheme:(lambda (a0 a1) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-scalar-tape)) (set-top-level-value! 'idris-ffi-tensor-create-scalar-tape (foreign-procedure \"tensor_create_scalar_tape\" (double int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-scalar-tape) a0 a1))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__createScalarTape : Double -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-tape)) (set-top-level-value! 'idris-ffi-tensor-create-tape (foreign-procedure \"tensor_create_tape\" (void* void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-tape) a0 a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__createTape : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-free-tape)) (set-top-level-value! 'idris-ffi-tensor-free-tape (foreign-procedure \"tensor_free_tape\" (void*) void))) ((top-level-value 'idris-ffi-tensor-free-tape) (vector-ref a0 2)))"
prim__freeTape : AnyPtr -> ()

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-item-tape)) (set-top-level-value! 'idris-ffi-tensor-item-tape (foreign-procedure \"tensor_item_tape\" (void*) double))) ((top-level-value 'idris-ffi-tensor-item-tape) (vector-ref a0 2)))"
prim__itemTape : AnyPtr -> Double

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-item-1d-tape)) (set-top-level-value! 'idris-ffi-tensor-item-1d-tape (foreign-procedure \"tensor_item_1d_tape\" (void* int) double))) ((top-level-value 'idris-ffi-tensor-item-1d-tape) (vector-ref a0 2) a1))"
prim__item1dTape : AnyPtr -> Int -> Double

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-clone-tape)) (set-top-level-value! 'idris-ffi-tensor-clone-tape (foreign-procedure \"tensor_clone_tape\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-clone-tape) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__cloneTape : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-add-tape)) (set-top-level-value! 'idris-ffi-tensor-add-tape (foreign-procedure \"tensor_add_tape\" (void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-add-tape) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__addTape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-sub-tape)) (set-top-level-value! 'idris-ffi-tensor-sub-tape (foreign-procedure \"tensor_sub_tape\" (void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-sub-tape) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__subTape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-mul-tape)) (set-top-level-value! 'idris-ffi-tensor-mul-tape (foreign-procedure \"tensor_mul_tape\" (void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-mul-tape) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__mulTape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-div-tape)) (set-top-level-value! 'idris-ffi-tensor-div-tape (foreign-procedure \"tensor_div_tape\" (void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-div-tape) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__divTape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-neg-tape)) (set-top-level-value! 'idris-ffi-tensor-neg-tape (foreign-procedure \"tensor_neg_tape\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-neg-tape) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__negTape : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-abs-tape)) (set-top-level-value! 'idris-ffi-tensor-abs-tape (foreign-procedure \"tensor_abs_tape\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-abs-tape) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__absTape : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-exp-tape)) (set-top-level-value! 'idris-ffi-tensor-exp-tape (foreign-procedure \"tensor_exp_tape\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-exp-tape) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__expTape : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-log-tape)) (set-top-level-value! 'idris-ffi-tensor-log-tape (foreign-procedure \"tensor_log_tape\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-log-tape) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__logTape : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-sqrt-tape)) (set-top-level-value! 'idris-ffi-tensor-sqrt-tape (foreign-procedure \"tensor_sqrt_tape\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-sqrt-tape) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__sqrtTape : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-pow-tape)) (set-top-level-value! 'idris-ffi-tensor-pow-tape (foreign-procedure \"tensor_pow_tape\" (void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-pow-tape) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__powTape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-sigmoid-tape)) (set-top-level-value! 'idris-ffi-tensor-sigmoid-tape (foreign-procedure \"tensor_sigmoid_tape\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-sigmoid-tape) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__sigmoidTape : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-tanh-tape)) (set-top-level-value! 'idris-ffi-tensor-tanh-tape (foreign-procedure \"tensor_tanh_tape\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-tanh-tape) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__tanhTape : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-add-scalar-tape)) (set-top-level-value! 'idris-ffi-tensor-add-scalar-tape (foreign-procedure \"tensor_add_scalar_tape\" (void* double) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-add-scalar-tape) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__addScalarTape : AnyPtr -> Double -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-mul-scalar-tape)) (set-top-level-value! 'idris-ffi-tensor-mul-scalar-tape (foreign-procedure \"tensor_mul_scalar_tape\" (void* double) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-mul-scalar-tape) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__mulScalarTape : AnyPtr -> Double -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-clamp-min-tape)) (set-top-level-value! 'idris-ffi-tensor-clamp-min-tape (foreign-procedure \"tensor_clamp_min_tape\" (void* double) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-clamp-min-tape) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__clampMinTape : AnyPtr -> Double -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-clamp-tape)) (set-top-level-value! 'idris-ffi-tensor-clamp-tape (foreign-procedure \"tensor_clamp_tape\" (void* double double) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-clamp-tape) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__clampTape : AnyPtr -> Double -> Double -> AnyPtr

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-round-tape)) (set-top-level-value! 'idris-ffi-tensor-round-tape (foreign-procedure \"tensor_round_tape\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-round-tape) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__roundTape : AnyPtr -> AnyPtr


----------------------------------------------------------------------
-- TapeExecutor type + UserExecutorCore instance
----------------------------------------------------------------------

||| The tape backend's `UserExecutorCore` instance head. An empty type
||| — it has no values; `Tensor [..] TapeExecutor` is just a typed tag for
||| "this tensor lives on the tape backend".
public export
data TapeExecutor : Type where MkTapeExecutor : TapeExecutor

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-item-2d-tape)) (set-top-level-value! 'idris-ffi-tensor-item-2d-tape (foreign-procedure \"tensor_item_2d_tape\" (void* int int) double))) ((top-level-value 'idris-ffi-tensor-item-2d-tape) (vector-ref a0 2) a1 a2))"
prim__item2dTape : AnyPtr -> Int -> Int -> Double
%foreign "scheme:(lambda (a0 a1 a2) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-mnist-get-image-tape)) (set-top-level-value! 'idris-ffi-mnist-get-image-tape (foreign-procedure \"mnist_get_image_tape\" (void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-mnist-get-image-tape) a0 a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__mnistGetImageTape : AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-one-hot-tape)) (set-top-level-value! 'idris-ffi-tensor-one-hot-tape (foreign-procedure \"tensor_one_hot_tape\" (void* int int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-one-hot-tape) a0 a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__oneHotTape : AnyPtr -> Int -> Int -> Int -> AnyPtr

public export
UserExecutorCore TapeExecutor where
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
  primClamp        = prim__clampTape
  primRound        = prim__roundTape

----------------------------------------------------------------------
-- Linear-slice FFI bindings (tape-suffixed)
----------------------------------------------------------------------

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-mv-tape)) (set-top-level-value! 'idris-ffi-tensor-mv-tape (foreign-procedure \"tensor_mv_tape\" (void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-mv-tape) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__mvTape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-mm-tape)) (set-top-level-value! 'idris-ffi-tensor-mm-tape (foreign-procedure \"tensor_mm_tape\" (void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-mm-tape) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__mmTape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-matmul-tape)) (set-top-level-value! 'idris-ffi-tensor-matmul-tape (foreign-procedure \"tensor_matmul_tape\" (void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-matmul-tape) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__matmulTape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-linear-tape)) (set-top-level-value! 'idris-ffi-tensor-linear-tape (foreign-procedure \"tensor_linear_tape\" (void* void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-linear-tape) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__linearTape : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-dot-tape)) (set-top-level-value! 'idris-ffi-tensor-dot-tape (foreign-procedure \"tensor_dot_tape\" (void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-dot-tape) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__dotTape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-outer-tape)) (set-top-level-value! 'idris-ffi-tensor-outer-tape (foreign-procedure \"tensor_outer_tape\" (void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-outer-tape) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__outerTape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-bmm-tape)) (set-top-level-value! 'idris-ffi-tensor-bmm-tape (foreign-procedure \"tensor_bmm_tape\" (void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-bmm-tape) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__bmmTape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-linear-2d-tape)) (set-top-level-value! 'idris-ffi-tensor-linear-2d-tape (foreign-procedure \"tensor_linear_2d_tape\" (void* void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-linear-2d-tape) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__linear2dTape : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-sum-tape)) (set-top-level-value! 'idris-ffi-tensor-sum-tape (foreign-procedure \"tensor_sum_tape\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-sum-tape) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__sumTape : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-mean-tape)) (set-top-level-value! 'idris-ffi-tensor-mean-tape (foreign-procedure \"tensor_mean_tape\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-mean-tape) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__meanTape : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-min-tape)) (set-top-level-value! 'idris-ffi-tensor-min-tape (foreign-procedure \"tensor_min_tape\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-min-tape) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__tensorMinTape : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-max-tape)) (set-top-level-value! 'idris-ffi-tensor-max-tape (foreign-procedure \"tensor_max_tape\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-max-tape) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__tensorMaxTape : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-sum-dim-tape)) (set-top-level-value! 'idris-ffi-tensor-sum-dim-tape (foreign-procedure \"tensor_sum_dim_tape\" (void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-sum-dim-tape) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__sumDimTape : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-select-tape)) (set-top-level-value! 'idris-ffi-tensor-select-tape (foreign-procedure \"tensor_select_tape\" (void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-select-tape) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__selectTape : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-unsqueeze-tape)) (set-top-level-value! 'idris-ffi-tensor-unsqueeze-tape (foreign-procedure \"tensor_unsqueeze_tape\" (void* int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-unsqueeze-tape) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__unsqueezeTape : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-squeeze-tape)) (set-top-level-value! 'idris-ffi-tensor-squeeze-tape (foreign-procedure \"tensor_squeeze_tape\" (void* int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-squeeze-tape) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__squeezeTape : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-stack-tape)) (set-top-level-value! 'idris-ffi-tensor-stack-tape (foreign-procedure \"tensor_stack_tape\" (void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-stack-tape) a0 a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__stackTape : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-view-1d-tape)) (set-top-level-value! 'idris-ffi-tensor-view-1d-tape (foreign-procedure \"tensor_view_1d_tape\" (void* int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-view-1d-tape) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__view1dTape : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-view-2d-tape)) (set-top-level-value! 'idris-ffi-tensor-view-2d-tape (foreign-procedure \"tensor_view_2d_tape\" (void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-view-2d-tape) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__view2dTape : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_reshape_1d_tape\" (void* int) void*) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle_tape\" (void*) void) raw_r) wr)))"
prim__reshape1dTape : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-reshape-2d-tape)) (set-top-level-value! 'idris-ffi-tensor-reshape-2d-tape (foreign-procedure \"tensor_reshape_2d_tape\" (void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-reshape-2d-tape) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__reshape2dTape : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3)  (when (not (top-level-bound? 'idris-ffi-tensor-reshape-3d-tape)) (set-top-level-value! 'idris-ffi-tensor-reshape-3d-tape (foreign-procedure \"tensor_reshape_3d_tape\" (void* int int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-reshape-3d-tape) (vector-ref a0 2) a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__reshape3dTape : AnyPtr -> Int -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (when (not (top-level-bound? 'idris-ffi-tensor-reshape-4d-tape)) (set-top-level-value! 'idris-ffi-tensor-reshape-4d-tape (foreign-procedure \"tensor_reshape_4d_tape\" (void* int int int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-reshape-4d-tape) (vector-ref a0 2) a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__reshape4dTape : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-tile-2d-tape)) (set-top-level-value! 'idris-ffi-tensor-tile-2d-tape (foreign-procedure \"tensor_tile_2d_tape\" (void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-tile-2d-tape) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__tile2dTape : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3)  (when (not (top-level-bound? 'idris-ffi-tensor-narrow-tape)) (set-top-level-value! 'idris-ffi-tensor-narrow-tape (foreign-procedure \"tensor_narrow_tape\" (void* int int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-narrow-tape) (vector-ref a0 2) a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__narrowTape : AnyPtr -> Int -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-transpose-last2-tape)) (set-top-level-value! 'idris-ffi-tensor-transpose-last2-tape (foreign-procedure \"tensor_transpose_last2_tape\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-transpose-last2-tape) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__transposeLast2Tape : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-transpose-2d-tape)) (set-top-level-value! 'idris-ffi-tensor-transpose-2d-tape (foreign-procedure \"tensor_transpose_2d_tape\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-transpose-2d-tape) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__transpose2dTape : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-cat-tape)) (set-top-level-value! 'idris-ffi-tensor-cat-tape (foreign-procedure \"tensor_cat_tape\" (void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-cat-tape) a0 a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__catTape : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-cat2-tape)) (set-top-level-value! 'idris-ffi-tensor-cat2-tape (foreign-procedure \"tensor_cat2_tape\" (void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-cat2-tape) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__cat2Tape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-concat-2d-axis1-tape)) (set-top-level-value! 'idris-ffi-tensor-concat-2d-axis1-tape (foreign-procedure \"tensor_concat_2d_axis1_tape\" (void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-concat-2d-axis1-tape) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__concat2dAxis1Tape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-gather-tape)) (set-top-level-value! 'idris-ffi-tensor-gather-tape (foreign-procedure \"tensor_gather_tape\" (void* void* int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-gather-tape) (vector-ref a0 2) (vector-ref a1 2) a2))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__gatherTape : AnyPtr -> AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-scatter-add-tape)) (set-top-level-value! 'idris-ffi-tensor-scatter-add-tape (foreign-procedure \"tensor_scatter_add_tape\" (void* void* int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-scatter-add-tape) (vector-ref a0 2) (vector-ref a1 2) a2))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__scatterAddTape : AnyPtr -> AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-argsort-tape)) (set-top-level-value! 'idris-ffi-tensor-argsort-tape (foreign-procedure \"tensor_argsort_tape\" (void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-argsort-tape) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__argsortTape : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-cumprod-tape)) (set-top-level-value! 'idris-ffi-tensor-cumprod-tape (foreign-procedure \"tensor_cumprod_tape\" (void* int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-cumprod-tape) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__cumprodTape : AnyPtr -> Int -> AnyPtr


public export
UserExecutorLinear TapeExecutor where
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

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-gelu-tape)) (set-top-level-value! 'idris-ffi-tensor-gelu-tape (foreign-procedure \"tensor_gelu_tape\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-gelu-tape) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__geluTape : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-leaky-relu-tape)) (set-top-level-value! 'idris-ffi-tensor-leaky-relu-tape (foreign-procedure \"tensor_leaky_relu_tape\" (void* double) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-leaky-relu-tape) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__leakyReluTape : AnyPtr -> Double -> AnyPtr
%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-silu-tape)) (set-top-level-value! 'idris-ffi-tensor-silu-tape (foreign-procedure \"tensor_silu_tape\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-silu-tape) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__siluTape : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-softplus-tape)) (set-top-level-value! 'idris-ffi-tensor-softplus-tape (foreign-procedure \"tensor_softplus_tape\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-softplus-tape) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__softplusTape : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-softmax-tape)) (set-top-level-value! 'idris-ffi-tensor-softmax-tape (foreign-procedure \"tensor_softmax_tape\" (void* int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-softmax-tape) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__softmaxTape : AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-log-softmax-tape)) (set-top-level-value! 'idris-ffi-tensor-log-softmax-tape (foreign-procedure \"tensor_log_softmax_tape\" (void* int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-log-softmax-tape) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__logSoftmaxTape : AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-softmax-2d-tape)) (set-top-level-value! 'idris-ffi-tensor-softmax-2d-tape (foreign-procedure \"tensor_softmax_2d_tape\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-softmax-2d-tape) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__softmax2dTape : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-log-softmax-2d-tape)) (set-top-level-value! 'idris-ffi-tensor-log-softmax-2d-tape (foreign-procedure \"tensor_log_softmax_2d_tape\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-log-softmax-2d-tape) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__logSoftmax2dTape : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-softmax-3d-tape)) (set-top-level-value! 'idris-ffi-tensor-softmax-3d-tape (foreign-procedure \"tensor_softmax_3d_tape\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-softmax-3d-tape) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__softmax3dTape : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-masked-fill-tape)) (set-top-level-value! 'idris-ffi-tensor-masked-fill-tape (foreign-procedure \"tensor_masked_fill_tape\" (void* void* double) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-masked-fill-tape) (vector-ref a0 2) (vector-ref a1 2) a2))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__maskedFillTape : AnyPtr -> AnyPtr -> Double -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-expand-mask-tape)) (set-top-level-value! 'idris-ffi-tensor-expand-mask-tape (foreign-procedure \"tensor_expand_mask_tape\" (void* int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-expand-mask-tape) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__expandMaskTape : AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3)  (when (not (top-level-bound? 'idris-ffi-tensor-layer-norm-2d-tape)) (set-top-level-value! 'idris-ffi-tensor-layer-norm-2d-tape (foreign-procedure \"tensor_layer_norm_2d_tape\" (void* void* void* double) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-layer-norm-2d-tape) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2) a3))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__layerNorm2dTape : AnyPtr -> AnyPtr -> AnyPtr -> Double -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6 a7 a8 a9)  (when (not (top-level-bound? 'idris-ffi-tensor-batch-norm-tape)) (set-top-level-value! 'idris-ffi-tensor-batch-norm-tape (foreign-procedure \"tensor_batch_norm_tape\" (void* void* void* void* void* int int int double double) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-batch-norm-tape) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2) (vector-ref a3 2) (vector-ref a4 2) a5 a6 a7 a8 a9))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__batchNormTape : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> Double -> Double -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3)  (when (not (top-level-bound? 'idris-ffi-tensor-dropout-tape)) (set-top-level-value! 'idris-ffi-tensor-dropout-tape (foreign-procedure \"tensor_dropout_tape\" (void* double int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-dropout-tape) (vector-ref a0 2) a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__dropoutTape : AnyPtr -> Double -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3)  (when (not (top-level-bound? 'idris-ffi-tensor-embedding-tape)) (set-top-level-value! 'idris-ffi-tensor-embedding-tape (foreign-procedure \"tensor_embedding_tape\" (void* void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-embedding-tape) (vector-ref a0 2) (vector-ref a1 2) a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__embeddingTape : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3)  (when (not (top-level-bound? 'idris-ffi-tensor-embedding-2d-tape)) (set-top-level-value! 'idris-ffi-tensor-embedding-2d-tape (foreign-procedure \"tensor_embedding_2d_tape\" (void* void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-embedding-2d-tape) (vector-ref a0 2) (vector-ref a1 2) a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__embedding2dTape : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-cosine-similarity-tape)) (set-top-level-value! 'idris-ffi-tensor-cosine-similarity-tape (foreign-procedure \"tensor_cosine_similarity_tape\" (void* void* int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-cosine-similarity-tape) (vector-ref a0 2) (vector-ref a1 2) a2))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__cosineSimilarityTape : AnyPtr -> AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (when (not (top-level-bound? 'idris-ffi-tensor-cross-attention-tape)) (set-top-level-value! 'idris-ffi-tensor-cross-attention-tape (foreign-procedure \"tensor_cross_attention_tape\" (void* void* void* void* double) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-cross-attention-tape) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2) (vector-ref a3 2) a4))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__crossAttentionTape : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> Double -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-bce-with-logits-tape)) (set-top-level-value! 'idris-ffi-tensor-bce-with-logits-tape (foreign-procedure \"tensor_bce_with_logits_tape\" (void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-bce-with-logits-tape) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__bceWithLogitsTape : AnyPtr -> AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3)  (when (not (top-level-bound? 'idris-ffi-tensor-gru-cell-tape)) (set-top-level-value! 'idris-ffi-tensor-gru-cell-tape (foreign-procedure \"tensor_gru_cell_tape\" (void* void* void* int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-gru-cell-tape) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2) a3))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__gruCellTape : AnyPtr -> AnyPtr -> AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-lstm-gates-pair-tape)) (set-top-level-value! 'idris-ffi-tensor-lstm-gates-pair-tape (foreign-procedure \"tensor_lstm_gates_pair_tape\" (void* void* int) void*))) ((top-level-value 'idris-ffi-tensor-lstm-gates-pair-tape) (vector-ref a0 2) (vector-ref a1 2) a2))"
prim__lstmGatesPairTape : AnyPtr -> AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-pair-first-tape)) (set-top-level-value! 'idris-ffi-tensor-pair-first-tape (foreign-procedure \"tensor_pair_first_tape\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-pair-first-tape) a0))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__pairFirstTape : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-pair-second-tape)) (set-top-level-value! 'idris-ffi-tensor-pair-second-tape (foreign-procedure \"tensor_pair_second_tape\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-pair-second-tape) a0))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__pairSecondTape : AnyPtr -> AnyPtr


-- Fused inference ops (used by `UserExecutorNN` below; FFI decls
-- moved up from the legacy `Training` slice region so they
-- precede their first use in the NN instance.)
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6)  (when (not (top-level-bound? 'idris-ffi-tensor-sdpa-2d-tape)) (set-top-level-value! 'idris-ffi-tensor-sdpa-2d-tape (foreign-procedure \"tensor_sdpa_2d_tape\" (void* void* void* int int int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-sdpa-2d-tape) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2) a3 a4 a5 a6))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__sdpa2dTape : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-rms-norm-2d-tape)) (set-top-level-value! 'idris-ffi-tensor-rms-norm-2d-tape (foreign-procedure \"tensor_rms_norm_2d_tape\" (void* void* double) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-rms-norm-2d-tape) (vector-ref a0 2) (vector-ref a1 2) a2))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__rmsNorm2dTape : AnyPtr -> AnyPtr -> Double -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-swiglu-2d-tape)) (set-top-level-value! 'idris-ffi-tensor-swiglu-2d-tape (foreign-procedure \"tensor_swiglu_2d_tape\" (void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-swiglu-2d-tape) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__swiGlu2dTape : AnyPtr -> AnyPtr -> AnyPtr

public export
UserExecutorNN TapeExecutor where
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
  primEmbedding2d      = prim__embedding2dTape
  primCosineSimilarity = prim__cosineSimilarityTape
  primCrossAttention   = prim__crossAttentionTape
  primBceWithLogits    = prim__bceWithLogitsTape
  primGruCell          = prim__gruCellTape
  primLstmGatesPair    = prim__lstmGatesPairTape
  primPairFirst        = prim__pairFirstTape
  primPairSecond       = prim__pairSecondTape

  -- Fused inference ops (lifted from legacy Training slice)
  primSdpa2d                   = prim__sdpa2dTape
  primRmsNorm2d                = prim__rmsNorm2dTape
  primSwiGlu2d                 = prim__swiGlu2dTape


----------------------------------------------------------------------
-- Conv-slice FFI bindings (tape-suffixed)
----------------------------------------------------------------------

%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (when (not (top-level-bound? 'idris-ffi-tensor-conv1d-tape)) (set-top-level-value! 'idris-ffi-tensor-conv1d-tape (foreign-procedure \"tensor_conv1d_tape\" (void* void* void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-conv1d-tape) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2) a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__conv1dTape : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-conv1d-circular-tape)) (set-top-level-value! 'idris-ffi-tensor-conv1d-circular-tape (foreign-procedure \"tensor_conv1d_circular_tape\" (void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-conv1d-circular-tape) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__conv1dCircularTape : AnyPtr -> AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-avg-pool1d-tape)) (set-top-level-value! 'idris-ffi-tensor-avg-pool1d-tape (foreign-procedure \"tensor_avg_pool1d_tape\" (void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-avg-pool1d-tape) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__avgPool1dTape : AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (when (not (top-level-bound? 'idris-ffi-tensor-max-pool1d-tape)) (set-top-level-value! 'idris-ffi-tensor-max-pool1d-tape (foreign-procedure \"tensor_max_pool1d_tape\" (void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-max-pool1d-tape) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__maxPool1dTape : AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6)  (when (not (top-level-bound? 'idris-ffi-tensor-conv2d-tape)) (set-top-level-value! 'idris-ffi-tensor-conv2d-tape (foreign-procedure \"tensor_conv2d_tape\" (void* void* void* int int int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-conv2d-tape) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2) a3 a4 a5 a6))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__conv2dTape : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6)  (when (not (top-level-bound? 'idris-ffi-tensor-conv2d-batched-tape)) (set-top-level-value! 'idris-ffi-tensor-conv2d-batched-tape (foreign-procedure \"tensor_conv2d_batched_tape\" (void* void* void* int int int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-conv2d-batched-tape) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2) a3 a4 a5 a6))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__conv2dBatchedTape : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (when (not (top-level-bound? 'idris-ffi-tensor-avg-pool2d-tape)) (set-top-level-value! 'idris-ffi-tensor-avg-pool2d-tape (foreign-procedure \"tensor_avg_pool2d_tape\" (void* int int int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-avg-pool2d-tape) (vector-ref a0 2) a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__avgPool2dTape : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (when (not (top-level-bound? 'idris-ffi-tensor-max-pool2d-tape)) (set-top-level-value! 'idris-ffi-tensor-max-pool2d-tape (foreign-procedure \"tensor_max_pool2d_tape\" (void* int int int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-max-pool2d-tape) (vector-ref a0 2) a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__maxPool2dTape : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (when (not (top-level-bound? 'idris-ffi-tensor-max-pool2d-batched-tape)) (set-top-level-value! 'idris-ffi-tensor-max-pool2d-batched-tape (foreign-procedure \"tensor_max_pool2d_batched_tape\" (void* int int int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-max-pool2d-batched-tape) (vector-ref a0 2) a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__maxPool2dBatchedTape : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr


public export
UserExecutorConv TapeExecutor where
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

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-requires-grad-tape)) (set-top-level-value! 'idris-ffi-tensor-requires-grad-tape (foreign-procedure \"tensor_requires_grad_tape\" (void*) int))) ((top-level-value 'idris-ffi-tensor-requires-grad-tape) (vector-ref a0 2)))"
prim__requiresGradTape : AnyPtr -> Int
%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-set-requires-grad-tape)) (set-top-level-value! 'idris-ffi-tensor-set-requires-grad-tape (foreign-procedure \"tensor_set_requires_grad_tape\" (void* int) void))) ((top-level-value 'idris-ffi-tensor-set-requires-grad-tape) (vector-ref a0 2) a1))"
prim__setRequiresGradTape : AnyPtr -> Int -> PrimIO ()
%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-backward-tape)) (set-top-level-value! 'idris-ffi-tensor-backward-tape (foreign-procedure \"tensor_backward_tape\" (void*) void))) ((top-level-value 'idris-ffi-tensor-backward-tape) (vector-ref a0 2)))"
prim__backwardTape : AnyPtr -> PrimIO ()
%foreign "C:tensor_no_grad_begin_tape,libidrisml"
prim__noGradBeginTape : PrimIO ()
%foreign "C:tensor_no_grad_end_tape,libidrisml"
prim__noGradEndTape : PrimIO ()
%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-detach-tape)) (set-top-level-value! 'idris-ffi-tensor-detach-tape (foreign-procedure \"tensor_detach_tape\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-detach-tape) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__detachTape : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-with-grad-tape)) (set-top-level-value! 'idris-ffi-tensor-with-grad-tape (foreign-procedure \"tensor_with_grad_tape\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-with-grad-tape) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__withGradTape : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-dim-tape)) (set-top-level-value! 'idris-ffi-tensor-dim-tape (foreign-procedure \"tensor_dim_tape\" (void*) int))) ((top-level-value 'idris-ffi-tensor-dim-tape) (vector-ref a0 2)))"
prim__tensorDimTape : AnyPtr -> Int
%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-size-tape)) (set-top-level-value! 'idris-ffi-tensor-size-tape (foreign-procedure \"tensor_size_tape\" (void* int) int))) ((top-level-value 'idris-ffi-tensor-size-tape) (vector-ref a0 2) a1))"
prim__tensorSizeAtTape : AnyPtr -> Int -> Int
%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-param-register-return-tape)) (set-top-level-value! 'idris-ffi-param-register-return-tape (foreign-procedure \"param_register_return_tape\" (string void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-param-register-return-tape) a0 (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__paramRegisterTape : String -> AnyPtr -> AnyPtr
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
%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (when (not (top-level-bound? 'idris-ffi-native-train-step-tape)) (set-top-level-value! 'idris-ffi-native-train-step-tape (foreign-procedure \"native_train_step_tape\" (void* int double void* double) double))) ((top-level-value 'idris-ffi-native-train-step-tape) a0 a1 a2 (vector-ref a3 2) a4))"
prim__nativeTrainStepTape : AnyPtr -> Int -> Double -> AnyPtr -> Double -> Double
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5)  (when (not (top-level-bound? 'idris-ffi-native-train-step-scaled-tape)) (set-top-level-value! 'idris-ffi-native-train-step-scaled-tape (foreign-procedure \"native_train_step_scaled_tape\" (void* int double void* double double) double))) ((top-level-value 'idris-ffi-native-train-step-scaled-tape) a0 a1 a2 (vector-ref a3 2) a4 a5))"
prim__nativeTrainStepScaledTape : AnyPtr -> Int -> Double -> AnyPtr -> Double -> Double -> Double
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
%foreign "C:backend_release_all_persistent_tape,libidrisml"
prim__releaseAllPersistentTape : PrimIO ()
%foreign "C:backend_reset_for_eval_tape,libidrisml"
prim__resetForEvalTape : PrimIO ()
%foreign "C:tensor_live_count_tape,libidrisml"
prim__liveCountTape : Int -> Int
%foreign "C:tensor_peak_live_count_tape,libidrisml"
prim__peakLiveCountTape : Int -> Int
%foreign "C:tensor_perf_reset_tape,libidrisml"
prim__perfResetTape : PrimIO ()
%foreign "C:tensor_perf_op_count_tape,libidrisml"
prim__perfOpCountTape : PrimIO Int




%foreign "scheme:(lambda (a0 a1 a2 a3) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-scalar-streamed-tape)) (set-top-level-value! 'idris-ffi-tensor-create-scalar-streamed-tape (foreign-procedure \"tensor_create_scalar_streamed_tape\" (double int int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-scalar-streamed-tape) a0 a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__createScalarStreamedTape : Double -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-streamed-tape)) (set-top-level-value! 'idris-ffi-tensor-create-streamed-tape (foreign-procedure \"tensor_create_streamed_tape\" (void* void* int int int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-streamed-tape) a0 a1 a2 a3 a4 a5))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__createStreamedTape : AnyPtr -> AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-1d-streamed-tape)) (set-top-level-value! 'idris-ffi-tensor-create-1d-streamed-tape (foreign-procedure \"tensor_create_1d_streamed_tape\" (int void* int int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-1d-streamed-tape) a0 a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__create1dStreamedTape : Int -> AnyPtr -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-2d-streamed-tape)) (set-top-level-value! 'idris-ffi-tensor-create-2d-streamed-tape (foreign-procedure \"tensor_create_2d_streamed_tape\" (int int void* int int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-2d-streamed-tape) a0 a1 a2 a3 a4 a5))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__create2dStreamedTape : Int -> Int -> AnyPtr -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-param-1d-streamed-tape)) (set-top-level-value! 'idris-ffi-tensor-create-param-1d-streamed-tape (foreign-procedure \"tensor_create_param_1d_streamed_tape\" (int void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-param-1d-streamed-tape) a0 a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__createParam1dStreamedTape : Int -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-param-2d-streamed-tape)) (set-top-level-value! 'idris-ffi-tensor-create-param-2d-streamed-tape (foreign-procedure \"tensor_create_param_2d_streamed_tape\" (int int void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-param-2d-streamed-tape) a0 a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__createParam2dStreamedTape : Int -> Int -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-param-3d-streamed-tape)) (set-top-level-value! 'idris-ffi-tensor-create-param-3d-streamed-tape (foreign-procedure \"tensor_create_param_3d_streamed_tape\" (int int int void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-param-3d-streamed-tape) a0 a1 a2 a3 a4 a5))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__createParam3dStreamedTape : Int -> Int -> Int -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-param-4d-streamed-tape)) (set-top-level-value! 'idris-ffi-tensor-create-param-4d-streamed-tape (foreign-procedure \"tensor_create_param_4d_streamed_tape\" (int int int int void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-param-4d-streamed-tape) a0 a1 a2 a3 a4 a5 a6))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__createParam4dStreamedTape : Int -> Int -> Int -> Int -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-state-1d-streamed-tape)) (set-top-level-value! 'idris-ffi-tensor-create-state-1d-streamed-tape (foreign-procedure \"tensor_create_state_1d_streamed_tape\" (int void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-state-1d-streamed-tape) a0 a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__createState1dStreamedTape : Int -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-state-2d-streamed-tape)) (set-top-level-value! 'idris-ffi-tensor-create-state-2d-streamed-tape (foreign-procedure \"tensor_create_state_2d_streamed_tape\" (int int void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-state-2d-streamed-tape) a0 a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__createState2dStreamedTape : Int -> Int -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-cast-dtype-streamed-tape)) (set-top-level-value! 'idris-ffi-tensor-cast-dtype-streamed-tape (foreign-procedure \"tensor_cast_dtype_streamed_tape\" (void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-cast-dtype-streamed-tape) (vector-ref a0 2) a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__castStreamedTape : AnyPtr -> Int -> Int -> AnyPtr

-- Fused param create + in-place init. Tape's C-side port slots for
-- these stay nullptr until Phase 7 lands the batched-fill impl; the
-- shared trampoline in `dtype_streamed.c` aborts loud with a clear
-- "this backend hasn't wired the fused-init port methods yet" message
-- if any of these is called. Idris declares the FFIs so the typeclass
-- instance type-checks; calls trip the C-side abort, not a silent
-- mis-dispatch.
%foreign "scheme:(lambda (a0 a1 a2 a3 a4) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-param-1d-normal-streamed-tape)) (set-top-level-value! 'idris-ffi-tensor-create-param-1d-normal-streamed-tape (foreign-procedure \"tensor_create_param_1d_normal_streamed_tape\" (int double double int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-param-1d-normal-streamed-tape) a0 a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__createParam1dNormalStreamedTape : Int -> Double -> Double -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-param-2d-normal-streamed-tape)) (set-top-level-value! 'idris-ffi-tensor-create-param-2d-normal-streamed-tape (foreign-procedure \"tensor_create_param_2d_normal_streamed_tape\" (int int double double int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-param-2d-normal-streamed-tape) a0 a1 a2 a3 a4 a5))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__createParam2dNormalStreamedTape : Int -> Int -> Double -> Double -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-param-3d-normal-streamed-tape)) (set-top-level-value! 'idris-ffi-tensor-create-param-3d-normal-streamed-tape (foreign-procedure \"tensor_create_param_3d_normal_streamed_tape\" (int int int double double int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-param-3d-normal-streamed-tape) a0 a1 a2 a3 a4 a5 a6))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__createParam3dNormalStreamedTape : Int -> Int -> Int -> Double -> Double -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6 a7) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-param-4d-normal-streamed-tape)) (set-top-level-value! 'idris-ffi-tensor-create-param-4d-normal-streamed-tape (foreign-procedure \"tensor_create_param_4d_normal_streamed_tape\" (int int int int double double int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-param-4d-normal-streamed-tape) a0 a1 a2 a3 a4 a5 a6 a7))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__createParam4dNormalStreamedTape : Int -> Int -> Int -> Int -> Double -> Double -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-param-1d-const-streamed-tape)) (set-top-level-value! 'idris-ffi-tensor-create-param-1d-const-streamed-tape (foreign-procedure \"tensor_create_param_1d_const_streamed_tape\" (int double int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-param-1d-const-streamed-tape) a0 a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__createParam1dConstStreamedTape : Int -> Double -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-param-2d-const-streamed-tape)) (set-top-level-value! 'idris-ffi-tensor-create-param-2d-const-streamed-tape (foreign-procedure \"tensor_create_param_2d_const_streamed_tape\" (int int double int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-param-2d-const-streamed-tape) a0 a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__createParam2dConstStreamedTape : Int -> Int -> Double -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-param-3d-const-streamed-tape)) (set-top-level-value! 'idris-ffi-tensor-create-param-3d-const-streamed-tape (foreign-procedure \"tensor_create_param_3d_const_streamed_tape\" (int int int double int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-param-3d-const-streamed-tape) a0 a1 a2 a3 a4 a5))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__createParam3dConstStreamedTape : Int -> Int -> Int -> Double -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-param-4d-const-streamed-tape)) (set-top-level-value! 'idris-ffi-tensor-create-param-4d-const-streamed-tape (foreign-procedure \"tensor_create_param_4d_const_streamed_tape\" (int int int int double int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-param-4d-const-streamed-tape) a0 a1 a2 a3 a4 a5 a6))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__createParam4dConstStreamedTape : Int -> Int -> Int -> Int -> Double -> Int -> Int -> AnyPtr
%foreign "C:tensor_set_init_seed_streamed_tape,libidrisml"
prim__setInitSeedStreamedTape : Bits64 -> Int -> PrimIO ()

public export
UserExecutorAutograd TapeExecutor where
  primRequiresGrad         = prim__requiresGradTape
  primSetRequiresGrad      = prim__setRequiresGradTape
  primBackward             = prim__backwardTape
  primNoGradBegin          = prim__noGradBeginTape
  primNoGradEnd            = prim__noGradEndTape
  primDetach               = prim__detachTape
  primWithGrad             = prim__withGradTape

public export
UserExecutorParamRegistry TapeExecutor where
  primParamRegister        = prim__paramRegisterTape
  primPolyakBlend          = prim__polyakBlendTape
  primParamCount           = prim__paramCountTape
  primParamName            = prim__paramNameTape
  primParamGradItemAt      = prim__paramGradItemAtTape
  primParamZeroAll         = prim__paramZeroAllTape

public export
UserExecutorOptimizer TapeExecutor where
  primOptimizerCreateSgd       = prim__optimizerCreateSgdTape
  primOptimizerCreateRmsprop   = prim__optimizerCreateRmspropTape
  primOptimizerCreateAdam      = prim__optimizerCreateAdamTape
  primOptimizerCreateAdamGroup = prim__optimizerCreateAdamGroupTape
  primOptimizerCreateAdamW     = prim__optimizerCreateAdamWTape
  primOptimizerSetLr           = prim__optimizerSetLrTape
  primOptimizerSetParamLr      = prim__optimizerSetParamLrTape
  primNativeTrainStep          = prim__nativeTrainStepTape
  primNativeTrainStepScaled    = prim__nativeTrainStepScaledTape

public export
UserExecutorSerialize TapeExecutor where
  primParamSave                = prim__paramSaveTape
  primParamLoad                = prim__paramLoadTape
  primParamLoadWithPolicy      = prim__paramLoadWithPolicyTape
  primOptimizerSave            = prim__optimizerSaveTape
  primOptimizerLoad            = prim__optimizerLoadTape

public export
UserExecutorProfiling TapeExecutor where
  primProfileReset             = prim__profileResetTape
  primProfileReport            = prim__profileReportTape
  primEpochBegin               = prim__epochBeginTape
  primEpochEnd                 = prim__epochEndTape
  primReleaseAllPersistent     = prim__releaseAllPersistentTape
  primResetForEval             = prim__resetForEvalTape
  primLiveCount                = prim__liveCountTape
  primPeakLiveCount            = prim__peakLiveCountTape
  primPerfReset                = prim__perfResetTape
  primPerfOpCount              = prim__perfOpCountTape

public export
UserExecutorTensorCreate TapeExecutor where
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
  primCreateParam1dNormalStreamed = prim__createParam1dNormalStreamedTape
  primCreateParam2dNormalStreamed = prim__createParam2dNormalStreamedTape
  primCreateParam3dNormalStreamed = prim__createParam3dNormalStreamedTape
  primCreateParam4dNormalStreamed = prim__createParam4dNormalStreamedTape
  primCreateParam1dConstStreamed  = prim__createParam1dConstStreamedTape
  primCreateParam2dConstStreamed  = prim__createParam2dConstStreamedTape
  primCreateParam3dConstStreamed  = prim__createParam3dConstStreamedTape
  primCreateParam4dConstStreamed  = prim__createParam4dConstStreamedTape
  primSetInitSeedStreamed         = prim__setInitSeedStreamedTape
  primTensorDim            = prim__tensorDimTape
  primTensorSizeAt         = prim__tensorSizeAtTape
  primItem2d               = prim__item2dTape
  primMnistGetImage        = prim__mnistGetImageTape
  primOneHot               = prim__oneHotTape

public export
UserExecutorTraining TapeExecutor where
----------------------------------------------------------------------
-- UserExecutorTransfer instance (cross-backend transfer surface)
--
-- Tape lives entirely on host CPU; there are no hardware variants
-- to switch between, so `primIntraMigrate` is a literal no-op (the
-- C-side `tensor_to_device_tape` returns the input handle as-is).
----------------------------------------------------------------------

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-to-doubles-tape)) (set-top-level-value! 'idris-ffi-tensor-to-doubles-tape (foreign-procedure \"tensor_to_doubles_tape\" (void* void*) void))) ((top-level-value 'idris-ffi-tensor-to-doubles-tape) (vector-ref a0 2) a1))"
prim__toHostTape : AnyPtr -> AnyPtr -> AnyPtr

-- The host buffer helpers (alloc / free / write-return for doubles
-- and ints) are byte-identical across all three backends, so they
-- live as unified definitions in `packages/backends/shared_utils.c`.
-- All three `UserExecutorTransfer` instances bind through the same
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

%foreign "scheme:(lambda (a0 a1 a2 a3) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-tape)) (set-top-level-value! 'idris-ffi-tensor-create-tape (foreign-procedure \"tensor_create_tape\" (void* void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-tape) a0 a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__createFromHostTape : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-to-device-tape)) (set-top-level-value! 'idris-ffi-tensor-to-device-tape (foreign-procedure \"tensor_to_device_tape\" (void* string) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-to-device-tape) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__intraMigrateTape : AnyPtr -> String -> AnyPtr

public export
UserExecutorTransfer TapeExecutor where
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
-- UserExecutorQuant instance (#411 BitNet b1.58)
----------------------------------------------------------------------
--
-- Tape stores ternary weights as packed 2-bit codes (4 values/byte)
-- — the pioneering sub-byte storage path. See design-decisions.md
-- "Per-backend ternary storage" + backend_tape/nn/quantization/
-- bitlinear.c for the layout + decode loop.

%foreign "scheme:(lambda (a0 a1 a2 a3 a4) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-ternary-packed-2d-tape)) (set-top-level-value! 'idris-ffi-tensor-create-ternary-packed-2d-tape (foreign-procedure \"tensor_create_ternary_packed_2d_tape\" (void* int int int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-ternary-packed-2d-tape) a0 a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__createTernaryPacked2dTape : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3)  (when (not (top-level-bound? 'idris-ffi-tensor-bitlinear-fwd-tape)) (set-top-level-value! 'idris-ffi-tensor-bitlinear-fwd-tape (foreign-procedure \"tensor_bitlinear_fwd_tape\" (void* void* void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-bitlinear-fwd-tape) (vector-ref a0 2) (vector-ref a1 2) (vector-ref a2 2) (vector-ref a3 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__bitlinearFwdTape : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (when (not (top-level-bound? 'idris-ffi-tensor-absmean-per-row-2d-tape)) (set-top-level-value! 'idris-ffi-tensor-absmean-per-row-2d-tape (foreign-procedure \"tensor_absmean_per_row_2d_tape\" (void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-absmean-per-row-2d-tape) (vector-ref a0 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__absmeanPerRow2dTape : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-ternary-quant-with-scale-2d-tape)) (set-top-level-value! 'idris-ffi-tensor-ternary-quant-with-scale-2d-tape (foreign-procedure \"tensor_ternary_quant_with_scale_2d_tape\" (void* void*) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-ternary-quant-with-scale-2d-tape) (vector-ref a0 2) (vector-ref a1 2)))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__ternaryQuantWithScale2dTape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-ternary-from-hf-packed-2d-tape)) (set-top-level-value! 'idris-ffi-tensor-create-ternary-from-hf-packed-2d-tape (foreign-procedure \"tensor_create_ternary_from_hf_packed_2d_tape\" (void* int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-ternary-from-hf-packed-2d-tape) a0 a1 a2))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__createTernaryFromHfPacked2dTape : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6)  (when (not (top-level-bound? 'idris-ffi-tensor-bitlinear-fwd-hf-quant-tape)) (set-top-level-value! 'idris-ffi-tensor-bitlinear-fwd-hf-quant-tape (foreign-procedure \"tensor_bitlinear_fwd_hf_quant_tape\" (void* double void* void* int void* double) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-bitlinear-fwd-hf-quant-tape) (vector-ref a0 2) a1 (vector-ref a2 2) (vector-ref a3 2) a4 (vector-ref a5 2) a6))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__bitlinearFwdHfQuantTape : AnyPtr -> Double -> AnyPtr -> AnyPtr -> Int -> AnyPtr -> Double -> AnyPtr

public export
UserExecutorQuant TapeExecutor where
  primCreateTernaryPacked2d       = prim__createTernaryPacked2dTape
  primBitlinearFwd                = prim__bitlinearFwdTape
  primBitlinearFwdHfQuant         = prim__bitlinearFwdHfQuantTape
  primAbsmeanPerRow2d             = prim__absmeanPerRow2dTape
  primTernaryQuantWithScale2d     = prim__ternaryQuantWithScale2dTape
  primCreateTernaryFromHfPacked2d = prim__createTernaryFromHfPacked2dTape


----------------------------------------------------------------------
-- Compatible (TapeExecutor, dt).
--
-- F64 + F32 are trainable: every public `tensor_*` kernel routes F32
-- through `tape_load_d` / `make_tensor_arena_f32` after Phase 3 + 3b.
-- The inference-only dtypes (BF16/F16/I8/I16/I32/I64/U8/Bool) store
-- packed bytes via the `double` lingua franca in `tape_round_to_dtype`
-- + the lifted bf16/f16 bit helpers (Phase 4 step 2); they never enter
-- a kernel that does arithmetic, only create/cast/readout. The
-- I64-magnitudes-above-2^53 caveat applies to tape's lingua-franca
-- path the same way it applies to safetensors I/O (documented, not
-- fixed).
----------------------------------------------------------------------

public export
Compatible TapeExecutor F64 where

public export
Compatible TapeExecutor F32 where

-- Inference-only dtypes — storage + cast + readout, never trainable.
public export
Compatible TapeExecutor BF16 where
public export
Compatible TapeExecutor F16 where
public export
Compatible TapeExecutor I8 where
public export
Compatible TapeExecutor I16 where
public export
Compatible TapeExecutor I32 where
public export
Compatible TapeExecutor I64 where
public export
Compatible TapeExecutor U8 where
public export
Compatible TapeExecutor Bool where

-- Sub-byte quantization dtypes (#411 BitNet b1.58). Storage + pack/
-- unpack are backend-agnostic (`shared_utils.c::ternary_pack`); per-
-- backend kernels land in B3. Tape is the pioneering arena for sub-
-- byte storage — `DT_TERNARY` / `DT_BINARY` are reserved in tape's
-- internal enum but no kernel writes them in B1.
public export
Compatible TapeExecutor Ternary where
public export
Compatible TapeExecutor Binary where


----------------------------------------------------------------------
-- HardwareClass: the tape backend runs on the host CPU.
----------------------------------------------------------------------

public export
HardwareClassed TapeExecutor where
  hardwareClass = HostCpu


----------------------------------------------------------------------
-- Hardware (type-level): tape runs on Cpu.
----------------------------------------------------------------------

public export
RunsOn TapeExecutor Cpu where


----------------------------------------------------------------------
-- Backend (type-level): tape is provided by TapeBackend.
----------------------------------------------------------------------

public export
RunsVia TapeExecutor TapeBackend where


----------------------------------------------------------------------
-- Preset: tape on Cpu defaults to TapeExecutor + F64.
-- Tape on AppleGpu / Cuda is intentionally undeclared — picking those
-- with PRIMARY=tape becomes a "no instance Preset TapeBackend …"
-- compile error at the example.
----------------------------------------------------------------------

public export
Preset TapeBackend Cpu where
  presetExecutor = TapeExecutor
  presetDType    = F64
