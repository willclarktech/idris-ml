||| Executor type + Core / Streamed / HardwareClassed instance
||| slices (lifecycle, elementwise arithmetic, stream tag, hw class).
module Executor.Tape.Core

import BackendLib
import DType.Core
import Executor.Core
import Hardware
import Preset

----------------------------------------------------------------------
-- Per-symbol bindings to the tape backend's suffixed C exports
----------------------------------------------------------------------

%foreign "scheme:(lambda (a0 a1) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-create-scalar-tape)) (set-top-level-value! 'idris-ffi-tensor-create-scalar-tape (foreign-procedure \"tensor_create_scalar_tape\" (double int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-create-scalar-tape) a0 a1))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__createScalarTape : Double -> Int -> AnyPtr

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
export
prim__item2dTape : AnyPtr -> Int -> Int -> Double
%foreign "scheme:(lambda (a0 a1 a2 a3) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (when (not (top-level-bound? 'idris-drain-once)) (when (not (top-level-bound? 'idris-release-cache)) (set-top-level-value! 'idris-release-cache (make-hashtable string-hash string=?))) (set-top-level-value! 'idris-drain-once (lambda () (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if (not d) #f (let ((tag (vector-ref d 1)) (raw (vector-ref d 2)) (cache (top-level-value 'idris-release-cache))) (let ((rel (or (hashtable-ref cache tag #f) (let ((sym (if (string=? tag \"primary\") \"tensor_release_handle\" (string-append \"tensor_release_handle_\" tag)))) (let ((fp (foreign-procedure sym (void*) void))) (hashtable-set! cache tag fp) fp))))) (rel raw) #t))))))) (when (not (top-level-bound? 'idris-ffi-tensor-one-hot-tape)) (set-top-level-value! 'idris-ffi-tensor-one-hot-tape (foreign-procedure \"tensor_one_hot_tape\" (void* int int int) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-one-hot-tape) a0 a1 a2 a3))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
export
prim__oneHotTape : AnyPtr -> Int -> Int -> Int -> AnyPtr

public export
UserExecutorCore TapeExecutor where
  deviceName       = "tape"
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primAbs          = prim__absTape
  primAdd          = prim__addTape
  primAddScalar    = prim__addScalarTape
  primClamp        = prim__clampTape
  primClampMin     = prim__clampMinTape
  primClone        = prim__cloneTape
  primCreateScalar = prim__createScalarTape
  primDiv          = prim__divTape
  primExp          = prim__expTape
  primFree         = prim__freeTape
  primItem         = prim__itemTape
  primItem1d       = prim__item1dTape
  primLog          = prim__logTape
  primMul          = prim__mulTape
  primMulScalar    = prim__mulScalarTape
  primNeg          = prim__negTape
  primPow          = prim__powTape
  primRound        = prim__roundTape
  primSigmoid      = prim__sigmoidTape
  primSqrt         = prim__sqrtTape
  primSub          = prim__subTape
  primTanh         = prim__tanhTape
  -- <<< END GENERATED <<<

public export
UserExecutorStreamed TapeExecutor where
  deviceStreamTag = 0

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
