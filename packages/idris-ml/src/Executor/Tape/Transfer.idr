||| Cross-backend transfer + quantization instance slices.
module Executor.Tape.Transfer

import BackendLib
import DType.Core
import Executor.Core
import public Executor.Tape.Training
import Hardware
import Preset

----------------------------------------------------------------------
-- UserExecutorTransfer instance (cross-backend transfer surface)
--
-- Tape lives entirely on host CPU; there are no hardware variants
-- to switch between, so `primIntraMigrate` is a literal no-op (the
-- C-side `tensor_to_device_tape` returns the input handle as-is).
----------------------------------------------------------------------

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-to-doubles-return-tape)) (set-top-level-value! 'idris-ffi-tensor-to-doubles-return-tape (foreign-procedure \"tensor_to_doubles_return_tape\" (void* void*) void*))) ((top-level-value 'idris-ffi-tensor-to-doubles-return-tape) (vector-ref a0 2) a1))"
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

||| Dtag-aware create-from-host: delegates to the dtag-dispatch
||| `prim__createStreamedTape` (stream pinned to 0 — tape has no
||| stream concept) so destination storage matches the type-level
||| `dt` instead of unconditionally constructing F64.
prim__createFromHostTape : AnyPtr -> AnyPtr -> Int -> Int -> Int -> AnyPtr
prim__createFromHostTape dat sh rank rg dtag =
  prim__createStreamedTape dat sh rank rg 0 dtag

%foreign "scheme:(lambda (a0 a1)  (when (not (top-level-bound? 'idris-ffi-tensor-to-device-tape)) (set-top-level-value! 'idris-ffi-tensor-to-device-tape (foreign-procedure \"tensor_to_device_tape\" (void* string) void*))) (when (not (top-level-bound? 'idris-ffi-tensor-retain-handle-tape)) (set-top-level-value! 'idris-ffi-tensor-retain-handle-tape (foreign-procedure \"tensor_retain_handle_tape\" (void*) void))) (let ((raw_r ((top-level-value 'idris-ffi-tensor-to-device-tape) (vector-ref a0 2) a1))) (let ((wr (vector 'tensor-handle-v2 \"tape\" raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((top-level-value 'idris-ffi-tensor-retain-handle-tape) raw_r) wr)))"
prim__intraMigrateTape : AnyPtr -> String -> AnyPtr

public export
UserExecutorTransfer TapeExecutor where
  backendTag         = "tape"
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primAllocHost    = prim__allocHostTape
  primAllocIntHost = prim__allocIntHostTape
  primFreeHost     = prim__freeHostTape
  primFreeIntHost  = prim__freeIntHostTape
  primIntraMigrate = prim__intraMigrateTape
  primSetIntHost   = prim__setIntHostTape
  primToHost       = prim__toHostTape
  -- <<< END GENERATED <<<
  -- Hand-written overrides:
  primCreateFromHost = prim__createFromHostTape

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
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  primAbsmeanPerRow2d             = prim__absmeanPerRow2dTape
  primBitlinearFwd                = prim__bitlinearFwdTape
  primBitlinearFwdHfQuant         = prim__bitlinearFwdHfQuantTape
  primCreateTernaryFromHfPacked2d = prim__createTernaryFromHfPacked2dTape
  primCreateTernaryPacked2d       = prim__createTernaryPacked2dTape
  primTernaryQuantWithScale2d     = prim__ternaryQuantWithScale2dTape
  -- <<< END GENERATED <<<
