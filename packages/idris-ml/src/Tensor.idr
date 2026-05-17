module Tensor

import Data.List
import Data.Maybe
import Data.SortedMap
import Data.Vect
import Compat.Random

import DataPoint
import Device
import public DType.Core
import public GradMode
import Floating
import Array
import Util


----------------------------------------------------------------------
-- Backend FFI (libtorch via libidrisml)
----------------------------------------------------------------------

-- ----------------------------------------------------------------------
-- Managed-handle plumbing (Chez guardian + foreign-procedure drain).
-- See docs/develop/tensor-lifecycle-plan.md for the design.
--
-- Every Tensor-returning FFI's Scheme wrapper wraps the C return in a
-- Chez vector (`#(tensor-handle raw_ptr)`) and registers it with a
-- top-level guardian. The vector IS the Tensor's runtime identity in
-- Chez — the Idris-Chez compiler can't elide it without eliding the
-- value itself. When the wrap becomes GC-unreachable,
-- prim__drainManagedHandles pops it and calls tensor_release_handle on
-- the raw pointer.
--
-- Idris does not distinguish "raw AnyPtr" from "wrapped AnyPtr" at the
-- type level — both are AnyPtr. Every %foreign "scheme:..." wrapper
-- internally `vector-ref`s its Tensor args to extract the raw pointer
-- before calling the C function. The wrap layer is invisible to Idris.
-- ----------------------------------------------------------------------

-- Self-init: creates the guardian and ensures libidrisml is loaded with
-- RTLD_GLOBAL so subsequent `foreign-procedure` lookups for C symbols
-- (tensor_retain_handle, tensor_release_handle, mlx_set_gc_drain_callback)
-- succeed. The %foreign "C:..." declarations elsewhere in this file
-- also trigger libidrisml load on first call, but the wrapped-handle ABI
-- shifts many FFIs to %foreign "scheme:..." with embedded
-- foreign-procedure calls; loading the lib here removes the ordering
-- dependency on a stray %foreign "C:..." call firing first.
%foreign "scheme:(lambda (dummy) (if (top-level-bound? 'idris-tensor-guardian) 0 (begin (load-shared-object \"libidrisml.dylib\") (set-top-level-value! 'idris-tensor-guardian (make-guardian)) 1)))"
prim__initManagedHandlesC : Int -> PrimIO Int

-- Drain the guardian: pop dead wrappers, call C tensor_release_handle on
-- each raw pointer. Returns the number drained. Uses (foreign-procedure
-- ...) which resolves tensor_release_handle from the dlopened libidrisml
-- at first call (cached thereafter). Self-initializing — yields 0 if
-- the guardian doesn't exist yet (no managed-handle wraps have happened).
%foreign "scheme:(lambda (dummy) (if (not (top-level-bound? 'idris-tensor-guardian)) 0 (let ((rel (foreign-procedure \"tensor_release_handle\" (void*) void))) (let loop ((n 0)) (let ((d ((top-level-value 'idris-tensor-guardian)))) (if d (begin (rel (vector-ref d 1)) (loop (+ n 1))) n))))))"
prim__drainManagedHandlesC : Int -> PrimIO Int

-- Force a Chez major GC. Use sparingly — only at known-safe drain points.
%foreign "scheme:(lambda (dummy) (collect 4) 0)"
prim__forceMajorGcC : Int -> PrimIO Int

||| Initialize the managed-handle guardian. Idempotent. Returns 1 the
||| first time it runs, 0 thereafter. Call once at backend init / first
||| tensor creation — wrapHandle assumes the guardian exists.
export
initManagedHandles : IO Int
initManagedHandles = primIO (prim__initManagedHandlesC 0)

||| Drain the guardian. Pops dead wrappers and calls
||| tensor_release_handle on each. Returns the number drained.
export
drainManagedHandles : IO Int
drainManagedHandles = primIO (prim__drainManagedHandlesC 0)

||| Force a Chez major GC. Combined with drainManagedHandles, this is the
||| reclamation mechanism for eval-phase tight loops. Expensive — only
||| call at boundaries like no_grad_end or every Nth FFI in heavy code.
export
forceMajorGc : IO ()
forceMajorGc = do
  _ <- primIO (prim__forceMajorGcC 0)
  pure ()

-- Lifecycle
--
-- Wrapped-handle ABI (mlx): every Tensor-returning FFI's Scheme wrapper
-- wraps the C return in a Chez vector + registers it with the
-- idris-tensor-guardian + retains via tensor_retain_handle. Every
-- Tensor-consuming FFI's Scheme wrapper extracts the raw pointer via
-- vector-ref. The Idris-level value (AnyPtr) is the wrap; the wrap is
-- the Tensor's identity in the Chez runtime — the Idris-Chez compiler
-- can't elide it without eliding the value itself.
--
-- See docs/develop/tensor-lifecycle-plan.md.
%foreign "scheme:(lambda (val rg) (when (not (top-level-bound? 'idris-libidrisml-loaded)) (load-shared-object \"libidrisml.dylib\") (set-top-level-value! 'idris-libidrisml-loaded #t)) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_create_scalar\" (double int) void*) val rg))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export prim__createScalar : Double -> Int -> AnyPtr

%foreign "scheme:(lambda (a0)  ((foreign-procedure \"tensor_free\" (void*) void) (vector-ref a0 1)))"
prim__free : AnyPtr -> ()

%foreign "scheme:(lambda (a0)  ((foreign-procedure \"tensor_item\" (void*) double) (vector-ref a0 1)))"
export prim__item : AnyPtr -> Double

-- Device transfer
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_to_device\" (void* string) void*) (vector-ref a0 1) a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export prim__toDevice : AnyPtr -> String -> AnyPtr

%foreign "scheme:(lambda (a0)  ((foreign-procedure \"tensor_device\" (void*) string) (vector-ref a0 1)))"
export prim__tensorDevice : AnyPtr -> String

-- Arithmetic (all return new tensors — libtorch builds autograd graph)
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_add\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export prim__add : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_sub\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export prim__sub : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_mul\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export prim__mul : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_div\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export prim__div : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_neg\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export prim__neg : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_abs\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export prim__abs : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_exp\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export prim__exp : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_log\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export prim__log : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_sqrt\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__sqrt : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_pow\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export prim__pow : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_sigmoid\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export prim__sigmoid : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_tanh\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export prim__tanh : AnyPtr -> AnyPtr

-- Linear algebra
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_mv\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export prim__mv : AnyPtr -> AnyPtr -> AnyPtr

-- Fused 1D linear: y = W @ x + bias. Eliminates the per-call FFI
-- overhead of separate prim__mv + prim__add.
%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_linear\" (void* void* void*) void*) (vector-ref a0 1) (vector-ref a1 1) (vector-ref a2 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export prim__linear : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_dot\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__dot : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_outer\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export prim__outer : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_matmul\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export prim__matmul : AnyPtr -> AnyPtr -> AnyPtr

-- Activation
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_softmax\" (void* int) void*) (vector-ref a0 1) a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export prim__softmax : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_log_softmax\" (void* int) void*) (vector-ref a0 1) a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export prim__logSoftmax : AnyPtr -> Int -> AnyPtr

-- Loss
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_bce_with_logits\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__bceWithLogits : AnyPtr -> AnyPtr -> AnyPtr

-- Reduction
%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_sum\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export prim__sum : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_mean\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export prim__mean : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_min\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export prim__tensorMin : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_max\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export prim__tensorMax : AnyPtr -> AnyPtr

-- Array creation/accessors
%foreign "scheme:(lambda (a0 a1 a2 a3) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_create\" (void* void* int int) void*) a0 a1 a2 a3))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__create : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0)  ((foreign-procedure \"tensor_numel\" (void*) int) (vector-ref a0 1)))"
prim__numel : AnyPtr -> Int

%foreign "scheme:(lambda (a0 a1)  ((foreign-procedure \"tensor_size\" (void* int) int) (vector-ref a0 1) a1))"
prim__size : AnyPtr -> Int -> Int

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_select\" (void* int int) void*) (vector-ref a0 1) a1 a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export prim__select : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_unsqueeze\" (void* int) void*) (vector-ref a0 1) a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export prim__unsqueeze : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_stack\" (void* int int) void*) a0 a1 a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__stack : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_detach\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__detach : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_with_grad\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__withGrad : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_mul_scalar\" (void* double) void*) (vector-ref a0 1) a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export prim__mulScalar : AnyPtr -> Double -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_add_scalar\" (void* double) void*) (vector-ref a0 1) a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export prim__addScalar : AnyPtr -> Double -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_clamp_min\" (void* double) void*) (vector-ref a0 1) a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export prim__clampMin : AnyPtr -> Double -> AnyPtr

-- NTM
%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_cosine_similarity\" (void* void* int) void*) (vector-ref a0 1) (vector-ref a1 1) a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export prim__cosineSimilarity : AnyPtr -> AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_conv1d_circular\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export prim__conv1dCircular : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_gelu\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export prim__gelu : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_leaky_relu\" (void* double) void*) (vector-ref a0 1) a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export prim__leakyRelu : AnyPtr -> Double -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_silu\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export prim__silu : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_softplus\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export prim__softplus : AnyPtr -> AnyPtr

-- Cross-attention: Q @ K^T * scale [+ mask] -> softmax -> @ V
%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (let ((raw_r ((foreign-procedure \"tensor_cross_attention\" (void* void* void* void* double) void*) (vector-ref a0 1) (vector-ref a1 1) (vector-ref a2 1) (vector-ref a3 1) a4))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export
prim__crossAttention : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> Double -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_gru_cell\" (void* void* void* int) void*) (vector-ref a0 1) (vector-ref a1 1) (vector-ref a2 1) a3))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export prim__gruCell : AnyPtr -> AnyPtr -> AnyPtr -> Int -> AnyPtr

-- Embedding
%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_embedding\" (void* void* int int) void*) (vector-ref a0 1) (vector-ref a1 1) a2 a3))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export
prim__embedding : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr

-- Batch Norm
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6 a7 a8 a9)  (let ((raw_r ((foreign-procedure \"tensor_batch_norm\" (void* void* void* void* void* int int int double double) void*) (vector-ref a0 1) (vector-ref a1 1) (vector-ref a2 1) (vector-ref a3 1) (vector-ref a4 1) a5 a6 a7 a8 a9))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export
prim__batchNorm : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> Double -> Double -> AnyPtr

-- Dropout
%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_dropout\" (void* double int int) void*) (vector-ref a0 1) a1 a2 a3))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export
prim__dropout : AnyPtr -> Double -> Int -> Int -> AnyPtr

-- Shape / info queries
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_squeeze\" (void* int) void*) (vector-ref a0 1) a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export prim__squeeze : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_clone\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export prim__clone : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  ((foreign-procedure \"tensor_dim\" (void*) int) (vector-ref a0 1)))"
export prim__tensorDim : AnyPtr -> Int

%foreign "scheme:(lambda (a0 a1)  ((foreign-procedure \"tensor_size\" (void* int) int) (vector-ref a0 1) a1))"
export prim__tensorSizeAt : AnyPtr -> Int -> Int

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_sum_dim\" (void* int int) void*) (vector-ref a0 1) a1 a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export prim__sumDim : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0)  ((foreign-procedure \"tensor_requires_grad\" (void*) int) (vector-ref a0 1)))"
export prim__requiresGrad : AnyPtr -> Int

%foreign "scheme:(lambda (a0 a1)  ((foreign-procedure \"tensor_set_requires_grad\" (void* int) void) (vector-ref a0 1) a1))"
export prim__setRequiresGrad : AnyPtr -> Int -> PrimIO ()

-- Gather / Scatter
%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_gather\" (void* void* int) void*) (vector-ref a0 1) (vector-ref a1 1) a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export prim__gather : AnyPtr -> AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_scatter_add\" (void* void* int) void*) (vector-ref a0 1) (vector-ref a1 1) a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export prim__scatterAdd : AnyPtr -> AnyPtr -> Int -> AnyPtr

-- Sort / Scan
%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_argsort\" (void* int int) void*) (vector-ref a0 1) a1 a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export prim__argsort : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_cumprod\" (void* int) void*) (vector-ref a0 1) a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export prim__cumprod : AnyPtr -> Int -> AnyPtr

-- Average Pooling
%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_avg_pool1d\" (void* int int) void*) (vector-ref a0 1) a1 a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export
prim__avgPool1d : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (let ((raw_r ((foreign-procedure \"tensor_avg_pool2d\" (void* int int int int) void*) (vector-ref a0 1) a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export
prim__avgPool2d : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr

-- Conv1D / MaxPool1D
%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (let ((raw_r ((foreign-procedure \"tensor_conv1d\" (void* void* void* int int) void*) (vector-ref a0 1) (vector-ref a1 1) (vector-ref a2 1) a3 a4))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export
prim__conv1d : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_max_pool1d\" (void* int int) void*) (vector-ref a0 1) a1 a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export
prim__maxPool1d : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_create_param_3d\" (int int int void*) void*) a0 a1 a2 a3))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export
prim__createParam3d : Int -> Int -> Int -> AnyPtr -> AnyPtr

-- Conv2D / MaxPool2D
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6)  (let ((raw_r ((foreign-procedure \"tensor_conv2d\" (void* void* void* int int int int) void*) (vector-ref a0 1) (vector-ref a1 1) (vector-ref a2 1) a3 a4 a5 a6))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export
prim__conv2d : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6)  (let ((raw_r ((foreign-procedure \"tensor_conv2d_batched\" (void* void* void* int int int int) void*) (vector-ref a0 1) (vector-ref a1 1) (vector-ref a2 1) a3 a4 a5 a6))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export
prim__conv2dBatched : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (let ((raw_r ((foreign-procedure \"tensor_max_pool2d\" (void* int int int int) void*) (vector-ref a0 1) a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export
prim__maxPool2d : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (let ((raw_r ((foreign-procedure \"tensor_max_pool2d_batched\" (void* int int int int) void*) (vector-ref a0 1) a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export
prim__maxPool2dBatched : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr

-- MNIST data loading
%foreign "C:mnist_load,libidrisml"
export
prim__mnistLoad : String -> String -> AnyPtr

%foreign "C:mnist_count,libidrisml"
export
prim__mnistCount : AnyPtr -> Int

%foreign "scheme:(lambda (a0 a1) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"mnist_get_image\" (void* int) void*) a0 a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export
prim__mnistGetImage : AnyPtr -> Int -> AnyPtr

%foreign "C:mnist_get_label,libidrisml"
export
prim__mnistGetLabel : AnyPtr -> Int -> Int

-- Parameter registry
-- Registers a parameter: enables requires_grad and adds to the registry.
-- Returns the tensorPtr for threading (prevents dead code elimination).
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"param_register_return\" (string void*) void*) a0 (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export
prim__paramRegister : String -> AnyPtr -> AnyPtr

-- In-place scalar subtract on a tensor (under no_grad). Returns tensor for threading.
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_subtract_scalar_inplace\" (void* double) void*) (vector-ref a0 1) a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export
prim__tensorSubScalarInplace : AnyPtr -> Double -> AnyPtr

-- Array-level parameter creation
%foreign "scheme:(lambda (a0 a1 a2) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_create_param_2d\" (int int void*) void*) a0 a1 a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export
prim__createParam2d : Int -> Int -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_create_param_1d\" (int void*) void*) a0 a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export
prim__createParam1d : Int -> AnyPtr -> AnyPtr

-- State tensors (non-learnable, non-grad). Both init-time permanent state
-- (NTM mask, BatchNorm running stats, transformer PE, DNC mask) AND
-- per-sequence transient state (Ntm/Dnc zeroState) flow through this
-- single path. mlx: is_state=1, refcount-driven; the Idris-side wrap is
-- the only stable holder, so the Tensor lives as long as the holder does.
-- tape/torch: the backend's own arena/shared_ptr handles freeing.
%foreign "scheme:(lambda (a0 a1 a2) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_create_state_2d\" (int int void*) void*) a0 a1 a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export
prim__createState2d : Int -> Int -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_create_state_1d\" (int void*) void*) a0 a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export
prim__createState1d : Int -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_view_2d\" (void* int int) void*) (vector-ref a0 1) a1 a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export
prim__view2d : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_view_1d\" (void* int) void*) (vector-ref a0 1) a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export
prim__view1d : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  ((foreign-procedure \"tensor_item_2d\" (void* int int) double) (vector-ref a0 1) a1 a2))"
export
prim__item2d : AnyPtr -> Int -> Int -> Double

%foreign "scheme:(lambda (a0 a1)  ((foreign-procedure \"tensor_item_1d\" (void* int) double) (vector-ref a0 1) a1))"
export
prim__item1d : AnyPtr -> Int -> Double

-- Fused LSTM gates: takes combined [4*o] tensor + prev_cell [o], returns pair handle
%foreign "scheme:(lambda (a0 a1 a2)  ((foreign-procedure \"tensor_lstm_gates_pair\" (void* void* int) void*) (vector-ref a0 1) (vector-ref a1 1) a2))"
export
prim__lstmGatesPair : AnyPtr -> AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_pair_first\" (void*) void*) a0))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export
prim__pairFirst : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_pair_second\" (void*) void*) a0))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export
prim__pairSecond : AnyPtr -> AnyPtr

-- Array-level forward ops (used by layers with consolidated weight tensors)
||| Matrix-vector multiply on raw tensor pointers.
export
tensorMv : AnyPtr -> AnyPtr -> AnyPtr
tensorMv = prim__mv

||| Add two raw tensor pointers.
export
tensorAdd : AnyPtr -> AnyPtr -> AnyPtr
tensorAdd = prim__add

-- No-grad scope. Push/pop a counter on the C side; mirrors PyTorch's
-- torch.no_grad(). When depth > 0, ops skip tape append on tape/mlx
-- and torch's NoGradGuard suppresses autograd graph construction.
-- PrimIO sequencing keeps the calls in order; same pattern as the
-- other side-effecting prims (prim__backwardC, prim__zeroAllGradsC).
%foreign "C:tensor_no_grad_begin,libidrisml"
prim__noGradBeginC : PrimIO ()

%foreign "C:tensor_no_grad_end,libidrisml"
prim__noGradEndC : PrimIO ()

||| Run an `IO` action with autograd disabled. Inside the action,
||| every tensor op skips tape/autograd graph construction, so the
||| results have no path to backward. Standard for RL rollouts and
||| any inference pass. Mirrors PyTorch's `with torch.no_grad():`.
||| Nested calls are stacked: only the outermost begin/end pair
||| toggles tracking, so library code can call this without checking
||| whether the caller already disabled grad.
export
withNoGrad : IO a -> IO a
withNoGrad act = do
  primIO prim__noGradBeginC
  result <- act
  primIO prim__noGradEndC
  -- Eval phases (typically wrapped in `withNoGrad`) can churn through
  -- thousands of per-sequence managed state Tensors. On mlx that drives
  -- the Metal MTLBuffer count past the paravirtualized-Metal ceiling on
  -- Tart / GHA macOS runners. Force a Chez major GC + drain the guardian
  -- here so dropped state Tensors release their C-side refs immediately.
  -- Non-mlx backends: drain is a no-op (no shadows registered).
  forceMajorGc
  _ <- drainManagedHandles
  pure result

----------------------------------------------------------------------
-- Sequencing helper
----------------------------------------------------------------------

-- Force evaluation of first arg, return second.
-- Must use concrete AnyPtr types (not polymorphic) to avoid
-- argument count issues at the FFI boundary.
%foreign "scheme:(lambda (a0 a1)  ((foreign-procedure \"idrisml_seq\" (void* void*) void*) a0 a1))"
export
prim__seq : AnyPtr -> AnyPtr -> AnyPtr


----------------------------------------------------------------------
-- C-side allocation + bulk-load helpers
----------------------------------------------------------------------

%foreign "C:tensor_alloc_doubles,libidrisml"
export prim__allocDoubles : Int -> AnyPtr

%foreign "C:tensor_read_double,libidrisml"
prim__readDouble : AnyPtr -> Int -> Double

-- Wrapper that returns the buffer pointer for threading through let chains
%foreign "C:tensor_write_double_return,libidrisml"
export
prim__setDouble : AnyPtr -> Int -> Double -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_one_hot\" (void* int int) void*) a0 a1 a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export
prim__oneHot : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_bmm\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export
prim__bmm : AnyPtr -> AnyPtr -> AnyPtr

-- 3D batched attention ops
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_bmm_3x3\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export
prim__bmm3x3 : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_softmax_3d\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export
prim__softmax3d : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_transpose_last2\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export
prim__transposeLast2 : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_reshape_3d\" (void* int int int) void*) (vector-ref a0 1) a1 a2 a3))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export
prim__reshape3d : AnyPtr -> Int -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (let ((raw_r ((foreign-procedure \"tensor_reshape_4d\" (void* int int int int) void*) (vector-ref a0 1) a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export
prim__reshape4d : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_expand_mask\" (void* int) void*) (vector-ref a0 1) a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export
prim__expandMask : AnyPtr -> Int -> AnyPtr

||| Tile a 2D tensor: `[m, n] -> [m*rep0, n*rep1]`. Element `(i, j)` in the
||| output equals element `(i mod m, j mod n)` in the input.
%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_tile_2d\" (void* int int) void*) (vector-ref a0 1) a1 a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export
prim__tile2d : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_create_1d\" (int void* int) void*) a0 a1 a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export
prim__create1d : Int -> AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_create_2d\" (int int void* int) void*) a0 a1 a2 a3))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export prim__create2d : Int -> Int -> AnyPtr -> Int -> AnyPtr

-- Array pointer array: stack scalar Tensor tensorPtrs to create
-- a 1D/2D tensor that preserves the autograd graph.
%foreign "C:tensor_ptr_array_alloc,libidrisml"
prim__ptrArrayAlloc : Int -> AnyPtr

-- Returns the array for threading
%foreign "scheme:(lambda (a0 a1 a2)  ((foreign-procedure \"tensor_ptr_array_set_return\" (void* int void*) void*) a0 a1 (vector-ref a2 1)))"
prim__ptrArraySet : AnyPtr -> Int -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_stack_from_array\" (void* int int) void*) a0 a1 a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__stackFromArray : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_cat_from_array\" (void* int int) void*) a0 a1 a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export
prim__catFromArray : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_cat2\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export
prim__cat2 : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_concat_2d_axis1\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export
prim__concat2dAxis1 : AnyPtr -> AnyPtr -> AnyPtr

-- N-ary cat: caller retains ownership of the handle array.
-- See tensor_cat in backend.h.
%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_cat\" (void* int int) void*) a0 a1 a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export
prim__cat : AnyPtr -> Int -> Int -> AnyPtr

-- Batch [...] tensors into [count, ...]. Equivalent to stack at dim=0.
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_batch\" (void* int) void*) a0 a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export
prim__batch : AnyPtr -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_narrow\" (void* int int int) void*) (vector-ref a0 1) a1 a2 a3))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export
prim__narrow : AnyPtr -> Int -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_mm\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export
prim__mm : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_linear_2d\" (void* void* void*) void*) (vector-ref a0 1) (vector-ref a1 1) (vector-ref a2 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export
prim__linear2d : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_transpose_2d\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export
prim__transpose2d : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_softmax_2d\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export
prim__softmax2d : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_masked_fill\" (void* void* double) void*) (vector-ref a0 1) (vector-ref a1 1) a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export
prim__maskedFill : AnyPtr -> AnyPtr -> Double -> AnyPtr

%foreign "scheme:(lambda (a0) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_causal_mask\" (int) void*) a0))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export
prim__causalMask : Int -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_log_softmax_2d\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export
prim__logSoftmax2d : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_layer_norm_2d\" (void* void* void* double) void*) (vector-ref a0 1) (vector-ref a1 1) (vector-ref a2 1) a3))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export
prim__layerNorm2d : AnyPtr -> AnyPtr -> AnyPtr -> Double -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_reshape\" (void* void* int) void*) (vector-ref a0 1) a1 a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__reshape : AnyPtr -> AnyPtr -> Int -> AnyPtr

%foreign "C:tensor_alloc_ints,libidrisml"
export
prim__allocInts : Int -> AnyPtr

%foreign "C:tensor_write_int_return,libidrisml"
export
prim__setInt : AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_reshape_2d\" (void* int int) void*) (vector-ref a0 1) a1 a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export prim__reshape2d : AnyPtr -> Int -> Int -> AnyPtr

-- Reshape to 1D: flatten any tensor to [n]
export
prim__reshape1d : AnyPtr -> Int -> AnyPtr
prim__reshape1d t n =
  let shape = prim__allocInts 1
      shape' = prim__setInt shape 0 n
  in prim__reshape t shape' 1

%foreign "scheme:(lambda (a0 a1 a2 a3 a4) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_create_param_4d\" (int int int int void*) void*) a0 a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
export
prim__createParam4d : Int -> Int -> Int -> Int -> AnyPtr -> AnyPtr


----------------------------------------------------------------------
-- Backpropagation: prims for native optimizer
----------------------------------------------------------------------

%foreign "scheme:(lambda (a0)  ((foreign-procedure \"tensor_backward_conditional\" (void*) int) (vector-ref a0 1)))"
prim__backwardAndCount : AnyPtr -> Int

----------------------------------------------------------------------
-- Native Optimizer
----------------------------------------------------------------------

%foreign "C:optimizer_create_sgd,libidrisml"
prim__optimizerCreateSgd : Double -> AnyPtr

%foreign "C:optimizer_create_rmsprop,libidrisml"
prim__optimizerCreateRmsprop : Double -> Double -> Double -> Double -> Double -> AnyPtr

%foreign "C:optimizer_create_adam,libidrisml"
prim__optimizerCreateAdam : Double -> Double -> Double -> Double -> AnyPtr

%foreign "C:optimizer_create_adam_group,libidrisml"
export
prim__optimizerCreateAdamGroup : Double -> Double -> Double -> Double -> String -> AnyPtr

%foreign "C:polyak_blend,libidrisml"
export
prim__polyakBlend : Double -> String -> String -> PrimIO Int

||| Polyak soft update for twin-network param groups registered under
||| `onlineScope` vs `targetScope`: for each online param, finds the
||| matching target param (same suffix after scope prefix) and blends
|||   target_data ← (1 − tau) · target_data + tau · online_data
||| in-place. Returns the number of param pairs blended. Used by SAC to
||| track target-Q networks.
export
polyakUpdate : (tau : Double) -> (onlineScope : String) -> (targetScope : String) -> IO Int
polyakUpdate tau onlineScope targetScope =
  primIO (prim__polyakBlend tau onlineScope targetScope)


public export
data ClipMode = NoClip | ValueClip Double | NormClip Double

||| Native libtorch optimizer. Single step() call updates all parameters.
public export
record NativeOptimizer where
  constructor MkNativeOptimizer
  handle : AnyPtr
  clipMode : ClipMode

||| Create a native SGD optimizer.
export
nativeSgd : Double -> NativeOptimizer
nativeSgd lr = MkNativeOptimizer (prim__optimizerCreateSgd lr) NoClip

||| Create a native RMSprop optimizer (matches PyTorch defaults).
export
nativeRmsprop : (lr : Double) -> (alpha : Double) -> (eps : Double) ->
                (clipVal : Double) -> (momentum : Double) -> NativeOptimizer
nativeRmsprop lr alpha eps clipVal momentum =
  MkNativeOptimizer
    (prim__optimizerCreateRmsprop lr alpha eps 0.0 momentum)
    (ValueClip clipVal)

||| Create a native Adam optimizer with global norm clipping.
export
nativeAdamGlobalClip : (lr : Double) -> (beta1 : Double) -> (beta2 : Double) ->
                       (eps : Double) -> (maxNorm : Double) -> NativeOptimizer
nativeAdamGlobalClip lr beta1 beta2 eps maxNorm =
  MkNativeOptimizer
    (prim__optimizerCreateAdam lr beta1 beta2 eps)
    (NormClip maxNorm)

||| Create a native Adam optimizer that only manages params whose registry
||| paramId starts with `scope`. Empty scope behaves like
||| `nativeAdamGlobalClip`. Used for multi-network setups where each
||| network (e.g. SAC actor / q1 / q2) needs its own optimizer so that
||| gradient leakage from one network's loss doesn't update another
||| network's weights (matches PyTorch's one-optimizer-per-net pattern).
export
nativeAdamGroup : (scope : String) ->
                  (lr : Double) -> (beta1 : Double) -> (beta2 : Double) ->
                  (eps : Double) -> (maxNorm : Double) -> NativeOptimizer
nativeAdamGroup scope lr beta1 beta2 eps maxNorm =
  MkNativeOptimizer
    (prim__optimizerCreateAdamGroup lr beta1 beta2 eps scope)
    (NormClip maxNorm)

%foreign "C:optimizer_create_adamw,libidrisml"
prim__optimizerCreateAdamW : Double -> Double -> Double -> Double -> Double -> AnyPtr

||| Create a native AdamW optimizer (decoupled weight decay) with global norm clipping.
export
nativeAdamW : (lr : Double) -> (beta1 : Double) -> (beta2 : Double) ->
              (eps : Double) -> (weightDecay : Double) -> (maxNorm : Double) -> NativeOptimizer
nativeAdamW lr beta1 beta2 eps wd maxNorm =
  MkNativeOptimizer
    (prim__optimizerCreateAdamW lr beta1 beta2 eps wd)
    (NormClip maxNorm)

%foreign "C:optimizer_set_param_lr,libidrisml"
prim__optimizerSetParamLR : AnyPtr -> String -> Double -> PrimIO ()

||| Set a per-parameter learning rate override. Parameters matching the given
||| name will use this LR instead of the optimizer's base LR.
||| Use LR=0 to freeze a parameter. Set LR<0 to revert to base LR.
export
setParamLR : NativeOptimizer -> String -> Double -> IO ()
setParamLR opt name lr = primIO (prim__optimizerSetParamLR opt.handle name lr)

%foreign "C:optimizer_set_lr,libidrisml"
prim__optimizerSetLrC : AnyPtr -> Double -> PrimIO ()

||| Update the optimizer's base (global) learning rate. Per-parameter
||| overrides set via `setParamLR` remain in effect; only un-overridden
||| params pick up the new base LR. Used to apply LR schedules per epoch.
export
setLearningRate : NativeOptimizer -> Double -> IO ()
setLearningRate opt lr = primIO (prim__optimizerSetLrC opt.handle lr)

-- Fused native train step: zero_grad → backward → clip → step.
-- Fused: zero_grad → backward → clip → step in single C call.
-- Returns loss value (read before step, so not stale).
--
-- After the C call returns, force a Chez minor GC + drain the
-- managed-handle guardian. This is the training-loop drain trigger
-- that lets the mlx refcount-driven lifecycle reclaim per-step
-- intermediate Tensors — without it, the wrap-and-retain on each
-- Tensor's creation keeps its refcount at >=1 indefinitely (Chez
-- doesn't auto-GC under foreign-side pressure alone, and drain is
-- only otherwise called at withNoGrad exit). On tape/torch the drain
-- is essentially a no-op (their retain/release are stubs).
%foreign "scheme:(lambda (a0 a1 a2 a3 a4) (let ((result ((foreign-procedure \"native_train_step\" (void* int double void* double) double) a0 a1 a2 (vector-ref a3 1) a4))) (collect 0) (when (top-level-bound? 'idris-tensor-guardian) (let ((rel (foreign-procedure \"tensor_release_handle\" (void*) void))) (let loop () (let ((d ((top-level-value 'idris-tensor-guardian)))) (when d (rel (vector-ref d 1)) (loop)))))) result))"
prim__nativeTrainStep : AnyPtr -> Int -> Double -> AnyPtr -> Double -> Double

----------------------------------------------------------------------
-- GC / RSS
----------------------------------------------------------------------

%foreign "C:backend_supports_tensor_params,libidrisml"
export
prim__backendSupportsTensorParams : Int

export
forceGC : IO ()
forceGC = pure ()

%foreign "C:get_rss_mb,libidrisml"
prim__getRssMB : Int

%foreign "C:get_current_rss_mb,libidrisml"
prim__getCurrentRssMB : Int

export
getRssMB : Nat -> Int
getRssMB _ = prim__getRssMB

export
getCurrentRssMB : Nat -> Int
getCurrentRssMB _ = prim__getCurrentRssMB

%foreign "C:backend_memory_report_return,libidrisml"
prim__memoryReport : Int -> PrimIO Int


||| Bulk-convert a Vector of Doubles to a C tensor handle.
||| The C tensor_create_1d function frees the input buffer after copying.
export
bulkToTensor : {n : Nat} -> Vector n Double -> AnyPtr
bulkToTensor {n} (VArray elems) =
  let nI = cast {to=Int} n
      buf = prim__allocDoubles nI
      buf' = packDoubleBuf buf 0 elems
  in prim__create1d nI buf' 0
  where
    packDoubleBuf : AnyPtr -> Int -> Vect k (Scalar Double) -> AnyPtr
    packDoubleBuf buf _ [] = buf
    packDoubleBuf buf off (SArray v :: rest) =
      let buf' = prim__setDouble buf off v
      in packDoubleBuf buf' (off + 1) rest

||| Bulk-convert a Vect of Vectors of Doubles to a [b, i] C tensor handle.
||| The C tensor_create_2d function frees the input buffer after copying.
||| Use to stack a per-sample input batch into a single batched tensor.
export
bulkToTensor2d : {b, i : Nat} -> Vect b (Vector i Double) -> AnyPtr
bulkToTensor2d {b} {i} rows =
  let bI = cast {to=Int} b
      iI = cast {to=Int} i
      buf = prim__allocDoubles (bI * iI)
      buf' = packRows buf 0 rows
  in prim__create2d bI iI buf' 0
  where
    packRow : AnyPtr -> Int -> Vect k (Scalar Double) -> AnyPtr
    packRow buf _ [] = buf
    packRow buf off (SArray v :: rest) =
      let buf' = prim__setDouble buf off v
      in packRow buf' (off + 1) rest
    packRows : AnyPtr -> Int -> Vect k (Vector i Double) -> AnyPtr
    packRows buf _ [] = buf
    packRows buf off (VArray row :: rest) =
      let buf' = packRow buf off row
      in packRows buf' (off + cast {to=Int} i) rest

||| Bulk-convert a Vector of Doubles to a persistent C tensor handle.
||| Persistent tensors survive tape resets — use when data is created once
||| and reused across training epochs.
export
vectorToTensorPersistent : {n : Nat} -> Vector n Double -> AnyPtr
vectorToTensorPersistent {n} (VArray elems) =
  let nI = cast {to=Int} n
      buf = prim__allocDoubles nI
      buf' = packBuf buf 0 elems
  in prim__createState1d nI buf'
  where
    packBuf : AnyPtr -> Int -> Vect k (Scalar Double) -> AnyPtr
    packBuf buf _ [] = buf
    packBuf buf off (SArray v :: rest) = packBuf (prim__setDouble buf off v) (off + 1) rest

||| Convert a DataPoint with Doubles to a TensorDataPoint with persistent C tensors.
export
toTDP : {i, o : Nat} -> DataPoint i o Double -> TensorDataPoint i o
toTDP dp = MkTensorDataPoint (vectorToTensorPersistent (x dp)) (vectorToTensorPersistent (y dp))

||| Print detailed memory breakdown to stderr.
export
memoryReport : IO ()
memoryReport = do
  _ <- primIO (prim__memoryReport 0)
  pure ()

%foreign "scheme:(lambda (a0)  ((foreign-procedure \"tensor_backward\" (void*) void) (vector-ref a0 1)))"
prim__backwardC : AnyPtr -> PrimIO ()

%foreign "C:param_zero_all_grads,libidrisml"
prim__zeroAllGradsC : PrimIO ()

-- runBackward is defined post-Tensor record below; the type-level
-- gate (Tensor [] d dt WithGrad-> IO ()) lives there.

%foreign "C:param_count,libidrisml"
prim__paramCountC : PrimIO Int

%foreign "C:param_name,libidrisml"
prim__paramNameC : Int -> PrimIO String

%foreign "C:param_grad_item_at,libidrisml"
prim__paramGradItemAtC : Int -> Int -> PrimIO Double

||| Get parameter count (for gradient inspection).
export
getParamCount : IO Int
getParamCount = primIO prim__paramCountC

||| Get parameter name by index.
export
getParamName : Int -> IO String
getParamName i = primIO (prim__paramNameC i)

||| Get gradient element for param i, element j.
export
getParamGradAt : Int -> Int -> IO Double
getParamGradAt i j = primIO (prim__paramGradItemAtC i j)

||| Zero all parameter gradients.
export
zeroAllGrads : IO ()
zeroAllGrads = primIO prim__zeroAllGradsC

%foreign "C:backend_name,libidrisml"
prim__backendName : String

||| Get the name of the active backend ("tape", "mlx", "torch").
export
backendName : String
backendName = prim__backendName

%foreign "C:backend_profile_reset,libidrisml"
prim__profileResetC : PrimIO ()

%foreign "C:backend_profile_report,libidrisml"
prim__profileReportC : PrimIO ()

||| Reset profiling counters.
export
profileReset : IO ()
profileReset = primIO prim__profileResetC

||| Print profiling breakdown to stderr.
export
profileReport : IO ()
profileReport = primIO prim__profileReportC

----------------------------------------------------------------------
-- Path C P3-1 spike: rank-aware Tensor
----------------------------------------------------------------------
--
-- Today's `Tensor d` is shape-erased and packed into the outer
-- `Array dims (Tensor d)` via Vect-of-Vect, scalarising at every
-- op. `Tensor dims d` lifts shape onto the Tensor itself: one tensor
-- handle per typed shape, no per-element packing.
--
-- `Tensor []` is the scalar — distinguished from `Tensor [n]` etc. by
-- type. Loss naturally types as `Tensor [] d`.
--
-- Keep `paramId`: the C-side optimizer registry is keyed on it.
-- Drop the cached `value : Double` — read at the boundary via
-- `tensorItem`.
--
-- Spike-only; lives in a parallel layer/example axis.

||| The autograd handle. Under the wrapped-handle ABI, `tensorPtr` is
||| not a raw pointer but a Chez vector `#(tensor-handle raw_ptr)`
||| produced by the creating FFI's Scheme glue and registered with the
||| `idris-tensor-guardian`. The vector IS the Tensor's runtime
||| identity — Idris-Chez codegen can't elide it without eliding the
||| Tensor value itself. C FFIs internally `vector-ref` to extract the
||| raw pointer, so this layer is invisible above the FFI boundary.
||| See docs/develop/tensor-lifecycle-plan.md.
public export
record Tensor (dims : Vect rank Nat) (0 d : Device) (0 dt : DType) (0 g : GradMode) where
  constructor MkTensor
  tensorPtr : AnyPtr
  paramId   : Maybe String

||| Transfer a tensor to a different device. The one place where
||| device types intentionally change. Wraps `prim__toDevice` with
||| the target device's `deviceName` as the C-side tag. `paramId` is
||| preserved (the C-side parameter registry tracks the moved
||| handle).
|||
||| Phase 2.1b: the target device is now a *type* with a
||| `UserDeviceCore` instance, not a `Device`-sum value. The instance
||| supplies the C-side string via its `deviceName` method.
export
toDevice : {0 d1 : Type} -> (0 d2 : Type) -> UserDeviceCore d2 =>
           Tensor dims d1 dt WithGrad -> IO (Tensor dims d2 dt WithGrad)
toDevice d2 t =
  pure (MkTensor (prim__toDevice t.tensorPtr (deviceName {d = d2}))
                 t.paramId)

||| Mark a tensor as no-grad: flips the C-side `requires_grad` flag to
||| false and retypes the handle as `NoGrad`. After this, downstream
||| ops on the tensor build no tape entries (per-backend semantics:
||| tape sets the field, torch calls `set_requires_grad_(false)`, mlx
||| sets the bool). For parameter tensors this effectively freezes
||| them — gradients no longer flow back to update their value. For
||| activation tensors it's harmless (they aren't graph leaves).
||| Mirrors PyTorch's `tensor.requires_grad_(False)`.
|||
||| Linear in its input: consumes the original tensor reference at
||| compile time, so a caller can't accidentally use the pre-weaken
||| variable afterwards (the runtime state has changed under it).
||| Closes the "freeze then keep using the original WithGrad type"
||| aliasing footgun.
export
weakenGrad : (1 _ : Tensor dims d dt g) -> IO (Tensor dims d dt NoGrad)
weakenGrad (MkTensor ptr pid) = do
  primIO (prim__setRequiresGrad ptr 0)
  pure (MkTensor ptr pid)

||| Pure type-level cast between grad-modes. `g` is 0-quantity in
||| `Tensor`'s declaration, so `MkTensor` is polymorphic in `g` and
||| destructure-reconstruct flips the type tag with no runtime work
||| and no type-system bypass. Used by `unfreezeLayer` impls to
||| retype tensor fields after flipping the C-side `requires_grad`
||| flag. Not a control surface for users — to *change* the runtime
||| flag, use `weakenGrad`.
export
retypeGrad : Tensor dims d dt g1 -> Tensor dims d dt g2
retypeGrad (MkTensor ptr pid) = MkTensor ptr pid


----------------------------------------------------------------------
-- Cross-dtype conversion: lossless via `UpcastableTo`, lossy via
-- explicit `tcast`.
----------------------------------------------------------------------

||| Lossless precision upcast within a single dtype family
||| (`F32 → F64`, `Int 16 → Int 32`, `BFloat 16 → BFloat 32`, …).
||| The `UpcastableTo from to` constraint is solved by Idris's
||| auto-search via per-family `LTE m n` instances in `DType.Core`;
||| narrowing casts (`F64 → F32`) and cross-family casts
||| (`UInt 8 → F16`) have no `UpcastableTo` instance and use
||| `tcast` (below) instead.
|||
||| Runtime support is deferred — the body is a hole that will be
||| filled when the C-side cast primitives land (`tensor_cast_dtype`).
||| Type signature is stable; calls type-check today and will
||| activate at runtime when the primitive arrives.
export
tcastSafe : (UpcastableTo from to, IsDType from, IsDType to) =>
            Tensor dims d from g -> IO (Tensor dims d to g)
tcastSafe v = ?tcastSafe_impl

||| Explicit precision/dtype cast in ANY direction, including
||| narrowing (`F64 → F32`) and cross-family (`UInt 8 → F16`).
||| The caller takes responsibility for any precision loss or
||| representation change — calling `tcast` is the explicit signal
||| that the conversion was intentional.
|||
||| For lossless conversions, prefer `tcastSafe` so the compiler
||| verifies via `UpcastableTo` that no information is lost. Use
||| `tcast` only when the conversion is deliberately narrowing or
||| cross-family.
|||
||| Runtime support is deferred — see `tcastSafe`.
export
tcast : (0 to : DType) -> (IsDType from, IsDType to) =>
        Tensor dims d from g -> IO (Tensor dims d to g)
tcast _ v = ?tcast_impl

||| Type-level aliases for common Tensor shapes. Aliases route shape
||| arithmetic (e.g. `4 * o`) through a Nat-argument slot rather than
||| inlining inside a Vect literal — the latter triggers an Idris 2
||| type-checker hang on multiplicative Nat expressions.
||| (`Tensor [4 * o, i] d` hangs; `TMat (4 * o) i d` works.)
public export
0 TVec : Nat -> Device -> DType -> GradMode -> Type
TVec n d dt g = Tensor [n] d dt g

public export
0 TMat : Nat -> Nat -> Device -> DType -> GradMode -> Type
TMat m n d dt g = Tensor [m, n] d dt g

-- Smart constructors --------------------------------------------------

||| Lift a pure expression into an IO action whose body is RE-EVALUATED
||| on every sequencing (NOT memoized like `Lazy a`). The correct
||| primitive for "FFI side effect deferred until IO is run". Every
||| Tensor smart constructor below uses this — their bodies are pure
||| expressions whose evaluation triggers FFI side effects, so wrapping
||| in `ioRerun` lets IO sequencing control when those side effects fire
||| (specifically: makes `withNoGrad (do ...)` properly bracket them).
export %inline
ioRerun : (() -> a) -> IO a
ioRerun f = primIO (\w => MkIORes (f ()) w)

||| Create a registered learnable [o, i] parameter from a flat (row-major)
||| double buffer. Mirrors Linear.nameLayer's tensor path.
export
tparam2d : {o, i : Nat} -> (paramId : String) -> AnyPtr -> IO (Tensor [o, i] d dt WithGrad)
tparam2d {o} {i} pid buf = ioRerun (\_ =>
  let oI = cast {to=Int} o
      iI = cast {to=Int} i
      reg = prim__paramRegister pid (prim__createParam2d oI iI buf)
  in MkTensor reg (Just pid))

||| Create a registered learnable [n] parameter from a double buffer.
export
tparam1d : {n : Nat} -> (paramId : String) -> AnyPtr -> IO (Tensor [n] d dt WithGrad)
tparam1d {n} pid buf = ioRerun (\_ =>
  let nI = cast {to=Int} n
      reg = prim__paramRegister pid (prim__createParam1d nI buf)
  in MkTensor reg (Just pid))

||| Wrap an existing 1D tensor handle as a non-parameter input.
||| Pure — no FFI side effect, just record construction.
export
tinput1d : {n : Nat} -> AnyPtr -> Tensor [n] d dt WithGrad
tinput1d t = MkTensor t Nothing

||| Wrap an existing 2D tensor handle as a non-parameter input.
||| Pure — no FFI side effect, just record construction.
export
tinput2d : {m, n : Nat} -> AnyPtr -> Tensor [m, n] d dt WithGrad
tinput2d t = MkTensor t Nothing

-- Arithmetic / linear algebra (autograd-tracked) ----------------------

||| Elementwise addition. Both operands share shape.
||| `%inline`: inlines to a direct `prim__add` + `MkTensor` allocation
||| at every call site. Critical for hot-path layers (LSTM/NTM/DNC
||| call this many times per timestep); without inlining, Idris2's
||| Chez codegen wraps each invocation in a closure dispatch that
||| adds ~20µs of Scheme-side overhead per call, accumulating to a
||| 2× regression on recurrent models.
export %inline
tadd : {0 d : Device} -> UserDeviceCore d => Tensor dims d dt g -> Tensor dims d dt g -> IO (Tensor dims d dt g)
tadd a b = ioRerun (\_ => MkTensor (primAdd {d} a.tensorPtr b.tensorPtr) Nothing)

||| Matrix-vector multiply: [m, n] · [n] -> [m]. `%inline` for the
||| same reason as `tadd` (hot path in recurrent forward passes).
export %inline
tmv : {0 d : Device} -> UserDeviceTape d =>
      Tensor [m, n] d dt g -> Tensor [n] d dt g -> IO (Tensor [m] d dt g)
tmv w x = ioRerun (\_ => MkTensor (primMv {d} w.tensorPtr x.tensorPtr) Nothing)

||| Fused 1D linear: y = W[m,n] · x[n] + bias[m]. One C call instead
||| of `tadd (tmv W x) bias` — collapses two FFI hops into one and
||| eliminates the intermediate Idris-side glue. Used by Layer.Linear's
||| applyVar and by NTM/DNC FCs.
export %inline
tlinear : {0 d : Device} -> UserDeviceTape d =>
          Tensor [o, i] d dt g -> Tensor [i] d dt g -> Tensor [o] d dt g -> IO (Tensor [o] d dt g)
tlinear w x bias = ioRerun (\_ =>
  MkTensor (primLinear {d} w.tensorPtr x.tensorPtr bias.tensorPtr) Nothing)

||| Fused batched linear: W[o,i] · X^T[b,i] + bias[o] -> [b, o].
export %inline
tlinear2d : {0 d : Device} -> UserDeviceTape d =>
            Tensor [o, i] d dt g -> Tensor [b, i] d dt g -> Tensor [o] d dt g -> IO (Tensor [b, o] d dt g)
tlinear2d w x bias = ioRerun (\_ =>
  MkTensor (primLinear2d {d} w.tensorPtr x.tensorPtr bias.tensorPtr) Nothing)

-- Per-sample extraction + scalar arithmetic (used by batched RL loss
-- builders: pluck a row from a [b, o] result, then a scalar from the
-- row, then build (q - target)^2 etc.) ---------------------------------

||| Select row `k` from a [b, n] Tensor, returning the n-vector slice.
||| Wraps `prim__select` on dim 0; preserves the autograd graph.
export
trowSelect : {0 d : Device} -> UserDeviceTape d => {b, n : Nat} ->
             Tensor [b, n] d dt g -> Int -> IO (Tensor [n] d dt g)
trowSelect t k = ioRerun (\_ => MkTensor (primSelect {d} t.tensorPtr 0 k) Nothing)

||| Select element `i` from an n-vector, returning a scalar Tensor.
export
telemSelect : {0 d : Device} -> UserDeviceTape d => {n : Nat} ->
              Tensor [n] d dt g -> Int -> IO (Tensor [] d dt g)
telemSelect t i = ioRerun (\_ => MkTensor (primSelect {d} t.tensorPtr 0 i) Nothing)

||| Scalar Tensor from a Double. Takes the value as a runtime argument
||| so Idris/Chez does NOT memoise the FFI result as a module-level
||| constant — same defence as `freshZeroLossT`. Non-grad: the C
||| backend creates a non-persistent scalar that is freed by the next
||| `tape_reset` (i.e. fine to call inside an epoch's loss builder).
export
||| Note: keeps the unified `prim__createScalar` (Phase 1 alias to
||| the primary backend) rather than dispatching via
||| `UserDeviceCore.primCreateScalar`. The op has no Tensor input, so
||| `d` would need to be inferred from the result's use-site and Idris
||| 2's bidirectional inference doesn't reliably push the instance
||| constraint through every call site that just lets-binds the
||| result. For built-in devices this matches the previous behavior
||| (alias to primary); for user-supplied devices, users should
||| construct scalars via their own `UserDeviceCore.primCreateScalar`
||| directly. Same compromise applies to `tparamScalar` and
||| `freshZeroLossT`.
tconstScalar : {0 d : Device} -> Double -> IO (Tensor [] d dt WithGrad)
tconstScalar v = ioRerun (\_ => MkTensor (prim__createScalar v 0) Nothing)

||| Subtract two equally-shaped Tensors (autograd-tracked).
export %inline
tsub : {0 d : Device} -> UserDeviceCore d => Tensor dims d dt g -> Tensor dims d dt g -> IO (Tensor dims d dt g)
tsub a b = ioRerun (\_ => MkTensor (primSub {d} a.tensorPtr b.tensorPtr) Nothing)

||| Elementwise multiply two equally-shaped Tensors (autograd-tracked).
export %inline
tmul : {0 d : Device} -> UserDeviceCore d => Tensor dims d dt g -> Tensor dims d dt g -> IO (Tensor dims d dt g)
tmul a b = ioRerun (\_ => MkTensor (primMul {d} a.tensorPtr b.tensorPtr) Nothing)

||| Negate a Tensor (autograd-tracked).
export %inline
tneg : {0 d : Device} -> UserDeviceCore d => Tensor dims d dt g -> IO (Tensor dims d dt g)
tneg a = ioRerun (\_ => MkTensor (primNeg {d} a.tensorPtr) Nothing)

||| Scale a Tensor by a Double (broadcasts the scalar; autograd-tracked).
||| Useful for mean-reduction (`tmulScalar loss (1.0 / cast n)`) and for
||| building per-sample loss expressions where one side of a product is
||| a runtime Double (e.g. DQN target value).
export %inline
tmulScalar : {0 d : Device} -> UserDeviceCore d => Tensor dims d dt g -> Double -> IO (Tensor dims d dt g)
tmulScalar v s = ioRerun (\_ => MkTensor (primMulScalar {d} v.tensorPtr s) Nothing)

||| Elementwise exponential (autograd-tracked).
export %inline
texp : {0 d : Device} -> UserDeviceCore d => Tensor dims d dt g -> IO (Tensor dims d dt g)
texp v = ioRerun (\_ => MkTensor (primExp {d} v.tensorPtr) Nothing)

||| Elementwise natural log (autograd-tracked).
export %inline
tlog : {0 d : Device} -> UserDeviceCore d => Tensor dims d dt g -> IO (Tensor dims d dt g)
tlog v = ioRerun (\_ => MkTensor (primLog {d} v.tensorPtr) Nothing)

||| Create a registered learnable scalar parameter (e.g. SAC's
||| state-independent log_std). Mirrors V1's `param`. The optimizer
||| picks it up automatically by paramId scope.
export
tparamScalar : {0 d : Device} -> (paramId : String) -> (val : Double) -> IO (Tensor [] d dt WithGrad)
tparamScalar pid val = ioRerun (\_ =>
  let ptr = prim__createScalar val 1                  -- requires_grad=true
      reg = prim__paramRegister pid ptr
  in MkTensor reg (Just pid))

||| Concatenate two [b, m] / [b, n] TVars along axis 1, producing
||| [b, m + n]. Wraps `prim__concat2dAxis1`. Used by SAC's actor loss
||| to build a [B, ObsDim + ActDim] Q-input from obs + reparametrized
||| action while preserving the autograd path through the action.
export
tconcat2dAxis1 : {0 d : Device} -> UserDeviceTape d => {b, m, n : Nat} ->
                 Tensor [b, m] d dt g -> Tensor [b, n] d dt g ->
                 IO (Tensor [b, m + n] d dt g)
tconcat2dAxis1 a b = ioRerun (\_ => MkTensor (primConcat2dAxis1 {d} a.tensorPtr b.tensorPtr) Nothing)

-- Activations (shape-preserving, pass-through autograd) ---------------
-- All `%inline` for hot-path performance — see `tadd` rationale.

export %inline
ttanh : {0 d : Device} -> UserDeviceCore d => Tensor dims d dt g -> IO (Tensor dims d dt g)
ttanh v = ioRerun (\_ => MkTensor (primTanh {d} v.tensorPtr) Nothing)

export %inline
tsigmoid : {0 d : Device} -> UserDeviceCore d => Tensor dims d dt g -> IO (Tensor dims d dt g)
tsigmoid v = ioRerun (\_ => MkTensor (primSigmoid {d} v.tensorPtr) Nothing)

export %inline
trelu : {0 d : Device} -> UserDeviceCore d => Tensor dims d dt g -> IO (Tensor dims d dt g)
trelu v = ioRerun (\_ => MkTensor (primClampMin {d} v.tensorPtr 0.0) Nothing)

export %inline
tgelu : {0 d : Device} -> UserDeviceTape d => Tensor dims d dt g -> IO (Tensor dims d dt g)
tgelu v = ioRerun (\_ => MkTensor (primGelu {d} v.tensorPtr) Nothing)

export %inline
tsilu : {0 d : Device} -> UserDeviceTape d => Tensor dims d dt g -> IO (Tensor dims d dt g)
tsilu v = ioRerun (\_ => MkTensor (primSilu {d} v.tensorPtr) Nothing)

export %inline
tleakyRelu : {0 d : Device} -> UserDeviceTape d => Double -> Tensor dims d dt g -> IO (Tensor dims d dt g)
tleakyRelu slope v = ioRerun (\_ => MkTensor (primLeakyRelu {d} v.tensorPtr slope) Nothing)

||| Softmax along axis 0 (1D vector).
export %inline
tsoftmax1d : {0 d : Device} -> UserDeviceTape d => {n : Nat} -> Tensor [n] d dt g -> IO (Tensor [n] d dt g)
tsoftmax1d v = ioRerun (\_ => MkTensor (primSoftmax {d} v.tensorPtr 0) Nothing)

||| Log-softmax along axis 0 (1D vector).
export %inline
tlogSoftmax1d : {0 d : Device} -> UserDeviceTape d => {n : Nat} -> Tensor [n] d dt g -> IO (Tensor [n] d dt g)
tlogSoftmax1d v = ioRerun (\_ => MkTensor (primLogSoftmax {d} v.tensorPtr 0) Nothing)

||| Fused LSTM gate computation: combined gates [4 * n] + previous cell [n]
||| → (new hidden [n], new cell [n]). Wraps `prim__lstmGatesPair`.
|||
||| The gate-vector size is encoded statically as `TVec (4 * n) d`
||| (alias for `Tensor [4 * n] d`). Routing the `4 * n` through the
||| `TVec` alias avoids the type-checker hang that direct
||| `Tensor [4 * n] d` triggers.
export
tlstmGatesPair : {n : Nat} -> TVec (4 * n) d dt g -> TVec n d dt g ->
                 IO (TVec n d dt g, TVec n d dt g)
tlstmGatesPair {n} combined prevCell = ioRerun (\_ =>
  let nI = cast {to=Int} n
      pair = prim__lstmGatesPair combined.tensorPtr prevCell.tensorPtr nI
  in (MkTensor (prim__pairFirst pair) Nothing, MkTensor (prim__pairSecond pair) Nothing))

||| Allocate a zero-initialised persistent state Tensor of size [n].
||| Use for LSTM/RNN/GRU initial hidden + cell state. Persistent =
||| survives tape reset.
export
tzeroState1d : {n : Nat} -> IO (Tensor [n] d dt g)
tzeroState1d {n} = ioRerun (\_ =>
  let nI = cast {to=Int} n
      buf = prim__allocDoubles nI
  in MkTensor (prim__createState1d nI buf) Nothing)

||| GRU cell — `nn.GRU` equation. Takes the two `[3 * n]` half-sums:
|||   ih = W_ih @ x + b_ih
|||   hh = W_hh @ h + b_hh
||| (computed by the caller via `tlinear`) plus the previous hidden
||| state. Internally:
|||   z = sigmoid(ih_z + hh_z),  r = sigmoid(ih_r + hh_r)
|||   n = tanh(ih_n + r * hh_n)
|||   h' = (1 - z) * n + z * prev
||| Pre-2026-05-09 this took a single fused `combined = ih + hh`
||| and ignored r (simplified GRU); aligned to the standard
||| `nn.GRU` equation so the example matches what library users
||| expect.
export
tgruCell : {n : Nat} -> TVec (3 * n) d dt g -> TVec (3 * n) d dt g -> TVec n d dt g -> IO (TVec n d dt g)
tgruCell {n} ih hh prevH = ioRerun (\_ =>
  let nI = cast {to=Int} n
  in MkTensor (prim__gruCell ih.tensorPtr hh.tensorPtr prevH.tensorPtr nI) Nothing)

-- Scalar boundary --------------------------------------------------

||| Read the scalar value out of a `Tensor [] d`.
export
tensorItem : Tensor [] d dt g -> Double
tensorItem v = prim__item v.tensorPtr

||| Run backward on a loss tensor. The loss MUST be `WithGrad` —
||| a `NoGrad` scalar can't have come from a path the autograd tape
||| recorded, so backward would be a silent no-op at best and a
||| malformed-tape crash at worst. Rejecting at the type level
||| catches "loss computed inside `withNoGrad`, then fed to training"
||| — the bug class the entire `GradMode` refactor exists to prevent.
export
runBackward : Tensor [] d dt WithGrad -> IO ()
runBackward t = primIO (prim__backwardC t.tensorPtr)

-- Loss (vector targets → scalar loss) ---------------------------------

||| MSE loss over a 1D prediction/target pair. Sum-reduced.
export
tmseLoss : {n : Nat} -> Tensor [n] d dt g -> Tensor [n] d dt g -> IO (Tensor [] d dt g)
tmseLoss p t = ioRerun (\_ =>
  let diff = prim__sub p.tensorPtr t.tensorPtr in
  let sqDiff = prim__mul diff diff in
  MkTensor (prim__sum sqDiff) Nothing)

||| NLL loss against a one-hot target. Mirrors
||| `Example.Supervised.nllLossTensor` (divide by n to match the
||| reference's mean reduction).
export
tnllLoss : {n : Nat} -> Tensor [n] d dt g -> Tensor [n] d dt g -> IO (Tensor [] d dt g)
tnllLoss {n} p t = ioRerun (\_ =>
  let logP = prim__logSoftmax p.tensorPtr 0 in
  let prod = prim__mul logP t.tensorPtr in
  let neg = prim__neg (prim__sum prod) in
  MkTensor (prim__mulScalar neg (1.0 / cast n)) Nothing)

||| Binary cross-entropy with logits, mean-reduced. Numerically stable
||| (wraps `prim__bceWithLogits`). For multi-element predictions/targets
||| use `tbceLoss : Tensor [n] d dt g-> Tensor [n] d dt g-> Tensor [] d dt g`;
||| the C op internally averages. Polymorphic in `g`: the loss's
||| grad-mode matches the predictions / targets, so a no-grad eval
||| `tbceLoss` (e.g. inside `withNoGrad`) returns a `NoGrad` scalar
||| that the type system will reject if accidentally fed to
||| `nativeTrainStep`.
export
tbceLoss : {n : Nat} -> Tensor [n] d dt g -> Tensor [n] d dt g -> IO (Tensor [] d dt g)
tbceLoss p t = ioRerun (\_ =>
  MkTensor (prim__bceWithLogits p.tensorPtr t.tensorPtr) Nothing)

-- Optimizer shim ------------------------------------------------------

||| Fused native train step on a Tensor loss: zero_grad → backward →
||| clip → step. Reads `prim__item` BEFORE the step so the returned
||| scalar is not stale. Mirrors `nativeTrainStep`.
export
nativeTrainStep : {0 d : Device} -> NativeOptimizer -> Tensor [] d dt WithGrad -> IO Double
nativeTrainStep opt loss = ioRerun (\_ =>
  let clipMode : Int
      clipMode = case opt.clipMode of NoClip => 0; ValueClip _ => 1; NormClip _ => 2
      clipVal  : Double
      clipVal  = case opt.clipMode of NoClip => 0.0; ValueClip v => v; NormClip v => v
      lossVal  = prim__item loss.tensorPtr
  in prim__nativeTrainStep opt.handle clipMode clipVal loss.tensorPtr lossVal)
