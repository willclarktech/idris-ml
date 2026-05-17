||| Device tags for type-safe tensor placement.
|||
||| **Phase 2.1b change**: `Device` was a closed sum
||| (`CPU | CUDA Nat | MPS`); it's now a kind-level slot. `CPU`,
||| `CUDA n`, `MPS` are *types*, not values. `Tensor`'s phantom
||| parameter is now `0 d : Type` (was `0 d : Device`).
|||
||| Each of `CPU` / `CUDA n` / `MPS` has a `UserDeviceCore` instance
||| forwarding to the build's primary backend via unified-name C
||| symbols (`tensor_add` aliased to `tensor_add_<primary>` per
||| Phase 1's rename + alias mechanism). For backend-specific
||| dispatch, use `Device.Tape` / `Device.Torch` / `Device.Mlx`'s
||| `TapeDev` / `TorchDev` / `MlxDev` tags directly — those bind to
||| the suffixed symbols.
|||
||| Users implementing a custom backend declare their own type and
||| `UserDeviceCore` instance; see `docs/develop/design-decisions.md`
||| "Pluggable Device" for the recipe.
module Device

import public Device.Core
import public DType.Core


----------------------------------------------------------------------
-- Default device tags (host CPU / CUDA / MPS) — Phase 2.1b
--
-- These are *types*, not values. Existing `Tensor [..] CPU` keeps
-- compiling because Tensor's phantom is `0 d : Type`. The
-- `UserDeviceCore` instances forward to unified-name C symbols
-- (`tensor_add` — aliased by Phase 1's link step to the primary
-- backend's `tensor_add_<primary>`), so any `Tensor [..] CPU` op
-- transparently runs on whatever the build's primary backend is.
----------------------------------------------------------------------

||| Host CPU device tag. Forwards to the primary backend's CPU-side
||| operations.
public export
data CPU : Type where MkCPU : CPU

||| CUDA device tag, parameterised by device index. Untested (the
||| torch backend's CUDA path is wired but never exercised in CI as
||| of 2026-05-13).
public export
data CUDA : Nat -> Type where MkCUDA : (n : Nat) -> CUDA n

||| MPS (Apple Metal Performance Shaders) device tag. Untested.
public export
data MPS : Type where MkMPS : MPS


----------------------------------------------------------------------
-- HasDeviceIndex — runtime-observable parameter for parameterized
-- devices (e.g. `CUDA n`)
--
-- `UserDeviceCore` declares its `d` parameter at 0-quantity (it's a
-- pure type-level dispatch tag), so an instance method body cannot
-- observe the value of `d`'s type-level parameters. That makes
-- writing `deviceName = "cuda:" ++ show n` for `UserDeviceCore
-- (CUDA n)` impossible directly — `n` is erased.
--
-- `HasDeviceIndex` carries the runtime index separately: a
-- non-erased typeclass over the device. The method `deviceIndex`
-- returns the `Nat` parameter, so `UserDeviceCore (CUDA n)`'s
-- `deviceName` can call `deviceIndex` to recover it.
--
-- See `docs/grad-mode-and-device-typing.md` "Parameterized devices"
-- and `docs/develop/design-decisions.md` "Open `d` parameter".
----------------------------------------------------------------------

||| Devices whose type carries a runtime-observable Nat index (CUDA's
||| device number is the canonical example). Methods of
||| `UserDeviceCore` that need to see the parameter — most commonly
||| `deviceName` — call `deviceIndex` to recover it.
public export
interface HasDeviceIndex (d : Device) where
  deviceIndex : Nat

public export
{n : Nat} -> HasDeviceIndex (CUDA n) where
  deviceIndex = n


----------------------------------------------------------------------
-- Unified-name FFI bindings (Phase 1's primary-backend aliases)
----------------------------------------------------------------------

%foreign "scheme:(lambda (a0 a1) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_create_scalar\" (double int) void*) a0 a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__createScalarUnified : Double -> Int -> AnyPtr

%foreign "scheme:(lambda (a0 a1 a2 a3) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_create\" (void* void* int int) void*) a0 a1 a2 a3))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__createUnified : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr

%foreign "scheme:(lambda (a0)  ((foreign-procedure \"tensor_free\" (void*) void) (vector-ref a0 1)))"
prim__freeUnified : AnyPtr -> ()

%foreign "scheme:(lambda (a0)  ((foreign-procedure \"tensor_item\" (void*) double) (vector-ref a0 1)))"
prim__itemUnified : AnyPtr -> Double

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_clone\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__cloneUnified : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_add\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__addUnified : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_sub\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__subUnified : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_mul\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__mulUnified : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_div\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__divUnified : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_neg\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__negUnified : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_abs\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__absUnified : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_exp\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__expUnified : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_log\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__logUnified : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_sqrt\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__sqrtUnified : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_pow\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__powUnified : AnyPtr -> AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_sigmoid\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__sigmoidUnified : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_tanh\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__tanhUnified : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_add_scalar\" (void* double) void*) (vector-ref a0 1) a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__addScalarUnified : AnyPtr -> Double -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_mul_scalar\" (void* double) void*) (vector-ref a0 1) a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__mulScalarUnified : AnyPtr -> Double -> AnyPtr

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_clamp_min\" (void* double) void*) (vector-ref a0 1) a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__clampMinUnified : AnyPtr -> Double -> AnyPtr


----------------------------------------------------------------------
-- UserDeviceCore instances — all three default tags forward through
-- the same unified-name FFI surface (Phase 1's primary-backend
-- aliases). This makes `Tensor [..] CPU` portable across builds:
-- whichever backend is primary handles the dispatch.
----------------------------------------------------------------------

public export
UserDeviceCore CPU where
  deviceName       = "cpu"
  primCreateScalar = prim__createScalarUnified
  primCreate       = prim__createUnified
  primFree         = prim__freeUnified
  primItem         = prim__itemUnified
  primClone        = prim__cloneUnified
  primAdd          = prim__addUnified
  primSub          = prim__subUnified
  primMul          = prim__mulUnified
  primDiv          = prim__divUnified
  primNeg          = prim__negUnified
  primAbs          = prim__absUnified
  primExp          = prim__expUnified
  primLog          = prim__logUnified
  primSqrt         = prim__sqrtUnified
  primPow          = prim__powUnified
  primSigmoid      = prim__sigmoidUnified
  primTanh         = prim__tanhUnified
  primAddScalar    = prim__addScalarUnified
  primMulScalar    = prim__mulScalarUnified
  primClampMin     = prim__clampMinUnified

public export
{n : Nat} -> UserDeviceCore (CUDA n) where
  deviceName       = "cuda:" ++ show n
  primCreateScalar = prim__createScalarUnified
  primCreate       = prim__createUnified
  primFree         = prim__freeUnified
  primItem         = prim__itemUnified
  primClone        = prim__cloneUnified
  primAdd          = prim__addUnified
  primSub          = prim__subUnified
  primMul          = prim__mulUnified
  primDiv          = prim__divUnified
  primNeg          = prim__negUnified
  primAbs          = prim__absUnified
  primExp          = prim__expUnified
  primLog          = prim__logUnified
  primSqrt         = prim__sqrtUnified
  primPow          = prim__powUnified
  primSigmoid      = prim__sigmoidUnified
  primTanh         = prim__tanhUnified
  primAddScalar    = prim__addScalarUnified
  primMulScalar    = prim__mulScalarUnified
  primClampMin     = prim__clampMinUnified

public export
UserDeviceCore MPS where
  deviceName       = "mps"
  primCreateScalar = prim__createScalarUnified
  primCreate       = prim__createUnified
  primFree         = prim__freeUnified
  primItem         = prim__itemUnified
  primClone        = prim__cloneUnified
  primAdd          = prim__addUnified
  primSub          = prim__subUnified
  primMul          = prim__mulUnified
  primDiv          = prim__divUnified
  primNeg          = prim__negUnified
  primAbs          = prim__absUnified
  primExp          = prim__expUnified
  primLog          = prim__logUnified
  primSqrt         = prim__sqrtUnified
  primPow          = prim__powUnified
  primSigmoid      = prim__sigmoidUnified
  primTanh         = prim__tanhUnified
  primAddScalar    = prim__addScalarUnified
  primMulScalar    = prim__mulScalarUnified
  primClampMin     = prim__clampMinUnified


----------------------------------------------------------------------
-- UserDeviceLinear instances (Phase 2.2). All three default tags
-- forward through unified-name FFI symbols, just like UserDeviceCore.
----------------------------------------------------------------------

%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_mv\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__mvUnified : AnyPtr -> AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_matmul\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__matmulUnified : AnyPtr -> AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_linear\" (void* void* void*) void*) (vector-ref a0 1) (vector-ref a1 1) (vector-ref a2 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__linearUnified : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_dot\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__dotUnified : AnyPtr -> AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_outer\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__outerUnified : AnyPtr -> AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_bmm\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__bmmUnified : AnyPtr -> AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_linear_2d\" (void* void* void*) void*) (vector-ref a0 1) (vector-ref a1 1) (vector-ref a2 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__linear2dUnified : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_sum\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__sumUnified : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_mean\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__meanUnified : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_min\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__tensorMinUnified : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_max\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__tensorMaxUnified : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_sum_dim\" (void* int int) void*) (vector-ref a0 1) a1 a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__sumDimUnified : AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_select\" (void* int int) void*) (vector-ref a0 1) a1 a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__selectUnified : AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_unsqueeze\" (void* int) void*) (vector-ref a0 1) a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__unsqueezeUnified : AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_squeeze\" (void* int) void*) (vector-ref a0 1) a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__squeezeUnified : AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_stack\" (void* int int) void*) a0 a1 a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__stackUnified : AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_view_1d\" (void* int) void*) (vector-ref a0 1) a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__view1dUnified : AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_view_2d\" (void* int int) void*) (vector-ref a0 1) a1 a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__view2dUnified : AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_reshape_2d\" (void* int int) void*) (vector-ref a0 1) a1 a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__reshape2dUnified : AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_reshape_3d\" (void* int int int) void*) (vector-ref a0 1) a1 a2 a3))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__reshape3dUnified : AnyPtr -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (let ((raw_r ((foreign-procedure \"tensor_reshape_4d\" (void* int int int int) void*) (vector-ref a0 1) a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__reshape4dUnified : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_narrow\" (void* int int int) void*) (vector-ref a0 1) a1 a2 a3))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__narrowUnified : AnyPtr -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_transpose_last2\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__transposeLast2Unified : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_transpose_2d\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__transpose2dUnified : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_cat\" (void* int int) void*) a0 a1 a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__catUnified : AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_cat2\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__cat2Unified : AnyPtr -> AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_concat_2d_axis1\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__concat2dAxis1Unified : AnyPtr -> AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_gather\" (void* void* int) void*) (vector-ref a0 1) (vector-ref a1 1) a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__gatherUnified : AnyPtr -> AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_scatter_add\" (void* void* int) void*) (vector-ref a0 1) (vector-ref a1 1) a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__scatterAddUnified : AnyPtr -> AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_argsort\" (void* int int) void*) (vector-ref a0 1) a1 a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__argsortUnified : AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_cumprod\" (void* int) void*) (vector-ref a0 1) a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__cumprodUnified : AnyPtr -> Int -> AnyPtr


public export
UserDeviceLinear CPU where
  primMv = prim__mvUnified
  primMatmul = prim__matmulUnified
  primLinear = prim__linearUnified
  primDot = prim__dotUnified
  primOuter = prim__outerUnified
  primBmm = prim__bmmUnified
  primLinear2d = prim__linear2dUnified
  primSum = prim__sumUnified
  primMean = prim__meanUnified
  primTensorMin = prim__tensorMinUnified
  primTensorMax = prim__tensorMaxUnified
  primSumDim = prim__sumDimUnified
  primSelect = prim__selectUnified
  primUnsqueeze = prim__unsqueezeUnified
  primSqueeze = prim__squeezeUnified
  primStack = prim__stackUnified
  primView1d = prim__view1dUnified
  primView2d = prim__view2dUnified
  primReshape2d = prim__reshape2dUnified
  primReshape3d = prim__reshape3dUnified
  primReshape4d = prim__reshape4dUnified
  primNarrow = prim__narrowUnified
  primTransposeLast2 = prim__transposeLast2Unified
  primTranspose2d = prim__transpose2dUnified
  primCat = prim__catUnified
  primCat2 = prim__cat2Unified
  primConcat2dAxis1 = prim__concat2dAxis1Unified
  primGather = prim__gatherUnified
  primScatterAdd = prim__scatterAddUnified
  primArgsort = prim__argsortUnified
  primCumprod = prim__cumprodUnified

public export
{n : Nat} -> UserDeviceLinear (CUDA n) where
  primMv = prim__mvUnified
  primMatmul = prim__matmulUnified
  primLinear = prim__linearUnified
  primDot = prim__dotUnified
  primOuter = prim__outerUnified
  primBmm = prim__bmmUnified
  primLinear2d = prim__linear2dUnified
  primSum = prim__sumUnified
  primMean = prim__meanUnified
  primTensorMin = prim__tensorMinUnified
  primTensorMax = prim__tensorMaxUnified
  primSumDim = prim__sumDimUnified
  primSelect = prim__selectUnified
  primUnsqueeze = prim__unsqueezeUnified
  primSqueeze = prim__squeezeUnified
  primStack = prim__stackUnified
  primView1d = prim__view1dUnified
  primView2d = prim__view2dUnified
  primReshape2d = prim__reshape2dUnified
  primReshape3d = prim__reshape3dUnified
  primReshape4d = prim__reshape4dUnified
  primNarrow = prim__narrowUnified
  primTransposeLast2 = prim__transposeLast2Unified
  primTranspose2d = prim__transpose2dUnified
  primCat = prim__catUnified
  primCat2 = prim__cat2Unified
  primConcat2dAxis1 = prim__concat2dAxis1Unified
  primGather = prim__gatherUnified
  primScatterAdd = prim__scatterAddUnified
  primArgsort = prim__argsortUnified
  primCumprod = prim__cumprodUnified

public export
UserDeviceLinear MPS where
  primMv = prim__mvUnified
  primMatmul = prim__matmulUnified
  primLinear = prim__linearUnified
  primDot = prim__dotUnified
  primOuter = prim__outerUnified
  primBmm = prim__bmmUnified
  primLinear2d = prim__linear2dUnified
  primSum = prim__sumUnified
  primMean = prim__meanUnified
  primTensorMin = prim__tensorMinUnified
  primTensorMax = prim__tensorMaxUnified
  primSumDim = prim__sumDimUnified
  primSelect = prim__selectUnified
  primUnsqueeze = prim__unsqueezeUnified
  primSqueeze = prim__squeezeUnified
  primStack = prim__stackUnified
  primView1d = prim__view1dUnified
  primView2d = prim__view2dUnified
  primReshape2d = prim__reshape2dUnified
  primReshape3d = prim__reshape3dUnified
  primReshape4d = prim__reshape4dUnified
  primNarrow = prim__narrowUnified
  primTransposeLast2 = prim__transposeLast2Unified
  primTranspose2d = prim__transpose2dUnified
  primCat = prim__catUnified
  primCat2 = prim__cat2Unified
  primConcat2dAxis1 = prim__concat2dAxis1Unified
  primGather = prim__gatherUnified
  primScatterAdd = prim__scatterAddUnified
  primArgsort = prim__argsortUnified
  primCumprod = prim__cumprodUnified


----------------------------------------------------------------------
-- UserDeviceNN — unified-name FFI bindings (Phase 2.3) + 3 instances.
----------------------------------------------------------------------

%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_gelu\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__geluUnified : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_leaky_relu\" (void* double) void*) (vector-ref a0 1) a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__leakyReluUnified : AnyPtr -> Double -> AnyPtr
%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_silu\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__siluUnified : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_softplus\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__softplusUnified : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_softmax\" (void* int) void*) (vector-ref a0 1) a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__softmaxUnified : AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_log_softmax\" (void* int) void*) (vector-ref a0 1) a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__logSoftmaxUnified : AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_softmax_2d\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__softmax2dUnified : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_log_softmax_2d\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__logSoftmax2dUnified : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_softmax_3d\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__softmax3dUnified : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_masked_fill\" (void* void* double) void*) (vector-ref a0 1) (vector-ref a1 1) a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__maskedFillUnified : AnyPtr -> AnyPtr -> Double -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_expand_mask\" (void* int) void*) (vector-ref a0 1) a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__expandMaskUnified : AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_causal_mask\" (int) void*) a0))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__causalMaskUnified : Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_layer_norm_2d\" (void* void* void* double) void*) (vector-ref a0 1) (vector-ref a1 1) (vector-ref a2 1) a3))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__layerNorm2dUnified : AnyPtr -> AnyPtr -> AnyPtr -> Double -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6 a7 a8 a9)  (let ((raw_r ((foreign-procedure \"tensor_batch_norm\" (void* void* void* void* void* int int int double double) void*) (vector-ref a0 1) (vector-ref a1 1) (vector-ref a2 1) (vector-ref a3 1) (vector-ref a4 1) a5 a6 a7 a8 a9))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__batchNormUnified : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> Double -> Double -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_dropout\" (void* double int int) void*) (vector-ref a0 1) a1 a2 a3))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__dropoutUnified : AnyPtr -> Double -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_embedding\" (void* void* int int) void*) (vector-ref a0 1) (vector-ref a1 1) a2 a3))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__embeddingUnified : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_cosine_similarity\" (void* void* int) void*) (vector-ref a0 1) (vector-ref a1 1) a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__cosineSimilarityUnified : AnyPtr -> AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (let ((raw_r ((foreign-procedure \"tensor_cross_attention\" (void* void* void* void* double) void*) (vector-ref a0 1) (vector-ref a1 1) (vector-ref a2 1) (vector-ref a3 1) a4))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__crossAttentionUnified : AnyPtr -> AnyPtr -> AnyPtr -> AnyPtr -> Double -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_bce_with_logits\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__bceWithLogitsUnified : AnyPtr -> AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3)  (let ((raw_r ((foreign-procedure \"tensor_gru_cell\" (void* void* void* int) void*) (vector-ref a0 1) (vector-ref a1 1) (vector-ref a2 1) a3))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__gruCellUnified : AnyPtr -> AnyPtr -> AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  ((foreign-procedure \"tensor_lstm_gates_pair\" (void* void* int) void*) (vector-ref a0 1) (vector-ref a1 1) a2))"
prim__lstmGatesPairUnified : AnyPtr -> AnyPtr -> Int -> AnyPtr
%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_pair_first\" (void*) void*) a0))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__pairFirstUnified : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_pair_second\" (void*) void*) a0))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__pairSecondUnified : AnyPtr -> AnyPtr

public export
UserDeviceNN CPU where
  primGelu             = prim__geluUnified
  primLeakyRelu        = prim__leakyReluUnified
  primSilu             = prim__siluUnified
  primSoftplus         = prim__softplusUnified
  primSoftmax          = prim__softmaxUnified
  primLogSoftmax       = prim__logSoftmaxUnified
  primSoftmax2d        = prim__softmax2dUnified
  primLogSoftmax2d     = prim__logSoftmax2dUnified
  primSoftmax3d        = prim__softmax3dUnified
  primMaskedFill       = prim__maskedFillUnified
  primExpandMask       = prim__expandMaskUnified
  primCausalMask       = prim__causalMaskUnified
  primLayerNorm2d      = prim__layerNorm2dUnified
  primBatchNorm        = prim__batchNormUnified
  primDropout          = prim__dropoutUnified
  primEmbedding        = prim__embeddingUnified
  primCosineSimilarity = prim__cosineSimilarityUnified
  primCrossAttention   = prim__crossAttentionUnified
  primBceWithLogits    = prim__bceWithLogitsUnified
  primGruCell          = prim__gruCellUnified
  primLstmGatesPair    = prim__lstmGatesPairUnified
  primPairFirst        = prim__pairFirstUnified
  primPairSecond       = prim__pairSecondUnified

public export
{n : Nat} -> UserDeviceNN (CUDA n) where
  primGelu             = prim__geluUnified
  primLeakyRelu        = prim__leakyReluUnified
  primSilu             = prim__siluUnified
  primSoftplus         = prim__softplusUnified
  primSoftmax          = prim__softmaxUnified
  primLogSoftmax       = prim__logSoftmaxUnified
  primSoftmax2d        = prim__softmax2dUnified
  primLogSoftmax2d     = prim__logSoftmax2dUnified
  primSoftmax3d        = prim__softmax3dUnified
  primMaskedFill       = prim__maskedFillUnified
  primExpandMask       = prim__expandMaskUnified
  primCausalMask       = prim__causalMaskUnified
  primLayerNorm2d      = prim__layerNorm2dUnified
  primBatchNorm        = prim__batchNormUnified
  primDropout          = prim__dropoutUnified
  primEmbedding        = prim__embeddingUnified
  primCosineSimilarity = prim__cosineSimilarityUnified
  primCrossAttention   = prim__crossAttentionUnified
  primBceWithLogits    = prim__bceWithLogitsUnified
  primGruCell          = prim__gruCellUnified
  primLstmGatesPair    = prim__lstmGatesPairUnified
  primPairFirst        = prim__pairFirstUnified
  primPairSecond       = prim__pairSecondUnified

public export
UserDeviceNN MPS where
  primGelu             = prim__geluUnified
  primLeakyRelu        = prim__leakyReluUnified
  primSilu             = prim__siluUnified
  primSoftplus         = prim__softplusUnified
  primSoftmax          = prim__softmaxUnified
  primLogSoftmax       = prim__logSoftmaxUnified
  primSoftmax2d        = prim__softmax2dUnified
  primLogSoftmax2d     = prim__logSoftmax2dUnified
  primSoftmax3d        = prim__softmax3dUnified
  primMaskedFill       = prim__maskedFillUnified
  primExpandMask       = prim__expandMaskUnified
  primCausalMask       = prim__causalMaskUnified
  primLayerNorm2d      = prim__layerNorm2dUnified
  primBatchNorm        = prim__batchNormUnified
  primDropout          = prim__dropoutUnified
  primEmbedding        = prim__embeddingUnified
  primCosineSimilarity = prim__cosineSimilarityUnified
  primCrossAttention   = prim__crossAttentionUnified
  primBceWithLogits    = prim__bceWithLogitsUnified
  primGruCell          = prim__gruCellUnified
  primLstmGatesPair    = prim__lstmGatesPairUnified
  primPairFirst        = prim__pairFirstUnified
  primPairSecond       = prim__pairSecondUnified


----------------------------------------------------------------------
-- UserDeviceConv — unified-name FFI bindings + 3 instances.
----------------------------------------------------------------------

%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (let ((raw_r ((foreign-procedure \"tensor_conv1d\" (void* void* void* int int) void*) (vector-ref a0 1) (vector-ref a1 1) (vector-ref a2 1) a3 a4))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__conv1dUnified : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"tensor_conv1d_circular\" (void* void*) void*) (vector-ref a0 1) (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__conv1dCircularUnified : AnyPtr -> AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_avg_pool1d\" (void* int int) void*) (vector-ref a0 1) a1 a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__avgPool1dUnified : AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2)  (let ((raw_r ((foreign-procedure \"tensor_max_pool1d\" (void* int int) void*) (vector-ref a0 1) a1 a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__maxPool1dUnified : AnyPtr -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6)  (let ((raw_r ((foreign-procedure \"tensor_conv2d\" (void* void* void* int int int int) void*) (vector-ref a0 1) (vector-ref a1 1) (vector-ref a2 1) a3 a4 a5 a6))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__conv2dUnified : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4 a5 a6)  (let ((raw_r ((foreign-procedure \"tensor_conv2d_batched\" (void* void* void* int int int int) void*) (vector-ref a0 1) (vector-ref a1 1) (vector-ref a2 1) a3 a4 a5 a6))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__conv2dBatchedUnified : AnyPtr -> AnyPtr -> AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (let ((raw_r ((foreign-procedure \"tensor_avg_pool2d\" (void* int int int int) void*) (vector-ref a0 1) a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__avgPool2dUnified : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (let ((raw_r ((foreign-procedure \"tensor_max_pool2d\" (void* int int int int) void*) (vector-ref a0 1) a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__maxPool2dUnified : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3 a4)  (let ((raw_r ((foreign-procedure \"tensor_max_pool2d_batched\" (void* int int int int) void*) (vector-ref a0 1) a1 a2 a3 a4))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__maxPool2dBatchedUnified : AnyPtr -> Int -> Int -> Int -> Int -> AnyPtr

public export
UserDeviceConv CPU where
  primConv1d           = prim__conv1dUnified
  primConv1dCircular   = prim__conv1dCircularUnified
  primAvgPool1d        = prim__avgPool1dUnified
  primMaxPool1d        = prim__maxPool1dUnified
  primConv2d           = prim__conv2dUnified
  primConv2dBatched    = prim__conv2dBatchedUnified
  primAvgPool2d        = prim__avgPool2dUnified
  primMaxPool2d        = prim__maxPool2dUnified
  primMaxPool2dBatched = prim__maxPool2dBatchedUnified

public export
{n : Nat} -> UserDeviceConv (CUDA n) where
  primConv1d           = prim__conv1dUnified
  primConv1dCircular   = prim__conv1dCircularUnified
  primAvgPool1d        = prim__avgPool1dUnified
  primMaxPool1d        = prim__maxPool1dUnified
  primConv2d           = prim__conv2dUnified
  primConv2dBatched    = prim__conv2dBatchedUnified
  primAvgPool2d        = prim__avgPool2dUnified
  primMaxPool2d        = prim__maxPool2dUnified
  primMaxPool2dBatched = prim__maxPool2dBatchedUnified

public export
UserDeviceConv MPS where
  primConv1d           = prim__conv1dUnified
  primConv1dCircular   = prim__conv1dCircularUnified
  primAvgPool1d        = prim__avgPool1dUnified
  primMaxPool1d        = prim__maxPool1dUnified
  primConv2d           = prim__conv2dUnified
  primConv2dBatched    = prim__conv2dBatchedUnified
  primAvgPool2d        = prim__avgPool2dUnified
  primMaxPool2d        = prim__maxPool2dUnified
  primMaxPool2dBatched = prim__maxPool2dBatchedUnified


----------------------------------------------------------------------
-- UserDeviceTape — unified-name FFI bindings + 3 instances.
----------------------------------------------------------------------

%foreign "scheme:(lambda (a0)  ((foreign-procedure \"tensor_requires_grad\" (void*) int) (vector-ref a0 1)))"
prim__requiresGradUnified : AnyPtr -> Int
%foreign "scheme:(lambda (a0 a1)  ((foreign-procedure \"tensor_set_requires_grad\" (void* int) void) (vector-ref a0 1) a1))"
prim__setRequiresGradUnified : AnyPtr -> Int -> PrimIO ()
%foreign "C:tensor_no_grad_begin,libidrisml"
prim__noGradBeginUnified : PrimIO ()
%foreign "C:tensor_no_grad_end,libidrisml"
prim__noGradEndUnified : PrimIO ()
%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_detach\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__detachUnified : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0)  (let ((raw_r ((foreign-procedure \"tensor_with_grad\" (void*) void*) (vector-ref a0 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__withGradUnified : AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0)  ((foreign-procedure \"tensor_dim\" (void*) int) (vector-ref a0 1)))"
prim__tensorDimUnified : AnyPtr -> Int
%foreign "scheme:(lambda (a0 a1)  ((foreign-procedure \"tensor_size\" (void* int) int) (vector-ref a0 1) a1))"
prim__tensorSizeAtUnified : AnyPtr -> Int -> Int
%foreign "scheme:(lambda (a0 a1)  (let ((raw_r ((foreign-procedure \"param_register_return\" (string void*) void*) a0 (vector-ref a1 1)))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__paramRegisterUnified : String -> AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_create_param_1d\" (int void*) void*) a0 a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__createParam1dUnified : Int -> AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_create_param_2d\" (int int void*) void*) a0 a1 a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__createParam2dUnified : Int -> Int -> AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2 a3) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_create_param_3d\" (int int int void*) void*) a0 a1 a2 a3))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__createParam3dUnified : Int -> Int -> Int -> AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_create_state_1d\" (int void*) void*) a0 a1))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__createState1dUnified : Int -> AnyPtr -> AnyPtr
%foreign "scheme:(lambda (a0 a1 a2) (when (not (top-level-bound? 'idris-tensor-guardian)) (set-top-level-value! 'idris-tensor-guardian (make-guardian))) (let ((raw_r ((foreign-procedure \"tensor_create_state_2d\" (int int void*) void*) a0 a1 a2))) (let ((wr (vector 'tensor-handle raw_r))) ((top-level-value 'idris-tensor-guardian) wr) ((foreign-procedure \"tensor_retain_handle\" (void*) void) raw_r) wr)))"
prim__createState2dUnified : Int -> Int -> AnyPtr -> AnyPtr
%foreign "C:tensor_alloc_doubles,libidrisml"
prim__allocDoublesUnified : Int -> AnyPtr
%foreign "C:tensor_read_double,libidrisml"
prim__readDoubleUnified : AnyPtr -> Int -> Double

public export
UserDeviceTape CPU where
  primRequiresGrad         = prim__requiresGradUnified
  primSetRequiresGrad      = prim__setRequiresGradUnified
  primNoGradBegin          = prim__noGradBeginUnified
  primNoGradEnd            = prim__noGradEndUnified
  primDetach               = prim__detachUnified
  primWithGrad             = prim__withGradUnified
  primTensorDim            = prim__tensorDimUnified
  primTensorSizeAt         = prim__tensorSizeAtUnified
  primParamRegister        = prim__paramRegisterUnified
  primCreateParam1d        = prim__createParam1dUnified
  primCreateParam2d        = prim__createParam2dUnified
  primCreateParam3d        = prim__createParam3dUnified
  primCreateState1d        = prim__createState1dUnified
  primCreateState2d        = prim__createState2dUnified
  primAllocDoubles         = prim__allocDoublesUnified
  primReadDouble           = prim__readDoubleUnified

public export
{n : Nat} -> UserDeviceTape (CUDA n) where
  primRequiresGrad         = prim__requiresGradUnified
  primSetRequiresGrad      = prim__setRequiresGradUnified
  primNoGradBegin          = prim__noGradBeginUnified
  primNoGradEnd            = prim__noGradEndUnified
  primDetach               = prim__detachUnified
  primWithGrad             = prim__withGradUnified
  primTensorDim            = prim__tensorDimUnified
  primTensorSizeAt         = prim__tensorSizeAtUnified
  primParamRegister        = prim__paramRegisterUnified
  primCreateParam1d        = prim__createParam1dUnified
  primCreateParam2d        = prim__createParam2dUnified
  primCreateParam3d        = prim__createParam3dUnified
  primCreateState1d        = prim__createState1dUnified
  primCreateState2d        = prim__createState2dUnified
  primAllocDoubles         = prim__allocDoublesUnified
  primReadDouble           = prim__readDoubleUnified

public export
UserDeviceTape MPS where
  primRequiresGrad         = prim__requiresGradUnified
  primSetRequiresGrad      = prim__setRequiresGradUnified
  primNoGradBegin          = prim__noGradBeginUnified
  primNoGradEnd            = prim__noGradEndUnified
  primDetach               = prim__detachUnified
  primWithGrad             = prim__withGradUnified
  primTensorDim            = prim__tensorDimUnified
  primTensorSizeAt         = prim__tensorSizeAtUnified
  primParamRegister        = prim__paramRegisterUnified
  primCreateParam1d        = prim__createParam1dUnified
  primCreateParam2d        = prim__createParam2dUnified
  primCreateParam3d        = prim__createParam3dUnified
  primCreateState1d        = prim__createState1dUnified
  primCreateState2d        = prim__createState2dUnified
  primAllocDoubles         = prim__allocDoublesUnified
  primReadDouble           = prim__readDoubleUnified


----------------------------------------------------------------------
-- Compatible (default device, dt) instances
--
-- The closed-sum aliases (`CPU` / `CUDA n` / `MPS`) support dt only.
-- Backend-specific dtype combinations live in their respective
-- backend modules (`Device.Mlx` exposes the (MlxCpu/MlxGpu, F32)
-- pairs).
----------------------------------------------------------------------

public export
Compatible CPU F64 where

public export
Compatible (CUDA n) F64 where

public export
Compatible MPS F64 where
