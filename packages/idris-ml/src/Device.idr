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

||| CUDA device tag, parameterised by device index. Untested as of
||| Phase 2.1b — the torch backend's CUDA path is wired but never
||| exercised in CI.
public export
data CUDA : Nat -> Type where MkCUDA : (n : Nat) -> CUDA n

||| MPS (Apple Metal Performance Shaders) device tag. Untested.
public export
data MPS : Type where MkMPS : MPS


----------------------------------------------------------------------
-- Unified-name FFI bindings (Phase 1's primary-backend aliases)
----------------------------------------------------------------------

%foreign "C:tensor_create_scalar,libidrisml"
prim__createScalarUnified : Double -> Int -> AnyPtr

%foreign "C:tensor_create,libidrisml"
prim__createUnified : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr

%foreign "C:tensor_free,libidrisml"
prim__freeUnified : AnyPtr -> ()

%foreign "C:tensor_item,libidrisml"
prim__itemUnified : AnyPtr -> Double

%foreign "C:tensor_clone,libidrisml"
prim__cloneUnified : AnyPtr -> AnyPtr

%foreign "C:tensor_add,libidrisml"
prim__addUnified : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_sub,libidrisml"
prim__subUnified : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_mul,libidrisml"
prim__mulUnified : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_div,libidrisml"
prim__divUnified : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_neg,libidrisml"
prim__negUnified : AnyPtr -> AnyPtr

%foreign "C:tensor_abs,libidrisml"
prim__absUnified : AnyPtr -> AnyPtr

%foreign "C:tensor_exp,libidrisml"
prim__expUnified : AnyPtr -> AnyPtr

%foreign "C:tensor_log,libidrisml"
prim__logUnified : AnyPtr -> AnyPtr

%foreign "C:tensor_sqrt,libidrisml"
prim__sqrtUnified : AnyPtr -> AnyPtr

%foreign "C:tensor_pow,libidrisml"
prim__powUnified : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_sigmoid,libidrisml"
prim__sigmoidUnified : AnyPtr -> AnyPtr

%foreign "C:tensor_tanh,libidrisml"
prim__tanhUnified : AnyPtr -> AnyPtr

%foreign "C:tensor_add_scalar,libidrisml"
prim__addScalarUnified : AnyPtr -> Double -> AnyPtr

%foreign "C:tensor_mul_scalar,libidrisml"
prim__mulScalarUnified : AnyPtr -> Double -> AnyPtr

%foreign "C:tensor_clamp_min,libidrisml"
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
UserDeviceCore (CUDA n) where
  deviceName       = "cuda"  -- index dropped at this layer; toDevice picks it up
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
