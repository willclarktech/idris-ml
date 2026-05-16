||| `TapeDev` — `UserDeviceCore` instance for the tape backend.
|||
||| Forwards to the tape-suffixed C symbols emitted under Phase 1's
||| `rename_tape.h` (e.g. `tensor_add_tape`). Only resolvable at
||| runtime if the build's BACKEND list includes `tape`.
module Device.Tape

import Device.Core


----------------------------------------------------------------------
-- Per-symbol bindings to the tape backend's suffixed C exports
----------------------------------------------------------------------

%foreign "C:tensor_create_scalar_tape,libidrisml"
prim__createScalarTape : Double -> Int -> AnyPtr

%foreign "C:tensor_create_tape,libidrisml"
prim__createTape : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr

%foreign "C:tensor_free_tape,libidrisml"
prim__freeTape : AnyPtr -> ()

%foreign "C:tensor_item_tape,libidrisml"
prim__itemTape : AnyPtr -> Double

%foreign "C:tensor_clone_tape,libidrisml"
prim__cloneTape : AnyPtr -> AnyPtr

%foreign "C:tensor_add_tape,libidrisml"
prim__addTape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_sub_tape,libidrisml"
prim__subTape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_mul_tape,libidrisml"
prim__mulTape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_div_tape,libidrisml"
prim__divTape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_neg_tape,libidrisml"
prim__negTape : AnyPtr -> AnyPtr

%foreign "C:tensor_abs_tape,libidrisml"
prim__absTape : AnyPtr -> AnyPtr

%foreign "C:tensor_exp_tape,libidrisml"
prim__expTape : AnyPtr -> AnyPtr

%foreign "C:tensor_log_tape,libidrisml"
prim__logTape : AnyPtr -> AnyPtr

%foreign "C:tensor_sqrt_tape,libidrisml"
prim__sqrtTape : AnyPtr -> AnyPtr

%foreign "C:tensor_pow_tape,libidrisml"
prim__powTape : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_sigmoid_tape,libidrisml"
prim__sigmoidTape : AnyPtr -> AnyPtr

%foreign "C:tensor_tanh_tape,libidrisml"
prim__tanhTape : AnyPtr -> AnyPtr

%foreign "C:tensor_add_scalar_tape,libidrisml"
prim__addScalarTape : AnyPtr -> Double -> AnyPtr

%foreign "C:tensor_mul_scalar_tape,libidrisml"
prim__mulScalarTape : AnyPtr -> Double -> AnyPtr

%foreign "C:tensor_clamp_min_tape,libidrisml"
prim__clampMinTape : AnyPtr -> Double -> AnyPtr


----------------------------------------------------------------------
-- TapeDev type + UserDeviceCore instance
----------------------------------------------------------------------

||| The tape backend's `UserDeviceCore` instance head. An empty type
||| — it has no values; `Tensor [..] TapeDev` is just a typed tag for
||| "this tensor lives on the tape backend".
public export
data TapeDev : Type where MkTapeDev : TapeDev

public export
UserDeviceCore TapeDev where
  deviceName       = "tape"
  primCreateScalar = prim__createScalarTape
  primCreate       = prim__createTape
  primFree         = prim__freeTape
  primItem         = prim__itemTape
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
