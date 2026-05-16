||| `MlxDev` — `UserDeviceCore` instance for the mlx backend.
|||
||| Forwards to the mlx-suffixed C symbols emitted under Phase 1's
||| `rename_mlx.h` (e.g. `tensor_add_mlx`). Only resolvable at runtime
||| if the build's BACKEND list includes `mlx` (Apple-only).
module Device.Mlx

import Device.Core


----------------------------------------------------------------------
-- Per-symbol bindings to the mlx backend's suffixed C exports
----------------------------------------------------------------------

%foreign "C:tensor_create_scalar_mlx,libidrisml"
prim__createScalarMlx : Double -> Int -> AnyPtr

%foreign "C:tensor_create_mlx,libidrisml"
prim__createMlx : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr

%foreign "C:tensor_free_mlx,libidrisml"
prim__freeMlx : AnyPtr -> ()

%foreign "C:tensor_item_mlx,libidrisml"
prim__itemMlx : AnyPtr -> Double

%foreign "C:tensor_clone_mlx,libidrisml"
prim__cloneMlx : AnyPtr -> AnyPtr

%foreign "C:tensor_add_mlx,libidrisml"
prim__addMlx : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_sub_mlx,libidrisml"
prim__subMlx : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_mul_mlx,libidrisml"
prim__mulMlx : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_div_mlx,libidrisml"
prim__divMlx : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_neg_mlx,libidrisml"
prim__negMlx : AnyPtr -> AnyPtr

%foreign "C:tensor_abs_mlx,libidrisml"
prim__absMlx : AnyPtr -> AnyPtr

%foreign "C:tensor_exp_mlx,libidrisml"
prim__expMlx : AnyPtr -> AnyPtr

%foreign "C:tensor_log_mlx,libidrisml"
prim__logMlx : AnyPtr -> AnyPtr

%foreign "C:tensor_sqrt_mlx,libidrisml"
prim__sqrtMlx : AnyPtr -> AnyPtr

%foreign "C:tensor_pow_mlx,libidrisml"
prim__powMlx : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_sigmoid_mlx,libidrisml"
prim__sigmoidMlx : AnyPtr -> AnyPtr

%foreign "C:tensor_tanh_mlx,libidrisml"
prim__tanhMlx : AnyPtr -> AnyPtr

%foreign "C:tensor_add_scalar_mlx,libidrisml"
prim__addScalarMlx : AnyPtr -> Double -> AnyPtr

%foreign "C:tensor_mul_scalar_mlx,libidrisml"
prim__mulScalarMlx : AnyPtr -> Double -> AnyPtr

%foreign "C:tensor_clamp_min_mlx,libidrisml"
prim__clampMinMlx : AnyPtr -> Double -> AnyPtr


----------------------------------------------------------------------
-- MlxDev type + UserDeviceCore instance
----------------------------------------------------------------------

public export
data MlxDev : Type where MkMlxDev : MlxDev

public export
UserDeviceCore MlxDev where
  deviceName       = "mlx"
  primCreateScalar = prim__createScalarMlx
  primCreate       = prim__createMlx
  primFree         = prim__freeMlx
  primItem         = prim__itemMlx
  primClone        = prim__cloneMlx
  primAdd          = prim__addMlx
  primSub          = prim__subMlx
  primMul          = prim__mulMlx
  primDiv          = prim__divMlx
  primNeg          = prim__negMlx
  primAbs          = prim__absMlx
  primExp          = prim__expMlx
  primLog          = prim__logMlx
  primSqrt         = prim__sqrtMlx
  primPow          = prim__powMlx
  primSigmoid      = prim__sigmoidMlx
  primTanh         = prim__tanhMlx
  primAddScalar    = prim__addScalarMlx
  primMulScalar    = prim__mulScalarMlx
  primClampMin     = prim__clampMinMlx
