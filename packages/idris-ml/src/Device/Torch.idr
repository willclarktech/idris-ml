||| `TorchDev` — `UserDeviceCore` instance for the libtorch backend.
|||
||| Forwards to the torch-suffixed C symbols emitted under Phase 1's
||| `rename_torch.h` (e.g. `tensor_add_torch`). Only resolvable at
||| runtime if the build's BACKEND list includes `torch`.
module Device.Torch

import Device.Core


----------------------------------------------------------------------
-- Per-symbol bindings to the torch backend's suffixed C exports
----------------------------------------------------------------------

%foreign "C:tensor_create_scalar_torch,libidrisml"
prim__createScalarTorch : Double -> Int -> AnyPtr

%foreign "C:tensor_create_torch,libidrisml"
prim__createTorch : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr

%foreign "C:tensor_free_torch,libidrisml"
prim__freeTorch : AnyPtr -> ()

%foreign "C:tensor_item_torch,libidrisml"
prim__itemTorch : AnyPtr -> Double

%foreign "C:tensor_clone_torch,libidrisml"
prim__cloneTorch : AnyPtr -> AnyPtr

%foreign "C:tensor_add_torch,libidrisml"
prim__addTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_sub_torch,libidrisml"
prim__subTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_mul_torch,libidrisml"
prim__mulTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_div_torch,libidrisml"
prim__divTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_neg_torch,libidrisml"
prim__negTorch : AnyPtr -> AnyPtr

%foreign "C:tensor_abs_torch,libidrisml"
prim__absTorch : AnyPtr -> AnyPtr

%foreign "C:tensor_exp_torch,libidrisml"
prim__expTorch : AnyPtr -> AnyPtr

%foreign "C:tensor_log_torch,libidrisml"
prim__logTorch : AnyPtr -> AnyPtr

%foreign "C:tensor_sqrt_torch,libidrisml"
prim__sqrtTorch : AnyPtr -> AnyPtr

%foreign "C:tensor_pow_torch,libidrisml"
prim__powTorch : AnyPtr -> AnyPtr -> AnyPtr

%foreign "C:tensor_sigmoid_torch,libidrisml"
prim__sigmoidTorch : AnyPtr -> AnyPtr

%foreign "C:tensor_tanh_torch,libidrisml"
prim__tanhTorch : AnyPtr -> AnyPtr

%foreign "C:tensor_add_scalar_torch,libidrisml"
prim__addScalarTorch : AnyPtr -> Double -> AnyPtr

%foreign "C:tensor_mul_scalar_torch,libidrisml"
prim__mulScalarTorch : AnyPtr -> Double -> AnyPtr

%foreign "C:tensor_clamp_min_torch,libidrisml"
prim__clampMinTorch : AnyPtr -> Double -> AnyPtr


----------------------------------------------------------------------
-- TorchDev type + UserDeviceCore instance
----------------------------------------------------------------------

public export
data TorchDev : Type where MkTorchDev : TorchDev

public export
UserDeviceCore TorchDev where
  deviceName       = "torch"
  primCreateScalar = prim__createScalarTorch
  primCreate       = prim__createTorch
  primFree         = prim__freeTorch
  primItem         = prim__itemTorch
  primClone        = prim__cloneTorch
  primAdd          = prim__addTorch
  primSub          = prim__subTorch
  primMul          = prim__mulTorch
  primDiv          = prim__divTorch
  primNeg          = prim__negTorch
  primAbs          = prim__absTorch
  primExp          = prim__expTorch
  primLog          = prim__logTorch
  primSqrt         = prim__sqrtTorch
  primPow          = prim__powTorch
  primSigmoid      = prim__sigmoidTorch
  primTanh         = prim__tanhTorch
  primAddScalar    = prim__addScalarTorch
  primMulScalar    = prim__mulScalarTorch
  primClampMin     = prim__clampMinTorch
