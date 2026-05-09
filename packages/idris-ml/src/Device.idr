-- | Device types for type-safe tensor placement.
-- |
-- | Phantom parameter on Tensor prevents mixing tensors from
-- | different devices at compile time.

module Device

import Decidable.Equality


----------------------------------------------------------------------
-- Device Type
----------------------------------------------------------------------

||| Target device for tensor computation.
public export
data Device = CPU | CUDA Nat | MPS


----------------------------------------------------------------------
-- Instances
----------------------------------------------------------------------

public export
Show Device where
  show CPU = "CPU"
  show (CUDA n) = "CUDA:" ++ show n
  show MPS = "MPS"

public export
Eq Device where
  CPU == CPU = True
  (CUDA n) == (CUDA m) = n == m
  MPS == MPS = True
  _ == _ = False

public export
DecEq Device where
  decEq CPU CPU = Yes Refl
  decEq MPS MPS = Yes Refl
  decEq (CUDA n) (CUDA m) with (decEq n m)
    decEq (CUDA n) (CUDA n) | Yes Refl = Yes Refl
    decEq (CUDA n) (CUDA m) | No contra = No (\case Refl => contra Refl)
  decEq CPU (CUDA _) = No (\case Refl impossible)
  decEq CPU MPS = No (\case Refl impossible)
  decEq (CUDA _) CPU = No (\case Refl impossible)
  decEq (CUDA _) MPS = No (\case Refl impossible)
  decEq MPS CPU = No (\case Refl impossible)
  decEq MPS (CUDA _) = No (\case Refl impossible)


----------------------------------------------------------------------
-- Helpers
----------------------------------------------------------------------

||| Convert device to C backend string ("cpu", "cuda:0", "mps").
public export
deviceToString : Device -> String
deviceToString CPU = "cpu"
deviceToString (CUDA n) = "cuda:" ++ show n
deviceToString MPS = "mps"
