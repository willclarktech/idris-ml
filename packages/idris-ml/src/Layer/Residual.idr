-- | Residual connection wrapper.
-- |
-- | Wraps an inner layer and adds a skip connection:
-- |   output = input + inner(input)
-- |
-- | Input and output dimensions must be equal (standard residual).

module Layer.Residual

import Data.Vect

import Device
import Endofunctor
import Floating
import Layer.Core
import Math
import Tensor
import Variable


----------------------------------------------------------------------
-- Residual State
----------------------------------------------------------------------

||| Residual wrapper: output = input + inner(input).
||| The inner layer must have the same input and output dimensions.
public export
data ResidualState : Nat -> Nat -> Type -> Type where
  MkResidual : AnyLayer n n ty -> ResidualState n n ty


----------------------------------------------------------------------
-- LayerLike Instance
----------------------------------------------------------------------

%default partial
export
LayerLike ResidualState where
  applyGeneric _ _ = idris_crash "Residual: use tensor path"
  applyVar {d} _ _ = idris_crash "Residual: use tensor path"

  applyVarTensor {d} (MkResidual inner) inputT =
    let innerPair = applyVarTensorAny inner inputT
        innerOut = snd innerPair
        inner' = fst innerPair
    in (MkResidual inner', tensorAdd inputT innerOut)

  emapLayer f (MkResidual inner) = MkResidual (emap f inner)

  showLayer (MkResidual inner) = "Residual<" ++ show inner ++ ">"

  nameLayer {d} pfx (MkResidual (MkAnyLayer l @{dict} layer)) =
    MkResidual (MkAnyLayer l @{dict} (nameLayer @{dict} pfx layer))

  layerPrefix _ = "res"

  toDoubleLayer {d} (MkResidual (MkAnyLayer l @{dict} layer)) =
    MkResidual (MkAnyLayer l @{dict} (toDoubleLayer @{dict} layer))

  setTraining mode (MkResidual (MkAnyLayer l @{dict} layer)) =
    MkResidual (MkAnyLayer l @{dict} (setTraining @{dict} mode layer))

  debugApply _ _ = idris_crash "Residual: use tensor path"


----------------------------------------------------------------------
-- Constructor
----------------------------------------------------------------------

||| Wrap a same-dim layer with a residual (skip) connection.
||| Output = input + inner(input).
export
residualLayer : {n : Nat} -> AnyLayer n n ty -> AnyLayer n n ty
residualLayer inner = MkAnyLayer ResidualState (MkResidual inner)
