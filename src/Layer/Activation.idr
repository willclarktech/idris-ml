module Layer.Activation

import Data.Vect

import Floating
import Layer.Core
import Math
import Tensor
import Variable


----------------------------------------------------------------------
-- Activation State
----------------------------------------------------------------------

||| Activation layer: input and output sizes must be equal.
public export
data ActivationState : Nat -> Nat -> Type -> Type where
  MkActivation : String -> ActivationFunction ty -> ActivationState n n ty


----------------------------------------------------------------------
-- LayerLike Instance
----------------------------------------------------------------------

export
LayerLike ActivationState where
  applyGeneric st@(MkActivation _ f) xs = (st, map f xs)

  applyVar st@(MkActivation _ f) xs = (st, map f xs)

  -- Tensor-level dispatch: apply prim__tanh/prim__sigmoid directly on the
  -- tensor instead of unpacking to scalars. 1 tape entry vs ~7n entries.
  applyVarTensor st@(MkActivation "tanh" _) inputT = (st, prim__tanh inputT)
  applyVarTensor st@(MkActivation "sigmoid" _) inputT = (st, prim__sigmoid inputT)
  applyVarTensor st@(MkActivation "relu" _) inputT = (st, prim__clampMin inputT 0.0)
  applyVarTensor st@(MkActivation "gelu" _) inputT = (st, prim__gelu inputT)
  applyVarTensor {i} st inputT =
    let input = VTensor (tensorToScalars inputT 0 i)
        (st', VTensor outElems) = applyVar st input
    in (st', vecStackTensor outElems)

  emapLayer _ st = st

  showLayer (MkActivation name _) = "Activation<" ++ name ++ ">"

  nameLayer _ st = st

  toDoubleLayer (MkActivation "sigmoid" _) = MkActivation "sigmoid" sigmoid
  toDoubleLayer (MkActivation "tanh" _) = MkActivation "tanh" Math.tanh
  toDoubleLayer (MkActivation "relu" _) = MkActivation "relu" (\x => max x (fromDouble 0.0))
  toDoubleLayer (MkActivation "gelu" _) = MkActivation "gelu" id  -- approx for Double eval
  toDoubleLayer (MkActivation name _) = MkActivation name id

  debugApply st inp =
    let (updated, out) = applyGeneric st inp
    in (updated, out, MkDebugEntry ("Activation<" ++ case st of MkActivation n _ => n ++ ">") [])


----------------------------------------------------------------------
-- Constructors
----------------------------------------------------------------------

export
sigmoidLayer : (FromDouble ty, Neg ty, Fractional ty, Floating ty) => AnyLayer n n ty
sigmoidLayer = MkAnyLayer ActivationState (MkActivation "sigmoid" sigmoid)

export
tanhLayer : (FromDouble ty, Neg ty, Fractional ty, Floating ty) => AnyLayer n n ty
tanhLayer = MkAnyLayer ActivationState (MkActivation "tanh" Math.tanh)

export
reluLayer : (Ord ty, FromDouble ty) => AnyLayer n n ty
reluLayer = MkAnyLayer ActivationState (MkActivation "relu" (\x => max x (fromDouble 0.0)))

export
geluLayer : AnyLayer n n ty
geluLayer = MkAnyLayer ActivationState (MkActivation "gelu" id)
