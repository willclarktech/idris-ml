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

  emapLayer _ st = st

  showLayer (MkActivation name _) = "Activation<" ++ name ++ ">"

  nameLayer _ st = st

  toDoubleLayer (MkActivation "sigmoid" _) = MkActivation "sigmoid" sigmoid
  toDoubleLayer (MkActivation "tanh" _) = MkActivation "tanh" Math.tanh
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
