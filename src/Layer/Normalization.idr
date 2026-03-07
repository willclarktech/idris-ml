module Layer.Normalization

import Data.Vect

import Floating
import Layer.Core
import Math
import Tensor
import Variable


----------------------------------------------------------------------
-- Normalization State
----------------------------------------------------------------------

||| Normalization layer: input and output sizes must be equal.
public export
data NormalizationState : Nat -> Nat -> Type -> Type where
  MkNormalization : String -> NormalizationFunction ty -> NormalizationState n n ty


----------------------------------------------------------------------
-- LayerLike Instance
----------------------------------------------------------------------

export
LayerLike NormalizationState where
  applyGeneric st@(MkNormalization _ f) xs = (st, f xs)

  -- Variable-specialized: dispatch to C-backed kernels for softmax/logSoftmax
  applyVar st@(MkNormalization "softmax" _) xs = (st, softmaxVar xs)
  applyVar st@(MkNormalization "logSoftmax" _) xs = (st, logSoftmaxVar xs)
  applyVar st@(MkNormalization _ f) xs = (st, f xs)

  emapLayer _ st = st

  showLayer (MkNormalization name _) = "Normalization<" ++ name ++ ">"

  nameLayer _ st = st

  toDoubleLayer (MkNormalization "softmax" _) = MkNormalization "softmax" softmax
  toDoubleLayer (MkNormalization "logSoftmax" _) = MkNormalization "logSoftmax" logSoftmax
  toDoubleLayer (MkNormalization name _) = MkNormalization name id

  debugApply st inp =
    let (updated, out) = applyGeneric st inp
    in (updated, out, MkDebugEntry ("Normalization<" ++ case st of MkNormalization n _ => n ++ ">") [])


----------------------------------------------------------------------
-- Constructors
----------------------------------------------------------------------

export
softmaxLayer : (Fractional ty, Floating ty) => AnyLayer n n ty
softmaxLayer = MkAnyLayer NormalizationState (MkNormalization "softmax" softmax)

export
logSoftmaxLayer : (FromDouble ty, Cast ty Double, Neg ty, Floating ty, Fractional ty) => AnyLayer n n ty
logSoftmaxLayer = MkAnyLayer NormalizationState (MkNormalization "logSoftmax" logSoftmax)
