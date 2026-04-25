module Layer.Normalization

import Data.Vect

import Device
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
  applyVar {d} st@(MkNormalization "softmax" _) xs = (st, softmaxVar xs)
  applyVar {d} st@(MkNormalization "logSoftmax" _) xs = (st, logSoftmaxVar xs)
  applyVar {d} st@(MkNormalization _ f) xs = (st, f xs)

  -- Tensor-level: direct C kernel call, no scalar packing
  applyVarTensor {d} st@(MkNormalization "softmax" _) inputT = (st, prim__softmax inputT 0)
  applyVarTensor {d} st@(MkNormalization "logSoftmax" _) inputT = (st, prim__logSoftmax inputT 0)
  applyVarTensor {d} {i} {o} st inputT =
    let input = VTensor (tensorToScalars inputT 0 i)
        (st', VTensor outElems) = applyVar st input
    in (st', vecStackTensor outElems)

  emapLayer _ st = st

  showLayer (MkNormalization name _) = "Normalization<" ++ name ++ ">"

  nameLayer {d} _ st = st

  toDoubleLayer {d} (MkNormalization "softmax" _) = MkNormalization "softmax" softmax
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
