module Layer.Activation

import Data.Vect

import Device
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

  applyVar {d} st@(MkActivation _ f) xs = (st, map f xs)

  -- Tensor-level dispatch: apply prim__tanh/prim__sigmoid directly on the
  -- tensor instead of unpacking to scalars. 1 tape entry vs ~7n entries.
  applyVarTensor {d} st@(MkActivation "tanh" _) inputT = (st, prim__tanh inputT)
  applyVarTensor {d} st@(MkActivation "sigmoid" _) inputT = (st, prim__sigmoid inputT)
  applyVarTensor {d} st@(MkActivation "relu" _) inputT = (st, prim__clampMin inputT 0.0)
  applyVarTensor {d} st@(MkActivation "gelu" _) inputT = (st, prim__gelu inputT)
  applyVarTensor {d} st@(MkActivation "silu" _) inputT = (st, prim__silu inputT)
  applyVarTensor {d} st@(MkActivation "leaky_relu" _) inputT = (st, prim__leakyRelu inputT 0.01)
  applyVarTensor {d} {i} st inputT =
    let input = VTensor (tensorToScalars inputT 0 i)
        (st', VTensor outElems) = applyVar st input
    in (st', vecStackTensor outElems)

  -- Batched dispatch: element-wise activations are shape-agnostic, so the
  -- same primitives apply on a [B, n] input and produce [B, n] output —
  -- still one tape entry total (no per-row loop).
  applyVarTensorBatch {d} st@(MkActivation "tanh" _) _ inputBT = (st, prim__tanh inputBT)
  applyVarTensorBatch {d} st@(MkActivation "sigmoid" _) _ inputBT = (st, prim__sigmoid inputBT)
  applyVarTensorBatch {d} st@(MkActivation "relu" _) _ inputBT = (st, prim__clampMin inputBT 0.0)
  applyVarTensorBatch {d} st@(MkActivation "gelu" _) _ inputBT = (st, prim__gelu inputBT)
  applyVarTensorBatch {d} st@(MkActivation "silu" _) _ inputBT = (st, prim__silu inputBT)
  applyVarTensorBatch {d} st@(MkActivation "leaky_relu" _) _ inputBT = (st, prim__leakyRelu inputBT 0.01)
  -- Unknown activation: fall back to per-row loop (matches interface default).
  applyVarTensorBatch {d} {i} st@(MkActivation _ _) b inputBT =
    let (st', outs) = goRows st 0 b
    in (st', stackRowTensors outs)
    where
      goRows : ActivationState i i (Variable d) -> Int -> (k : Nat) -> (ActivationState i i (Variable d), Vect k AnyPtr)
      goRows st _ Z = (st, [])
      goRows (MkActivation name f) off (S n) =
        let row = prim__select inputBT 0 off
            (st', outRow) = applyVarTensor (MkActivation name f) row
            (st'', rest) = goRows st' (off + 1) n
        in (st'', outRow :: rest)

  emapLayer _ st = st

  showLayer (MkActivation name _) = "Activation<" ++ name ++ ">"

  nameLayer {d} _ st = st

  toDoubleLayer {d} (MkActivation "sigmoid" _) = MkActivation "sigmoid" sigmoid
  toDoubleLayer (MkActivation "tanh" _) = MkActivation "tanh" Math.tanh
  toDoubleLayer (MkActivation "relu" _) = MkActivation "relu" (\x => max x (fromDouble 0.0))
  toDoubleLayer (MkActivation "gelu" _) = MkActivation "gelu" id  -- approx for Double eval
  toDoubleLayer (MkActivation "leaky_relu" _) = MkActivation "leaky_relu" (\x => if x >= fromDouble 0.0 then x else fromDouble 0.01 * x)
  toDoubleLayer (MkActivation "silu" _) = MkActivation "silu" (\x => x * sigmoid x)
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

export
leakyReluLayer : (Ord ty, FromDouble ty, Num ty) => AnyLayer n n ty
leakyReluLayer = MkAnyLayer ActivationState (MkActivation "leaky_relu" (\x => if x >= fromDouble 0.0 then x else fromDouble 0.01 * x))

export
siluLayer : (FromDouble ty, Neg ty, Fractional ty, Floating ty) => AnyLayer n n ty
siluLayer = MkAnyLayer ActivationState (MkActivation "silu" (\x => x * sigmoid x))
