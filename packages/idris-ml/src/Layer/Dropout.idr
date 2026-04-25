-- | Dropout layer with inverted dropout scaling.
-- |
-- | Training: zeros elements with probability p, scales survivors by 1/(1-p).
-- | Eval: identity (pass-through).
-- | Toggle via setTraining.

module Layer.Dropout

import Data.Vect

import Device
import Endofunctor
import Floating
import Layer.Core
import Tensor
import Variable


----------------------------------------------------------------------
-- Helper: get a pseudo-random seed
----------------------------------------------------------------------

-- Random seed for dropout mask. The dummy arg prevents CSE.
%foreign "C:dropout_random_seed,libidrisml"
dropoutSeed : Int -> Int


----------------------------------------------------------------------
-- Dropout State
----------------------------------------------------------------------

||| Dropout layer. Input and output sizes must be equal.
public export
record DropoutState (n : Nat) (inputSize : Nat) (outputSize : Nat) (ty : Type) where
  constructor MkDropout
  0 dimPrf : inputSize = n
  0 outPrf : outputSize = n
  dropProb : Double
  training : Bool


----------------------------------------------------------------------
-- LayerLike Instance
----------------------------------------------------------------------

%default partial
export
{n : Nat} -> LayerLike (DropoutState n) where
  applyGeneric _ _ = idris_crash "Dropout: use tensor path"
  applyVar {d} _ _ = idris_crash "Dropout: use tensor path"

  applyVarTensor {d} {i} {o} st@(MkDropout dp op p t) inputT =
    if t
      then let seed = dropoutSeed 0
           in (st, prim__dropout inputT p 1 seed)
      else (st, inputT)

  emapLayer _ st = st
  showLayer (MkDropout _ _ p _) = "Dropout<p=" ++ show p ++ ">"
  nameLayer {d} _ st = st
  layerPrefix _ = "drop"
  toDoubleLayer {d} (MkDropout dp op p _) = MkDropout dp op p False

  setTraining mode (MkDropout dp op p _) = MkDropout dp op p mode

  debugApply _ _ = idris_crash "Dropout: use tensor path"


----------------------------------------------------------------------
-- Constructor
----------------------------------------------------------------------

||| Create a dropout layer with given drop probability.
||| Starts in training mode.
export
dropoutLayer : {n : Nat} -> (p : Double) -> AnyLayer n n ty
dropoutLayer p = MkAnyLayer (DropoutState n) (MkDropout Refl Refl p True)
