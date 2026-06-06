module Layer.Dropout

import Data.Vect

import Executor
import Layer.Core
import Tensor


-- Random seed for dropout mask. Dummy arg prevents CSE — the FFI
-- binding is shared with the V1 dropout layer (declared there but we
-- don't import that module to keep V1/ paths independent).
%foreign "C:dropout_random_seed,libidrisml"
dropoutSeed : Int -> Int


----------------------------------------------------------------------
-- Dropout — typed-surface dropout (Path C)
----------------------------------------------------------------------
--
-- Inverted dropout: training mode zeros elements with probability p
-- and scales survivors by 1/(1-p); eval mode is identity. Toggle
-- via `setTraining`. No learnable params, so no `nameLayer` work.
--
-- Parameterised by both input and output sizes (with `i = n` and
-- `o = n` enforced by the constructor) so the type fits LayerLike's
-- `Nat -> Nat -> Executor -> Type` arity.

public export
data DropoutState : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type where
  MkDropout : (p : Double) -> (training : Bool) -> DropoutState n n ex dt g


----------------------------------------------------------------------
-- Forward
----------------------------------------------------------------------

%default partial

export
applyDropout : {0 ex : Executor} -> UserExecutorTraining ex => UserExecutorCore ex => RuntimeDType dt => Linked ex => Compatible ex dt => {n : Nat} ->
                 DropoutState n n ex dt g ->
                 TVec n ex dt g ->
                 IO (DropoutState n n ex dt g, TVec n ex dt g)
applyDropout st@(MkDropout p training) input = ioRerun (\_ =>
  if training
    then
      let seed = dropoutSeed 0
          outPtr = primDropout {ex} input.tensorPtr p 1 seed
      in (st, MkTensor outPtr Nothing)
    else (st, input))


----------------------------------------------------------------------
-- Constructor
----------------------------------------------------------------------

||| Create a Dropout with given drop probability. Starts in training
||| mode; flip to eval via `setTraining False`.
export
dropoutLayer : {n : Nat} -> (p : Double) -> DropoutState n n ex dt g
dropoutLayer p = MkDropout p True

||| Toggle training/eval mode.
export
setTraining : Bool -> DropoutState n n ex dt g -> DropoutState n n ex dt g
setTraining mode (MkDropout p _) = MkDropout p mode


----------------------------------------------------------------------
-- LayerLike instance
----------------------------------------------------------------------

public export
LayerLike DropoutState where
  -- Pattern match on MkDropout unifies i = o = n, so we can pass
  -- through to applyDropout which works on the same-size form.
  applyVar st@(MkDropout _ _) input = applyDropout st input

  -- prim__dropout is rank-agnostic, so the batched form is identical
  -- to the single-sample form — same dropout rate, fresh per-call seed.
  applyVarBatch st@(MkDropout p training) input = ioRerun (\_ =>
    if training
      then let seed = dropoutSeed 0
               outPtr = primDropout {ex} input.tensorPtr p 1 seed
           in (st, MkTensor outPtr Nothing)
      else (st, input))

  layerPrefix _ = "drop"

  -- Dropout is stateless (no params); freeze/unfreeze just retypes.
  freezeLayer (MkDropout p t) = pure (MkDropout p t)
  unfreezeLayer (MkDropout p t) = pure (MkDropout p t)

||| Wrap a Dropout in `AnyLayer`.
export
dropoutLayerAny : {n : Nat} -> (p : Double) -> AnyLayer n n ex dt g
dropoutLayerAny p = MkAnyLayer DropoutState (dropoutLayer p)
