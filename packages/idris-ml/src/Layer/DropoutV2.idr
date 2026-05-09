module Layer.DropoutV2

import Data.Vect

import Device
import Layer.CoreV2
import Variable


-- Random seed for dropout mask. Dummy arg prevents CSE — the FFI
-- binding is shared with the V1 dropout layer (declared there but we
-- don't import that module to keep V1/V2 paths independent).
%foreign "C:dropout_random_seed,libidrisml"
dropoutSeed : Int -> Int


----------------------------------------------------------------------
-- DropoutV2 — typed-surface dropout (Path C)
----------------------------------------------------------------------
--
-- Inverted dropout: training mode zeros elements with probability p
-- and scales survivors by 1/(1-p); eval mode is identity. Toggle
-- via `setTrainingV2`. No learnable params, so no `nameLayer` work.
--
-- Parameterised by both input and output sizes (with `i = n` and
-- `o = n` enforced by the constructor) so the type fits LayerLikeV2's
-- `Nat -> Nat -> Device -> Type` arity.

public export
data DropoutStateV2 : Nat -> Nat -> (0 _ : Device) -> Type where
  MkDropoutV2 : (p : Double) -> (training : Bool) -> DropoutStateV2 n n d


----------------------------------------------------------------------
-- Forward
----------------------------------------------------------------------

%default partial

export
applyDropoutV2 : {n : Nat} ->
                 DropoutStateV2 n n d ->
                 TVec n d ->
                 (DropoutStateV2 n n d, TVec n d)
applyDropoutV2 st@(MkDropoutV2 p training) input =
  if training
    then
      let seed = dropoutSeed 0
          outPtr = prim__dropout input.tensorPtr p 1 seed
      in (st, MkTVar outPtr Nothing)
    else (st, input)


----------------------------------------------------------------------
-- Constructor
----------------------------------------------------------------------

||| Create a DropoutV2 with given drop probability. Starts in training
||| mode; flip to eval via `setTrainingV2 False`.
export
dropoutLayerV2 : {n : Nat} -> (p : Double) -> DropoutStateV2 n n d
dropoutLayerV2 p = MkDropoutV2 p True

||| Toggle training/eval mode.
export
setTrainingV2 : Bool -> DropoutStateV2 n n d -> DropoutStateV2 n n d
setTrainingV2 mode (MkDropoutV2 p _) = MkDropoutV2 p mode


----------------------------------------------------------------------
-- LayerLikeV2 instance
----------------------------------------------------------------------

public export
LayerLikeV2 DropoutStateV2 where
  -- Pattern match on MkDropoutV2 unifies i = o = n, so we can pass
  -- through to applyDropoutV2 which works on the same-size form.
  applyTVar st@(MkDropoutV2 _ _) input = applyDropoutV2 st input
  layerPrefixV2 _ = "dropV2"

||| Wrap a DropoutV2 in `AnyLayerV2`.
export
dropoutLayerV2Any : {n : Nat} -> (p : Double) -> AnyLayerV2 n n d
dropoutLayerV2Any p = MkAnyLayerV2 DropoutStateV2 (dropoutLayerV2 p)
