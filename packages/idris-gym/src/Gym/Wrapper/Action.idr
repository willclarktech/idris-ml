module Gym.Wrapper.Action

import Data.Vect
import Decidable.Equality
import Gym.Space


----------------------------------------------------------------------
-- Per-scalar helpers
----------------------------------------------------------------------

clampScalar : Double -> Double -> Double -> Double
clampScalar lo hi x =
  if x < lo then lo else if x > hi then hi else x

||| Linear rescale of x from source range [lo1, hi1] into target [lo2, hi2].
rescaleScalar : Double -> Double -> Double -> Double -> Double -> Double
rescaleScalar lo1 hi1 lo2 hi2 x =
  let span1 = hi1 - lo1
      span2 = hi2 - lo2
      t = if span1 == 0.0 then 0.0 else (x - lo1) / span1
  in lo2 + t * span2


----------------------------------------------------------------------
-- Vector-level action transforms
----------------------------------------------------------------------

||| Clip each element of an action vector to the bounds of a Box space.
||| Non-Box spaces (or length mismatches) pass the action through unchanged.
export
clipAction : Space -> {m : Nat} -> Vect m Double -> Vect m Double
clipAction (Box {n} lo hi) v =
  case decEq n m of
    Yes Refl => zipWith (\loI, (xI, hiI) => clampScalar loI hiI xI)
                        lo (zip v hi)
    No _     => v
  where
    zip : Vect k a -> Vect k b -> Vect k (a, b)
    zip [] [] = []
    zip (x :: xs) (y :: ys) = (x, y) :: zip xs ys
clipAction _ v = v

||| Rescale each element of an action vector from fromSp to toSp bounds.
||| Requires both spaces to be Box with matching dimension.
export
rescaleAction : (fromSp : Space) -> (toSp : Space) ->
                {m : Nat} -> Vect m Double -> Vect m Double
rescaleAction (Box {n=nF} loF hiF) (Box {n=nT} loT hiT) v =
  case decEq nF m of
    No _ => v
    Yes Refl => case decEq nF nT of
      No _ => v
      Yes Refl => mapQuad (\lF, hF, lT, hT, x => rescaleScalar lF hF lT hT x) loF hiF loT hiT v
  where
    mapQuad : (Double -> Double -> Double -> Double -> Double -> Double) ->
              Vect k Double -> Vect k Double ->
              Vect k Double -> Vect k Double ->
              Vect k Double -> Vect k Double
    mapQuad _ [] [] [] [] [] = []
    mapQuad f (a :: as) (b :: bs) (c :: cs) (d :: ds) (e :: es) =
      f a b c d e :: mapQuad f as bs cs ds es
rescaleAction _ _ v = v


----------------------------------------------------------------------
-- Scalar-action convenience (for Box {n=1} envs)
----------------------------------------------------------------------

||| Clip a scalar action to the first dimension of a Box space.
export
clipScalarAction : Space -> Double -> Double
clipScalarAction (Box (lo :: _) (hi :: _)) x = clampScalar lo hi x
clipScalarAction _ x = x

||| Rescale a scalar action from fromSp to toSp first-dim bounds.
export
rescaleScalarAction : Space -> Space -> Double -> Double
rescaleScalarAction (Box (lF :: _) (hF :: _)) (Box (lT :: _) (hT :: _)) x =
  rescaleScalar lF hF lT hT x
rescaleScalarAction _ _ x = x
