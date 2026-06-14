module Gym.Space

import Data.Vect
import Decidable.Equality

----------------------------------------------------------------------
-- Space ADT
----------------------------------------------------------------------

||| Description of a valid action or observation space.
||| Values carry runtime bounds; the type system does not enforce them.
||| Spaces are informational metadata for wrappers and loggers.
public export
data Space : Type where
  ||| Integer in {0, ..., n-1}. Matches Gymnasium's Discrete(n).
  Discrete  : (n : Nat) -> Space
  ||| Bounded real vector of length n with per-dim [low, high] ranges.
  Box       : {n : Nat} -> (low : Vect n Double) -> (high : Vect n Double) -> Space
  ||| Binary vector of length n. Matches Gymnasium's MultiBinary(n).
  MultiBin  : (n : Nat) -> Space
  ||| Product of discrete spaces with per-dim cardinalities.
  MultiDisc : {k : Nat} -> (nvec : Vect k Nat) -> Space

----------------------------------------------------------------------
-- Helpers
----------------------------------------------------------------------

||| Cardinality of a finite space, if any. Box is uncountable; MultiDisc
||| overflows quickly so we leave it as Nothing.
export
spaceSize : Space -> Maybe Nat
spaceSize (Discrete n)   = Just n
spaceSize (Box _ _)      = Nothing
spaceSize (MultiBin n)   = Just (power 2 n)
  where
    power : Nat -> Nat -> Nat
    power _ Z     = 1
    power b (S k) = b * power b k
spaceSize (MultiDisc _)  = Nothing

||| Shape of a space as a list of dimensions (empty for scalars).
export
spaceShape : Space -> List Nat
spaceShape (Discrete _)    = []
spaceShape (Box {n} _ _)   = [n]
spaceShape (MultiBin n)    = [n]
spaceShape (MultiDisc {k} _) = [k]

||| Does a Nat belong to a Discrete space?
export
containsNat : Space -> Nat -> Bool
containsNat (Discrete n) k = k < n
containsNat _            _ = False

||| Does a vector of doubles lie within a Box space (element-wise)?
export
containsBox : Space -> {m : Nat} -> (v : Vect m Double) -> Bool
containsBox (Box {n} lo hi) v =
  case decEq n m of
    Yes Refl => checkAll lo hi v
    No _     => False
  where
    checkAll : Vect k Double -> Vect k Double -> Vect k Double -> Bool
    checkAll []        []        []        = True
    checkAll (l :: ls) (h :: hs) (x :: xs) =
      x >= l && x <= h && checkAll ls hs xs
containsBox _ _ = False
