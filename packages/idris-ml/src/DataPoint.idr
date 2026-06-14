module DataPoint

import Data.Vect

import Array

public export
record DataPoint i o ty where
  constructor MkDataPoint
  x : Vector i ty
  y : Vector o ty

public export
implementation Functor (DataPoint i o) where
  map f (MkDataPoint x y) = MkDataPoint (map f x) (map f y)

public export
record RecurrentDataPoint i o ty where
  constructor MkRecurrentDataPoint
  xs : List (Vector i ty)
  ys : List (Vector o ty)

public export
implementation {i, o : Nat} -> Show ty => Show (RecurrentDataPoint i o ty) where
  show (MkRecurrentDataPoint xs ys) = "RecurrentDataPoint<" ++ show i ++ "," ++ show o ++ ">(" ++ show xs ++ "," ++ show ys ++ ")"

public export
implementation Functor (RecurrentDataPoint i o) where
  map f (MkRecurrentDataPoint xs ys) = MkRecurrentDataPoint (map (map f) xs) (map (map f) ys)

||| Two-phase data point for encoding/output-phase tasks (e.g., NTM copy/recall).
||| During encoding, outputs are discarded. During the output phase,
||| zero inputs are fed and outputs are compared against targets.
public export
record TwoPhaseDataPoint i o ty where
  constructor MkTwoPhaseDataPoint
  encodingInputs : List (Vector i ty)
  targets : List (Vector o ty)

public export
implementation Functor (TwoPhaseDataPoint i o) where
  map f (MkTwoPhaseDataPoint xs ys) = MkTwoPhaseDataPoint (map (map f) xs) (map (map f) ys)

||| Data point with pre-allocated C tensor handles.
||| Bypasses all scalar packing — the tensors are created directly from raw data.
||| The type parameters i and o are phantom (for type safety only).
public export
record TensorDataPoint (i : Nat) (o : Nat) where
  constructor MkTensorDataPoint
  inputTensor : AnyPtr   -- 1D tensor [i]
  targetTensor : AnyPtr  -- 1D tensor [o]
