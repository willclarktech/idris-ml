module DataPoint

import Data.Vect

import Tensor


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
