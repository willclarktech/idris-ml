module Network

import Data.Vect
import Data.Zippable

import Endofunctor
import Layer
import Math
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
data Network : (inputDims : Nat) -> (hiddenDims : List Nat) -> (outputDims : Nat) -> Type -> Type where
  OutputLayer : Layer i o a -> Network i [] o a
  (~>) : Layer i h a -> Network h hs o a -> Network i (h :: hs) o a

infixr 5 ~>

public export
implementation {i, o : Nat} -> Show ty => Show (Network i [] o ty) where
  show (OutputLayer layer) = show layer

public export
implementation {i, h : Nat} -> (Show ty, Show (Network h hs o ty)) => Show (Network i (h :: hs) o ty) where
  show (layer ~> layers) = show layer ++ " ~> " ++ show layers

public export
implementation Endofunctor (Network i hs o) where
  emap f (OutputLayer layer) = OutputLayer (emap f layer)
  emap f (layer ~> layers) = emap f layer ~> emap f layers

export
forward : (Num ty) => {i, o : Nat} -> {hs : List Nat} -> Network i hs o ty -> Vector i ty -> Vector o ty
forward (OutputLayer layer) = applyLayer layer
forward {hs = h :: _} (layer ~> layers) = forward layers . applyLayer layer

export
calculateLoss : (Num ty, Fractional ty) => {i, o, n : Nat} -> {hs : List Nat} -> LossFunction ty -> Network i hs o ty -> Vect n (DataPoint i o ty) -> ty
calculateLoss lossFn m dataPoints =
  let
    xs = map x dataPoints
    ys = map y dataPoints
    predictions = map (forward m) xs
    losses = zipWith lossFn predictions ys
  in mean $ VTensor $ map STensor losses
