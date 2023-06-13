module Network

import Data.Vect
import Data.Zippable

import Endofunctor
import Layer
import Math
import Tensor
import Util


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
forward : (Num ty) => {i, o : Nat} -> {hs : List Nat} -> Network i hs o ty -> Vector i ty -> (Network i hs o ty, Vector o ty)
forward (OutputLayer layer) xs =
  let (updatedLayer, output) = applyLayer layer xs
  in (OutputLayer updatedLayer, output)
forward {hs = h :: _} (layer ~> layers) xs =
  let
    (updatedLayer, layerOutput) = applyLayer layer xs
    (updatedNetwork, networkOutput) = forward layers layerOutput
  in (updatedLayer ~> updatedNetwork, networkOutput)

export
evaluate : Num ty => {i, o : Nat} -> {hs : List Nat} -> Network i hs o ty -> Vect n (DataPoint i o ty) -> Vect n (Vector o ty)
evaluate model = map (snd . (forward model) . x)

export
forwardMany : (Num ty) => {i, o : Nat} -> {hs : List Nat} -> (Network i hs o ty, Vect n (Vector o ty)) -> Vector i ty -> (Network i hs o ty, Vect (S n) (Vector o ty))
forwardMany (model, outputs) input =
  let (updatedModel, newOutput) = forward model input
  in rewrite plusCommutative 1 n in (updatedModel, outputs ++ [newOutput])

export
calculateLoss : (Num ty, Fractional ty) => {i, o, n : Nat} -> {hs : List Nat} -> LossFunction ty -> Network i hs o ty -> Vect n (DataPoint i o ty) -> ty
calculateLoss lossFn m dataPoints =
  let
    xs = map x dataPoints
    ys = map y dataPoints
    (updatedNetwork, predictions) = foldlD (\k => (Network i hs o ty, Vect k (Vector o ty))) forwardMany (m, []) xs
    losses = zipWith lossFn predictions ys
  in mean $ VTensor $ map STensor losses
