module Network

import Data.List
import Data.Vect
import Data.Zippable

import DataPoint
import Endofunctor
import Layer
import Math
import Tensor
import Util


public export
data Network : (inputDims : Nat) -> (hiddenDims : List Nat) -> (outputDims : Nat) -> Type -> Type where
  OutputLayer : Layer i o ty -> Network i [] o ty
  (~>) : Layer i h ty -> Network h hs o ty -> Network i (h :: hs) o ty

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
forward (OutputLayer layer) x =
  let (updatedLayer, output) = applyLayer layer x
  in (OutputLayer updatedLayer, output)
forward {hs = h :: _} (layer ~> layers) x =
  let
    (updatedLayer, layerOutput) = applyLayer layer x
    (updatedNetwork, networkOutput) = forward layers layerOutput
  in (updatedLayer ~> updatedNetwork, networkOutput)

evaluateSingleDataPoint : Num ty => {i, o : Nat} -> {hs : List Nat} -> Network i hs o ty -> DataPoint i o ty -> Vector o ty
evaluateSingleDataPoint model = snd . (forward model) . x

export
evaluate : Num ty => {i, o : Nat} -> {hs : List Nat} -> Network i hs o ty -> Vect n (DataPoint i o ty) -> Vect n (Vector o ty)
evaluate model = map (evaluateSingleDataPoint model)

forwardNext : (Num ty) => {i, o : Nat} -> {hs : List Nat} -> (Network i hs o ty, Vect n (Vector o ty)) -> Vector i ty -> (Network i hs o ty, Vect (S n) (Vector o ty))
forwardNext (nn, outputs) inp =
  let (updatedModel, newOutput) = forward nn inp
  in (updatedModel, snoc outputs newOutput)

forwardMany : (Num ty) => {i, o : Nat} -> {hs : List Nat} -> Network i hs o ty -> Vect n (Vector i ty) -> (Network i hs o ty, Vect n (Vector o ty))
forwardMany network xs = foldlD (\k => (Network i hs o ty, Vect k (Vector o ty))) forwardNext (network, []) xs

export
calculateLoss : (Num ty, Fractional ty) => {i, o, n : Nat} -> {hs : List Nat} -> LossFunction ty -> Network i hs o ty -> Vect n (DataPoint i o ty) -> ty
calculateLoss lossFn model dataPoints =
  let
    xs = map x dataPoints
    ys = map y dataPoints
    (updatedNetwork, predictions) = forwardMany model xs
    losses = zipWith lossFn predictions ys
  in mean $ VTensor $ map STensor losses

recur : (Num ty) => {i, o : Nat} -> {hs : List Nat} -> (Network i hs o ty, List (Vector o ty)) -> Vector i ty -> (Network i hs o ty, List (Vector o ty))
recur (m, os) i =
  let (updatedModel, output) = forward m i
  in (updatedModel, snoc os output)

export
forwardRecurrent : (Num ty) => {i, o : Nat} -> {hs : List Nat} -> Network i hs o ty -> List (Vector i ty) -> (Network i hs o ty, List (Vector o ty))
forwardRecurrent model = foldl recur (model, [])

evaluateSingleRecurrentDataPoint : Num ty => {i, o : Nat} -> {hs : List Nat} -> Network i hs o ty -> RecurrentDataPoint i o ty -> List (Vector o ty)
evaluateSingleRecurrentDataPoint model dataPoints = snd $ (forwardRecurrent model) dataPoints.xs

export
evaluateRecurrent : Num ty => {i, o : Nat} -> {hs : List Nat} -> Network i hs o ty -> Vect n (RecurrentDataPoint i o ty) -> Vect n (List (Vector o ty))
evaluateRecurrent model dataPoints = map (evaluateSingleRecurrentDataPoint model) dataPoints

forwardNextRecurrent : (Num ty) => {i, o : Nat} -> {hs : List Nat} -> (Network i hs o ty, Vect n (List (Vector o ty))) -> List (Vector i ty) -> (Network i hs o ty, Vect (1 + n) (List (Vector o ty)))
forwardNextRecurrent (nn, outputs) inps =
  let (updatedModel, newOutput) = forwardRecurrent nn inps
  in (updatedModel, snoc outputs newOutput)

forwardManyRecurrent : (Num ty) => {i, o : Nat} -> {hs : List Nat} -> Network i hs o ty -> Vect n (List (Vector i ty)) -> (Network i hs o ty, Vect n (List (Vector o ty)))
forwardManyRecurrent network xs = foldlD (\k => (Network i hs o ty, Vect k (List (Vector o ty)))) forwardNextRecurrent (network, []) xs

export
calculateLossRecurrent : (Num ty, Fractional ty) => {i, o, n : Nat} -> {hs : List Nat} -> LossFunction ty -> Network i hs o ty -> Vect n (RecurrentDataPoint i o ty) -> ty
calculateLossRecurrent lossFn model dataPoints =
  let
    xs = map xs dataPoints
    ys = map ys dataPoints
    (updatedNetwork, predictions) = forwardManyRecurrent model xs
    losses = zipWith (zipWith lossFn) predictions ys
  in mean . VTensor $ map (STensor . mean) losses
