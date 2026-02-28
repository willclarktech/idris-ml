module Backprop

import Data.SortedMap
import Data.Vect

import DataPoint
import Endofunctor
import Floating
import Math
import Layer
import Tensor
import Variable


----------------------------------------------------------------------
-- Gradient Application
----------------------------------------------------------------------

clipGrad : Double -> Double -> Double
clipGrad maxAbs g = max (-maxAbs) (min maxAbs g)

applyGrads : Double -> Double -> SortedMap String Double -> Variable -> Variable
applyGrads lr maxGrad grads v = case v.paramId of
  Just pid => case lookup pid grads of
    Just g  => let val = v.value - lr * clipGrad maxGrad g
               in Var (nextNodeId val) (Just pid) val 0 (const []) []
    Nothing => Var (nextNodeId v.value) v.paramId v.value 0 (const []) []
  Nothing  => v

----------------------------------------------------------------------
-- Supervised Training
----------------------------------------------------------------------

export
epoch :
  {i, o, n : Nat} ->
  {hs : List Nat} ->
  Double ->
  Double ->
  Vect n (DataPoint i o Variable) ->
  LossFunction Variable ->
  Network i hs o Variable ->
  Network i hs o Variable
epoch lr maxGrad dataPoints lossFn model =
  let loss = calculateLoss lossFn model dataPoints
      grads = collectGrads 1.0 loss
  in emap (applyGrads lr maxGrad grads) model

export
train :
  {i, o, n : Nat} ->
  {hs : List Nat} ->
  Double ->
  Double ->
  Network i hs o Variable ->
  Vect n (DataPoint i o Variable) ->
  LossFunction Variable ->
  Int ->
  Network i hs o Variable
train lr maxGrad model dataPoints lossFn epochs = foldl (\m, _ => epoch lr maxGrad dataPoints lossFn m) model [1 .. epochs]

----------------------------------------------------------------------
-- Recurrent Training
----------------------------------------------------------------------

export
epochRecurrent :
  {i, o, n : Nat} ->
  {hs : List Nat} ->
  Double ->
  Double ->
  Vect n (RecurrentDataPoint i o Variable) ->
  LossFunction Variable ->
  Network i hs o Variable ->
  Network i hs o Variable
epochRecurrent lr maxGrad dataPoints lossFn model =
  let loss = calculateLossRecurrent lossFn model dataPoints
      grads = collectGrads 1.0 loss
  in emap (applyGrads lr maxGrad grads) model

export
trainRecurrent :
  {i, o, n : Nat} ->
  {hs : List Nat} ->
  Double ->
  Double ->
  Network i hs o Variable ->
  Vect n (RecurrentDataPoint i o Variable) ->
  LossFunction Variable ->
  Int ->
  Network i hs o Variable
trainRecurrent lr maxGrad model dataPoints lossFn epochs = foldl (\m, _ => epochRecurrent lr maxGrad dataPoints lossFn m) model [1 .. epochs]
