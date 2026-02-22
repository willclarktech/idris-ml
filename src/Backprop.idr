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

applyGrads : Double -> SortedMap String Double -> Variable -> Variable
applyGrads lr grads v = case v.paramId of
  Just pid => case lookup pid grads of
    Just g  => Var (Just pid) (v.value - lr * g) 0 (const []) []
    Nothing => Var v.paramId v.value 0 (const []) []
  Nothing  => v

----------------------------------------------------------------------
-- Supervised Training
----------------------------------------------------------------------

export
epoch :
  {i, o, n : Nat} ->
  {hs : List Nat} ->
  Double ->
  Vect n (DataPoint i o Variable) ->
  LossFunction Variable ->
  Network i hs o Variable ->
  Network i hs o Variable
epoch lr dataPoints lossFn model =
  let loss = calculateLoss lossFn model dataPoints
      grads = collectGrads 1.0 loss
  in emap (applyGrads lr grads) model

export
train :
  {i, o, n : Nat} ->
  {hs : List Nat} ->
  Double ->
  Network i hs o Variable ->
  Vect n (DataPoint i o Variable) ->
  LossFunction Variable ->
  Int ->
  Network i hs o Variable
train lr model dataPoints lossFn epochs = foldl (\m, _ => epoch lr dataPoints lossFn m) model [1 .. epochs]

----------------------------------------------------------------------
-- Recurrent Training
----------------------------------------------------------------------

export
epochRecurrent :
  {i, o, n : Nat} ->
  {hs : List Nat} ->
  Double ->
  Vect n (RecurrentDataPoint i o Variable) ->
  LossFunction Variable ->
  Network i hs o Variable ->
  Network i hs o Variable
epochRecurrent lr dataPoints lossFn model =
  let loss = calculateLossRecurrent lossFn model dataPoints
      grads = collectGrads 1.0 loss
  in emap (applyGrads lr grads) model

export
trainRecurrent :
  {i, o, n : Nat} ->
  {hs : List Nat} ->
  Double ->
  Network i hs o Variable ->
  Vect n (RecurrentDataPoint i o Variable) ->
  LossFunction Variable ->
  Int ->
  Network i hs o Variable
trainRecurrent lr model dataPoints lossFn epochs = foldl (\m, _ => epochRecurrent lr dataPoints lossFn m) model [1 .. epochs]
