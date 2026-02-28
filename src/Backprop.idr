module Backprop

import Data.SortedMap
import Data.Vect

import DataPoint
import Endofunctor
import Floating
import Math
import Layer
import Optimizer
import Tensor
import Variable


----------------------------------------------------------------------
-- Supervised Training
----------------------------------------------------------------------

export
epoch :
  {i, o, n : Nat} ->
  {hs : List Nat} ->
  Optimizer ->
  Vect n (DataPoint i o Variable) ->
  LossFunction Variable ->
  Network i hs o Variable ->
  OptimizerState ->
  (Network i hs o Variable, OptimizerState)
epoch opt dataPoints lossFn model st =
  let loss = calculateLossVar lossFn model dataPoints
      grads = collectGrads 1.0 loss
      (deltas, st') = opt.step grads st
      model' = syncNetworkBuffers (emap (applyDeltas deltas) model)
  in (model', st')

export
trainFrom :
  {i, o, n : Nat} ->
  {hs : List Nat} ->
  Optimizer ->
  Network i hs o Variable ->
  Vect n (DataPoint i o Variable) ->
  LossFunction Variable ->
  Int ->
  OptimizerState ->
  (Network i hs o Variable, OptimizerState)
trainFrom opt model dataPoints lossFn epochs st =
  foldl (\(m, s), _ => epoch opt dataPoints lossFn m s) (model, st) [1 .. epochs]

export
train :
  {i, o, n : Nat} ->
  {hs : List Nat} ->
  Optimizer ->
  Network i hs o Variable ->
  Vect n (DataPoint i o Variable) ->
  LossFunction Variable ->
  Int ->
  Network i hs o Variable
train opt model dataPoints lossFn epochs =
  fst $ trainFrom opt model dataPoints lossFn epochs initState

----------------------------------------------------------------------
-- Recurrent Training
----------------------------------------------------------------------

export
epochRecurrent :
  {i, o, n : Nat} ->
  {hs : List Nat} ->
  Optimizer ->
  Vect n (RecurrentDataPoint i o Variable) ->
  LossFunction Variable ->
  Network i hs o Variable ->
  OptimizerState ->
  (Network i hs o Variable, OptimizerState)
epochRecurrent opt dataPoints lossFn model st =
  let loss = calculateLossRecurrentVar lossFn model dataPoints
      grads = collectGrads 1.0 loss
      (deltas, st') = opt.step grads st
      model' = syncNetworkBuffers (emap (applyDeltas deltas) model)
  in (model', st')

export
trainRecurrentFrom :
  {i, o, n : Nat} ->
  {hs : List Nat} ->
  Optimizer ->
  Network i hs o Variable ->
  Vect n (RecurrentDataPoint i o Variable) ->
  LossFunction Variable ->
  Int ->
  OptimizerState ->
  (Network i hs o Variable, OptimizerState)
trainRecurrentFrom opt model dataPoints lossFn epochs st =
  foldl (\(m, s), _ => epochRecurrent opt dataPoints lossFn m s) (model, st) [1 .. epochs]

export
trainRecurrent :
  {i, o, n : Nat} ->
  {hs : List Nat} ->
  Optimizer ->
  Network i hs o Variable ->
  Vect n (RecurrentDataPoint i o Variable) ->
  LossFunction Variable ->
  Int ->
  Network i hs o Variable
trainRecurrent opt model dataPoints lossFn epochs =
  fst $ trainRecurrentFrom opt model dataPoints lossFn epochs initState
