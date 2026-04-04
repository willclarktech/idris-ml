module Backprop

import Data.SortedMap
import Data.Vect

import DataPoint
import Endofunctor
import Floating
import Math
import Layer
import Optimizer
import Schedule
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
  (Network i hs o Variable, OptimizerState, Double)
epoch opt dataPoints lossFn model st =
  let loss = calculateLossVar lossFn model dataPoints
      grads = collectGrads 1.0 loss
      (deltas, st') = opt.step grads st
      model' = emap (applyDeltas deltas) model
  in (model', st', loss.value)

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
  foldl (\(m, s), _ =>
    let (m', s', _) = epoch opt dataPoints lossFn m s
    in (m', s')) (model, st) [1 .. epochs]

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
  (Network i hs o Variable, OptimizerState, Double)
epochRecurrent opt dataPoints lossFn model st =
  let loss = calculateLossRecurrentVar lossFn model dataPoints
      grads = collectGrads 1.0 loss
      (deltas, st') = opt.step grads st
      model' = emap (applyDeltas deltas) model
  in (model', st', loss.value)

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
  foldl (\(m, s), _ =>
    let (m', s', _) = epochRecurrent opt dataPoints lossFn m s
    in (m', s')) (model, st) [1 .. epochs]

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


----------------------------------------------------------------------
-- Two-Phase Training
----------------------------------------------------------------------

export
epochTwoPhase :
  {i, o, n : Nat} ->
  {hs : List Nat} ->
  Optimizer ->
  Vect n (TwoPhaseDataPoint i o Variable) ->
  LossFunction Variable ->
  Network i hs o Variable ->
  OptimizerState ->
  (Network i hs o Variable, OptimizerState, Double)
epochTwoPhase opt dataPoints lossFn model st =
  let loss = calculateLossTwoPhaseVar lossFn model dataPoints
      grads = collectGrads 1.0 loss
      (deltas, st') = opt.step grads st
      model' = emap (applyDeltas deltas) model
  in (model', st', loss.value)


----------------------------------------------------------------------
-- Scheduled Training with Early Stopping
----------------------------------------------------------------------

||| Train with a learning rate schedule and optional early stopping.
||| Returns (model, optimizerState, epochsCompleted).
export
trainScheduledFrom :
  {i, o, n : Nat} ->
  {hs : List Nat} ->
  (Double -> Optimizer) ->
  Schedule ->
  Network i hs o Variable ->
  Vect n (DataPoint i o Variable) ->
  LossFunction Variable ->
  (totalEpochs : Nat) ->
  (patience : Nat) ->
  OptimizerState ->
  (Network i hs o Variable, OptimizerState, Nat)
trainScheduledFrom makeOpt schedule model dataPoints lossFn totalEpochs patience st =
  go 0 model st (1.0/0.0) 0
  where
    minDelta : Double
    minDelta = 0.001
    go : Nat -> Network i hs o Variable -> OptimizerState -> Double -> Nat ->
         (Network i hs o Variable, OptimizerState, Nat)
    go ep m s bestLoss staleCount =
      if ep >= totalEpochs then (m, s, ep)
      else
        let lr = schedule ep
            opt = makeOpt lr
            (m', s', loss) = epoch opt dataPoints lossFn m s
        in if loss /= loss then (m', s', ep + 1)  -- NaN check: diverged
           else let improved = loss < bestLoss - minDelta
                    bestLoss' = if improved then loss else bestLoss
                    staleCount' : Nat
                    staleCount' = if improved then 0 else staleCount + 1
                in if patience > 0 && staleCount' >= patience
                   then (m', s', ep + 1)
                   else go (ep + 1) m' s' bestLoss' staleCount'

||| Recurrent training with a learning rate schedule and optional early stopping.
||| Returns (model, optimizerState, epochsCompleted).
export
trainRecurrentScheduledFrom :
  {i, o, n : Nat} ->
  {hs : List Nat} ->
  (Double -> Optimizer) ->
  Schedule ->
  Network i hs o Variable ->
  Vect n (RecurrentDataPoint i o Variable) ->
  LossFunction Variable ->
  (totalEpochs : Nat) ->
  (patience : Nat) ->
  OptimizerState ->
  (Network i hs o Variable, OptimizerState, Nat)
trainRecurrentScheduledFrom makeOpt schedule model dataPoints lossFn totalEpochs patience st =
  go 0 model st (1.0/0.0) 0
  where
    minDelta : Double
    minDelta = 0.001
    go : Nat -> Network i hs o Variable -> OptimizerState -> Double -> Nat ->
         (Network i hs o Variable, OptimizerState, Nat)
    go ep m s bestLoss staleCount =
      if ep >= totalEpochs then (m, s, ep)
      else
        let lr = schedule ep
            opt = makeOpt lr
            (m', s', loss) = epochRecurrent opt dataPoints lossFn m s
        in if loss /= loss then (m', s', ep + 1)  -- NaN check: diverged
           else let improved = loss < bestLoss - minDelta
                    bestLoss' = if improved then loss else bestLoss
                    staleCount' : Nat
                    staleCount' = if improved then 0 else staleCount + 1
                in if patience > 0 && staleCount' >= patience
                   then (m', s', ep + 1)
                   else go (ep + 1) m' s' bestLoss' staleCount'


----------------------------------------------------------------------
-- Scheduled Two-Phase Training with Early Stopping
----------------------------------------------------------------------

||| Two-phase training with a learning rate schedule and optional early stopping.
||| Returns (model, optimizerState, epochsCompleted).
export
trainTwoPhaseScheduledFrom :
  {i, o, n : Nat} ->
  {hs : List Nat} ->
  (Double -> Optimizer) ->
  Schedule ->
  Network i hs o Variable ->
  Vect n (TwoPhaseDataPoint i o Variable) ->
  LossFunction Variable ->
  (totalEpochs : Nat) ->
  (patience : Nat) ->
  OptimizerState ->
  (Network i hs o Variable, OptimizerState, Nat)
trainTwoPhaseScheduledFrom makeOpt schedule model dataPoints lossFn totalEpochs patience st =
  go 0 model st (1.0/0.0) 0
  where
    minDelta : Double
    minDelta = 0.001
    go : Nat -> Network i hs o Variable -> OptimizerState -> Double -> Nat ->
         (Network i hs o Variable, OptimizerState, Nat)
    go ep m s bestLoss staleCount =
      if ep >= totalEpochs then (m, s, ep)
      else
        let lr = schedule ep
            opt = makeOpt lr
            (m', s', loss) = epochTwoPhase opt dataPoints lossFn m s
        in if loss /= loss then (m', s', ep + 1)
           else let improved = loss < bestLoss - minDelta
                    bestLoss' = if improved then loss else bestLoss
                    staleCount' : Nat
                    staleCount' = if improved then 0 else staleCount + 1
                in if patience > 0 && staleCount' >= patience
                   then (m', s', ep + 1)
                   else go (ep + 1) m' s' bestLoss' staleCount'


----------------------------------------------------------------------
-- Native Training (libtorch optimizer)
----------------------------------------------------------------------

||| Native two-phase epoch with BCE loss.
||| Uses libtorch optimizer directly: zero_grad → backward → clip → step.
||| No collectGrads, no SortedMap, no applyDeltas.
export
epochTwoPhaseBceNative :
  {i, o, n : Nat} ->
  {hs : List Nat} ->
  NativeOptimizer ->
  Vect n (TwoPhaseDataPoint i o Variable) ->
  Network i hs o Variable ->
  (Network i hs o Variable, Double)
epochTwoPhaseBceNative opt dataPoints model =
  let loss = calculateLossTwoPhaseVarBce model dataPoints
      lossVal = nativeTrainStep opt loss
  -- Model structure unchanged; tensor values mutated in-place by optimizer
  in (model, lossVal)

||| Native epoch for supervised training.
export
epochNative :
  {i, o, n : Nat} ->
  {hs : List Nat} ->
  NativeOptimizer ->
  Vect n (DataPoint i o Variable) ->
  LossFunction Variable ->
  Network i hs o Variable ->
  (Network i hs o Variable, Double)
epochNative opt dataPoints lossFn model =
  let loss = calculateLossVar lossFn model dataPoints
      lossVal = nativeTrainStep opt loss
  in (model, lossVal)

||| Native epoch for recurrent training.
export
epochRecurrentNative :
  {i, o, n : Nat} ->
  {hs : List Nat} ->
  NativeOptimizer ->
  Vect n (RecurrentDataPoint i o Variable) ->
  LossFunction Variable ->
  Network i hs o Variable ->
  (Network i hs o Variable, Double)
epochRecurrentNative opt dataPoints lossFn model =
  let loss = calculateLossRecurrentVar lossFn model dataPoints
      lossVal = nativeTrainStep opt loss
  in (model, lossVal)

||| Simple native training loop: run N epochs, return final model.
export
trainNative :
  {i, o, n : Nat} ->
  {hs : List Nat} ->
  NativeOptimizer ->
  Network i hs o Variable ->
  Vect n (DataPoint i o Variable) ->
  LossFunction Variable ->
  Int ->
  Network i hs o Variable
trainNative opt model dataPoints lossFn epochs =
  foldl (\m, _ =>
    let (m', _) = epochNative opt dataPoints lossFn m
    in m') model [1 .. epochs]

