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

||| Tensor-level loss function: takes raw tensor handles (pred, target) -> Variable.
public export
0 LossFnTensor : Type
LossFnTensor = AnyPtr -> AnyPtr -> Variable

||| Tensor-level native epoch: bypasses scalar packing/unpacking entirely.
||| Accepts DataPoint Double (raw data), bulk-converts to C tensors,
||| forwards through network at tensor level, computes loss on tensors.
||| ~99% fewer FFI calls than epochNative.
export
epochNativeTensor :
  {i, o, n : Nat} ->
  {hs : List Nat} ->
  NativeOptimizer ->
  Vect n (DataPoint i o Double) ->
  LossFnTensor ->
  Network i hs o Variable ->
  (Network i hs o Variable, Double)
epochNativeTensor opt dataPoints lossFn model =
  let -- Bulk-convert inputs and targets to C tensors (1 FFI call each)
      tensorPairs = map (\dp => (bulkToTensor (x dp), bulkToTensor (y dp))) dataPoints
      -- Forward each through network at tensor level (no scalar wrapping)
      (_, losses) = foldl (\(m, acc), (inT, tgtT) =>
        let (m', outT) = forwardVarTensor m inT
            loss = lossFn outT tgtT
        in (m', loss :: acc))
        (the (Network i hs o Variable, List Variable) (model, [])) tensorPairs
      -- Mean loss
      totalLoss = foldl (\acc, l => acc + l) (the Variable (fromDouble 0.0)) losses
      nf = fromDouble (cast (natToInteger n))
      avgLoss : Variable
      avgLoss = totalLoss / nf
      lossVal = nativeTrainStep opt avgLoss
  in (model, lossVal)

||| Tensor-level native epoch with pre-allocated tensor data.
||| Zero conversion overhead — tensors are created directly by the data generator.
export
epochNativeTensorPre :
  {i, o, n : Nat} ->
  {hs : List Nat} ->
  NativeOptimizer ->
  Vect n (TensorDataPoint i o) ->
  LossFnTensor ->
  Network i hs o Variable ->
  (Network i hs o Variable, Double)
epochNativeTensorPre opt dataPoints lossFn model =
  let (_, losses) = foldl (\(m, acc), dp =>
        let (m', outT) = forwardVarTensor m (inputTensor dp)
            loss = lossFn outT (targetTensor dp)
        in (m', loss :: acc))
        (the (Network i hs o Variable, List Variable) (model, [])) dataPoints
      totalLoss = foldl (\acc, l => acc + l) (the Variable (fromDouble 0.0)) losses
      nf = fromDouble (cast (natToInteger n))
      avgLoss : Variable
      avgLoss = totalLoss / nf
      lossVal = nativeTrainStep opt avgLoss
  in (model, lossVal)

||| Batched tensor-level epoch: forwards all data points through a
||| batch-aware forward function, computes per-element loss, averages.
||| The batchFwd function takes a list of input tensor handles and the
||| batch size, returns a list of output tensor handles.
export
epochNativeTensorBatch :
  {i, o, n : Nat} ->
  {hs : List Nat} ->
  NativeOptimizer ->
  Vect n (TensorDataPoint i o) ->
  (List AnyPtr -> Int -> List AnyPtr) ->
  LossFnTensor ->
  Network i hs o Variable ->
  (Network i hs o Variable, Double)
epochNativeTensorBatch {n} opt dataPoints batchFwd lossFn model =
  let nI = cast {to=Int} (natToInteger n)
      inputs = toList (map inputTensor dataPoints)
      targets = toList (map targetTensor dataPoints)
      outputs = batchFwd inputs nI
      losses = zipLoss outputs targets
      totalLoss = foldl (\acc, l => acc + l) (the Variable (fromDouble 0.0)) losses
      nf = fromDouble (cast (natToInteger n))
      avgLoss : Variable
      avgLoss = totalLoss / nf
      lossVal = nativeTrainStep opt avgLoss
  in (model, lossVal)
  where
    zipLoss : List AnyPtr -> List AnyPtr -> List Variable
    zipLoss [] _ = []
    zipLoss _ [] = []
    zipLoss (o :: os) (t :: ts) = lossFn o t :: zipLoss os ts

||| Tensor-level recurrent epoch.
||| Each RecurrentDataPoint has xs (list of input tensors) and ys (list of target tensors).
||| Converts from RecurrentDataPoint Double by bulk-creating tensors.
export
epochRecurrentNativeTensor :
  {i, o, n : Nat} ->
  {hs : List Nat} ->
  NativeOptimizer ->
  Vect n (RecurrentDataPoint i o Double) ->
  LossFnTensor ->
  Network i hs o Variable ->
  (Network i hs o Variable, Double)
epochRecurrentNativeTensor opt dataPoints lossFn model =
  let -- Process each sequence: reset state, forward timesteps, compute loss
      seqLosses = toList $ map (\dp =>
        let m0 = resetNetworkState model
            -- Forward each timestep, collect output tensors
            (_, outTs) = foldl (\(m, outs), x =>
              let inT = bulkToTensor x
                  (m', outT) = forwardVarTensor m inT
              in (m', outT :: outs))
              (the (Network i hs o Variable, List AnyPtr) (m0, []))
              (xs dp)
            revOuts = reverse outTs
            -- Compute per-timestep loss and average
            tgtTs = map bulkToTensor (ys dp)
            stepLosses = zipLoss revOuts tgtTs
            seqLoss = foldl (\acc, l => acc + l) (the Variable (fromDouble 0.0)) stepLosses
            nSteps = fromDouble (cast (length stepLosses))
        in seqLoss / nSteps) dataPoints
      -- Average across sequences
      totalLoss = foldl (\acc, l => acc + l) (the Variable (fromDouble 0.0)) seqLosses
      nSeqs = fromDouble (cast (natToInteger n))
      avgLoss : Variable
      avgLoss = totalLoss / nSeqs
      lossVal = nativeTrainStep opt avgLoss
  in (model, lossVal)
  where
    zipLoss : List AnyPtr -> List AnyPtr -> List Variable
    zipLoss [] _ = []
    zipLoss _ [] = []
    zipLoss (o :: os) (t :: ts) = lossFn o t :: zipLoss os ts

||| Tensor-level two-phase epoch (for NTM copy/recall).
||| Encode phase: forward each input timestep, discard outputs.
||| Decode phase: forward zeros, collect outputs, compute BCE loss.
export
epochTwoPhaseTensor :
  {i, o, n : Nat} ->
  {hs : List Nat} ->
  NativeOptimizer ->
  Vect n (TwoPhaseDataPoint i o Double) ->
  Network i hs o Variable ->
  (Network i hs o Variable, Double)
epochTwoPhaseTensor opt dataPoints model =
  let seqLosses = toList $ map (\dp =>
        let m0 = resetNetworkState model
            -- Encode phase: forward each input, discard outputs
            mEnc = foldl (\m, x =>
              let inT = bulkToTensor x
                  (m', _) = forwardVarTensor m inT
              in m') m0 (encodingInputs dp)
            -- Decode phase: forward zeros, compute BCE loss per timestep
            zeroT = prim__createState1d (cast {to=Int} i) (prim__allocDoubles (cast {to=Int} i))
            (_, outTs) = foldl (\(m, outs), _ =>
              let (m', outT) = forwardVarTensor m zeroT
              in (m', outT :: outs))
              (the (Network i hs o Variable, List AnyPtr) (mEnc, []))
              (targets dp)
            revOuts = reverse outTs
            tgtTs = map bulkToTensor (targets dp)
            stepLosses = zipLossBce revOuts tgtTs
            seqLoss = foldl (\acc, l => acc + l) (the Variable (fromDouble 0.0)) stepLosses
            nSteps = fromDouble (cast (length stepLosses))
        in seqLoss / nSteps) dataPoints
      totalLoss = foldl (\acc, l => acc + l) (the Variable (fromDouble 0.0)) seqLosses
      nSeqs = fromDouble (cast (natToInteger n))
      avgLoss : Variable
      avgLoss = totalLoss / nSeqs
      lossVal = nativeTrainStep opt avgLoss
  in (model, lossVal)
  where
    -- BCE with logits: max(x,0) - x*y + log(1+exp(-|x|))
    bceTensor : AnyPtr -> AnyPtr -> Variable
    bceTensor predT targetT =
      let relu_x = prim__clampMin predT 0.0
          xy = prim__mul predT targetT
          abs_x = prim__abs predT
          neg_abs_x = prim__neg abs_x
          exp_neg = prim__exp neg_abs_x
          one_plus_exp = tensorAdd exp_neg (prim__createScalar 1.0 0)
          log_term = prim__log one_plus_exp
          loss = tensorAdd (prim__sub relu_x xy) log_term
          result = prim__mean loss
          val = prim__item result
      in Var result Nothing val

    zipLossBce : List AnyPtr -> List AnyPtr -> List Variable
    zipLossBce [] _ = []
    zipLossBce _ [] = []
    zipLossBce (o :: os) (t :: ts) = bceTensor o t :: zipLossBce os ts

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

