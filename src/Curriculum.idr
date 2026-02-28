module Curriculum

import Data.Vect

import Backprop
import DataPoint
import Endofunctor
import Layer
import Math
import Optimizer
import Schedule
import Tensor
import Variable


----------------------------------------------------------------------
-- Stage
----------------------------------------------------------------------

||| A curriculum stage with a label, advancement threshold, and data generator.
public export
record Stage (i : Nat) (o : Nat) (n : Nat) where
  constructor MkStage
  label : String
  threshold : Double  -- loss below this advances to next stage (0.0 = never auto-advance)
  generate : IO (Vect n (RecurrentDataPoint i o Variable))


----------------------------------------------------------------------
-- Internal
----------------------------------------------------------------------

minDelta : Double
minDelta = 0.001

||| Run a chunk of training epochs with fixed data (pure).
runChunk : {i, o, n : Nat} -> {hs : List Nat} ->
           (Double -> Optimizer) -> Schedule ->
           Network i hs o Variable ->
           Vect n (RecurrentDataPoint i o Variable) ->
           LossFunction Variable ->
           Nat -> Nat -> OptimizerState -> Double -> Nat ->
           (Network i hs o Variable, OptimizerState, Double, Nat)
runChunk _ _ m _ _ Z _ s lastLoss sc = (m, s, lastLoss, sc)
runChunk mk sched m ds lossFn (S k) ep s bl sc =
  let lr = sched ep
      opt = mk lr
      (m', s', loss) = epochRecurrent opt ds lossFn m s
      improved = loss < bl - minDelta
      bl' = if improved then loss else bl
      sc' : Nat
      sc' = if improved then 0 else sc + 1
  in runChunk mk sched m' ds lossFn k (ep + 1) s' bl' sc'

||| Train one curriculum stage with periodic data regeneration.
trainStage : {i, o, n : Nat} -> {hs : List Nat} ->
             (Double -> Optimizer) -> Schedule ->
             Network i hs o Variable ->
             Stage i o n ->
             LossFunction Variable ->
             Nat -> Nat -> Nat -> Nat ->
             OptimizerState -> Double -> Nat ->
             IO (Network i hs o Variable, OptimizerState, Nat, Bool)
trainStage _ _ model _ _ _ _ Z done st _ _ = pure (model, st, done, False)
trainStage makeOpt schedule model stage lossFn chunkSz patience budget done st bestLoss staleCount = do
  -- Generate fresh training data
  dps <- stage.generate
  let chunk = min chunkSz budget
  -- Train the chunk
  let (model', st', loss, staleCount') = runChunk makeOpt schedule model dps lossFn chunk done st bestLoss staleCount
  putStrLn $ "  " ++ show (done + chunk) ++ ":\t" ++ show loss
  let bestLoss' = min bestLoss loss
  -- Check stage advancement
  if stage.threshold > 0.0 && loss < stage.threshold
    then do
      putStrLn $ "  -> Advancing (loss " ++ show loss ++ " < " ++ show stage.threshold ++ ")"
      pure (model', st', done + chunk, True)
    else if patience > 0 && staleCount' >= patience
    then do
      putStrLn $ "  Early stop at epoch " ++ show (done + chunk) ++ " (patience=" ++ show patience ++ ")"
      pure (model', st', done + chunk, False)
    else if loss /= loss
    then do
      putStrLn $ "  Diverged (NaN) at epoch " ++ show (done + chunk)
      pure (model', st', done + chunk, False)
    else trainStage makeOpt schedule model' stage lossFn chunkSz patience (minus budget chunk) (done + chunk) st' bestLoss' staleCount'


----------------------------------------------------------------------
-- Public API
----------------------------------------------------------------------

||| Run multi-stage curriculum training with periodic data regeneration.
||| Returns (trained model, optimizer state, total epochs completed).
export
runCurriculum :
  {i, o, n : Nat} -> {hs : List Nat} ->
  (Double -> Optimizer) ->
  Schedule ->
  Network i hs o Variable ->
  LossFunction Variable ->
  List (Stage i o n) ->
  (totalEpochs : Nat) ->
  (patience : Nat) ->
  (chunkSize : Nat) ->
  OptimizerState ->
  IO (Network i hs o Variable, OptimizerState, Nat)
runCurriculum _ _ model _ [] _ _ _ st = pure (model, st, 0)
runCurriculum makeOpt schedule model lossFn stages totalEpochs patience chunkSize st =
  go model stages totalEpochs 0 st
  where
    go : Network i hs o Variable -> List (Stage i o n) -> Nat -> Nat -> OptimizerState ->
         IO (Network i hs o Variable, OptimizerState, Nat)
    go m [] _ done s = pure (m, s, done)
    go m (stage :: rest) budget done s = do
      putStrLn $ "\n" ++ stage.label
      (m', s', done', advanced) <- trainStage makeOpt schedule m
        stage lossFn chunkSize patience budget done s (1.0/0.0) 0
      let remaining = minus budget (minus done' done)
      if advanced && remaining > 0
        then go m' rest remaining done' s'
        else pure (m', s', done')
