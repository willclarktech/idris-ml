||| Multi-stage curriculum training.
|||
||| A `Stage` bundles a label, an advancement threshold, and a data
||| generator. `runCurriculum` walks the stages in order: per stage it
||| trains chunks of `chunkSize` epochs (regenerating data each chunk),
||| advances when the chunk's loss falls below `threshold`, and
||| applies global patience-based early stopping across stages.
|||
||| The training inner loop runs on top of `epochRecurrentVar` from
||| `Backprop`. `Schedule` is bound to the optimizer via
||| `setLearningRate` per epoch (no per-epoch optimizer rebuild —
||| V2's C-side optimizer keeps its own state).
module Curriculum

import Data.Vect
import System.Clock

import Backprop
import DataPoint
import Layer.Core
import Schedule
import Tensor
import Util
import Device


----------------------------------------------------------------------
-- Stage
----------------------------------------------------------------------

||| A curriculum stage with a label, advancement threshold, and data
||| generator. `threshold > 0.0` advances when the chunk loss falls
||| below it; `threshold = 0.0` never auto-advances.
public export
record Stage (0 d : Device) (i : Nat) (o : Nat) (n : Nat) where
  constructor MkStage
  label : String
  threshold : Double
  generate : IO (Vect n (RecurrentDataPoint i o Double))


----------------------------------------------------------------------
-- Internal
----------------------------------------------------------------------

minDelta : Double
minDelta = 0.001

||| Run a chunk of training epochs over fixed data.
runChunk : {0 d : Device} -> UserDeviceTraining d => RuntimeDType dt => Linked d => Compatible d dt => IsFloating dt => {i, o, n : Nat} -> {hs : List Nat} ->
           NativeOptimizer d -> Schedule ->
           Network i hs o d dt WithGrad ->
           Vect n (RecurrentDataPoint i o Double) ->
           LossFn d dt o ->
           (chunkRemaining : Nat) ->
           (epoch : Nat) ->
           (bestLoss : Double) ->
           (staleCount : Nat) ->
           IO (Network i hs o d dt WithGrad, Double, Nat)
runChunk _ _ m _ _ Z _ bl sc = pure (m, bl, sc)
runChunk opt sched m ds lossFn (S k) ep bl sc = do
  setLearningRate opt (sched ep)
  (m', loss) <- epochRecurrentVar opt ds lossFn m
  let improved : Bool
      improved = loss < bl - minDelta
      bl' : Double
      bl' = if improved then loss else bl
      sc' : Nat
      sc' = if improved then 0 else sc + 1
  runChunk opt sched m' ds lossFn k (ep + 1) bl' sc'


||| Train one curriculum stage. Returns
||| (model, totalEpochs, advanced?). `advanced=True` means the
||| caller should move to the next stage.
trainStage : {0 d : Device} -> UserDeviceTraining d => RuntimeDType dt => Linked d => Compatible d dt => IsFloating dt => {i, o, n : Nat} -> {hs : List Nat} ->
             NativeOptimizer d -> Schedule ->
             Network i hs o d dt WithGrad ->
             Stage d i o n ->
             LossFn d dt o ->
             (chunkSize : Nat) ->
             (patience : Nat) ->
             (budget : Nat) ->
             (epochsSoFar : Nat) ->
             (bestLoss : Double) ->
             (staleCount : Nat) ->
             Clock Monotonic ->
             IO (Network i hs o d dt WithGrad, Nat, Bool)
trainStage _ _ model _ _ _ _ Z done _ _ _ = pure (model, done, False)
trainStage opt sched model stage lossFn chunkSz patience budget done bestLoss staleCount t0 = do
  dps <- stage.generate
  let chunk = min chunkSz budget
  (model', loss, staleCount') <-
    runChunk opt sched model dps lossFn chunk done bestLoss staleCount
  now <- clockTime Monotonic
  putStrLn $ "  " ++ formatElapsed t0 now ++ " " ++ show (done + chunk)
          ++ ":\t" ++ show loss
  let bestLoss' = min bestLoss loss
  if stage.threshold > 0.0 && loss < stage.threshold
    then do
      putStrLn $ "  " ++ formatElapsed t0 now
              ++ " -> Advancing (loss " ++ show loss
              ++ " < " ++ show stage.threshold ++ ")"
      pure (model', done + chunk, True)
    else if patience > 0 && staleCount' >= patience
    then do
      putStrLn $ "  " ++ formatElapsed t0 now
              ++ " Early stop at epoch " ++ show (done + chunk)
              ++ " (patience=" ++ show patience ++ ")"
      pure (model', done + chunk, False)
    else if loss /= loss
    then do
      putStrLn $ "  " ++ formatElapsed t0 now
              ++ " Diverged (NaN) at epoch " ++ show (done + chunk)
      pure (model', done + chunk, False)
    else trainStage opt sched model' stage lossFn chunkSz patience
                     (minus budget chunk) (done + chunk)
                     bestLoss' staleCount' t0


----------------------------------------------------------------------
-- Public API
----------------------------------------------------------------------

||| Run multi-stage curriculum training with periodic data regen.
||| Returns (trained model, total epochs completed).
|||
||| The `Schedule` is applied per-epoch via `setLearningRate` on the
||| supplied `NativeOptimizer` (V2 keeps optimizer state in C; no
||| per-epoch optimizer rebuild needed).
export
runCurriculum :
  {d : Device} -> UserDeviceTraining d => RuntimeDType dt => Linked d => Compatible d dt => IsFloating dt =>
  {i, o, n : Nat} -> {hs : List Nat} ->
  NativeOptimizer d ->
  Schedule ->
  Network i hs o d dt WithGrad ->
  LossFn d dt o ->
  List (Stage d i o n) ->
  (totalEpochs : Nat) ->
  (patience : Nat) ->
  (chunkSize : Nat) ->
  IO (Network i hs o d dt WithGrad, Nat)
runCurriculum _ _ model _ [] _ _ _ = pure (model, 0)
runCurriculum opt sched model lossFn stages totalEpochs patience chunkSize = do
  t0 <- clockTime Monotonic
  go model stages totalEpochs 0 t0
  where
    go : Network i hs o d dt WithGrad ->
         List (Stage d i o n) ->
         (budget : Nat) ->
         (epochsDone : Nat) ->
         Clock Monotonic ->
         IO (Network i hs o d dt WithGrad, Nat)
    go m [] _ done _ = pure (m, done)
    go m (stage :: rest) budget done t0 = do
      putStrLn $ "\n" ++ stage.label
      (m', done', advanced) <- trainStage opt sched m
        stage lossFn chunkSize patience budget done (1.0/0.0) 0 t0
      let remaining = minus budget (minus done' done)
      if advanced && remaining > 0
        then go m' rest remaining done' t0
        else pure (m', done')
