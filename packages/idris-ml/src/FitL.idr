||| The linear-resource (`L IO`) `fit` driver — the migration counterpart of
||| `Fit.idr`. The model is a single-owner linear resource: every step
||| consumes it and threads it back (`EpochStepL`), so reusing a stale handle
||| (freeze/eval then train) is a compile-time linearity error. Mirrors the
||| IO `fit`/`fitSupervised`/`fitSupervisedMixed`/`fitCustom` shapes; the
||| epoch loop is `Train.EngineL.runEpochLoopL`. Lives in its own module so
||| the linear imports stay clear of `Fit.idr`'s bare `Nat` arithmetic (which
||| re-defaults to `Integer` once `Data.Linear` is in scope); here the few
||| `Nat` accumulators use `Z`/`S` to sidestep that.
|||
||| Data stays plain `IO` (`DataStream.next` lifted via `liftIO1`) — data is
||| not a linear resource; model linearity is orthogonal to it. The optimizer
||| step functions (`nativeTrainStep`/`trainStepScaled`/`applyScale`/`tick`)
||| touch only the C registry, never the model value, so they stay `IO` and
||| are lifted at the call site. Coexists with `Fit.idr`; the IO surface is
||| deleted when every caller is on `L IO`.
module FitL

import Control.Linear.LIO
import Data.IORef
import Data.Linear
import Data.Maybe
import Data.Vect
import System.Clock

import Checkpoint
import DataStream
import Executor
import GradScaler
import Optimizer
import Tensor
import Train
import Train.Engine
import Train.EngineL
import Util
import Util.Log

-- `Data.Linear.Copies` exports `Nil`, making the `[]` in scalar `Tensor []`
-- (the loss type) ambiguous. Hide it — we never construct a `Copies`.
%hide Data.Linear.Copies.Nil

||| A linear training step: consume the model + a batch, do the task's
||| forward / backward / optimizer-step / state-threading, and return the
||| (rebuilt) model beside the banged loss. The `L IO` analogue of
||| `Fit.EpochStep`. Supervised rebuilds the model unchanged (params update
||| in the C registry); RL threads its state bundle.
public export
0 EpochStepL : Type -> Type -> Type
EpochStepL m batch = (1 _ : m) -> batch -> L IO {use = 1} (LPair (!* Double) m)

-- One full dataset pass for the linear loop: fold `step` over `steps`
-- batches, threading the model linearly and accumulating the mean of the
-- *finite* batch losses (NaN handling per `nanHalts`, as in `Fit.runPass`).
-- `Z`/`S` accumulators avoid bare `Nat` literals (see module header).
runPassL : {0 ex : Executor} -> UserExecutorTraining ex => UserExecutorTransfer ex =>
           {0 m : Type} -> {0 batch : Type} -> (nanHalts : Bool) ->
           EpochStepL m batch -> DataStream batch ->
           (steps : Nat) -> (accSum : Double) -> (accCount : Nat) -> (1 _ : m) ->
           L IO {use = 1} (LPair (!* Double) m)
runPassL _ _ _ Z accSum accCount m =
  pure1 (MkBang (case accCount of
                   Z     => 0.0 / 0.0
                   (S _) => accSum / cast accCount) # m)
runPassL nanHalts step s (S k) accSum accCount m = do
  b <- liftIO1 s.next
  (MkBang loss # m') <- step m b
  liftIO1 (when (backendTag {ex} == "mlx") $ do
             forceMajorGc
             ignore drainManagedHandles)
  if isDiverged loss
    then if nanHalts
           then pure1 (MkBang (0.0 / 0.0) # m')
           else runPassL {ex} nanHalts step s k accSum accCount m'
    else runPassL {ex} nanHalts step s k (accSum + loss) (S accCount) m'

||| Optimizer-free linear `fit`: the `L IO` analogue of `Fit.fitCustom`.
||| Same epoch-loop machinery (full pass, early stop, checkpoint, NaN
||| handling, mlx hygiene) with no optimizer and so no schedule tick — for
||| training whose updates live entirely in the (linear) step.
export
fitCustomL : {0 ex : Executor} -> UserExecutorTraining ex => UserExecutorTransfer ex =>
             {0 m : Type} -> {0 batch : Type} -> {default True nanHalts : Bool} ->
             EpochStepL m batch -> DataStream batch -> TrainConfig m -> (1 _ : m) ->
             L IO {use = 1} (LPair (!* (Nat, Double)) m)
fitCustomL {nanHalts} step s cfg m0 = do
  tStart  <- liftIO1 (clockTime Monotonic)
  liftIO1 (logInfo $ "Fitting... [backend=" ++ backendName {ex} ++ "]")
  bestRef <- liftIO1 (newIORef (the Double (1.0 / 0.0)))
  startEp <- liftIO1 (resumeFromCheckpoint cfg.checkpoint bestRef)
  let stepsPerEpoch : Nat := fromMaybe (the Nat 1) s.epochLen
  let (_ ** (esStep, esInit, esTerm)) = earlyStopMachine cfg.earlyStop
  (MkBang (epochsDone, loss) # mFin) <-
    runEpochLoopL {ex} cfg.totalEpochs cfg.logEvery cfg.metricsL cfg.checkpoint
                  bestRef nanHalts esStep esInit esTerm
                  (\mm, ep => do
                     liftIO1 (cfg.beforeEpoch ep)
                     runPassL {ex} nanHalts step s stepsPerEpoch 0.0 Z mm)
                  tStart startEp m0
  liftIO1 (returnBestCheckpoint cfg.checkpoint)
  tEnd <- liftIO1 (clockTime Monotonic)
  liftIO1 (logInfo $ formatTimingSummary tStart tEnd epochsDone)
  liftIO1 (logInfo $ formatPerfMsPerEp tStart tEnd epochsDone)
  liftIO1 (profileReport {ex})
  pure1 (MkBang (epochsDone, loss) # mFin)

||| Run linear training. The `L IO` analogue of `Fit.fit`: `fitCustomL` plus
||| a per-epoch `tick opt` to advance the optimizer's LR schedule.
export
fitL : {0 ex : Executor} -> UserExecutorTraining ex => UserExecutorTransfer ex =>
       {0 m : Type} -> {0 batch : Type} -> {default True nanHalts : Bool} ->
       EpochStepL m batch -> Optimizer ex -> DataStream batch -> TrainConfig m -> (1 _ : m) ->
       L IO {use = 1} (LPair (!* (Nat, Double)) m)
fitL {nanHalts} step opt s cfg m0 =
  fitCustomL {ex} {nanHalts} step s
             ({ beforeEpoch := \ep => do tick opt ep; cfg.beforeEpoch ep } cfg) m0

||| Supervised convenience for the linear loop: give a linear loss function
||| (consume the model, run `forwardL`, return the scalar loss + the model),
||| never call `nativeTrainStep`. Builds an `EpochStepL` doing one fused step
||| per batch (zero-grad → backward → clip → step) and threads the model. The
||| `L IO` analogue of `Fit.fitSupervised`.
export
fitSupervisedL : {0 ex : Executor} -> Backend ex dt => UserExecutorTransfer ex =>
                 IsFloating dt => {0 m : Type} -> {0 batch : Type} ->
                 Optimizer ex ->
                 ((1 _ : m) -> batch -> L IO {use = 1} (LPair (!* (Tensor [] ex dt WithGrad)) m)) ->
                 DataStream batch -> TrainConfig m -> (1 _ : m) ->
                 L IO {use = 1} (LPair (!* (Nat, Double)) m)
fitSupervisedL opt lossFn s cfg m0 =
  fitL (\mm, b => do
          (MkBang loss # m') <- lossFn mm b
          d <- liftIO1 (nativeTrainStep opt loss)
          pure1 (MkBang d # m'))
       opt s cfg m0

||| Mixed-precision supervised convenience for the linear loop: scales the
||| loss + steps via the `GradScaler` (overflow → step skipped, not
||| divergence). The `L IO` analogue of `Fit.fitSupervisedMixed`.
export
fitSupervisedMixedL : {0 ex : Executor} -> Backend ex dt => UserExecutorTransfer ex =>
                      IsFloating dt => {0 m : Type} -> {0 batch : Type} ->
                      Optimizer ex -> GradScaler ex dt ->
                      ((1 _ : m) -> batch -> L IO {use = 1} (LPair (!* (Tensor [] ex dt WithGrad)) m)) ->
                      DataStream batch -> TrainConfig m -> (1 _ : m) ->
                      L IO {use = 1} (LPair (!* (Nat, Double)) m)
fitSupervisedMixedL opt gs lossFn s cfg m0 =
  fitL {nanHalts = False}
       (\mm, b => do
          (MkBang loss # m') <- lossFn mm b
          scaled <- liftIO1 (applyScale gs loss)
          d <- liftIO1 (trainStepScaled opt gs scaled)
          pure1 (MkBang d # m'))
       opt s cfg m0
