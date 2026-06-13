||| The v1 unified training driver. One `fit` for supervised, recurrent,
||| two-phase, and RL — the step function owns control-flow + the
||| optimizer step + (optional) model-state threading; `fit` owns the
||| epoch loop, schedule tick, early stop, checkpointing, NaN handling,
||| and the mlx generation hygiene (all via Train.Engine.runEpochLoop,
||| the same engine the legacy runTrainingIO now uses). The supervised
||| 90% use `fitSupervised` / `fitSupervisedMixed` and never touch
||| `nativeTrainStep`; RL/custom loops pass their own `EpochStep`.
|||
||| This refines api-critique §N6: there `fit` *owns* the optimizer step
||| (Step returns just the loss), which can't express DQN's many-steps-
||| per-episode or PPO's K-epoch update. The shape that already unifies
||| every example — including RL — is runTrainingIO's `m -> dp ->
||| IO (m, Double)`: the step owns stepping + threading. We adopt that
||| and add the supervised convenience wrappers. (Recorded in
||| design-decisions.md.)
module Fit

import Data.IORef
import Data.Vect
import System.Clock

import Util
import Util.Log
import Executor
import Tensor
import Optimizer
import GradScaler
import Checkpoint
import Train          -- TrainConfig + its builders (reused as fit's config)
import Train.Engine
import DataStream

||| A training step: take the model + a batch, do whatever forward /
||| backward / optimizer-step / state-threading the task needs, and
||| return the (possibly updated) model and the loss to log. Supervised
||| returns the model unchanged; RL threads its state bundle.
public export
0 EpochStep : Type -> Type -> Type
EpochStep m batch = m -> batch -> IO (m, Double)

||| Run training. Pulls one batch per epoch from `stream`, calls `step`,
||| and drives the shared engine. `nanHalts` (default True) treats a NaN
||| loss as divergence; set False for the mixed-precision overflow-skip
||| semantics (the scaler returns NaN to mean "step skipped").
||| The optimizer carries the LR schedule (tick is called per epoch);
||| `cfg.beforeEpoch` remains as an extra hook.
export
fit : {0 ex : Executor} -> UserExecutorTraining ex => UserExecutorTransfer ex =>
      {0 m : Type} -> {0 batch : Type} -> {default True nanHalts : Bool} ->
      EpochStep m batch -> Optimizer ex -> DataStream batch -> TrainConfig m -> m ->
      IO (m, Nat, Double)
fit {nanHalts} step opt s cfg m0 = do
  tStart <- clockTime Monotonic
  logInfo $ "Fitting... [backend=" ++ backendName {ex} ++ "]"
  bestRef <- newIORef (the Double (1.0/0.0))
  startEp <- resumeFromCheckpoint cfg.checkpoint bestRef
  let perEpoch : m -> Nat -> IO (m, Double)
      perEpoch m ep = do
        tick opt ep
        cfg.beforeEpoch ep
        b <- s.next
        step m b
  let (_ ** (esStep, esInit, esTerm)) = earlyStopMachine cfg.earlyStop
  (mFin, epochsDone, loss) <-
    runEpochLoop {ex} cfg.totalEpochs cfg.logEvery cfg.metrics cfg.checkpoint
                 bestRef nanHalts esStep esInit esTerm perEpoch tStart startEp m0
  returnBestCheckpoint cfg.checkpoint
  tEnd <- clockTime Monotonic
  logInfo $ formatTimingSummary tStart tEnd epochsDone
  logInfo $ formatPerfMsPerEp tStart tEnd epochsDone
  profileReport {ex}
  pure (mFin, epochsDone, loss)

||| Supervised convenience: give a loss function, never call
||| `nativeTrainStep`. Builds an `EpochStep` that does one fused step per
||| batch (zero-grad → backward → clip → step) and returns the model
||| unchanged (params update in the registry).
export
fitSupervised : {0 ex : Executor} -> Backend ex dt => UserExecutorTransfer ex =>
                IsFloating dt => {0 m : Type} -> {0 batch : Type} ->
                Optimizer ex -> (m -> batch -> IO (Tensor [] ex dt WithGrad)) ->
                DataStream batch -> TrainConfig m -> m -> IO (m, Nat, Double)
fitSupervised opt lossFn s cfg m0 =
  fit (\m, b => do loss <- lossFn m b
                   d <- nativeTrainStep opt loss
                   pure (m, d))
      opt s cfg m0

||| Mixed-precision supervised convenience: scales the loss + steps via
||| the GradScaler (overflow → step skipped, not divergence). The loss
||| function builds the loss at the compute dtype `dt`.
export
fitSupervisedMixed : {0 ex : Executor} -> Backend ex dt => UserExecutorTransfer ex =>
                     IsFloating dt => {0 m : Type} -> {0 batch : Type} ->
                     Optimizer ex -> GradScaler ex dt ->
                     (m -> batch -> IO (Tensor [] ex dt WithGrad)) ->
                     DataStream batch -> TrainConfig m -> m -> IO (m, Nat, Double)
fitSupervisedMixed opt gs lossFn s cfg m0 =
  fit {nanHalts = False}
      (\m, b => do loss <- lossFn m b
                   scaled <- applyScale gs loss
                   d <- trainStepScaled opt gs scaled
                   pure (m, d))
      opt s cfg m0
