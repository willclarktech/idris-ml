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
import Util
import Util.Log

||| A training step: take the model + a batch, do whatever forward /
||| backward / optimizer-step / state-threading the task needs, and
||| return the (possibly updated) model and the loss to log. Supervised
||| returns the model unchanged; RL threads its state bundle.
public export
0 EpochStep : Type -> Type -> Type
EpochStep m batch = m -> batch -> IO (m, Double)

-- One full dataset pass: fold `step` over `steps` batches, accumulating
-- the mean of the *finite* batch losses. NaN handling honours `nanHalts`:
-- single precision (True) treats a NaN batch loss as divergence — short-
-- circuit, return NaN so the engine halts; mixed precision (False) means
-- the scaler skipped that step — drop it from the mean and continue. On
-- mlx, drain the per-step grad husks between batches so a long pass can't
-- outrun the MTLBuffer ceiling (no-op begin/end on tape/torch). Top-level
-- (not a where-clause) so its interface-constraint binding is unambiguous.
runPass : {0 ex : Executor} -> UserExecutorTraining ex => UserExecutorTransfer ex =>
          {0 m : Type} -> {0 batch : Type} -> (nanHalts : Bool) ->
          EpochStep m batch -> DataStream batch ->
          (steps : Nat) -> (accSum : Double) -> (accCount : Nat) -> m -> IO (m, Double)
runPass _ _ _ Z accSum accCount m =
  pure (m, if accCount == 0 then 0.0/0.0 else accSum / cast accCount)
runPass nanHalts step s (S k) accSum accCount m = do
  b <- s.next
  (m', loss) <- step m b
  when (backendTag {ex} == "mlx") $ do
    forceMajorGc
    ignore drainManagedHandles
  if isDiverged loss
    then if nanHalts
           then pure (m', 0.0/0.0)
           else runPass {ex} nanHalts step s k accSum accCount m'
    else runPass {ex} nanHalts step s k (accSum + loss) (accCount + 1) m'

||| Run training. Each epoch is one full dataset pass: `fit` pulls
||| `epochLen` batches from `stream` (one batch when the stream is
||| infinite — `epochLen = Nothing`, the RL/synthetic case), calls `step`
||| on each, and reports the mean batch loss to the shared engine. Early
||| stop / checkpoint cadence is therefore per-pass, matching PyTorch's
||| epoch. `nanHalts` (default True) treats a NaN loss as divergence; set
||| False for the mixed-precision overflow-skip semantics (the scaler
||| returns NaN to mean "step skipped"). The optimizer carries the LR
||| schedule (tick is called per epoch); `cfg.beforeEpoch` is an extra hook.
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
  let stepsPerEpoch : Nat := fromMaybe 1 s.epochLen
  let perEpoch : m -> Nat -> IO (m, Double)
      perEpoch m ep = do
        tick opt ep
        cfg.beforeEpoch ep
        runPass {ex} nanHalts step s stepsPerEpoch 0.0 0 m
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
