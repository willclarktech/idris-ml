||| The unified `fit` driver (`L IO`). The model is a single-owner linear
||| resource: every step consumes it and threads it back (`EpochStep`), so
||| reusing a stale handle (freeze/eval then train) is a compile-time linearity
||| error. One `fit` for supervised / recurrent / two-phase / RL — the step
||| function owns control-flow + the optimizer step + (optional) model-state
||| threading; `fit` owns the epoch loop, schedule tick, early stop,
||| checkpointing, NaN handling, and mlx generation hygiene (via
||| `Train.EngineL.runEpochLoop`). Supervised 90% use `fitSupervised` /
||| `fitSupervisedMixed` and never touch `trainStep`; RL/custom loops pass
||| their own `EpochStep`.
|||
||| Imports `Data.Linear.Notation` (for `MkBang`/`!*`), **not** full
||| `Data.Linear` — the latter re-exports `Copies`, whose `Nil`/`(::)` shadow
||| `[]`/`::` (breaking scalar `Tensor []` and list literals) and perturb
||| `Nat`-literal defaulting. `LPair`/`#` are `Builtin` (always in scope).
||| The `Z`/`S` accumulators below keep clear of bare `Nat` literals.
|||
||| Data stays plain `IO` (`DataStream.next` lifted via `liftIO1`) — data is
||| not a linear resource; model linearity is orthogonal to it. The optimizer
||| step functions (`trainStep`/`trainStepScaled`/`applyScale`/`tick`)
||| touch only the C registry, never the model value, so they stay `IO` and
||| are lifted at the call site.
module Ml.Fit

import Control.Linear.LIO
import Data.IORef
import Data.Linear.Notation
import Data.Maybe
import Data.Vect
import System.Clock

import Ml.Checkpoint
import Ml.DataStream
import Ml.Executor
import Ml.GradScaler
import Ml.Optimizer
import Ml.Tensor
import Ml.Train
import Ml.Train.Engine
import Ml.Train.EngineL
import Ml.Util
import Ml.Util.Log

||| A linear training step: consume the model + a batch, do the task's
||| forward / backward / optimizer-step / state-threading, and return the
||| (rebuilt) model beside the banged loss. The `L IO` analogue of
||| `Fit.EpochStep`. Supervised rebuilds the model unchanged (params update
||| in the C registry); RL threads its state bundle.
public export
0 EpochStep : Type -> Type -> Type
EpochStep m batch = (1 _ : m) -> batch -> L IO {use = 1} (LPair (!* Double) m)

-- One full dataset pass for the linear loop: fold `step` over `steps`
-- batches, threading the model linearly and accumulating the mean of the
-- *finite* batch losses (NaN handling per `nanHalts`, as in `Fit.runPass`).
-- `Z`/`S` accumulators avoid bare `Nat` literals (see module header).
runPass : {0 ex : Executor} -> UserExecutorTraining ex => UserExecutorTransfer ex =>
           {0 m : Type} -> {0 batch : Type} -> (nanHalts : Bool) ->
           EpochStep m batch -> DataStream batch ->
           (steps : Nat) -> (accSum : Double) -> (accCount : Nat) -> (1 _ : m) ->
           L IO {use = 1} (LPair (!* Double) m)
runPass _ _ _ Z accSum accCount m =
  pure1 (MkBang (case accCount of
                   Z     => 0.0 / 0.0
                   (S _) => accSum / cast accCount) # m)
runPass nanHalts step s (S k) accSum accCount m = do
  b <- liftIO1 s.next
  (MkBang loss # m') <- step m b
  liftIO1 (when (backendTag {ex} == "mlx") $ do
             forceMajorGc
             ignore drainManagedHandles)
  if isDiverged loss
    then if nanHalts
           then pure1 (MkBang (0.0 / 0.0) # m')
           else runPass {ex} nanHalts step s k accSum accCount m'
    else runPass {ex} nanHalts step s k (accSum + loss) (S accCount) m'

||| Optimizer-free linear `fit`: the `L IO` analogue of `Fit.fitCustom`.
||| Same epoch-loop machinery (full pass, early stop, checkpoint, NaN
||| handling, mlx hygiene) with no optimizer and so no schedule tick — for
||| training whose updates live entirely in the (linear) step.
export
fitCustom : {0 ex : Executor} -> UserExecutorTraining ex => UserExecutorTransfer ex =>
             {0 m : Type} -> {0 batch : Type} -> {default True nanHalts : Bool} ->
             EpochStep m batch -> DataStream batch -> TrainConfig m -> (1 _ : m) ->
             L IO {use = 1} (LPair (!* (Nat, Double)) m)
fitCustom {nanHalts} step s cfg m0 = do
  -- Both inert unless their env var is set, and both belong here rather than
  -- in the examples: this is the last point before any parameter is stepped,
  -- and every example that has parameters to dump or load reaches training
  -- through this function. Keeping them out of example bodies matters —
  -- examples are read to learn the library, and alignment-harness plumbing is
  -- not part of that. `runInitL` would be the more obvious home but is
  -- executor-agnostic, so hooking it would force an `{ex = ...}` annotation
  -- at every construction site.
  liftIO1 (maybeDumpInit {ex})
  _ <- liftIO1 (maybeLoadOracle {ex})
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
                     runPass {ex} nanHalts step s stepsPerEpoch 0.0 Z mm)
                  tStart startEp m0
  liftIO1 (returnBestCheckpoint cfg.checkpoint)
  tEnd <- liftIO1 (clockTime Monotonic)
  liftIO1 (logInfo $ formatTimingSummary tStart tEnd epochsDone)
  liftIO1 (logInfo $ formatPerfMsPerEp tStart tEnd epochsDone)
  -- The C-side report prints unconditionally, so the INFO gate lives
  -- here (timing reports are INFO-class output per Util.Log's scheme).
  liftIO1 (do lvl <- getLogLevel
              when (lvl >= levelInfo) (profileReport {ex}))
  pure1 (MkBang (epochsDone, loss) # mFin)

||| Run linear training. The `L IO` analogue of `Fit.fit`: `fitCustom` plus
||| a per-epoch `tick opt` to advance the optimizer's LR schedule.
export
fit : {0 ex : Executor} -> UserExecutorTraining ex => UserExecutorTransfer ex =>
       {0 m : Type} -> {0 batch : Type} -> {default True nanHalts : Bool} ->
       EpochStep m batch -> Optimizer ex -> DataStream batch -> TrainConfig m -> (1 _ : m) ->
       L IO {use = 1} (LPair (!* (Nat, Double)) m)
fit {nanHalts} step opt s cfg m0 =
  fitCustom {ex} {nanHalts} step s
             ({ beforeEpoch := \ep => do
                  -- Top of epoch 1 = exactly one epoch of training has run.
                  -- Inert unless IDRISML_ONE_STEP is set; see
                  -- `Checkpoint.maybeDumpAfterStep`.
                  when (ep == 1) (maybeDumpAfterStep {ex})
                  tick opt ep
                  cfg.beforeEpoch ep } cfg) m0

||| Supervised convenience for the linear loop: give a linear loss function
||| (consume the model, run `forwardL`, return the scalar loss + the model),
||| never call `trainStep`. Builds an `EpochStep` doing one fused step
||| per batch (zero-grad → backward → clip → step) and threads the model. The
||| `L IO` analogue of `Fit.fitSupervised`.
export
fitSupervised : {0 ex : Executor} -> Backend ex dt => UserExecutorTransfer ex =>
                 IsFloating dt => {0 m : Type} -> {0 batch : Type} ->
                 Optimizer ex ->
                 ((1 _ : m) -> batch -> L IO {use = 1} (LPair (!* (Tensor [] ex dt WithGrad)) m)) ->
                 DataStream batch -> TrainConfig m -> (1 _ : m) ->
                 L IO {use = 1} (LPair (!* (Nat, Double)) m)
fitSupervised opt lossFn s cfg m0 =
  fit (\mm, b => do
          (MkBang loss # m') <- lossFn mm b
          d <- liftIO1 (trainStep opt loss)
          pure1 (MkBang d # m'))
       opt s cfg m0

||| Mixed-precision supervised convenience for the linear loop: scales the
||| loss + steps via the `GradScaler` (overflow → step skipped, not
||| divergence). The `L IO` analogue of `Fit.fitSupervisedMixed`.
export
fitSupervisedMixed : {0 ex : Executor} -> Backend ex dt => UserExecutorTransfer ex =>
                      IsFloating dt => {0 m : Type} -> {0 batch : Type} ->
                      Optimizer ex -> GradScaler ex dt ->
                      ((1 _ : m) -> batch -> L IO {use = 1} (LPair (!* (Tensor [] ex dt WithGrad)) m)) ->
                      DataStream batch -> TrainConfig m -> (1 _ : m) ->
                      L IO {use = 1} (LPair (!* (Nat, Double)) m)
fitSupervisedMixed opt gs lossFn s cfg m0 =
  fit {nanHalts = False}
       (\mm, b => do
          (MkBang loss # m') <- lossFn mm b
          scaled <- liftIO1 (applyScale gs loss)
          d <- liftIO1 (trainStepScaled opt gs scaled)
          pure1 (MkBang d # m'))
       opt s cfg m0
