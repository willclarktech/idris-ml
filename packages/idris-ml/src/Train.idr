||| Unified training runner for all examples.
||| Handles epoch iteration, progress logging, NaN detection, early stopping,
||| CLI arg parsing, and result formatting.
module Train

import Data.IORef
import Data.List
import Data.Nat
import System.Clock

import Util
import Util.Log
import Executor
import Tensor
import Schedule
import Checkpoint
import public Train.Engine  -- MetricsFn, EarlyStopConfig, showFix + the shared engine pieces


----------------------------------------------------------------------
-- CLI Argument Parsing
----------------------------------------------------------------------

||| Specification for a single CLI flag.
public export
record ArgSpec a where
  constructor Arg
  flag : String
  apply : String -> a -> a

||| Parse CLI args using a list of flag specs.
||| Unknown flags are silently skipped.
export
parseArgs : a -> List (ArgSpec a) -> List String -> a
parseArgs defaults specs args = go args defaults
  where
    findSpec : String -> List (ArgSpec a) -> Maybe (ArgSpec a)
    findSpec _ [] = Nothing
    findSpec f (s :: rest) = if s.flag == f then Just s else findSpec f rest

    go : List String -> a -> a
    go [] c = c
    go (f :: v :: rest) c = case findSpec f specs of
      Just s => go rest (s.apply v c)
      Nothing => go (v :: rest) c
    go (_ :: rest) c = go rest c

||| Cast a String to Nat (via Integer).
export
castNat : String -> Nat
castNat s = cast (the Integer (cast s))

||| Cast a String to Bits64 (via Integer).
export
castBits64 : String -> Bits64
castBits64 s = cast (the Integer (cast s))


----------------------------------------------------------------------
-- Result Formatting
----------------------------------------------------------------------

||| Format a machine-readable RESULT line from key-value pairs.
export
formatResult : List (String, String) -> String
formatResult kvs = "RESULT" ++ concatMap (\(k, v) => "\t" ++ k ++ "=" ++ v) kvs

-- `showFix`, `EarlyStopConfig`, and `MetricsFn` moved to Train.Engine
-- (re-exported above via `import public Train.Engine`).


----------------------------------------------------------------------
-- Training Configuration
----------------------------------------------------------------------

||| Per-epoch return + rolling-window state for RL examples. Lets RL
||| epoch closures push the most recent episodic return into a metrics
||| callback for the unified `[hh:mm:ss] EP\tloss\tpeak\tcur\treturn\trecent_K`
||| log format. Pair with `rlMetricsFn` below.
public export
record RLMetricsState where
  constructor MkRLMetricsState
  lastReturn : IORef Double
  recentWindow : IORef (List Double)
  windowSize : Nat

||| Build a fresh `RLMetricsState` with an empty rolling window of size N.
export
newRLMetricsState : (windowSize : Nat) -> IO RLMetricsState
newRLMetricsState n = do
  r <- newIORef 0.0
  w <- newIORef []
  pure (MkRLMetricsState r w n)

||| Record this epoch's episodic return. Updates `lastReturn` and pushes
||| onto the rolling window, dropping the oldest entry when full.
export
recordReturn : RLMetricsState -> (ret : Double) -> IO ()
recordReturn s ret = do
  writeIORef s.lastReturn ret
  w <- readIORef s.recentWindow
  writeIORef s.recentWindow (take s.windowSize (ret :: w))

||| Read the latest return + recent-window mean as a `List (String, String)`
||| suitable for the per-epoch log metrics tail. Use inside an inline
||| `metrics := \_ => readRLMetrics "recent_100" s` callback. Returns
||| `[("return", X.X), (recentLabel, X.X)]` (both at 1 decimal place).
export
readRLMetrics : (recentLabel : String) -> RLMetricsState
              -> IO (List (String, String))
readRLMetrics recentLabel s = do
  r <- readIORef s.lastReturn
  w <- readIORef s.recentWindow
  let n      : Double = cast (the Integer (cast (length w)))
      recent : Double = if n == 0 then 0.0 else sum w / n
  pure [("return", showFix 1 r), (recentLabel, showFix 1 recent)]

||| Training configuration.
|||
||| `beforeEpoch` runs before each epoch's `epochFn`. Defaults to a no-op.
||| Use `applySchedule` (below) to bind a `Schedule` to a `NativeOptimizer`
||| as a beforeEpoch hook — that's how LR schedules attach to training.
public export
record TrainConfig (model : Type) where
  constructor MkTrainConfig
  totalEpochs : Nat
  logEvery : Nat
  earlyStop : EarlyStopConfig
  metrics : MetricsFn model
  beforeEpoch : Nat -> IO ()
  checkpoint : Maybe CheckpointPolicy

||| Simple config: run N epochs, log every 100, no early stopping.
export
simpleConfig : Nat -> TrainConfig model
simpleConfig n = MkTrainConfig n 100 NoEarlyStop (const (pure [])) (\_ => pure ()) Nothing

||| Config with patience-based early stopping.
export
patienceConfig : Nat -> Nat -> TrainConfig model
patienceConfig epochs pat =
  MkTrainConfig epochs 100 (Patience pat 0.001) (const (pure [])) (\_ => pure ()) Nothing

||| Config with windowed-average early stopping.
export
windowedConfig : Nat -> Double -> Nat -> Nat -> TrainConfig model
windowedConfig epochs threshold window pat =
  MkTrainConfig epochs 100 (WindowedAvg threshold window pat) (const (pure [])) (\_ => pure ()) Nothing

||| Config with windowed-percentile early stopping. Robust to bimodal
||| loss distributions (e.g. variable-length-sequence tasks where short
||| sequences quickly hit near-zero loss while long ones plateau higher).
export
windowedPercentileConfig : Nat -> Double -> Double -> Nat -> Nat -> TrainConfig model
windowedPercentileConfig epochs pct threshold window pat =
  MkTrainConfig epochs 100 (WindowedPercentile pct threshold window pat) (const (pure [])) (\_ => pure ()) Nothing

||| Attach a checkpoint policy to a config (auto-save / keep-best /
||| resume). Examples plug a `fileCheckpoint` policy in here.
export
withCheckpoint : CheckpointPolicy -> TrainConfig model -> TrainConfig model
withCheckpoint pol cfg = { checkpoint := Just pol } cfg

||| Build a TrainConfig with no checkpoint policy (the common case).
||| Examples opt into checkpointing by wrapping with `withCheckpoint`.
export
mkTrainConfig : Nat -> Nat -> EarlyStopConfig -> MetricsFn model -> (Nat -> IO ()) ->
                TrainConfig model
mkTrainConfig e l es m b = MkTrainConfig e l es m b Nothing

||| Bind a Schedule to a NativeOptimizer, producing a beforeEpoch hook.
||| Per epoch, sets the optimizer's base LR to `schedule epoch`. Plug into
||| `TrainConfig` via the `beforeEpoch` field:
|||   `let cfg = { beforeEpoch := applySchedule sched opt } (simpleConfig 1000)`
export
applySchedule : UserExecutorTraining ex => Schedule -> NativeOptimizer ex -> Nat -> IO ()
applySchedule sched opt ep = setLearningRate opt (sched ep)


----------------------------------------------------------------------
-- Training Runner
----------------------------------------------------------------------

||| Run training with an IO-based epoch function. Use this when the
||| per-epoch step needs IO (e.g. sampling a replay-buffer batch, running
||| a vectorised env rollout). For pure epochs, use `runTraining`.
export
runTrainingIO :
  {0 ex : Executor} -> UserExecutorTraining ex => UserExecutorTransfer ex =>
  {0 model : Type} -> {0 dp : Type} ->
  (epochFn : model -> dp -> IO (model, Double)) ->
  (dataSrc : IO dp) ->
  TrainConfig model ->
  model ->
  IO (model, Nat, Double)
runTrainingIO {model} epochFn dataSrc cfg model0 = do
  tStart <- clockTime Monotonic
  logInfo $ "Training... [backend=" ++ backendName {ex} ++ "]"
  bestRef <- newIORef (the Double (1.0/0.0))
  startEp <- resumeFromCheckpoint cfg.checkpoint bestRef
  -- perEpoch = the legacy beforeEpoch hook + data pull + epoch fn. The
  -- shared engine (Train.Engine.runEpochLoop) owns the generation
  -- bracket, logging, NaN handling, checkpointing, and early stop.
  let perEpoch : model -> Nat -> IO (model, Double)
      perEpoch m ep = do cfg.beforeEpoch ep; d <- dataSrc; epochFn m d
  let (_ ** (esStep, esInit, esTerm)) = earlyStopMachine cfg.earlyStop
  (m, epochsDone, loss) <-
    runEpochLoop {ex} cfg.totalEpochs cfg.logEvery cfg.metrics cfg.checkpoint
                 bestRef True esStep esInit esTerm perEpoch tStart startEp model0
  -- Return-best: reload the best checkpoint so the returned model is
  -- the best seen, not the last (Lightning semantics). loadState
  -- mutates the registry in place; the model structure is unchanged.
  returnBestCheckpoint cfg.checkpoint
  let finalModel = m
  tEnd <- clockTime Monotonic
  logInfo $ formatTimingSummary tStart tEnd epochsDone
  logInfo $ formatPerfMsPerEp tStart tEnd epochsDone
  liveH <- primIO (primLiveCount {ex})
  peakH <- primIO (primPeakLiveCount {ex})
  logInfo $ "Peak RSS: " ++ show (getRssMB 0) ++ " MB"
          ++ "\tCurrent RSS: " ++ show (getCurrentRssMB 0) ++ " MB"
          ++ "\tLive handles: " ++ show liveH
          ++ "\tPeak handles: " ++ show peakH
  profileReport {ex}
  pure (finalModel, epochsDone, loss)

||| Run training with an IO-typed epoch function. After the
||| smart-constructor IO refactor, epochVar etc. all
||| return `IO (model, Double)`, so this is now a thin alias for
||| `runTrainingIO`.
export
runTraining :
  {0 ex : Executor} -> UserExecutorTraining ex => UserExecutorTransfer ex =>
  {0 model : Type} -> {0 dp : Type} ->
  (epochFn : model -> dp -> IO (model, Double)) ->
  (dataSrc : IO dp) ->
  TrainConfig model ->
  model ->
  IO (model, Nat, Double)
runTraining = runTrainingIO {ex}
