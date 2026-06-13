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
  startEp <- case cfg.checkpoint of
    Nothing  => pure 0
    Just pol => do
      mst <- pol.loadState (pol.dir ++ "/last")
      case mst of
        Nothing => pure 0
        Just (ep0, best0) => do
          writeIORef bestRef best0
          logInfo $ "  Resuming from epoch " ++ show ep0
                   ++ " (best=" ++ showFix 6 best0 ++ ")"
          pure ep0
  result@(m, epochsDone, loss) <- case cfg.earlyStop of
    NoEarlyStop => goSimple bestRef startEp model0 0.0 tStart
    Patience pat minD => goPatience bestRef startEp model0 (1.0/0.0) 0 tStart pat minD
    WindowedAvg thresh win pat => goWindowed bestRef startEp model0 0.0 0 [] 0 tStart thresh win pat
    WindowedPercentile pct thresh win pat =>
      goWindowedPercentile bestRef startEp model0 [] 0 0 tStart pct thresh win pat
  -- Return-best: reload the best checkpoint so the returned model is
  -- the best seen, not the last (Lightning semantics). loadState
  -- mutates the registry in place; the model structure is unchanged.
  finalModel <- case cfg.checkpoint of
    Just pol => if pol.keepBest
                  then do _ <- pol.loadState (pol.dir ++ "/best"); pure m
                  else pure m
    Nothing  => pure m
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
  where
    shouldLog : Nat -> Bool
    shouldLog ep = case cfg.logEvery of
      Z => False
      S k => modNatNZ ep (S k) ItIsSucc == 0

    fmtMetrics : List (String, String) -> String
    fmtMetrics [] = ""
    fmtMetrics ((k, v) :: rest) = "\t" ++ k ++ "=" ++ v ++ fmtMetrics rest

    -- Force every character of the metric strings. The strings are built
    -- lazily from tensor reads (`show (primItem …)`, `argmaxAtPtr …`), so
    -- this drags those reads to happen *now* — inside the withNoGrad
    -- bracket below, before its exit drain frees the eval tensors. Without
    -- it the thunks dangle on mlx (where the generation sweep is real, not
    -- a no-op like tape/torch). Always returns False (sum of lengths ≥ 0).
    forceMetrics : List (String, String) -> Bool
    forceMetrics xs = foldl (\acc, (k, v) => acc + length k + length v) 0 xs < 0

    logEpoch : Clock Monotonic -> Nat -> Double -> model -> IO ()
    logEpoch t0 ep loss m = do
      now <- clockTime Monotonic
      -- Per-epoch metrics are pure evaluation: run them with autograd off
      -- so they build no tape, and force the result strings inside the
      -- bracket (see `forceMetrics`). A grad-mode metrics pass leaves a
      -- live tape whose tensors the epoch-end generation sweep then frees,
      -- crashing the next epoch on mlx (use-after-free).
      extra <- withNoGrad {ex} $ do
                 e <- cfg.metrics m
                 if forceMetrics e then pure e else pure e
      liveH <- primIO (primLiveCount {ex})
      peakH <- primIO (primPeakLiveCount {ex})
      let memSuffix = "\tpeak=" ++ show (getRssMB 0) ++ "MB"
                   ++ "\tcur=" ++ show (getCurrentRssMB 0) ++ "MB"
                   ++ "\thandles=" ++ show liveH
                   ++ "\tpeakhandles=" ++ show peakH
      logInfo $ "  " ++ formatElapsed t0 now ++ " " ++ show ep
               ++ "\tloss=" ++ showFix 6 loss ++ memSuffix ++ fmtMetrics extra

    -- Per-epoch generation-scoped free. `epochBegin` marks the tensor
    -- generation; `epochEnd` frees the epoch's wrap-only (rc==1) grad
    -- intermediates, sparing registry params (rc>1) and pre-epoch state.
    -- Bounds the training-side live-handle count the same way withNoGrad
    -- bounds eval. No-op on tape/torch (no buffer ceiling). Defined here so
    -- they capture the device `d` (the loops shadow it with `d <- dataSrc`).
    epochBegin : IO ()
    epochBegin = primIO (primEpochBegin {ex})
    -- On mlx, force a major GC + drain the managed-handle guardian *before*
    -- the sweep, mirroring withNoGrad. The epoch's grad intermediates are
    -- only reachable until the epoch fn returns, so the per-step
    -- `(collect 0)` minor GC in `nativeTrainStep` can't collect them; here,
    -- post-return, the major GC makes their dead wraps unreachable, drain
    -- releases them (rc 1->0), and `primEpochEnd`'s sweep then frees the
    -- husks via its rc==0 path. Without this the mlx husks accumulate
    -- across epochs (the long-grad-mode handle leak). tape/torch skip it:
    -- their release is a no-op stub and primEpochEnd is a no-op, so a
    -- per-epoch full GC would be pure overhead.
    epochEnd : IO ()
    epochEnd = do
      when (backendTag {ex} == "mlx") $ do
        forceMajorGc
        _ <- drainManagedHandles
        pure ()
      primIO (primEpochEnd {ex})

    divisibleBy : Nat -> Nat -> Bool
    divisibleBy _ Z     = False
    divisibleBy n (S k) = modNatNZ n (S k) ItIsSucc == 0

    -- After each completed (non-NaN) epoch: keep-best save to
    -- `<dir>/best` if the monitored scalar improved (lower is better),
    -- then periodic save to `<dir>/last`. `bestRef` holds the best
    -- metric seen; the sidecar stores `S ep` as the resume point. The
    -- save runs outside any withNoGrad bracket (it reads registry
    -- params, rc>1, so it's sweep-safe) except the optional monitor
    -- eval, which is bracketed.
    postEpoch : IORef Double -> Nat -> Double -> IO ()
    postEpoch bestRef ep loss =
      case cfg.checkpoint of
        Nothing  => pure ()
        Just pol => do
          when pol.keepBest $ do
            cur <- case pol.monitor of
                     Nothing => pure loss
                     Just f  => withNoGrad {ex} f
            b <- readIORef bestRef
            when (cur < b) $ do
              writeIORef bestRef cur
              ignore $ pol.saveState (pol.dir ++ "/best") (S ep) cur
          when (divisibleBy (S ep) pol.everyN) $ do
            b <- readIORef bestRef
            ignore $ pol.saveState (pol.dir ++ "/last") (S ep) b

    -- Simple: no early stopping
    goSimple : IORef Double -> Nat -> model -> Double -> Clock Monotonic -> IO (model, Nat, Double)
    goSimple bestRef ep m lastLoss t0 =
      if ep >= cfg.totalEpochs then pure (m, ep, lastLoss)
      else do
        epochBegin
        cfg.beforeEpoch ep
        d <- dataSrc
        (m', loss) <- epochFn m d
        when (shouldLog ep) $ logEpoch t0 ep loss m'
        epochEnd
        if loss /= loss
          then do now <- clockTime Monotonic
                  logWarn $ "  " ++ formatElapsed t0 now ++ " Diverged (NaN) at epoch " ++ show ep
                  pure (m', ep, loss)
          else do postEpoch bestRef ep loss
                  goSimple bestRef (S ep) m' loss t0

    diverged : Clock Monotonic -> Nat -> model -> Double -> IO (model, Nat, Double)
    diverged t0 ep m loss = do
      now <- clockTime Monotonic
      logWarn $ "  " ++ formatElapsed t0 now ++ " Diverged (NaN) at epoch " ++ show ep
      pure (m, ep, loss)

    -- Patience-based early stopping
    goPatience : IORef Double -> Nat -> model -> Double -> Nat -> Clock Monotonic -> Nat -> Double ->
                 IO (model, Nat, Double)
    goPatience bestRef ep m bestLoss stale t0 pat minD =
      if ep >= cfg.totalEpochs then pure (m, ep, bestLoss)
      else do
        epochBegin
        cfg.beforeEpoch ep
        d <- dataSrc
        (m', loss) <- epochFn m d
        when (shouldLog ep) $ logEpoch t0 ep loss m'
        epochEnd
        if loss /= loss
          then diverged t0 ep m' loss
          else do
            postEpoch bestRef ep loss
            let improved = loss < bestLoss - minD
                best' = if improved then loss else bestLoss
                stale' : Nat
                stale' = if improved then 0 else stale + 1
            if pat > 0 && stale' >= pat
              then do now <- clockTime Monotonic
                      logInfo $ "  " ++ formatElapsed t0 now ++ " Early stop at epoch "
                               ++ show (ep + 1) ++ " (patience=" ++ show pat ++ ")"
                      pure (m', ep + 1, loss)
              else goPatience bestRef (S ep) m' best' stale' t0 pat minD

    -- Windowed-average early stopping
    goWindowed : IORef Double -> Nat -> model -> Double -> Nat -> List Double -> Nat ->
                 Clock Monotonic -> Double -> Nat -> Nat ->
                 IO (model, Nat, Double)
    goWindowed bestRef ep m iSum iCount avgs convCount t0 thresh win pat =
      if ep >= cfg.totalEpochs then pure (m, ep, 0.0)
      else do
        epochBegin
        cfg.beforeEpoch ep
        d <- dataSrc
        (m', loss) <- epochFn m d
        when (shouldLog ep) $ logEpoch t0 ep loss m'
        epochEnd
        if loss /= loss
          then diverged t0 ep m' loss
          else postEpoch bestRef ep loss >>
               (let iSum' = iSum + loss
                    iCount' = iCount + 1
                in if iCount' < 100
                 then goWindowed bestRef (S ep) m' iSum' iCount' avgs convCount t0 thresh win pat
                 else let avg = iSum' / 100.0
                          avgs' = avg :: avgs
                          wc = max 1 (div win 100)
                      in if length avgs' < wc
                        then goWindowed bestRef (S ep) m' 0.0 0 avgs' convCount t0 thresh win pat
                        else let windowAvg = foldl (+) 0.0 (take wc avgs') / cast wc
                             in if windowAvg >= thresh
                               then goWindowed bestRef (S ep) m' 0.0 0 avgs' 0 t0 thresh win pat
                               else let cc = convCount + 1
                                    in if cc >= pat
                                      then do now <- clockTime Monotonic
                                              logInfo $ "  " ++ formatElapsed t0 now
                                                       ++ " Converged at epoch " ++ show (ep + 1)
                                                       ++ " (window_avg=" ++ show windowAvg ++ ")"
                                              pure (m', ep + 1, loss)
                                      else do now <- clockTime Monotonic
                                              logInfo $ "    " ++ formatElapsed t0 now
                                                       ++ " convergence " ++ show cc ++ "/" ++ show pat
                                                       ++ " (window_avg=" ++ show windowAvg ++ ")"
                                              goWindowed bestRef (S ep) m' 0.0 0 avgs' cc t0 thresh win pat)

    -- Windowed-percentile early stopping. Maintains a rolling window of
    -- the last `win` per-epoch losses. Every 100 epochs, sorts the window,
    -- picks the loss at the kth percentile (where k = floor(pct * len)),
    -- and compares to threshold. Fires when the percentile is below
    -- threshold for `pat` consecutive checks.
    --
    -- This computes percentile over RAW PER-EPOCH LOSSES, not chunk-means.
    -- That distinction matters for bimodal losses (variable-length-sequence
    -- tasks): individual epochs that happen to draw short sequences hit
    -- near-zero loss long before the chunk-MEAN does. p10 of raw losses is
    -- "the easiest 10% of recent epochs" — drops below threshold once the
    -- model handles those reliably.
    goWindowedPercentile : IORef Double -> Nat -> model -> List Double -> Nat -> Nat ->
                           Clock Monotonic -> Double -> Double -> Nat -> Nat ->
                           IO (model, Nat, Double)
    goWindowedPercentile bestRef ep m recent epochsSinceCheck convCount t0 pct thresh win pat =
      if ep >= cfg.totalEpochs then pure (m, ep, 0.0)
      else do
        epochBegin
        cfg.beforeEpoch ep
        d <- dataSrc
        (m', loss) <- epochFn m d
        when (shouldLog ep) $ logEpoch t0 ep loss m'
        epochEnd
        if loss /= loss
          then diverged t0 ep m' loss
          else postEpoch bestRef ep loss >>
               (let recent' = take win (loss :: recent)
                    esc' = epochsSinceCheck + 1
                in if esc' < 100 || length recent' < win
                 then goWindowedPercentile bestRef (S ep) m' recent' esc' convCount
                                            t0 pct thresh win pat
                 else let sorted = sort recent'
                          idx = min (minus win 1)
                                    (cast {to=Nat} (the Integer
                                           (cast (pct * cast win))))
                          pctVal = case drop idx sorted of
                                     (x :: _) => x
                                     [] => 0.0
                      in if pctVal >= thresh
                        then goWindowedPercentile bestRef (S ep) m' recent' 0 0
                                                   t0 pct thresh win pat
                        else let cc = convCount + 1
                             in if cc >= pat
                               then do now <- clockTime Monotonic
                                       logInfo $ "  " ++ formatElapsed t0 now
                                                ++ " Converged at epoch " ++ show (ep + 1)
                                                ++ " (p" ++ show (cast {to=Int} (pct * 100.0))
                                                ++ "_loss=" ++ show pctVal ++ ")"
                                       pure (m', ep + 1, loss)
                               else do now <- clockTime Monotonic
                                       logInfo $ "    " ++ formatElapsed t0 now
                                                ++ " convergence " ++ show cc ++ "/" ++ show pat
                                                ++ " (p" ++ show (cast {to=Int} (pct * 100.0))
                                                ++ "_loss=" ++ show pctVal ++ ")"
                                       goWindowedPercentile bestRef (S ep) m' recent' 0 cc
                                                             t0 pct thresh win pat)


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
