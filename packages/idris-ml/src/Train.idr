||| Unified training runner for all examples.
||| Handles epoch iteration, progress logging, NaN detection, early stopping,
||| CLI arg parsing, and result formatting.
module Train

import Data.IORef
import Data.List
import Data.Nat
import System.Clock

import Util
import Tensor
import Schedule


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

||| Show a Double with `digits` fixed decimal places (rounded half-up).
||| Mirrors Python's `f"{x:.<digits>f}"` so paired-side logs diff cleanly.
||| Handles negatives, NaN, and ±infinity.
export
showFix : (digits : Nat) -> Double -> String
showFix d x =
  if x /= x then "nan"
  else if x == 1.0/0.0 then "inf"
  else if x == -1.0/0.0 then "-inf"
  else
    let sign     : String  = if x < 0 then "-" else ""
        absX     : Double  = if x < 0 then -x else x
        scaleD   : Double  = pow 10.0 (cast {to=Double} (cast {to=Integer} d))
        scaledI  : Integer = cast {to=Integer} (absX * scaleD + 0.5)
        scaleI   : Integer = cast {to=Integer} scaleD
        intPart  : Integer = scaledI `div` scaleI
        fracPart : Integer = scaledI `mod` scaleI
        fracStr  : String  = padZeros d (show fracPart)
    in if d == 0
         then sign ++ show intPart
         else sign ++ show intPart ++ "." ++ fracStr
  where
    padZeros : Nat -> String -> String
    padZeros n s =
      if length s >= n
        then s
        else pack (List.replicate (n `minus` length s) '0') ++ s


----------------------------------------------------------------------
-- Early Stopping Strategies
----------------------------------------------------------------------

||| Early stopping configuration.
public export
data EarlyStopConfig
  = NoEarlyStop
  | Patience Nat Double              -- patience, minDelta
  | WindowedAvg Double Nat Nat       -- threshold, window, patience
  | WindowedPercentile Double Double Nat Nat
    -- percentile (0.0–1.0), threshold, window, patience.
    -- Splits `window` epochs into 100-epoch chunks (same as WindowedAvg),
    -- sorts the chunk-means, picks the chunk-mean at index
    -- `floor(percentile * num_chunks)`, and compares to `threshold`. With
    -- bimodal losses (variable-length-sequence tasks), picking p10 is the
    -- "best 100-epoch chunk in the window" — fires reliably once the model
    -- converges on at least the easier sequences.


----------------------------------------------------------------------
-- Training Configuration
----------------------------------------------------------------------

||| Extra metrics to log at each logging step (e.g. accuracy, memory).
public export
0 MetricsFn : Type -> Type
MetricsFn model = model -> IO (List (String, String))

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

||| Simple config: run N epochs, log every 100, no early stopping.
export
simpleConfig : Nat -> TrainConfig model
simpleConfig n = MkTrainConfig n 100 NoEarlyStop (const (pure [])) (\_ => pure ())

||| Config with patience-based early stopping.
export
patienceConfig : Nat -> Nat -> TrainConfig model
patienceConfig epochs pat =
  MkTrainConfig epochs 100 (Patience pat 0.001) (const (pure [])) (\_ => pure ())

||| Config with windowed-average early stopping.
export
windowedConfig : Nat -> Double -> Nat -> Nat -> TrainConfig model
windowedConfig epochs threshold window pat =
  MkTrainConfig epochs 100 (WindowedAvg threshold window pat) (const (pure [])) (\_ => pure ())

||| Config with windowed-percentile early stopping. Robust to bimodal
||| loss distributions (e.g. variable-length-sequence tasks where short
||| sequences quickly hit near-zero loss while long ones plateau higher).
export
windowedPercentileConfig : Nat -> Double -> Double -> Nat -> Nat -> TrainConfig model
windowedPercentileConfig epochs pct threshold window pat =
  MkTrainConfig epochs 100 (WindowedPercentile pct threshold window pat) (const (pure [])) (\_ => pure ())

||| Bind a Schedule to a NativeOptimizer, producing a beforeEpoch hook.
||| Per epoch, sets the optimizer's base LR to `schedule epoch`. Plug into
||| `TrainConfig` via the `beforeEpoch` field:
|||   `let cfg = { beforeEpoch := applySchedule sched opt } (simpleConfig 1000)`
export
applySchedule : Schedule -> NativeOptimizer -> Nat -> IO ()
applySchedule sched opt ep = setLearningRate opt (sched ep)


----------------------------------------------------------------------
-- Training Runner
----------------------------------------------------------------------

||| Run training with an IO-based epoch function. Use this when the
||| per-epoch step needs IO (e.g. sampling a replay-buffer batch, running
||| a vectorised env rollout). For pure epochs, use `runTraining`.
export
runTrainingIO :
  {0 model : Type} -> {0 dp : Type} ->
  (epochFn : model -> dp -> IO (model, Double)) ->
  (dataSrc : IO dp) ->
  TrainConfig model ->
  model ->
  IO (model, Nat, Double)
runTrainingIO {model} epochFn dataSrc cfg model0 = do
  tStart <- clockTime Monotonic
  putStrLn $ "Training... [backend=" ++ backendName ++ "]"
  result@(m, epochsDone, loss) <- case cfg.earlyStop of
    NoEarlyStop => goSimple 0 model0 0.0 tStart
    Patience pat minD => goPatience 0 model0 (1.0/0.0) 0 tStart pat minD
    WindowedAvg thresh win pat => goWindowed 0 model0 0.0 0 [] 0 tStart thresh win pat
    WindowedPercentile pct thresh win pat =>
      goWindowedPercentile 0 model0 [] 0 0 tStart pct thresh win pat
  tEnd <- clockTime Monotonic
  putStrLn $ formatTimingSummary tStart tEnd epochsDone
  putStrLn $ "Peak RSS: " ++ show (getRssMB 0) ++ " MB"
          ++ "\tCurrent RSS: " ++ show (getCurrentRssMB 0) ++ " MB"
  profileReport
  pure result
  where
    shouldLog : Nat -> Bool
    shouldLog ep = case cfg.logEvery of
      Z => False
      S k => modNatNZ ep (S k) ItIsSucc == 0

    fmtMetrics : List (String, String) -> String
    fmtMetrics [] = ""
    fmtMetrics ((k, v) :: rest) = "\t" ++ k ++ "=" ++ v ++ fmtMetrics rest

    logEpoch : Clock Monotonic -> Nat -> Double -> model -> IO ()
    logEpoch t0 ep loss m = do
      now <- clockTime Monotonic
      extra <- cfg.metrics m
      let memSuffix = "\tpeak=" ++ show (getRssMB 0) ++ "MB"
                   ++ "\tcur=" ++ show (getCurrentRssMB 0) ++ "MB"
      putStrLn $ "  " ++ formatElapsed t0 now ++ " " ++ show ep
               ++ "\tloss=" ++ showFix 6 loss ++ memSuffix ++ fmtMetrics extra

    -- Simple: no early stopping
    goSimple : Nat -> model -> Double -> Clock Monotonic -> IO (model, Nat, Double)
    goSimple ep m lastLoss t0 =
      if ep >= cfg.totalEpochs then pure (m, ep, lastLoss)
      else do
        cfg.beforeEpoch ep
        d <- dataSrc
        (m', loss) <- epochFn m d
        when (shouldLog ep) $ logEpoch t0 ep loss m'
        if loss /= loss
          then do now <- clockTime Monotonic
                  putStrLn $ "  " ++ formatElapsed t0 now ++ " Diverged (NaN) at epoch " ++ show ep
                  pure (m', ep, loss)
          else goSimple (S ep) m' loss t0

    diverged : Clock Monotonic -> Nat -> model -> Double -> IO (model, Nat, Double)
    diverged t0 ep m loss = do
      now <- clockTime Monotonic
      putStrLn $ "  " ++ formatElapsed t0 now ++ " Diverged (NaN) at epoch " ++ show ep
      pure (m, ep, loss)

    -- Patience-based early stopping
    goPatience : Nat -> model -> Double -> Nat -> Clock Monotonic -> Nat -> Double ->
                 IO (model, Nat, Double)
    goPatience ep m bestLoss stale t0 pat minD =
      if ep >= cfg.totalEpochs then pure (m, ep, bestLoss)
      else do
        cfg.beforeEpoch ep
        d <- dataSrc
        (m', loss) <- epochFn m d
        when (shouldLog ep) $ logEpoch t0 ep loss m'
        if loss /= loss
          then diverged t0 ep m' loss
          else do
            let improved = loss < bestLoss - minD
                best' = if improved then loss else bestLoss
                stale' : Nat
                stale' = if improved then 0 else stale + 1
            if pat > 0 && stale' >= pat
              then do now <- clockTime Monotonic
                      putStrLn $ "  " ++ formatElapsed t0 now ++ " Early stop at epoch "
                               ++ show (ep + 1) ++ " (patience=" ++ show pat ++ ")"
                      pure (m', ep + 1, loss)
              else goPatience (S ep) m' best' stale' t0 pat minD

    -- Windowed-average early stopping
    goWindowed : Nat -> model -> Double -> Nat -> List Double -> Nat ->
                 Clock Monotonic -> Double -> Nat -> Nat ->
                 IO (model, Nat, Double)
    goWindowed ep m iSum iCount avgs convCount t0 thresh win pat =
      if ep >= cfg.totalEpochs then pure (m, ep, 0.0)
      else do
        cfg.beforeEpoch ep
        d <- dataSrc
        (m', loss) <- epochFn m d
        when (shouldLog ep) $ logEpoch t0 ep loss m'
        if loss /= loss
          then diverged t0 ep m' loss
          else let iSum' = iSum + loss
                   iCount' = iCount + 1
               in if iCount' < 100
                 then goWindowed (S ep) m' iSum' iCount' avgs convCount t0 thresh win pat
                 else let avg = iSum' / 100.0
                          avgs' = avg :: avgs
                          wc = max 1 (div win 100)
                      in if length avgs' < wc
                        then goWindowed (S ep) m' 0.0 0 avgs' convCount t0 thresh win pat
                        else let windowAvg = foldl (+) 0.0 (take wc avgs') / cast wc
                             in if windowAvg >= thresh
                               then goWindowed (S ep) m' 0.0 0 avgs' 0 t0 thresh win pat
                               else let cc = convCount + 1
                                    in if cc >= pat
                                      then do now <- clockTime Monotonic
                                              putStrLn $ "  " ++ formatElapsed t0 now
                                                       ++ " Converged at epoch " ++ show (ep + 1)
                                                       ++ " (window_avg=" ++ show windowAvg ++ ")"
                                              pure (m', ep + 1, loss)
                                      else do now <- clockTime Monotonic
                                              putStrLn $ "    " ++ formatElapsed t0 now
                                                       ++ " convergence " ++ show cc ++ "/" ++ show pat
                                                       ++ " (window_avg=" ++ show windowAvg ++ ")"
                                              goWindowed (S ep) m' 0.0 0 avgs' cc t0 thresh win pat

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
    goWindowedPercentile : Nat -> model -> List Double -> Nat -> Nat ->
                           Clock Monotonic -> Double -> Double -> Nat -> Nat ->
                           IO (model, Nat, Double)
    goWindowedPercentile ep m recent epochsSinceCheck convCount t0 pct thresh win pat =
      if ep >= cfg.totalEpochs then pure (m, ep, 0.0)
      else do
        cfg.beforeEpoch ep
        d <- dataSrc
        (m', loss) <- epochFn m d
        when (shouldLog ep) $ logEpoch t0 ep loss m'
        if loss /= loss
          then diverged t0 ep m' loss
          else let recent' = take win (loss :: recent)
                   esc' = epochsSinceCheck + 1
               in if esc' < 100 || length recent' < win
                 then goWindowedPercentile (S ep) m' recent' esc' convCount
                                            t0 pct thresh win pat
                 else let sorted = sort recent'
                          idx = min (minus win 1)
                                    (cast {to=Nat} (the Integer
                                           (cast (pct * cast win))))
                          pctVal = case drop idx sorted of
                                     (x :: _) => x
                                     [] => 0.0
                      in if pctVal >= thresh
                        then goWindowedPercentile (S ep) m' recent' 0 0
                                                   t0 pct thresh win pat
                        else let cc = convCount + 1
                             in if cc >= pat
                               then do now <- clockTime Monotonic
                                       putStrLn $ "  " ++ formatElapsed t0 now
                                                ++ " Converged at epoch " ++ show (ep + 1)
                                                ++ " (p" ++ show (cast {to=Int} (pct * 100.0))
                                                ++ "_loss=" ++ show pctVal ++ ")"
                                       pure (m', ep + 1, loss)
                               else do now <- clockTime Monotonic
                                       putStrLn $ "    " ++ formatElapsed t0 now
                                                ++ " convergence " ++ show cc ++ "/" ++ show pat
                                                ++ " (p" ++ show (cast {to=Int} (pct * 100.0))
                                                ++ "_loss=" ++ show pctVal ++ ")"
                                       goWindowedPercentile (S ep) m' recent' 0 cc
                                                             t0 pct thresh win pat


||| Run training with an IO-typed epoch function. After the
||| smart-constructor IO refactor, epochVar / epochVarTensor etc. all
||| return `IO (model, Double)`, so this is now a thin alias for
||| `runTrainingIO`.
export
runTraining :
  {0 model : Type} -> {0 dp : Type} ->
  (epochFn : model -> dp -> IO (model, Double)) ->
  (dataSrc : IO dp) ->
  TrainConfig model ->
  model ->
  IO (model, Nat, Double)
runTraining = runTrainingIO
