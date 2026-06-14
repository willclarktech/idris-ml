||| Shared training-loop engine — the composable pieces behind both the
||| legacy `runTraining`/`runTrainingIO` (Train.idr) and the v1 `fit`
||| driver (Fit.idr). Extracting them here means one implementation of
||| the subtle parts (mlx generation hygiene, the withNoGrad eval
||| bracket, NaN handling, checkpoint resume/keep-best/return-best) that
||| RL/custom loops can also compose directly instead of reimplementing.
|||
||| Everything here is generic in the model type `m` (fully opaque — no
||| `Params` constraint; the engine never traverses the model) and the
||| per-epoch result is reduced to a `Double` loss before it reaches the
||| loop, so the engine touches tensors only inside `logEpoch`'s eval
||| bracket.
module Train.Engine

import Data.IORef
import Data.List
import Data.Nat
import System.Clock

import Checkpoint
import Executor
import Tensor
import Util
import Util.Log

----------------------------------------------------------------------
-- Shared types (were in Train.idr; re-exported there for source compat)
----------------------------------------------------------------------

||| Extra metrics to log at each logging step (e.g. accuracy, memory).
public export
0 MetricsFn : Type -> Type
MetricsFn model = model -> IO (List (String, String))

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
-- Formatting (shared with examples via Train re-export)
----------------------------------------------------------------------

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
-- Per-epoch generation bracket
----------------------------------------------------------------------

||| Whether to log at this epoch given the `logEvery` cadence.
export
shouldLog : (logEvery : Nat) -> (epoch : Nat) -> Bool
shouldLog Z     _  = False
shouldLog (S k) ep = modNatNZ ep (S k) ItIsSucc == 0

||| `n` divisible by `m` (m=0 → False). Used for periodic checkpointing.
export
divisibleBy : Nat -> Nat -> Bool
divisibleBy _ Z     = False
divisibleBy n (S k) = modNatNZ n (S k) ItIsSucc == 0

||| Per-epoch generation-scoped free: `primEpochBegin` marks the tensor
||| generation, run `act`, then (mlx only) force a major GC + drain the
||| managed-handle guardian *before* the sweep so the epoch's dead grad
||| husks are reclaimed, and `primEpochEnd` sweeps. No-op begin/end on
||| tape/torch (no buffer ceiling), so the mlx-gated GC is skipped there.
|||
||| Distinct from `Tensor.withGenFree`, which omits the major-GC+drain —
||| reusing that here would leak mlx grad husks across epochs (the
||| long-grad-mode handle leak). Keep this the per-epoch frame.
export
withEpoch : {0 ex : Executor} -> UserExecutorTraining ex => UserExecutorTransfer ex => IO a -> IO a
withEpoch act = do
  primIO (primEpochBegin {ex})
  result <- act
  when (backendTag {ex} == "mlx") $ do
    forceMajorGc
    _ <- drainManagedHandles
    pure ()
  primIO (primEpochEnd {ex})
  pure result

----------------------------------------------------------------------
-- Eval-bracket metrics logging
----------------------------------------------------------------------

fmtMetrics : List (String, String) -> String
fmtMetrics []               = ""
fmtMetrics ((k, v) :: rest) = "\t" ++ k ++ "=" ++ v ++ fmtMetrics rest

-- Force every character of the metric strings. The strings are built
-- lazily from tensor reads, so this drags those reads to happen *now* —
-- inside the withNoGrad bracket below, before its exit drain frees the
-- eval tensors. Without it the thunks dangle on mlx. Always False.
forceMetrics : List (String, String) -> Bool
forceMetrics xs = foldl (\acc, (k, v) => acc + length k + length v) 0 xs < 0

||| Log one epoch line: `[hh:mm:ss] EP\tloss\tpeak\tcur\thandles…\t<metrics>`.
||| Runs `metrics` with autograd off (builds no tape) and forces the
||| result strings inside the bracket — a grad-mode metrics pass would
||| leave a live tape whose tensors the epoch-end sweep then frees,
||| crashing the next epoch on mlx (use-after-free).
export
logEpoch : {0 ex : Executor} -> UserExecutorTraining ex => {0 m : Type} ->
           MetricsFn m -> Clock Monotonic -> (epoch : Nat) -> (loss : Double) -> m -> IO ()
logEpoch metrics t0 ep loss m = do
  now <- clockTime Monotonic
  extra <- withNoGrad {ex} $ do
             e <- metrics m
             if forceMetrics e then pure e else pure e
  liveH <- primIO (primLiveCount {ex})
  peakH <- primIO (primPeakLiveCount {ex})
  let memSuffix = "\tpeak=" ++ show (getRssMB 0) ++ "MB"
               ++ "\tcur=" ++ show (getCurrentRssMB 0) ++ "MB"
               ++ "\thandles=" ++ show liveH
               ++ "\tpeakhandles=" ++ show peakH
  logInfo $ "  " ++ formatElapsed t0 now ++ " " ++ show ep
           ++ "\tloss=" ++ showFix 6 loss ++ memSuffix ++ fmtMetrics extra

----------------------------------------------------------------------
-- NaN handling
----------------------------------------------------------------------

||| IEEE-754 NaN test (true only for NaN). In single precision a NaN
||| epoch loss means divergence; in mixed precision it means the scaler
||| detected overflow and skipped the step (not divergence).
export
isDiverged : Double -> Bool
isDiverged x = x /= x

||| Halt on divergence: log a warning, return the model + epoch + loss.
export
diverged : {0 m : Type} -> Clock Monotonic -> Nat -> m -> Double -> IO (m, Nat, Double)
diverged t0 ep m loss = do
  now <- clockTime Monotonic
  logWarn $ "  " ++ formatElapsed t0 now ++ " Diverged (NaN) at epoch " ++ show ep
  pure (m, ep, loss)

----------------------------------------------------------------------
-- Checkpoint resume / keep-best / return-best
----------------------------------------------------------------------

||| Resume from `<dir>/last` if a checkpoint policy is attached: seed
||| `bestRef` with the saved best metric and return the resume epoch
||| offset (0 for a fresh start).
export
resumeFromCheckpoint : Maybe CheckpointPolicy -> IORef Double -> IO Nat
resumeFromCheckpoint Nothing    _       = pure 0
resumeFromCheckpoint (Just pol) bestRef = do
  mst <- pol.loadState (pol.dir ++ "/last")
  case mst of
    Nothing           => pure 0
    Just (ep0, best0) => do
      writeIORef bestRef best0
      logInfo $ "  Resuming from epoch " ++ show ep0
               ++ " (best=" ++ showFix 6 best0 ++ ")"
      pure ep0

||| After a completed (non-NaN) epoch: keep-best save to `<dir>/best`
||| when the monitored scalar improves (lower better), then periodic
||| save to `<dir>/last` every `everyN`. The monitor eval is bracketed;
||| the saves read registry params (rc>1, sweep-safe).
export
postEpoch : {0 ex : Executor} -> UserExecutorTraining ex =>
            Maybe CheckpointPolicy -> IORef Double -> (epoch : Nat) -> (loss : Double) -> IO ()
postEpoch Nothing    _       _  _    = pure ()
postEpoch (Just pol) bestRef ep loss = do
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

||| Return-best: reload `<dir>/best` so the returned model is the best
||| seen, not the last (Lightning semantics). `loadState` mutates the
||| registry in place; the model structure is unchanged, so the caller
||| keeps its model value.
export
returnBestCheckpoint : Maybe CheckpointPolicy -> IO ()
returnBestCheckpoint Nothing    = pure ()
returnBestCheckpoint (Just pol) =
  when pol.keepBest $ ignore $ pol.loadState (pol.dir ++ "/best")

----------------------------------------------------------------------
-- Unified epoch loop + pluggable early-stop state machines
----------------------------------------------------------------------

||| One early-stop decision: keep iterating with updated state, or halt
||| now (the loop then returns the current model + `S epoch` + loss).
public export
data EsDecision : Type -> Type where
  EsKeep : s -> EsDecision s
  EsHalt : EsDecision s

||| Pluggable early-stop step: given (t0, epoch, loss, state), update the
||| state or halt. Does its own convergence/early-stop logging.
public export
0 EarlyStopStep : Type -> Type
EarlyStopStep s = Clock Monotonic -> (epoch : Nat) -> (loss : Double) -> s -> IO (EsDecision s)

-- NoEarlyStop: never halts; terminal loss is the last epoch's loss.
esNone : EarlyStopStep ()
esNone _ _ _ _ = pure (EsKeep ())

-- Patience: state = (bestLoss, stale). Halts after `pat` consecutive
-- non-improving epochs (improvement = loss < bestLoss - minDelta).
esPatience : (pat : Nat) -> (minD : Double) -> EarlyStopStep (Double, Nat)
esPatience pat minD t0 ep loss (bestLoss, stale) =
  let improved = loss < bestLoss - minD
      best'  = if improved then loss else bestLoss
      stale' = if improved then 0 else stale + 1
  in if pat > 0 && stale' >= pat
       then do now <- clockTime Monotonic
               logInfo $ "  " ++ formatElapsed t0 now ++ " Early stop at epoch "
                        ++ show (ep + 1) ++ " (patience=" ++ show pat ++ ")"
               pure EsHalt
       else pure (EsKeep (best', stale'))

-- Windowed-average: state = (iSum, iCount, avgs, convCount). Splits into
-- 100-epoch chunks, keeps the last `win/100` chunk-means, halts when
-- their average stays below `thresh` for `pat` consecutive checks.
esWindowedAvg : (thresh : Double) -> (win : Nat) -> (pat : Nat) ->
                EarlyStopStep (Double, Nat, List Double, Nat)
esWindowedAvg thresh win pat t0 ep loss (iSum, iCount, avgs, convCount) =
  let iSum'   = iSum + loss
      iCount' = iCount + 1
  in if iCount' < 100
       then pure (EsKeep (iSum', iCount', avgs, convCount))
       else let avg   = iSum' / 100.0
                avgs' = avg :: avgs
                wc    = max 1 (div win 100)
            in if length avgs' < wc
                 then pure (EsKeep (0.0, 0, avgs', convCount))
                 else let windowAvg = foldl (+) 0.0 (take wc avgs') / cast wc
                      in if windowAvg >= thresh
                           then pure (EsKeep (0.0, 0, avgs', 0))
                           else let cc = convCount + 1
                                in if cc >= pat
                                     then do now <- clockTime Monotonic
                                             logInfo $ "  " ++ formatElapsed t0 now
                                                      ++ " Converged at epoch " ++ show (ep + 1)
                                                      ++ " (window_avg=" ++ show windowAvg ++ ")"
                                             pure EsHalt
                                     else do now <- clockTime Monotonic
                                             logInfo $ "    " ++ formatElapsed t0 now
                                                      ++ " convergence " ++ show cc ++ "/" ++ show pat
                                                      ++ " (window_avg=" ++ show windowAvg ++ ")"
                                             pure (EsKeep (0.0, 0, avgs', cc))

-- Windowed-percentile: state = (recent, epochsSinceCheck, convCount).
-- Maintains the last `win` raw per-epoch losses; every 100 epochs sorts
-- them, takes the `pct` percentile, halts when it stays below `thresh`
-- for `pat` consecutive checks. (Raw losses, not chunk-means — robust
-- to bimodal variable-length-sequence losses.)
esWindowedPct : (pct : Double) -> (thresh : Double) -> (win : Nat) -> (pat : Nat) ->
                EarlyStopStep (List Double, Nat, Nat)
esWindowedPct pct thresh win pat t0 ep loss (recent, esc, convCount) =
  let recent' = take win (loss :: recent)
      esc'    = esc + 1
  in if esc' < 100 || length recent' < win
       then pure (EsKeep (recent', esc', convCount))
       else let sorted = sort recent'
                idx    = min (minus win 1)
                             (cast {to=Nat} (the Integer (cast (pct * cast win))))
                pctVal = case drop idx sorted of
                           (x :: _) => x
                           []       => 0.0
            in if pctVal >= thresh
                 then pure (EsKeep (recent', 0, 0))
                 else let cc = convCount + 1
                      in if cc >= pat
                           then do now <- clockTime Monotonic
                                   logInfo $ "  " ++ formatElapsed t0 now
                                            ++ " Converged at epoch " ++ show (ep + 1)
                                            ++ " (p" ++ show (cast {to=Int} (pct * 100.0))
                                            ++ "_loss=" ++ show pctVal ++ ")"
                                   pure EsHalt
                           else do now <- clockTime Monotonic
                                   logInfo $ "    " ++ formatElapsed t0 now
                                            ++ " convergence " ++ show cc ++ "/" ++ show pat
                                            ++ " (p" ++ show (cast {to=Int} (pct * 100.0))
                                            ++ "_loss=" ++ show pctVal ++ ")"
                                   pure (EsKeep (recent', 0, cc))

||| Dispatch an `EarlyStopConfig` to its (step, initial-state,
||| terminal-loss) triple. `terminal st lastLoss` is the loss returned
||| when training reaches `totalEpochs` without early-stopping:
||| NoEarlyStop yields the last epoch's loss, Patience the best seen,
||| the windowed strategies 0.0 (matching the legacy loops).
export
earlyStopMachine : EarlyStopConfig ->
                   (s ** (EarlyStopStep s, s, s -> Double -> Double))
earlyStopMachine NoEarlyStop =
  (() ** (esNone, (), \_, last => last))
earlyStopMachine (Patience pat minD) =
  ((Double, Nat) ** (esPatience pat minD, (1.0/0.0, 0), \(best, _), _ => best))
earlyStopMachine (WindowedAvg thresh win pat) =
  ((Double, Nat, List Double, Nat) **
   (esWindowedAvg thresh win pat, (0.0, 0, [], 0), \_, _ => 0.0))
earlyStopMachine (WindowedPercentile pct thresh win pat) =
  ((List Double, Nat, Nat) **
   (esWindowedPct pct thresh win pat, ([], 0, 0), \_, _ => 0.0))

-- The recursive driver. Top-level (not a where-clause) so its single
-- interface-constraint binding is unambiguous. Loop-invariant params
-- are threaded each call; (st, lastLoss, ep, m) vary.
epochLoopGo :
  {0 ex : Executor} -> UserExecutorTraining ex => UserExecutorTransfer ex =>
  {0 m : Type} -> {0 s : Type} ->
  (totalEpochs : Nat) -> (logEvery : Nat) -> MetricsFn m ->
  Maybe CheckpointPolicy -> IORef Double -> (nanIsDivergence : Bool) ->
  EarlyStopStep s -> (esTerminal : s -> Double -> Double) ->
  (perEpoch : m -> Nat -> IO (m, Double)) -> Clock Monotonic ->
  s -> (lastLoss : Double) -> Nat -> m -> IO (m, Nat, Double)
epochLoopGo totalEpochs logEvery metrics checkpoint bestRef nanIsDivergence
            esStep esTerminal perEpoch t0 st lastLoss ep m =
  if ep >= totalEpochs then pure (m, ep, esTerminal st lastLoss)
  else do
    (m', loss) <- withEpoch {ex} $ do
      (m', loss) <- perEpoch m ep
      when (shouldLog logEvery ep) $ logEpoch {ex} metrics t0 ep loss m'
      pure (m', loss)
    if nanIsDivergence && isDiverged loss
      then diverged t0 ep m' loss
      else do
        postEpoch {ex} checkpoint bestRef ep loss
        dec <- esStep t0 ep loss st
        case dec of
          EsHalt     => pure (m', S ep, loss)
          EsKeep st' => epochLoopGo {ex} totalEpochs logEvery metrics checkpoint bestRef
                                    nanIsDivergence esStep esTerminal perEpoch t0
                                    st' loss (S ep) m'

||| The unified epoch loop driving both `runTrainingIO` and `fit`.
||| Owns: the `totalEpochs` terminator, the `withEpoch` generation
||| bracket around `perEpoch` + conditional `logEpoch`, the NaN branch
||| (`nanIsDivergence` True = single precision halts on NaN; False =
||| mixed precision treats NaN as an overflow-skip and continues),
||| `postEpoch` checkpointing, and the early-stop state machine. The
||| caller supplies `perEpoch` (its own beforeEpoch/tick + data pull +
||| step) and the early-stop triple (from `earlyStopMachine`).
export
runEpochLoop :
  {0 ex : Executor} -> UserExecutorTraining ex => UserExecutorTransfer ex =>
  {0 m : Type} -> {0 s : Type} ->
  (totalEpochs : Nat) -> (logEvery : Nat) -> MetricsFn m ->
  Maybe CheckpointPolicy -> (bestRef : IORef Double) -> (nanIsDivergence : Bool) ->
  EarlyStopStep s -> (esInit : s) -> (esTerminal : s -> Double -> Double) ->
  (perEpoch : m -> Nat -> IO (m, Double)) ->
  Clock Monotonic -> (startEp : Nat) -> (m0 : m) ->
  IO (m, Nat, Double)
runEpochLoop totalEpochs logEvery metrics checkpoint bestRef nanIsDivergence
             esStep esInit esTerminal perEpoch t0 startEp m0 =
  epochLoopGo {ex} totalEpochs logEvery metrics checkpoint bestRef nanIsDivergence
              esStep esTerminal perEpoch t0 esInit 0.0 startEp m0
