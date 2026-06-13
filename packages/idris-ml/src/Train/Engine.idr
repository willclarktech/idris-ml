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

import Util
import Util.Log
import Executor
import Tensor
import Checkpoint

%default total


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
fmtMetrics [] = ""
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
    Nothing => pure 0
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
