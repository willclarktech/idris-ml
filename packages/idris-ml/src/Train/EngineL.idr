||| The linear-resource (`L IO`) epoch loop — the migration counterpart of
||| `Train.Engine`'s IO loop. Lives in its own module so the linear imports
||| (`Control.Linear.LIO` / `Data.Linear`) stay clear of `Train.Engine`'s
||| delicate `Nat` arithmetic, whose bare numeric literals re-default to
||| `Integer` once those modules are in scope (an Idris elaboration-order
||| fragility). It reuses every model-agnostic piece from `Train.Engine`
||| (early-stop machines, checkpoint resume/keep-best, NaN test, `shouldLog`)
||| and only re-expresses the two pieces that thread the model: the
||| generation bracket and the recursive epoch driver.
|||
||| The model `m` is a single-owner **linear** resource threaded through
||| `perEpoch` and each recursive call; it is never handed to metrics
||| (`MetricsFnL` is model-free — every real metrics callback ignores the
||| model anyway, reading C-registry / IORef state). This is what makes
||| "freeze/eval a model then reuse the stale handle" a compile-time error.
||| Coexists with `Train.Engine`; the IO loop is deleted when every caller
||| is on `L IO`.
module Train.EngineL

import Control.Linear.LIO
import Data.IORef
import Data.Linear
import System.Clock

import Checkpoint
import Executor
import Tensor
import Train.Engine
import Util
import Util.Log

||| Model-free metrics for the linear loop. The IO `MetricsFn` takes the
||| model, but **every** caller ignores it (`\_ => readRLMetrics …`, default
||| `const (pure [])`) — metrics read C-registry / IORef state, never the
||| model value. Dropping the model argument lets the loop thread the
||| (linear, single-owner) model through every step without ever handing it
||| to metrics (which it couldn't, short of an interface reflect, since the
||| engine is generic in `m`). Resolves risk #3 of the linear-types
||| migration (metrics peeking at the linear `m`).
public export
0 MetricsFnL : Type
MetricsFnL = IO (List (String, String))

||| Per-epoch generation bracket for the linear loop: the `L IO` analogue of
||| `Train.Engine.withEpoch`. Brackets `primEpochBegin`/`primEpochEnd` (+ the
||| mlx GC+drain) around a linear `act` that threads the model — `result` is
||| bound linearly (`act` is `use=1`) and returned once via `pure1`; the
||| begin/end/drain are ordinary `liftIO1` effects that don't touch it.
export
withEpochL : {0 ex : Executor} -> UserExecutorTraining ex => UserExecutorTransfer ex =>
             {0 a : Type} -> (1 act : L IO {use = 1} a) -> L IO {use = 1} a
withEpochL act = do
  liftIO1 (primIO (primEpochBegin {ex}))
  result <- act
  liftIO1 (when (backendTag {ex} == "mlx") $ do
             forceMajorGc
             _ <- drainManagedHandles
             pure ())
  liftIO1 (primIO (primEpochEnd {ex}))
  pure1 result

||| `logEpoch` for the linear loop: model-free (`MetricsFnL`), so it never
||| consumes the threaded model. Body mirrors `Train.Engine.logEpoch` minus
||| the `m` argument; the metrics pass still runs with autograd off. Plain
||| `IO` (no linear resources), lifted by the caller via `liftIO1`.
export
logEpochL : {0 ex : Executor} -> UserExecutorTraining ex =>
            MetricsFnL -> Clock Monotonic -> (epoch : Nat) -> (loss : Double) -> IO ()
logEpochL metrics t0 ep loss = do
  now <- clockTime Monotonic
  extra <- withNoGrad {ex} $ do
             e <- metrics
             if forceMetrics e then pure e else pure e
  liveH <- primIO (primLiveCount {ex})
  peakH <- primIO (primPeakLiveCount {ex})
  let memSuffix = "\tpeak=" ++ show (getRssMB 0) ++ "MB"
               ++ "\tcur=" ++ show (getCurrentRssMB 0) ++ "MB"
               ++ "\thandles=" ++ show liveH
               ++ "\tpeakhandles=" ++ show peakH
  logInfo $ "  " ++ formatElapsed t0 now ++ " " ++ show ep
           ++ "\tloss=" ++ showFix 6 loss ++ memSuffix ++ fmtMetrics extra

||| Halt-on-divergence for the linear loop: log the warning, thread the
||| (linear) model out beside the banged `(epoch, loss)`. The `L IO`
||| analogue of `Train.Engine.diverged`.
export
divergedL : {0 m : Type} -> Clock Monotonic -> Nat -> (1 _ : m) -> Double ->
            L IO {use = 1} (LPair (!* (Nat, Double)) m)
divergedL t0 ep m loss = do
  liftIO1 $ do
    now <- clockTime Monotonic
    logWarn $ "  " ++ formatElapsed t0 now ++ " Diverged (NaN) at epoch " ++ show ep
  pure1 (MkBang (ep, loss) # m)

-- The linear recursive driver: the `L IO` analogue of `epochLoopGo`. The
-- model `m` is threaded through `perEpoch` and each recursive call; it is
-- never handed to metrics. The result bangs `(epochsDone, loss)` and carries
-- the model out linearly.
epochLoopGoL :
  {0 ex : Executor} -> UserExecutorTraining ex => UserExecutorTransfer ex =>
  {0 m : Type} -> {0 s : Type} ->
  (totalEpochs : Nat) -> (logEvery : Nat) -> MetricsFnL ->
  Maybe CheckpointPolicy -> IORef Double -> (nanIsDivergence : Bool) ->
  EarlyStopStep s -> (esTerminal : s -> Double -> Double) ->
  (perEpoch : (1 _ : m) -> Nat -> L IO {use = 1} (LPair (!* Double) m)) -> Clock Monotonic ->
  s -> (lastLoss : Double) -> Nat -> (1 _ : m) ->
  L IO {use = 1} (LPair (!* (Nat, Double)) m)
epochLoopGoL totalEpochs logEvery metrics checkpoint bestRef nanIsDivergence
             esStep esTerminal perEpoch t0 st lastLoss ep m =
  if ep >= totalEpochs then pure1 (MkBang (ep, esTerminal st lastLoss) # m)
  else do
    (MkBang loss # m') <- withEpochL {ex} $ do
      (MkBang loss # m') <- perEpoch m ep
      liftIO1 (when (shouldLog logEvery ep) $ logEpochL {ex} metrics t0 ep loss)
      pure1 (MkBang loss # m')
    if nanIsDivergence && isDiverged loss
      then divergedL t0 ep m' loss
      else do
        liftIO1 (postEpoch {ex} checkpoint bestRef ep loss)
        dec <- liftIO1 (esStep t0 ep loss st)
        case dec of
          EsHalt     => pure1 (MkBang (S ep, loss) # m')
          EsKeep st' => epochLoopGoL {ex} totalEpochs logEvery metrics checkpoint bestRef
                                     nanIsDivergence esStep esTerminal perEpoch t0
                                     st' loss (S ep) m'

||| The unified linear epoch loop — the `L IO` analogue of
||| `Train.Engine.runEpochLoop`, driving the linear `fit`. Same
||| responsibilities (terminator, generation bracket, NaN branch,
||| checkpointing, early stop) with the model threaded as a single-owner
||| linear resource end to end.
export
runEpochLoopL :
  {0 ex : Executor} -> UserExecutorTraining ex => UserExecutorTransfer ex =>
  {0 m : Type} -> {0 s : Type} ->
  (totalEpochs : Nat) -> (logEvery : Nat) -> MetricsFnL ->
  Maybe CheckpointPolicy -> (bestRef : IORef Double) -> (nanIsDivergence : Bool) ->
  EarlyStopStep s -> (esInit : s) -> (esTerminal : s -> Double -> Double) ->
  (perEpoch : (1 _ : m) -> Nat -> L IO {use = 1} (LPair (!* Double) m)) ->
  Clock Monotonic -> (startEp : Nat) -> (1 m0 : m) ->
  L IO {use = 1} (LPair (!* (Nat, Double)) m)
runEpochLoopL totalEpochs logEvery metrics checkpoint bestRef nanIsDivergence
              esStep esInit esTerminal perEpoch t0 startEp m0 =
  epochLoopGoL {ex} totalEpochs logEvery metrics checkpoint bestRef nanIsDivergence
               esStep esTerminal perEpoch t0 esInit 0.0 startEp m0
