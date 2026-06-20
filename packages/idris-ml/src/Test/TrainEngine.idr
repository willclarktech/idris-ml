module Test.TrainEngine

import Data.IORef
import Data.Vect
import System.Clock

import Checkpoint
import Executor
import Optimizer
import Tensor
import Test.Config
import Test.Harness
import Train
import Train.Engine

----------------------------------------------------------------------
-- Early-stop oracle: drives `runEpochLoop` directly with the early-stop
-- triple from `earlyStopMachine` (the same composition runTrainingIO
-- used before it was deleted). A scripted epochFn (model = Nat epoch
-- counter, loss = scriptLoss epoch) isolates the LOOP logic from the
-- numerics, so any drift in epoch-count / early-stop firing /
-- final-loss surfaces here. Golden values are loop-behaviour invariants.
----------------------------------------------------------------------

scriptRun : TrainConfig Nat -> (Nat -> Double) -> IO (Nat, Double)
scriptRun cfg scriptLoss = do
  bestRef <- newIORef (the Double (1.0 / 0.0))
  t0 <- clockTime Monotonic
  let (_ ** (esStep, esInit, esTerm)) = earlyStopMachine cfg.earlyStop
  let perEpoch : Nat -> Nat -> IO (Nat, Double)
      perEpoch n ep = do cfg.beforeEpoch ep; pure (S n, scriptLoss n)
  (_, epochsDone, finalLoss) <-
    runEpochLoop {ex=TestExecutor} cfg.totalEpochs cfg.logEvery cfg.metrics cfg.checkpoint
                 bestRef True esStep esInit esTerm perEpoch t0 0 0
  pure (epochsDone, finalLoss)

oracleNoEarlyStop : IO Bool
oracleNoEarlyStop = do
  (ed, fl) <- scriptRun (simpleConfig 5) (\i => 0.5 - 0.1 * cast i)
  check ("oracle NoEarlyStop -> (" ++ show ed ++ ", " ++ showFix 6 fl ++ ")")
        (ed == 5 && abs (fl - 0.1) < 1.0e-9)

oraclePatience : IO Bool
oraclePatience = do
  (ed, fl) <- scriptRun (patienceConfig 20 2) (\i => if i < 3 then 1.0 - 0.1 * cast i else 0.7)
  check ("oracle Patience -> (" ++ show ed ++ ", " ++ showFix 6 fl ++ ")")
        (ed == 6 && abs (fl - 0.7) < 1.0e-9)

oracleWindowedAvg : IO Bool
oracleWindowedAvg = do
  (ed, fl) <- scriptRun (windowedConfig 150 0.1 100 1) (\_ => 0.05)
  check ("oracle WindowedAvg -> (" ++ show ed ++ ", " ++ showFix 6 fl ++ ")")
        (ed == 100 && abs (fl - 0.05) < 1.0e-9)

oracleWindowedPct : IO Bool
oracleWindowedPct = do
  (ed, fl) <- scriptRun (windowedPercentileConfig 150 0.5 0.1 100 1) (\_ => 0.05)
  check ("oracle WindowedPercentile -> (" ++ show ed ++ ", " ++ showFix 6 fl ++ ")")
        (ed == 100 && abs (fl - 0.05) < 1.0e-9)

-- Registered scalar param at the test backend/dtype (mirrors Test.Optimizer).
mkW : String -> Double -> IO (Tensor [] TestExecutor TestDType WithGrad)
mkW name v = do
  wptr <- ioRerun (\_ => primCreateScalar {ex=TestExecutor} v 1)
  _ <- ioRerun (\_ => primParamRegister {ex=TestExecutor} name wptr)
  pure (MkTensor wptr (Just name))

-- A hand-rolled 3-epoch loop composing the EXPORTED engine pieces
-- (`withEpoch` per-epoch bracket) with `trainStep` — the shape a
-- custom/RL loop reuses instead of reimplementing. loss = w*w from
-- w=1.0 at lr=0.1: each step w *= 0.8, so after 3 epochs w = 0.512.
handLoopConverges : IO Bool
handLoopConverges = do
  w <- mkW "te_hand_w" 1.0
  opt <- sgd {ex=TestExecutor} 0.1 defaultOpts
  let oneEpoch : IO Double
      oneEpoch = withEpoch {ex=TestExecutor} $ do
        loss <- tmul w w
        trainStep {ex=TestExecutor} {dt=TestDType} opt loss
  _ <- oneEpoch
  _ <- oneEpoch
  _ <- oneEpoch
  let v = tensorItem w
  -- tolerance is F32-safe: mlx (F32) yields 0.51199996, ~3.5e-8 off the
  -- F64-exact 0.512. This runs through the real backend tensor (unlike the
  -- pure-Double oracle checks above), so it must clear the F32 ULP floor.
  check ("withEpoch+trainStep hand loop converges (w=" ++ show v
         ++ ", expect 0.512)") (abs (v - 0.512) < 1.0e-5)

isDivergedDetectsNaN : IO Bool
isDivergedDetectsNaN =
  check "isDiverged: NaN True, finite False"
        (isDiverged (0.0/0.0) && not (isDiverged 1.0) && not (isDiverged 0.0))

shouldLogCadence : IO Bool
shouldLogCadence =
  check "shouldLog: every-epoch (1), mod cadence (3), off (0)"
        (  shouldLog 1 5      -- logEvery 1 → always
        && shouldLog 3 6      -- 6 mod 3 == 0
        && not (shouldLog 3 7)
        && not (shouldLog 0 5)) -- logEvery 0 → never

divisibleByCases : IO Bool
divisibleByCases =
  check "divisibleBy: 6|3, not 7|3, n|0 = False"
        (divisibleBy 6 3 && not (divisibleBy 7 3) && not (divisibleBy 5 0))

showFixRounds : IO Bool
showFixRounds =
  check "showFix matches f-string rounding"
        (  showFix 6 0.512 == "0.512000"
        && showFix 0 3.7 == "4"          -- 3.7 + 0.5 → 4
        && showFix 1 0.25 == "0.3"       -- half-up: 2.5 → 3
        && showFix 6 (0.0/0.0) == "nan")

----------------------------------------------------------------------
-- RL-reuse: the exported engine pieces compose into a hand-rolled
-- threaded loop WITHOUT fit — the migration path for state-threading RL
-- (DQN's DqnState etc.). runEpochLoop threads the model (here a Nat
-- "episode count"); the perEpoch does its own trainStep; a custom
-- EarlyStopStep halts after 3 epochs. No example touched.
----------------------------------------------------------------------

rlReuseComposesEngine : IO Bool
rlReuseComposesEngine = do
  w <- mkW "te_rl_w" 1.0
  opt <- sgd {ex=TestExecutor} 0.1 defaultOpts
  bestRef <- newIORef (the Double (1.0/0.0))
  t0 <- clockTime Monotonic
  let perEpoch : Nat -> Nat -> IO (Nat, Double)
      perEpoch episodes _ = do
        loss <- tmul w w
        d <- trainStep opt loss
        pure (S episodes, d)            -- thread state: DqnState-style
  let esStep : EarlyStopStep Nat
      esStep _ _ _ n = pure (if n >= 2 then EsHalt else EsKeep (S n))
  (finalEpisodes, epochsRun, _) <-
    runEpochLoop {ex=TestExecutor} 100 0 (const (pure [])) Nothing bestRef True
                 esStep 0 (\_, _ => 0.0) perEpoch t0 0 0
  let v = tensorItem w
  check ("RL-reuse: hand-rolled threaded loop via runEpochLoop (episodes="
         ++ show finalEpisodes ++ ", epochs=" ++ show epochsRun ++ ", w=" ++ show v ++ ")")
        (finalEpisodes == epochsRun && epochsRun == 3 && v < 1.0)

export
tests : List (IO Bool)
tests = [ handLoopConverges, isDivergedDetectsNaN, shouldLogCadence
        , divisibleByCases, showFixRounds
        , oracleNoEarlyStop, oraclePatience, oracleWindowedAvg, oracleWindowedPct
        , rlReuseComposesEngine ]
