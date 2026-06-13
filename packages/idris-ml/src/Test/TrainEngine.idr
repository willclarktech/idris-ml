module Test.TrainEngine

import Data.Vect

import Test.Harness
import Executor
import Optimizer
import Tensor
import Train.Engine
import Test.Config


-- Registered scalar param at the test backend/dtype (mirrors Test.Optimizer).
mkW : String -> Double -> IO (Tensor [] TestExecutor TestDType WithGrad)
mkW name v = do
  wptr <- ioRerun (\_ => primCreateScalar {ex=TestExecutor} v 1)
  _ <- ioRerun (\_ => primParamRegister {ex=TestExecutor} name wptr)
  pure (MkTensor wptr (Just name))

-- A hand-rolled 3-epoch loop composing the EXPORTED engine pieces
-- (`withEpoch` per-epoch bracket) with `nativeTrainStep` — the shape a
-- custom/RL loop reuses instead of reimplementing. loss = w*w from
-- w=1.0 at lr=0.1: each step w *= 0.8, so after 3 epochs w = 0.512.
handLoopConverges : IO Bool
handLoopConverges = do
  w <- mkW "te_hand_w" 1.0
  opt <- sgd {ex=TestExecutor} 0.1 defaultOpts
  let oneEpoch : IO Double
      oneEpoch = withEpoch {ex=TestExecutor} $ do
        loss <- tmul w w
        nativeTrainStep {ex=TestExecutor} {dt=TestDType} opt loss
  _ <- oneEpoch
  _ <- oneEpoch
  _ <- oneEpoch
  let v = tensorItem w
  check ("withEpoch+nativeTrainStep hand loop converges (w=" ++ show v
         ++ ", expect 0.512)") (abs (v - 0.512) < 1.0e-9)

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

export
tests : List (IO Bool)
tests = [ handLoopConverges, isDivergedDetectsNaN, shouldLogCadence
        , divisibleByCases, showFixRounds ]
