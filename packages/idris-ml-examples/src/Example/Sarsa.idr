module Example.Sarsa

import Control.Linear.LIO
import Data.Fin
import Data.Linear.Notation
import Data.List
import Data.Maybe
import Data.Vect
import System

import Gym.Env
import Gym.ToyText.CliffWalking
import Ml.Array
import Ml.Compat.Random
import Ml.DataStream
import Ml.Executor
import Ml.Fit
import Ml.Math
import Ml.Train

import BuildConfig

----------------------------------------------------------------------
-- Env dimensions
----------------------------------------------------------------------

NumStates  : Nat; NumStates = 48
NumActions : Nat; NumActions = 4
MaxSteps   : Nat; MaxSteps = 100

----------------------------------------------------------------------
-- Q-table as a Array
----------------------------------------------------------------------

QTable : Type
QTable = Array [NumStates, NumActions] Double

zeroQ : QTable
zeroQ = replicate {dims = [NumStates, NumActions]} 0.0

qGet : Fin NumStates -> Fin NumActions -> QTable -> Double
qGet i j q =
  let SArray v = index j (index i q)
  in v

qRowAt : Fin NumStates -> QTable -> Vector NumActions Double
qRowAt i q = index i q

qSet : Fin NumStates -> Fin NumActions -> Double -> QTable -> QTable
qSet i j v (VArray rows) =
  let (VArray cells) = Data.Vect.index i rows
      newRow = VArray (Data.Vect.replaceAt j (SArray v) cells)
  in VArray (Data.Vect.replaceAt i newRow rows)

----------------------------------------------------------------------
-- Nat -> Fin conversions
----------------------------------------------------------------------

toStateFin : Nat -> Fin NumStates
toStateFin n = case natToFin n NumStates of
  Just f  => f
  Nothing => FZ

----------------------------------------------------------------------
-- Epsilon-greedy
----------------------------------------------------------------------

epsGreedy : Double -> Vector NumActions Double -> Double -> Double -> Fin NumActions
epsGreedy eps qr u1 u2 =
  if u1 < eps
    then let idx : Nat
             idx = integerToNat (cast (u2 * cast NumActions))
         in case natToFin idx NumActions of
              Just f  => f
              Nothing => FZ
    else argmax qr

----------------------------------------------------------------------
-- Episode rollout with SARSA updates
--
-- Key difference from Q-learning: the TD target uses the NEXT action
-- selected under the current epsilon-greedy policy (on-policy), not
-- max over next actions (off-policy).
----------------------------------------------------------------------

-- Inner loop: we carry the current (state, action) pair. Each iteration
-- consumes 2 uniforms (for picking NEXT action).
sarsaLoop : Double -> Double -> Double ->
            CWState -> Fin NumActions -> QTable ->
            Nat -> List Double -> (QTable, Double)
sarsaLoop _ _ _ _ _ q Z _                                        = (q, 0.0)
sarsaLoop _ _ _ _ _ q _ []                                       = (q, 0.0)
sarsaLoop _ _ _ _ _ q _ [_]                                      = (q, 0.0)
sarsaLoop alpha gamma eps st aFin q (S steps) (u1 :: u2 :: rest) =
  let sIdx = toStateFin (cwObserve st)
      aNat = finToNat aFin
  in case cwStep st aNat of
       (reward, st', outcome, _) =>
         let oldQ = qGet sIdx aFin q
         in if done outcome
              then let newQ = oldQ + alpha * (reward - oldQ)
                       q'   = qSet sIdx aFin newQ q
                   in (q', reward)
              else let sNextIdx = toStateFin (cwObserve st')
                       aNextFin  = epsGreedy eps (qRowAt sNextIdx q) u1 u2
                       target    = reward + gamma * qGet sNextIdx aNextFin q
                       newQ      = oldQ + alpha * (target - oldQ)
                       q'        = qSet sIdx aFin newQ q
                       (qF, fut) = sarsaLoop alpha gamma eps st' aNextFin q' steps rest
                   in (qF, reward + fut)

-- Outer wrapper: picks initial action (consumes first 2 uniforms) and
-- hands off to sarsaLoop.
runEpisode : Double -> Double -> Double ->
             CWState -> QTable -> Nat -> List Double -> (QTable, Double)
runEpisode _ _ _ _ q _ []                                = (q, 0.0)
runEpisode _ _ _ _ q _ [_]                               = (q, 0.0)
runEpisode alpha gamma eps st q steps (u1 :: u2 :: rest) =
  let sIdx = toStateFin (cwObserve st)
      a0   = epsGreedy eps (qRowAt sIdx q) u1 u2
  in sarsaLoop alpha gamma eps st a0 q steps rest

----------------------------------------------------------------------
-- Config & training
----------------------------------------------------------------------

record Config where
  constructor MkConfig
  alpha   : Double
  gamma   : Double
  epsilon : Double
  epochs  : Nat
  seed    : Bits64

defaultConfig : Config
defaultConfig = MkConfig 0.5 1.0 0.1 1000 42

specs : List (ArgSpec Config)
specs = [ Arg "--alpha" (\v, c => { alpha := cast v } c)
        , Arg "--gamma" (\v, c => { gamma := cast v } c)
        , Arg "--epsilon" (\v, c => { epsilon := cast v } c)
        , Arg "--epochs" (\v, c => { epochs := castNat v } c)
        , Arg "--seed" (\v, c => { seed := castBits64 v } c)
        ]

epochSarsa : Config -> QTable -> List Double -> (QTable, Double)
epochSarsa cfg q noise =
  let (q', ret) = runEpisode cfg.alpha cfg.gamma cfg.epsilon (MkCW 3 0) q MaxSteps noise
  in (q', negate ret)

genNoise : Nat -> IO (List Double)
genNoise Z     = pure []
genNoise (S k) = do
  u <- randomRIO (the Double 0.0, 1.0)
  rest <- genNoise k
  pure (u :: rest)

----------------------------------------------------------------------
-- Greedy evaluation
----------------------------------------------------------------------

evalEpisode : QTable -> CWState -> Nat -> Double -> Double
evalEpisode _ _ Z acc      = acc
evalEpisode q st (S k) acc =
  let sIdx = toStateFin (cwObserve st)
      aNat = finToNat (argmax (qRowAt sIdx q))
  in case cwStep st aNat of
       (reward, st', outcome, _) =>
         if done outcome then acc + reward
         else evalEpisode q st' k (acc + reward)

evalN : QTable -> Nat -> Double -> Double
evalN _ Z acc     = acc
evalN q (S k) acc =
  evalN q k (acc + evalEpisode q (MkCW 3 0) MaxSteps 0.0)

----------------------------------------------------------------------
-- Main
----------------------------------------------------------------------

main : IO ()
main = do
  requireMachine {m = ChosenMachine}
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)
  srand cfg.seed

  putStrLn "=== SARSA on CliffWalking ==="
  putStrLn $ "Config: alpha=" ++ show cfg.alpha
           ++ " gamma=" ++ show cfg.gamma
           ++ " epsilon=" ++ show cfg.epsilon
           ++ " epochs=" ++ show cfg.epochs
           ++ " seed=" ++ show cfg.seed
  putStrLn ""

  metrics <- newRLMetricsState 100
  Control.Linear.LIO.run $ do
    (MkBang (epochsDone, _) # (VArray trows)) <- fitCustom {ex=ExampleExecutor}
      (\(VArray rows), d => do
         let (m', loss) = epochSarsa cfg (VArray rows) d
         liftIO1 (recordReturn metrics (negate loss))
         pure1 (MkBang loss # m'))
      (generate (genNoise (MaxSteps * 2 + 2)))
      ({ metricsL := readRLMetrics "recent_100" metrics }
         (simpleConfig {model = QTable} cfg.epochs))
      zeroQ

    let nEval = the Nat 100
        totalReturn = evalN (VArray trows) nEval 0.0
        avgReturn   = totalReturn / cast (natToInteger nEval)
    liftIO1 $ do
      putStrLn ""
      putStrLn $ "Eval (100 episodes, greedy): avg_return=" ++ show avgReturn
      putStrLn ""
      putStrLn $ formatResult [("avg_return", show avgReturn),
                               ("epochs", show epochsDone),
                               ("seed", show cfg.seed)]
