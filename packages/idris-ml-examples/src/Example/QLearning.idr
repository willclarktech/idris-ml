module Example.QLearning

import Data.List
import Data.Vect
import Data.Fin
import Data.Maybe
import System
import Compat.Random

import Gym.Env
import Gym.ToyText.CliffWalking
import Math
import Array
import Train
import Executor
import BuildConfig


----------------------------------------------------------------------
-- Env dimensions
----------------------------------------------------------------------

NumStates : Nat; NumStates = 48
NumActions : Nat; NumActions = 4
MaxSteps : Nat; MaxSteps = 100


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
-- Nat -> Fin conversions (env guarantees in-range)
----------------------------------------------------------------------

toStateFin : Nat -> Fin NumStates
toStateFin n = case natToFin n NumStates of
  Just f => f
  Nothing => FZ

toActionFin : Nat -> Fin NumActions
toActionFin n = case natToFin n NumActions of
  Just f => f
  Nothing => FZ


----------------------------------------------------------------------
-- Epsilon-greedy + max over a row
----------------------------------------------------------------------

rowMax : Vector NumActions Double -> Double
rowMax qr =
  let SArray v = index (argmax qr) qr
  in v

epsGreedy : Double -> Vector NumActions Double -> Double -> Double -> Fin NumActions
epsGreedy eps qr u1 u2 =
  if u1 < eps
    then let idx : Nat
             idx = integerToNat (cast (u2 * cast NumActions))
         in case natToFin idx NumActions of
              Just f => f
              Nothing => FZ
    else argmax qr


----------------------------------------------------------------------
-- Episode rollout with Q-learning updates
----------------------------------------------------------------------

-- Consumes 2 uniforms per step; returns (updated Q, episodic return).
runEpisode : Double -> Double -> Double ->
             CWState -> QTable -> Nat -> List Double -> (QTable, Double)
runEpisode _ _ _ _ q Z _ = (q, 0.0)
runEpisode _ _ _ _ q _ [] = (q, 0.0)
runEpisode _ _ _ _ q _ [_] = (q, 0.0)
runEpisode alpha gamma eps st q (S steps) (u1 :: u2 :: rest) =
  let sIdx  = toStateFin (cwObserve st)
      qr    = qRowAt sIdx q
      aFin  = epsGreedy eps qr u1 u2
      aNat  = finToNat aFin
  in case cwStep st aNat of
       (reward, st', outcome, _) =>
         let sNextIdx = toStateFin (cwObserve st')
             bootstrap = if done outcome then 0.0 else gamma * rowMax (qRowAt sNextIdx q)
             oldQ  = qGet sIdx aFin q
             newQ  = oldQ + alpha * (reward + bootstrap - oldQ)
             q'    = qSet sIdx aFin newQ q
         in if done outcome
              then (q', reward)
              else let (qF, fut) = runEpisode alpha gamma eps st' q' steps rest
                   in (qF, reward + fut)


----------------------------------------------------------------------
-- Config & training loop
----------------------------------------------------------------------

record Config where
  constructor MkConfig
  alpha : Double
  gamma : Double
  epsilon : Double
  epochs : Nat
  seed : Bits64

defaultConfig : Config
defaultConfig = MkConfig 0.5 1.0 0.1 500 42

specs : List (ArgSpec Config)
specs = [ Arg "--alpha" (\v, c => { alpha := cast v } c)
        , Arg "--gamma" (\v, c => { gamma := cast v } c)
        , Arg "--epsilon" (\v, c => { epsilon := cast v } c)
        , Arg "--epochs" (\v, c => { epochs := castNat v } c)
        , Arg "--seed" (\v, c => { seed := castBits64 v } c)
        ]

epochQLearning : Config -> QTable -> List Double -> (QTable, Double)
epochQLearning cfg q noise =
  let (q', ret) = runEpisode cfg.alpha cfg.gamma cfg.epsilon (MkCW 3 0) q MaxSteps noise
  in (q', negate ret)

genNoise : Nat -> IO (List Double)
genNoise Z = pure []
genNoise (S k) = do
  u <- randomRIO (the Double 0.0, 1.0)
  rest <- genNoise k
  pure (u :: rest)


----------------------------------------------------------------------
-- Greedy evaluation
----------------------------------------------------------------------

evalEpisode : QTable -> CWState -> Nat -> Double -> Double
evalEpisode _ _ Z acc = acc
evalEpisode q st (S k) acc =
  let sIdx = toStateFin (cwObserve st)
      aNat = finToNat (argmax (qRowAt sIdx q))
  in case cwStep st aNat of
       (reward, st', outcome, _) =>
         if done outcome then acc + reward
         else evalEpisode q st' k (acc + reward)

evalN : QTable -> Nat -> Double -> Double
evalN _ Z acc = acc
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

  putStrLn "=== Q-learning on CliffWalking ==="
  putStrLn $ "Config: alpha=" ++ show cfg.alpha
           ++ " gamma=" ++ show cfg.gamma
           ++ " epsilon=" ++ show cfg.epsilon
           ++ " epochs=" ++ show cfg.epochs
           ++ " seed=" ++ show cfg.seed
  putStrLn ""

  metrics <- newRLMetricsState 100
  (trained, epochsDone, _) <- runTrainingIO {ex=ExampleExecutor}
    (\m, d => do
       let (m', loss) = epochQLearning cfg m d
       recordReturn metrics (negate loss)
       pure (m', loss))
    (genNoise (MaxSteps * 2))
    ({ metrics := \_ => readRLMetrics "recent_100" metrics }
       (simpleConfig {model = QTable} cfg.epochs))
    zeroQ

  putStrLn ""
  let nEval = the Nat 100
      totalReturn = evalN trained nEval 0.0
      avgReturn = totalReturn / cast (natToInteger nEval)
  putStrLn $ "Eval (100 episodes, greedy): avg_return=" ++ show avgReturn
  putStrLn ""
  putStrLn $ formatResult [("avg_return", show avgReturn),
                            ("epochs", show epochsDone),
                            ("seed", show cfg.seed)]
