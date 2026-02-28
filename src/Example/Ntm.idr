module Example.Ntm

import Data.List
import Data.String
import Data.Vect
import System
import System.Random

import Backprop
import DataPoint
import Floating
import Layer
import Math
import Optimizer
import Schedule
import Tensor
import Variable


----------------------------------------------------------------------
-- Configuration
----------------------------------------------------------------------

||| Input/output size = number of symbols (0 = <BLANK>)
W : Nat
W = 3

||| Number of memory slots
N : Nat
N = 10

||| Controller hidden layer size
H : Nat
H = 20

||| Number of training examples
E : Nat
E = 13


----------------------------------------------------------------------
-- Copy Task Data
----------------------------------------------------------------------

||| Training sequences (symbols 1-2, 0 is reserved for <BLANK>)
||| Task: input [s, 0] -> output [0, s]
sequences : Vect E (List (Fin W))
sequences =
  [ [2, 1, 2]
  , [1, 2, 1, 2]
  , [1, 2, 2, 1]
  , [1, 1, 2, 2, 1]
  , [2, 2, 1, 1, 2]
  , [2, 1, 1, 2, 2, 1]
  , [1, 2, 2, 1, 2, 1]
  , [2, 1, 2, 1, 2, 1, 2]
  , [1, 1, 1, 2, 2, 2, 1]
  , [1, 2, 1, 1, 2, 2, 1, 2]
  , [2, 1, 1, 2, 1, 2, 2]
  , [2, 2, 1, 2, 1, 1, 2, 1]
  , [1, 1, 2, 1, 2, 2, 1, 2]
  ]

||| Held-out test sequence to check generalization
||| Longer than any training example — tests NTM memory capacity
testSequences : Vect 5 (List (Fin W))
testSequences =
  [ [1, 2, 2, 1, 1, 2, 1, 2]
  , [2, 1, 1, 2, 2, 1, 2, 1]
  , [1, 1, 2, 2, 1, 1]
  , [2, 2, 1, 1, 2, 2, 1]
  , [2, 1, 2, 1, 1, 2, 2, 1]
  ]

||| Convert a sequence to a RecurrentDataPoint for copy task
||| Input: sequence ++ blanks (write phase)
||| Output: blanks ++ sequence (read phase)
prep : List (Fin W) -> RecurrentDataPoint W W Double
prep sequence =
  let
    len = length sequence
    blank : Fin W
    blank = 0
    pad = Data.List.replicate len blank
    inp = sequence ++ pad
    outp = pad ++ sequence
    xs = map (oneHotEncode {n=W}) inp
    ys = map (oneHotEncode {n=W}) outp
    toDouble : Vector W Nat -> Vector W Double
    toDouble = map (fromInteger . natToInteger)
  in MkRecurrentDataPoint (map toDouble xs) (map toDouble ys)

rawData : Vect E (RecurrentDataPoint W W Double)
rawData = map prep sequences

rawTestData : Vect 5 (RecurrentDataPoint W W Double)
rawTestData = map prep testSequences


----------------------------------------------------------------------
-- Decode/Display Helpers
----------------------------------------------------------------------

decodeOutput : Vect n (List (Vector W Variable)) -> Vect n (List (Fin W))
decodeOutput = map (map argmax)

showSequences : Vect n (List (Fin W)) -> String
showSequences seqs = show $ map (map finToNat) seqs

matchCount : List (Fin W) -> List (Fin W) -> Nat
matchCount [] [] = 0
matchCount (x :: xs) (y :: ys) = (if x == y then 1 else 0) + matchCount xs ys
matchCount _ _ = 0

totalLen : Vect n (List a) -> Nat
totalLen [] = 0
totalLen (x :: xs) = length x + totalLen xs

accuracy : Vect n (List (Fin W)) -> Vect n (List (Fin W)) -> Double
accuracy preds targets =
  let len = totalLen targets
      correct = sum $ zipWith matchCount (toList preds) (toList targets)
  in if len == 0 then 0.0 else cast correct / cast len


----------------------------------------------------------------------
-- Training with progress reporting
----------------------------------------------------------------------

trainReport :
  (Double -> Optimizer) ->
  Schedule ->
  Network W [W] W Variable ->
  Vect E (RecurrentDataPoint W W Variable) ->
  Nat -> Nat -> Nat -> OptimizerState -> Double -> Nat ->
  IO (Network W [W] W Variable, OptimizerState, Nat)
trainReport _ _ model _ Z _ done st _ staleCount = pure (model, st, done)
trainReport makeOpt schedule model dps (S chunks) patience done st bestLoss staleCount = do
  -- Run 100 epochs with schedule
  let (model', st', loss, staleCount') = runChunk makeOpt schedule model dps 100 done st bestLoss staleCount
  putStrLn $ "  " ++ show (done + 100) ++ ":\t" ++ show loss
  let bestLoss' = if loss < bestLoss then loss else bestLoss
  -- Check early stopping
  if patience > 0 && staleCount' >= patience
    then do
      putStrLn $ "  Early stop at epoch " ++ show (done + 100) ++ " (patience=" ++ show patience ++ ")"
      pure (model', st', done + 100)
    else if loss /= loss  -- NaN check
    then do
      putStrLn $ "  Diverged (NaN) at epoch " ++ show (done + 100)
      pure (model', st', done + 100)
    else trainReport makeOpt schedule model' dps chunks patience (done + 100) st' bestLoss' staleCount'
  where
    minDelta : Double
    minDelta = 0.0001
    runChunk : (Double -> Optimizer) -> Schedule ->
               Network W [W] W Variable ->
               Vect E (RecurrentDataPoint W W Variable) ->
               Nat -> Nat -> OptimizerState -> Double -> Nat ->
               (Network W [W] W Variable, OptimizerState, Double, Nat)
    runChunk _ _ m _ Z _ s lastLoss sc = (m, s, lastLoss, sc)
    runChunk mk sched m ds (S k) ep s bl sc =
      let lr = sched ep
          opt = mk lr
          (m', s', loss) = epochRecurrent opt ds nllLoss m s
          improved = loss < bl - minDelta
          bl' = if improved then loss else bl
          sc' : Nat
          sc' = if improved then 0 else sc + 1
      in runChunk mk sched m' ds k (ep + 1) s' bl' sc'


----------------------------------------------------------------------
-- CLI Argument Parsing
----------------------------------------------------------------------

record Config where
  constructor MkConfig
  lr : Double
  maxNorm : Double
  beta1 : Double
  beta2 : Double
  eps : Double
  divFinal : Double
  epochs : Nat
  patience : Nat
  seed : Bits64

defaultConfig : Config
defaultConfig = MkConfig 0.001 5.0 0.9 0.999 (pow 10 (-8)) 10.0 6000 200 123456

parseConfig : List String -> Config
parseConfig args = go args defaultConfig
  where
    go : List String -> Config -> Config
    go [] c = c
    go ("--lr" :: v :: rest) c = go rest ({ lr := cast v } c)
    go ("--max-norm" :: v :: rest) c = go rest ({ maxNorm := cast v } c)
    go ("--beta1" :: v :: rest) c = go rest ({ beta1 := cast v } c)
    go ("--beta2" :: v :: rest) c = go rest ({ beta2 := cast v } c)
    go ("--eps" :: v :: rest) c = go rest ({ eps := cast v } c)
    go ("--div-final" :: v :: rest) c = go rest ({ divFinal := cast v } c)
    go ("--epochs" :: v :: rest) c = go rest ({ epochs := cast (cast {to=Integer} v) } c)
    go ("--patience" :: v :: rest) c = go rest ({ patience := cast (cast {to=Integer} v) } c)
    go ("--seed" :: v :: rest) c = go rest ({ seed := cast (cast {to=Integer} v) } c)
    go (_ :: rest) c = go rest c


----------------------------------------------------------------------
-- Main
----------------------------------------------------------------------

main : IO ()
main = do
  args <- getArgs
  let cfg = parseConfig (drop 1 args)

  srand cfg.seed

  putStrLn "=== NTM Copy Task ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr
           ++ " maxNorm=" ++ show cfg.maxNorm
           ++ " beta1=" ++ show cfg.beta1
           ++ " beta2=" ++ show cfg.beta2
           ++ " epochs=" ++ show cfg.epochs
           ++ " patience=" ++ show cfg.patience
           ++ " seed=" ++ show cfg.seed
           ++ " H=" ++ show H
  putStrLn ""

  -- Build NTM with logSoftmax output
  controllerHidden <- linearLayer {i = NtmInputWidth W, o = H}
  controllerOut <- linearLayer {i = H, o = NtmOutputWidth N W}
  let controller = controllerHidden ~> tanhLayer ~> OutputLayer controllerOut
  ntm <- ntmLayer {n = N, w = W} controller
  let model = nameNetworkParams "ntm" $ ntm ~> OutputLayer logSoftmaxLayer

  putStr "Model:\t\t"
  printLn model
  putStrLn ""

  -- Prepare data
  let dataPoints = map (map fromDouble) rawData
  let testPoints = map (map fromDouble) rawTestData
  let targets = map (map argmax . ys) dataPoints
  let testTargets = map (map argmax . ys) testPoints
  putStr "Targets:\t"
  putStrLn $ showSequences targets
  putStr "Test targets:\t"
  putStrLn $ showSequences testTargets

  -- Pre-training evaluation
  let loss = calculateLossRecurrent nllLoss model dataPoints
  putStr "Pre loss:\t"
  printLn $ value loss
  putStr "Predictions:\t"
  putStrLn $ showSequences $ decodeOutput $ evaluateRecurrent model dataPoints
  putStrLn ""

  -- One-cycle training
  let makeOpt = \lr => adamGlobalClip lr cfg.beta1 cfg.beta2 cfg.eps cfg.maxNorm
  let schedule = oneCycle cfg.lr 25.0 cfg.divFinal 0.25 cfg.epochs
  let chunks = cfg.epochs `div` 100
  putStrLn $ "Training (one-cycle, lrMax=" ++ show cfg.lr ++ ")..."
  (trained, finalSt, epochsDone) <- trainReport makeOpt schedule model dataPoints chunks cfg.patience 0 initState (1.0/0.0) 0

  putStrLn ""

  -- Final evaluation
  let trainPreds = decodeOutput $ evaluateRecurrent trained dataPoints
  let testPreds = decodeOutput $ evaluateRecurrent trained testPoints
  let trainAcc = accuracy trainPreds targets
  let testAcc = accuracy testPreds testTargets

  putStrLn "Train predictions:"
  putStr "  Predictions:\t"
  putStrLn $ showSequences trainPreds
  putStr "  Accuracy:\t"
  putStrLn $ show trainAcc

  putStrLn "Test eval:"
  putStr "  Predictions:\t"
  putStrLn $ showSequences testPreds
  putStr "  Accuracy:\t"
  putStrLn $ show testAcc

  -- Machine-readable result line for sweep script
  putStrLn $ "RESULT\t"
           ++ show cfg.lr ++ "\t"
           ++ show cfg.maxNorm ++ "\t"
           ++ show cfg.beta1 ++ "\t"
           ++ show cfg.beta2 ++ "\t"
           ++ show cfg.epochs ++ "\t"
           ++ show cfg.patience ++ "\t"
           ++ show epochsDone ++ "\t"
           ++ show cfg.seed ++ "\t"
           ++ show H ++ "\t"
           ++ show trainAcc ++ "\t"
           ++ show testAcc
