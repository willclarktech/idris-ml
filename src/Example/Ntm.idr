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
E = 5


----------------------------------------------------------------------
-- Copy Task Data
----------------------------------------------------------------------

||| Training sequences (symbols 1-2, 0 is reserved for <BLANK>)
||| Task: input [s, 0] -> output [0, s]
sequences : Vect E (List (Fin W))
sequences =
  [ [1, 2, 1, 2]
  , [1, 1, 2, 2, 1]
  , [2, 1, 1, 2, 2, 1]
  , [2, 1, 2, 1, 2, 1, 2]
  , [1, 2, 1, 1, 2, 2, 1, 2]
  ]

||| Held-out test sequence to check generalization
||| Longer than any training example — tests NTM memory capacity
testSequences : Vect 1 (List (Fin W))
testSequences = [ [1, 2, 2, 1, 1, 2, 1, 2] ]

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

rawTestData : Vect 1 (RecurrentDataPoint W W Double)
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
  Optimizer ->
  Network W [W] W Variable ->
  Vect E (RecurrentDataPoint W W Variable) ->
  Nat -> Nat -> OptimizerState ->
  IO (Network W [W] W Variable, OptimizerState)
trainReport _ model _ Z _ st = pure (model, st)
trainReport opt model dps (S chunks) done st = do
  let (model', st') = trainRecurrentFrom opt model dps nllLoss 100 st
  let loss = calculateLossRecurrent nllLoss model' dps
  putStrLn $ "  " ++ show (done + 100) ++ ":\t" ++ show (value loss)
  trainReport opt model' dps chunks (done + 100) st'


----------------------------------------------------------------------
-- CLI Argument Parsing
----------------------------------------------------------------------

record Config where
  constructor MkConfig
  lr1 : Double
  lr2 : Double
  maxNorm : Double
  epochs1 : Nat
  epochs2 : Nat
  seed : Bits64

defaultConfig : Config
defaultConfig = MkConfig 0.001 0.0003 5.0 3000 3000 123456

parseConfig : List String -> Config
parseConfig args = go args defaultConfig
  where
    go : List String -> Config -> Config
    go [] c = c
    go ("--lr1" :: v :: rest) c = go rest ({ lr1 := cast v } c)
    go ("--lr2" :: v :: rest) c = go rest ({ lr2 := cast v } c)
    go ("--max-norm" :: v :: rest) c = go rest ({ maxNorm := cast v } c)
    go ("--epochs1" :: v :: rest) c = go rest ({ epochs1 := cast (cast {to=Integer} v) } c)
    go ("--epochs2" :: v :: rest) c = go rest ({ epochs2 := cast (cast {to=Integer} v) } c)
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
  putStrLn $ "Config: lr1=" ++ show cfg.lr1
           ++ " lr2=" ++ show cfg.lr2
           ++ " maxNorm=" ++ show cfg.maxNorm
           ++ " epochs=" ++ show cfg.epochs1 ++ "+" ++ show cfg.epochs2
           ++ " seed=" ++ show cfg.seed
           ++ " H=" ++ show H
  putStrLn ""

  -- Build NTM with logSoftmax output
  controllerHidden <- linearLayer {i = NtmInputWidth W, o = H}
  controllerOut <- linearLayer {i = H, o = NtmOutputWidth N W}
  let controller = controllerHidden ~> sigmoidLayer ~> OutputLayer controllerOut
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

  -- Phase 1
  let opt1 = adamGlobalClip cfg.lr1 0.9 0.999 (pow 10 (-8)) cfg.maxNorm
  let chunks1 = cfg.epochs1 `div` 100
  putStrLn $ "Training (lr=" ++ show cfg.lr1 ++ ")..."
  (t1, s1) <- trainReport opt1 model dataPoints chunks1 0 initState

  putStrLn ""
  putStrLn "Midpoint predictions (train):"
  putStr "  Predictions:\t"
  putStrLn $ showSequences $ decodeOutput $ evaluateRecurrent t1 dataPoints
  putStrLn "Midpoint predictions (test):"
  putStr "  Predictions:\t"
  putStrLn $ showSequences $ decodeOutput $ evaluateRecurrent t1 testPoints
  putStrLn ""

  -- Phase 2 (carry Adam state)
  let opt2 = adamGlobalClip cfg.lr2 0.9 0.999 (pow 10 (-8)) cfg.maxNorm
  let chunks2 = cfg.epochs2 `div` 100
  putStrLn $ "Training (lr=" ++ show cfg.lr2 ++ ")..."
  (t2, _) <- trainReport opt2 t1 dataPoints chunks2 cfg.epochs1 s1

  putStrLn ""

  -- Final evaluation
  let trainPreds = decodeOutput $ evaluateRecurrent t2 dataPoints
  let testPreds = decodeOutput $ evaluateRecurrent t2 testPoints
  let trainAcc = accuracy trainPreds targets
  let testAcc = accuracy testPreds testTargets

  putStrLn "Train predictions:"
  putStr "  Predictions:\t"
  putStrLn $ showSequences trainPreds
  putStr "  Accuracy:\t"
  putStrLn $ show trainAcc

  putStrLn "Test eval ([1,2,2,1,1,2,1,2]):"
  putStr "  Predictions:\t"
  putStrLn $ showSequences testPreds
  putStr "  Accuracy:\t"
  putStrLn $ show testAcc

  -- Machine-readable result line for sweep script
  putStrLn $ "RESULT\t"
           ++ show cfg.lr1 ++ "\t"
           ++ show cfg.lr2 ++ "\t"
           ++ show cfg.maxNorm ++ "\t"
           ++ show cfg.epochs1 ++ "\t"
           ++ show cfg.epochs2 ++ "\t"
           ++ show cfg.seed ++ "\t"
           ++ show H ++ "\t"
           ++ show trainAcc ++ "\t"
           ++ show testAcc
