module Example.Ntm

import Data.List
import Data.Vect
import System.Random

import Backprop
import DataPoint
import Floating
import Layer
import Math
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
H = 8

||| Number of training examples
E : Nat
E = 3


----------------------------------------------------------------------
-- Copy Task Data
----------------------------------------------------------------------

||| Training sequences (symbols 1-2, 0 is reserved for <BLANK>)
||| Task: input [s, 0] -> output [0, s]
sequences : Vect E (List (Fin W))
sequences =
  [ [1, 2, 1, 2]
  , [2, 1, 1, 2, 2, 1]
  , [1, 1, 2, 2, 1]
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


----------------------------------------------------------------------
-- Main
----------------------------------------------------------------------

main : IO ()
main = do
  srand 123456

  putStrLn "=== NTM Copy Task ==="
  putStrLn ""

  -- Build NTM with softmax output
  controllerHidden <- linearLayer {i = NtmInputWidth W, o = H}
  controllerOut <- linearLayer {i = H, o = NtmOutputWidth N W}
  let controller = controllerHidden ~> sigmoidLayer ~> OutputLayer controllerOut
  ntm <- ntmLayer {n = N, w = W} controller
  let model = nameNetworkParams "ntm" $ ntm ~> OutputLayer softmaxLayer

  putStr "Model:\t\t"
  printLn model
  putStrLn ""

  -- Prepare data
  let dataPoints = map (map fromDouble) rawData
  let testPoints = map (map fromDouble) rawTestData
  putStr "Targets:\t"
  putStrLn $ showSequences $ map (map argmax . ys) dataPoints
  putStr "Test targets:\t"
  putStrLn $ showSequences $ map (map argmax . ys) testPoints

  let lossFn = crossEntropy

  -- Pre-training evaluation
  let loss = calculateLossRecurrent lossFn model dataPoints
  putStr "Pre loss:\t"
  printLn $ value loss
  putStr "Predictions:\t"
  putStrLn $ showSequences $ decodeOutput $ evaluateRecurrent model dataPoints
  putStrLn ""

  let lr = 0.5
  let maxGrad = 1.0

  -- Train in stages to show progress
  putStrLn "Training..."
  let t1 = trainRecurrent lr maxGrad model dataPoints lossFn 500
  let l1 = calculateLossRecurrent lossFn t1 dataPoints
  putStrLn $ "  500 epochs:\t" ++ show (value l1)

  let t2 = trainRecurrent lr maxGrad t1 dataPoints lossFn 500
  let l2 = calculateLossRecurrent lossFn t2 dataPoints
  putStrLn $ "  1000 epochs:\t" ++ show (value l2)

  let t3 = trainRecurrent lr maxGrad t2 dataPoints lossFn 500
  let l3 = calculateLossRecurrent lossFn t3 dataPoints
  putStrLn $ "  1500 epochs:\t" ++ show (value l3)

  putStrLn ""

  -- Post-training evaluation
  putStrLn "Train predictions:"
  putStr "  Predictions:\t"
  putStrLn $ showSequences $ decodeOutput $ evaluateRecurrent t3 dataPoints

  putStrLn "Test eval ([1,2,2,1,1,2,1,2]):"
  putStr "  Predictions:\t"
  putStrLn $ showSequences $ decodeOutput $ evaluateRecurrent t3 testPoints
