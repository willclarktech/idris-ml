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
W = 2

||| Number of memory slots
N : Nat
N = 2

||| Controller hidden layer size
H : Nat
H = 4

||| Number of training examples
E : Nat
E = 1


----------------------------------------------------------------------
-- Copy Task Data
----------------------------------------------------------------------

||| Training sequences (symbol 1, 0 is reserved for <BLANK>)
||| Task: input [1, <BLANK>] -> output [<BLANK>, 1]
sequences : Vect E (List (Fin W))
sequences =
  [ [1]
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


----------------------------------------------------------------------
-- Decode/Display Helpers
----------------------------------------------------------------------

decodeOutput : Vect E (List (Vector W Variable)) -> Vect E (List (Fin W))
decodeOutput = map (map argmax)

showSequences : Vect E (List (Fin W)) -> String
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
  putStr "Targets:\t"
  putStrLn $ showSequences $ map (map argmax . ys) dataPoints

  let lr = 0.03
  let lossFn = crossEntropy

  -- Pre-training evaluation
  let loss = calculateLossRecurrent lossFn model dataPoints
  putStr "Pre loss:\t"
  printLn $ value loss
  putStr "Predictions:\t"
  putStrLn $ showSequences $ decodeOutput $ evaluateRecurrent model dataPoints
  putStrLn ""

  -- Train in stages to show progress
  putStrLn "Training..."
  let t1 = trainRecurrent lr model dataPoints lossFn 200
  let l1 = calculateLossRecurrent lossFn t1 dataPoints
  putStrLn $ "  200 epochs:\t" ++ show (value l1)

  let t2 = trainRecurrent lr t1 dataPoints lossFn 200
  let l2 = calculateLossRecurrent lossFn t2 dataPoints
  putStrLn $ "  400 epochs:\t" ++ show (value l2)

  let t3 = trainRecurrent lr t2 dataPoints lossFn 200
  let l3 = calculateLossRecurrent lossFn t3 dataPoints
  putStrLn $ "  600 epochs:\t" ++ show (value l3)

  putStrLn ""

  -- Post-training evaluation
  putStr "Predictions:\t"
  putStrLn $ showSequences $ decodeOutput $ evaluateRecurrent t3 dataPoints
