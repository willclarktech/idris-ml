module Example.Bench

import Data.List
import Data.Stream
import Data.Vect
import System
import System.Clock
import System.Random

import Backprop
import DataPoint
import Floating
import Layer
import Math
import Optimizer
import Tensor
import Util
import Variable


----------------------------------------------------------------------
-- Timing
----------------------------------------------------------------------

elapsedMs : Clock Monotonic -> Clock Monotonic -> Double
elapsedMs t0 t1 =
  let s = cast {to=Double} (seconds t1 - seconds t0)
      ns = cast {to=Double} (nanoseconds t1 - nanoseconds t0)
  in s * 1000.0 + ns / 1000000.0


----------------------------------------------------------------------
-- Supervised (Linear + Softmax, same config as Supervised.idr)
----------------------------------------------------------------------

supervisedData : Vect 5 (DataPoint 2 3 Double)
supervisedData =
  [ MkDataPoint (VTensor [1.5, -2.7]) (VTensor [0, 1, 0])
  , MkDataPoint (VTensor [-3.2, 4.1]) (VTensor [0, 1, 0])
  , MkDataPoint (VTensor [5.7, 0]) (VTensor [0, 0, 1])
  , MkDataPoint (VTensor [-1.3, 8.8]) (VTensor [0, 1, 0])
  , MkDataPoint (VTensor [2.9, -1.4]) (VTensor [1, 0, 0])
  ]

benchSupervised : IO ()
benchSupervised = do
  ll <- nameParams "ll" <$> linearLayer
  let model = ll ~> OutputLayer softmaxLayer
  let prepared = map (map fromDouble) supervisedData
  let opt = sgd 0.03 (1.0/0.0)

  -- Warmup: 100 epochs
  let warmModel = train opt model prepared crossEntropy 100

  -- Benchmark: 1000 epochs
  t0 <- clockTime Monotonic
  let trained = train opt warmModel prepared crossEntropy 1000
  let loss = calculateLoss crossEntropy trained prepared
  t1 <- clockTime Monotonic

  putStrLn $ "Supervised (1000 epochs): " ++ show (elapsedMs t0 t1) ++ " ms"
  putStrLn $ "  Final loss: " ++ show (value loss)


----------------------------------------------------------------------
-- RNN (same config as Rnn.idr)
----------------------------------------------------------------------

generateRnnData : Nat -> (List Double, List Double)
generateRnnData n =
  let infinitePattern = cycle [0, 1, 0]
  in (take n infinitePattern, take n (drop 1 infinitePattern))

generateRnnDataSet : {n : Nat} -> Vect n (List Double, List Double)
generateRnnDataSet = map (generateRnnData . (+3) . finToNat) Data.Vect.Fin.range

rnnRawData : (n : Nat) -> Vect n (RecurrentDataPoint 1 1 Double)
rnnRawData n = map (\(is, os) => MkRecurrentDataPoint (prep is) (prep os)) $ generateRnnDataSet {n}
  where
    prep : (ns : List Double) -> List (Vector 1 Double)
    prep ns = map (flatten . STensor) ns

benchRnn : IO ()
benchRnn = do
  rnn <- nameParams "rnn" <$> rnnLayer
  let model = OutputLayer rnn
  let dataPoints = map (map fromDouble) (rnnRawData 8)
  let opt = sgd 0.03 (1.0/0.0)

  -- Warmup: 100 epochs
  let warmModel = trainRecurrent opt model dataPoints binaryCrossEntropyWithLogits 100

  -- Benchmark: 1000 epochs
  t0 <- clockTime Monotonic
  let trained = trainRecurrent opt warmModel dataPoints binaryCrossEntropyWithLogits 1000
  let loss = calculateLossRecurrent binaryCrossEntropyWithLogits trained dataPoints
  t1 <- clockTime Monotonic

  putStrLn $ "RNN (1000 epochs):        " ++ show (elapsedMs t0 t1) ++ " ms"
  putStrLn $ "  Final loss: " ++ show (value loss)


----------------------------------------------------------------------
-- NTM Copy Task (same config as Ntm.idr)
----------------------------------------------------------------------

W : Nat
W = 3

N : Nat
N = 10

H : Nat
H = 20

E : Nat
E = 5

ntmSequences : Vect E (List (Fin W))
ntmSequences =
  [ [1, 2, 1, 2]
  , [1, 1, 2, 2, 1]
  , [2, 1, 1, 2, 2, 1]
  , [2, 1, 2, 1, 2, 1, 2]
  , [1, 2, 1, 1, 2, 2, 1, 2]
  ]

prepNtm : List (Fin W) -> RecurrentDataPoint W W Double
prepNtm sequence =
  let len = length sequence
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

ntmRawData : Vect E (RecurrentDataPoint W W Double)
ntmRawData = map prepNtm ntmSequences

benchNtm : IO ()
benchNtm = do
  controllerHidden <- linearLayer {i = NtmInputWidth W, o = H}
  controllerOut <- linearLayer {i = H, o = NtmOutputWidth N W}
  let controller = controllerHidden ~> sigmoidLayer ~> OutputLayer controllerOut
  ntm <- ntmLayer {n = N, w = W} controller
  let model = nameNetworkParams "ntm" $ ntm ~> OutputLayer logSoftmaxLayer

  let dataPoints = map (map fromDouble) ntmRawData
  let opt = adamGlobalClip 0.001 0.9 0.999 (pow 10 (-8)) 5.0

  -- Warmup: 10 epochs
  let (warmModel, warmSt) = trainRecurrentFrom opt model dataPoints nllLoss 10 initState

  -- Benchmark: 100 epochs
  t0 <- clockTime Monotonic
  let (benchModel, _) = trainRecurrentFrom opt warmModel dataPoints nllLoss 100 warmSt
  let loss = calculateLossRecurrent nllLoss benchModel dataPoints
  t1 <- clockTime Monotonic

  putStrLn $ "NTM (100 epochs):         " ++ show (elapsedMs t0 t1) ++ " ms"
  putStrLn $ "  Final loss: " ++ show (value loss)


----------------------------------------------------------------------
-- Main
----------------------------------------------------------------------

main : IO ()
main = do
  srand 123456

  benchSupervised
  benchRnn
  benchNtm
