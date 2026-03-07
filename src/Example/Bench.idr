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
import Generate
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
  ll <- linearLayer
  let model = autoName $ ll ~> OutputLayer softmaxLayer
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
  putStrLn $ "  Peak RSS: " ++ show (getRssMB 0) ++ " MB"


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
  rnn <- rnnLayer
  let model = autoName $ OutputLayer rnn
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
  putStrLn $ "  Peak RSS: " ++ show (getRssMB 1) ++ " MB"


----------------------------------------------------------------------
-- NTM Copy Task (binary, two-phase, matching PyTorch benchmark)
----------------------------------------------------------------------

BenchW : Nat
BenchW = 3

BenchInputW : Nat
BenchInputW = S BenchW

BenchOutputW : Nat
BenchOutputW = BenchW

BenchN : Nat
BenchN = 10

BenchM : Nat
BenchM = 5

BenchH : Nat
BenchH = 20

BenchBatch : Nat
BenchBatch = 5

benchNtm : IO ()
benchNtm = do
  ntm <- ntmLayer {inputSize = BenchInputW, outputSize = BenchOutputW, n = BenchN, m = BenchM, h = BenchH}
  let model = autoName $ OutputLayer ntm

  -- Generate fixed training data
  batch <- copyTaskBinaryBatchVect {w = BenchW} BenchBatch 2 4
  let dataPoints = map (map fromDouble) batch
  let numPids = getNumPids 0
  let opt = rmspropValueClipDense 0.0001 0.95 1.0e-8 10.0
  let st0 = initDenseState numPids

  -- Warmup: 10 epochs
  let (warmModel, warmSt, _) = foldl
        (\(m, s, _), _ =>
          epochTwoPhaseDense opt dataPoints binaryCrossEntropyWithLogits m s)
        (model, st0, 0.0) [1..10]

  -- Benchmark: 100 epochs
  t0 <- clockTime Monotonic
  let (benchModel, _, benchLoss) = foldl
        (\(m, s, _), _ =>
          epochTwoPhaseDense opt dataPoints binaryCrossEntropyWithLogits m s)
        (warmModel, warmSt, 0.0) [1..100]
  t1 <- clockTime Monotonic

  putStrLn $ "NTM (100 epochs):         " ++ show (elapsedMs t0 t1) ++ " ms"
  putStrLn $ "  Final loss: " ++ show benchLoss
  putStrLn $ "  Peak RSS: " ++ show (getRssMB 2) ++ " MB"


----------------------------------------------------------------------
-- NTM Copy Production Scale (matching NtmCopy.idr architecture)
----------------------------------------------------------------------

CopyW : Nat
CopyW = 8

CopyInputW : Nat
CopyInputW = S CopyW

CopyOutputW : Nat
CopyOutputW = CopyW

CopyN : Nat
CopyN = 128

CopyM : Nat
CopyM = 20

CopyH : Nat
CopyH = 100

CopyBatch : Nat
CopyBatch = 16

benchNtmCopy : IO ()
benchNtmCopy = do
  ntm <- ntmLayer {inputSize = CopyInputW, outputSize = CopyOutputW, n = CopyN, m = CopyM, h = CopyH}
  let model = autoName $ OutputLayer ntm

  -- Generate fixed training data
  batch <- copyTaskBinaryBatchVect {w = CopyW} CopyBatch 1 20
  let dataPoints = map (map fromDouble) batch
  let numPids = getNumPids 0
  let opt = rmspropValueClipDense 0.0001 0.95 1.0e-8 10.0
  let st0 = initDenseState numPids

  -- Warmup: 10 epochs
  let (warmModel, warmSt, _) = foldl
        (\(m, s, _), _ =>
          epochTwoPhaseDense opt dataPoints binaryCrossEntropyWithLogits m s)
        (model, st0, 0.0) [1..10]

  -- Benchmark: 100 epochs
  t0 <- clockTime Monotonic
  let (benchModel, _, benchLoss) = foldl
        (\(m, s, _), _ =>
          epochTwoPhaseDense opt dataPoints binaryCrossEntropyWithLogits m s)
        (warmModel, warmSt, 0.0) [1..100]
  t1 <- clockTime Monotonic

  putStrLn $ "NTM-copy (100 epochs):    " ++ show (elapsedMs t0 t1) ++ " ms"
  putStrLn $ "  Final loss: " ++ show benchLoss
  putStrLn $ "  Peak RSS: " ++ show (getRssMB 3) ++ " MB"


----------------------------------------------------------------------
-- NTM Copy 1K (realistic: fresh data + GC, matching real training)
----------------------------------------------------------------------

copy1kEpoch : DenseOptimizer ->
              Network CopyInputW [] CopyOutputW Variable ->
              DenseOptimizerState ->
              IO (Network CopyInputW [] CopyOutputW Variable, DenseOptimizerState, Double)
copy1kEpoch opt m s = do
  batch <- copyTaskBinaryBatchVect {w = CopyW} CopyBatch 1 20
  let dps = map (map fromDouble) batch
  let res = epochTwoPhaseDense opt dps binaryCrossEntropyWithLogits m s
  pure res

copy1kLoop : DenseOptimizer -> Nat -> Nat ->
             Network CopyInputW [] CopyOutputW Variable ->
             DenseOptimizerState -> Double ->
             IO (Network CopyInputW [] CopyOutputW Variable, DenseOptimizerState, Double)
copy1kLoop opt numEpochs remaining m s loss =
  if remaining == 0 then pure (m, s, loss)
  else do
    (m', s', loss') <- copy1kEpoch opt m s
    let i = minus numEpochs remaining
    when (modNatNZ i 10 ItIsSucc == 0) forceGC
    copy1kLoop opt numEpochs (minus remaining 1) m' s' loss'

benchNtmCopy1k : IO ()
benchNtmCopy1k = do
  ntm <- ntmLayer {inputSize = CopyInputW, outputSize = CopyOutputW, n = CopyN, m = CopyM, h = CopyH}
  let model = autoName $ OutputLayer ntm
  let numPids = getNumPids 0
  let opt = rmspropValueClipMomentumDense 0.0001 0.95 1.0e-8 10.0 0.9
  let st0 = initDenseState numPids

  -- Warmup: 10 epochs (fresh data + GC)
  (warmModel, warmSt, _) <- copy1kLoop opt 10 10 model st0 0.0
  forceGC

  -- Benchmark: 1000 epochs (fresh data + GC every 10)
  t0 <- clockTime Monotonic
  (_, _, finalLoss) <- copy1kLoop opt 1000 1000 warmModel warmSt 0.0
  t1 <- clockTime Monotonic

  putStrLn $ "NTM-copy-1k (1000 epochs): " ++ show (elapsedMs t0 t1) ++ " ms"
  putStrLn $ "  Final loss: " ++ show finalLoss
  putStrLn $ "  Peak RSS: " ++ show (getRssMB 4) ++ " MB"


----------------------------------------------------------------------
-- Main
----------------------------------------------------------------------

main : IO ()
main = do
  srand 123456

  benchSupervised
  benchRnn
  benchNtm
  benchNtmCopy
  benchNtmCopy1k
