module Example.Bench

import Data.List
import Data.Stream
import Data.Vect
import System
import System.Clock
import System.Random

import Backprop
import DataPoint
import Endofunctor
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
  let opt = nativeSgd 0.03

  -- Warmup: 100 epochs
  let warmModel = trainNative opt model prepared crossEntropy 100

  -- Benchmark: 1000 epochs
  t0 <- clockTime Monotonic
  let trained = trainNative opt warmModel prepared crossEntropy 1000
  let dblModel = toDoubleNetwork (emap refreshValue trained)
  let loss = calculateLoss crossEntropy dblModel (map (map fromDouble) supervisedData)
  t1 <- clockTime Monotonic

  putStrLn $ "Supervised (1000 epochs): " ++ show (elapsedMs t0 t1) ++ " ms"
  putStrLn $ "  Final loss: " ++ show loss
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
  let opt = nativeSgd 0.03

  -- Warmup: 100 epochs
  let warmModel = foldl (\m, _ => fst (epochRecurrentNative opt dataPoints binaryCrossEntropyWithLogits m)) model [1..100]

  -- Benchmark: 1000 epochs
  t0 <- clockTime Monotonic
  let trained = foldl (\m, _ => fst (epochRecurrentNative opt dataPoints binaryCrossEntropyWithLogits m)) warmModel [1..1000]
  let dblModel = toDoubleNetwork (emap refreshValue trained)
  let loss = calculateLossRecurrent binaryCrossEntropyWithLogits dblModel (rnnRawData 8)
  t1 <- clockTime Monotonic

  putStrLn $ "RNN (1000 epochs):        " ++ show (elapsedMs t0 t1) ++ " ms"
  putStrLn $ "  Final loss: " ++ show loss
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
  let opt = nativeRmsprop 0.0001 0.95 1.0e-8 10.0 0.0

  -- Warmup: 10 epochs
  let (warmModel, _) = foldl
        (\(m, _), _ =>
          epochTwoPhaseBceNative opt dataPoints m)
        (model, 0.0) [1..10]

  -- Benchmark: 100 epochs
  t0 <- clockTime Monotonic
  let (benchModel, benchLoss) = foldl
        (\(m, _), _ =>
          epochTwoPhaseBceNative opt dataPoints m)
        (warmModel, 0.0) [1..100]
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
  let opt = nativeRmsprop 0.0001 0.95 1.0e-8 10.0 0.0

  -- Warmup: 10 epochs
  let (warmModel, _) = foldl
        (\(m, _), _ =>
          epochTwoPhaseBceNative opt dataPoints m)
        (model, 0.0) [1..10]

  -- Benchmark: 100 epochs
  t0 <- clockTime Monotonic
  let (benchModel, benchLoss) = foldl
        (\(m, _), _ =>
          epochTwoPhaseBceNative opt dataPoints m)
        (warmModel, 0.0) [1..100]
  t1 <- clockTime Monotonic

  putStrLn $ "NTM-copy (100 epochs):    " ++ show (elapsedMs t0 t1) ++ " ms"
  putStrLn $ "  Final loss: " ++ show benchLoss
  putStrLn $ "  Peak RSS: " ++ show (getRssMB 3) ++ " MB"


----------------------------------------------------------------------
-- NTM Copy 1K (realistic: fresh data + GC, matching real training)
----------------------------------------------------------------------

copy1kEpoch : NativeOptimizer ->
              Network CopyInputW [] CopyOutputW Variable ->
              IO (Network CopyInputW [] CopyOutputW Variable, Double)
copy1kEpoch opt m = do
  batch <- copyTaskBinaryBatchVect {w = CopyW} CopyBatch 1 20
  let dps = map (map fromDouble) batch
  let res = epochTwoPhaseBceNative opt dps m
  pure res

copy1kLoop : NativeOptimizer -> Nat -> Nat ->
             Network CopyInputW [] CopyOutputW Variable ->
             Double ->
             IO (Network CopyInputW [] CopyOutputW Variable, Double)
copy1kLoop opt numEpochs remaining m loss =
  if remaining == 0 then pure (m, loss)
  else do
    (m', loss') <- copy1kEpoch opt m
    let i = minus numEpochs remaining
    when (modNatNZ i 10 ItIsSucc == 0) forceGC
    copy1kLoop opt numEpochs (minus remaining 1) m' loss'

benchNtmCopy1k : IO ()
benchNtmCopy1k = do
  ntm <- ntmLayer {inputSize = CopyInputW, outputSize = CopyOutputW, n = CopyN, m = CopyM, h = CopyH}
  let model = autoName $ OutputLayer ntm
  let opt = nativeRmsprop 0.0001 0.95 1.0e-8 10.0 0.9

  -- Warmup: 10 epochs (fresh data + GC)
  (warmModel, _) <- copy1kLoop opt 10 10 model 0.0
  forceGC

  -- Benchmark: 1000 epochs (fresh data + GC every 10)
  t0 <- clockTime Monotonic
  (_, finalLoss) <- copy1kLoop opt 1000 1000 warmModel 0.0
  t1 <- clockTime Monotonic

  putStrLn $ "NTM-copy-1k (1000 epochs): " ++ show (elapsedMs t0 t1) ++ " ms"
  putStrLn $ "  Final loss: " ++ show finalLoss
  putStrLn $ "  Peak RSS: " ++ show (getRssMB 4) ++ " MB"


----------------------------------------------------------------------
-- NTM Recall (matching NtmAssociativeRecall.idr architecture)
----------------------------------------------------------------------

RecallW : Nat
RecallW = 6

RecallInputW : Nat
RecallInputW = S (S RecallW)

RecallOutputW : Nat
RecallOutputW = RecallW

RecallN : Nat
RecallN = 128

RecallM : Nat
RecallM = 20

RecallH : Nat
RecallH = 100

RecallBatch : Nat
RecallBatch = 16

benchNtmRecall : IO ()
benchNtmRecall = do
  ntm <- ntmLayer {inputSize = RecallInputW, outputSize = RecallOutputW, n = RecallN, m = RecallM, h = RecallH}
  let model = autoName $ OutputLayer ntm

  -- Generate fixed training data
  batch <- recallTaskBinaryBatchVect {w = RecallW} RecallBatch 2 6 3
  let dataPoints = map (map fromDouble) batch
  let opt = nativeRmsprop 0.0001 0.95 1.0e-8 10.0 0.9

  -- Warmup: 10 epochs
  let (warmModel, _) = foldl
        (\(m, _), _ =>
          epochTwoPhaseBceNative opt dataPoints m)
        (model, 0.0) [1..10]

  -- Benchmark: 100 epochs
  t0 <- clockTime Monotonic
  let (benchModel, benchLoss) = foldl
        (\(m, _), _ =>
          epochTwoPhaseBceNative opt dataPoints m)
        (warmModel, 0.0) [1..100]
  t1 <- clockTime Monotonic

  putStrLn $ "NTM-recall (100 epochs):  " ++ show (elapsedMs t0 t1) ++ " ms"
  putStrLn $ "  Final loss: " ++ show benchLoss
  putStrLn $ "  Peak RSS: " ++ show (getRssMB 5) ++ " MB"


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
  benchNtmRecall
