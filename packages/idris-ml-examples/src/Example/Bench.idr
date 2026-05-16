module Example.Bench

import Data.List
import Data.Stream
import Data.Vect
import System
import System.Clock
import Compat.Random

import Backprop
import DataPoint
import Floating
import Generate
import Layer.Core
import Layer.Linear
import Layer.Ntm
import Layer.Rnn
import Array
import Util
import Device
import Tensor


----------------------------------------------------------------------
-- Timing
----------------------------------------------------------------------

elapsedMs : Clock Monotonic -> Clock Monotonic -> Double
elapsedMs t0 t1 =
  let s = cast {to=Double} (seconds t1 - seconds t0)
      ns = cast {to=Double} (nanoseconds t1 - nanoseconds t0)
  in s * 1000.0 + ns / 1000000.0


----------------------------------------------------------------------
-- Supervised: Linear classifier, raw logits + BCE-with-logits loss
----------------------------------------------------------------------

supervisedData : Vect 5 (DataPoint 2 3 Double)
supervisedData =
  [ MkDataPoint (VArray [1.5, -2.7]) (VArray [0, 1, 0])
  , MkDataPoint (VArray [-3.2, 4.1]) (VArray [0, 1, 0])
  , MkDataPoint (VArray [5.7, 0]) (VArray [0, 0, 1])
  , MkDataPoint (VArray [-1.3, 8.8]) (VArray [0, 1, 0])
  , MkDataPoint (VArray [2.9, -1.4]) (VArray [1, 0, 0])
  ]

benchSupervised : IO ()
benchSupervised = do
  llAny <- linearLayerAny {i=2} {o=3} "ll"
  let model : Network 2 [] 3 CPU WithGrad
      model = OutputLayer llAny
  let opt = nativeSgd 0.03

  -- Warmup: 100 epochs
  let (warmModel, _) = foldl
        (\(m, _), _ => epochVar opt supervisedData tbceLoss m)
        (model, 0.0) [1..100]

  -- Benchmark: 1000 epochs
  t0 <- clockTime Monotonic
  let (_, finalLoss) = foldl
        (\(m, _), _ => epochVar opt supervisedData tbceLoss m)
        (warmModel, 0.0) [1..1000]
  t1 <- clockTime Monotonic

  putStrLn $ "Supervised (1000 epochs): " ++ show (elapsedMs t0 t1) ++ " ms"
  putStrLn $ "  Final loss: " ++ show finalLoss
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
    prep ns = map (flatten . SArray) ns

benchRnn : IO ()
benchRnn = do
  rnnAny <- rnnLayerAny {i=1} {o=1} "rnn"
  let model : Network 1 [] 1 CPU WithGrad
      model = OutputLayer rnnAny
  let dataPoints = rnnRawData 8
  let opt = nativeSgd 0.03

  -- Warmup: 100 epochs
  let (warmModel, _) = foldl
        (\(m, _), _ => epochRecurrentVar opt dataPoints tbceLoss m)
        (model, 0.0) [1..100]

  -- Benchmark: 1000 epochs
  t0 <- clockTime Monotonic
  let (_, finalLoss) = foldl
        (\(m, _), _ => epochRecurrentVar opt dataPoints tbceLoss m)
        (warmModel, 0.0) [1..1000]
  t1 <- clockTime Monotonic

  putStrLn $ "RNN (1000 epochs):        " ++ show (elapsedMs t0 t1) ++ " ms"
  putStrLn $ "  Final loss: " ++ show finalLoss
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
  ntmAny <- ntmLayerAny {i=BenchInputW, o=BenchOutputW, n=BenchN, m=BenchM, h=BenchH} "ntm"
  let model : Network BenchInputW [] BenchOutputW CPU WithGrad
      model = OutputLayer ntmAny

  -- Generate fixed training data (raw Doubles; epochTwoPhaseVar converts internally)
  batch <- copyTaskBinaryBatchVect {w = BenchW} BenchBatch 2 4
  let opt = nativeRmsprop 0.0001 0.95 1.0e-8 10.0 0.0

  -- Warmup: 10 epochs
  let (warmModel, _) = foldl
        (\(m, _), _ =>
          epochTwoPhaseVar opt batch tbceLoss m)
        (model, 0.0) [1..10]

  -- Benchmark: 100 epochs
  t0 <- clockTime Monotonic
  let (_, benchLoss) = foldl
        (\(m, _), _ =>
          epochTwoPhaseVar opt batch tbceLoss m)
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
  ntmAny <- ntmLayerAny {i=CopyInputW, o=CopyOutputW, n=CopyN, m=CopyM, h=CopyH} "ntm"
  let model : Network CopyInputW [] CopyOutputW CPU WithGrad
      model = OutputLayer ntmAny

  batch <- copyTaskBinaryBatchVect {w = CopyW} CopyBatch 1 20
  let opt = nativeRmsprop 0.0001 0.95 1.0e-8 10.0 0.0

  -- Warmup: 10 epochs
  let (warmModel, _) = foldl
        (\(m, _), _ =>
          epochTwoPhaseVar opt batch tbceLoss m)
        (model, 0.0) [1..10]

  -- Benchmark: 100 epochs
  t0 <- clockTime Monotonic
  let (_, benchLoss) = foldl
        (\(m, _), _ =>
          epochTwoPhaseVar opt batch tbceLoss m)
        (warmModel, 0.0) [1..100]
  t1 <- clockTime Monotonic

  putStrLn $ "NTM-copy (100 epochs):    " ++ show (elapsedMs t0 t1) ++ " ms"
  putStrLn $ "  Final loss: " ++ show benchLoss
  putStrLn $ "  Peak RSS: " ++ show (getRssMB 3) ++ " MB"


----------------------------------------------------------------------
-- NTM Copy 1K (realistic: fresh data + GC, matching real training)
----------------------------------------------------------------------

copy1kEpoch : NativeOptimizer ->
              Network CopyInputW [] CopyOutputW CPU WithGrad ->
              IO (Network CopyInputW [] CopyOutputW CPU WithGrad, Double)
copy1kEpoch opt m = do
  batch <- copyTaskBinaryBatchVect {w = CopyW} CopyBatch 1 20
  let res = epochTwoPhaseVar opt batch tbceLoss m
  pure res

copy1kLoop : NativeOptimizer -> Nat -> Nat ->
             Network CopyInputW [] CopyOutputW CPU WithGrad ->
             Double ->
             IO (Network CopyInputW [] CopyOutputW CPU WithGrad, Double)
copy1kLoop opt numEpochs remaining m loss =
  if remaining == 0 then pure (m, loss)
  else do
    (m', loss') <- copy1kEpoch opt m
    let i = minus numEpochs remaining
    when (modNatNZ i 10 ItIsSucc == 0) forceGC
    copy1kLoop opt numEpochs (minus remaining 1) m' loss'

benchNtmCopy1k : IO ()
benchNtmCopy1k = do
  ntmAny <- ntmLayerAny {i=CopyInputW, o=CopyOutputW, n=CopyN, m=CopyM, h=CopyH} "ntm"
  let model : Network CopyInputW [] CopyOutputW CPU WithGrad
      model = OutputLayer ntmAny
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
  ntmAny <- ntmLayerAny {i=RecallInputW, o=RecallOutputW, n=RecallN, m=RecallM, h=RecallH} "ntm"
  let model : Network RecallInputW [] RecallOutputW CPU WithGrad
      model = OutputLayer ntmAny

  batch <- recallTaskBinaryBatchVect {w = RecallW} RecallBatch 2 6 3
  let opt = nativeRmsprop 0.0001 0.95 1.0e-8 10.0 0.9

  -- Warmup: 10 epochs
  let (warmModel, _) = foldl
        (\(m, _), _ =>
          epochTwoPhaseVar opt batch tbceLoss m)
        (model, 0.0) [1..10]

  -- Benchmark: 100 epochs
  t0 <- clockTime Monotonic
  let (_, benchLoss) = foldl
        (\(m, _), _ =>
          epochTwoPhaseVar opt batch tbceLoss m)
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
  args <- getArgs
  case drop 1 args of
    [] => do
      benchSupervised
      benchRnn
      benchNtm
      benchNtmCopy
      benchNtmCopy1k
      benchNtmRecall
    ["supervised"]  => benchSupervised
    ["rnn"]         => benchRnn
    ["ntm"]         => benchNtm
    ["ntm-copy"]    => benchNtmCopy
    ["ntm-copy-1k"] => benchNtmCopy1k
    ["ntm-recall"]  => benchNtmRecall
    other => do
      putStrLn $ "unknown bench selector: " ++ show other
      putStrLn "valid: supervised | rnn | ntm | ntm-copy | ntm-copy-1k | ntm-recall"
      exitWith (ExitFailure 2)
