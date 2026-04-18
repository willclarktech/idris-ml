-- | NTM Copy Task
-- |
-- | Binary vector copy task with LSTM controller, interpolation write,
-- | sigmoid output + BCE loss, and RMSprop optimizer.

module Example.NtmCopy

import Data.List
import Data.String
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
import Train
import Util
import Variable


----------------------------------------------------------------------
-- Configuration
----------------------------------------------------------------------

W : Nat
W = 8

InputW : Nat
InputW = S W

OutputW : Nat
OutputW = W

N : Nat
N = 128

M : Nat
M = 20

H : Nat
H = 100

TestSize : Nat
TestSize = 20


----------------------------------------------------------------------
-- Display Helpers
----------------------------------------------------------------------

showBinaryVec : {w : Nat} -> Vector w Double -> String
showBinaryVec (VTensor xs) = "[" ++ go xs ++ "]"
  where
    go : Vect k (Scalar Double) -> String
    go [] = ""
    go [STensor x] = if x >= 0.5 then "1" else "0"
    go (STensor x :: rest) = (if x >= 0.5 then "1" else "0") ++ "," ++ go rest

showBinaryLogits : {w : Nat} -> Vector w Double -> String
showBinaryLogits (VTensor xs) = "[" ++ go xs ++ "]"
  where
    go : Vect k (Scalar Double) -> String
    go [] = ""
    go [STensor x] = if sigD x >= 0.5 then "1" else "0"
    go (STensor x :: rest) = (if sigD x >= 0.5 then "1" else "0") ++ "," ++ go rest


----------------------------------------------------------------------
-- CLI Argument Parsing
----------------------------------------------------------------------

record Config where
  constructor MkConfig
  lr : Double
  clipVal : Double
  alpha : Double
  eps : Double
  momentum : Double
  epochs : Nat
  esThreshold : Double
  esWindow : Nat
  esPatience : Nat
  seed : Bits64
  minLen : Nat
  maxLen : Nat
  batch : Nat

defaultConfig : Config
defaultConfig = MkConfig 0.0001 10.0 0.95 1.0e-8 0.9 50000 0.01 1000 3 42 1 20 1

specs : List (ArgSpec Config)
specs = [ Arg "--lr" (\v, c => { lr := cast v } c)
        , Arg "--clip" (\v, c => { clipVal := cast v } c)
        , Arg "--alpha" (\v, c => { alpha := cast v } c)
        , Arg "--eps" (\v, c => { eps := cast v } c)
        , Arg "--momentum" (\v, c => { momentum := cast v } c)
        , Arg "--epochs" (\v, c => { epochs := castNat v } c)
        , Arg "--es-threshold" (\v, c => { esThreshold := cast v } c)
        , Arg "--es-window" (\v, c => { esWindow := castNat v } c)
        , Arg "--es-patience" (\v, c => { esPatience := castNat v } c)
        , Arg "--seed" (\v, c => { seed := castBits64 v } c)
        , Arg "--min-len" (\v, c => { minLen := castNat v } c)
        , Arg "--max-len" (\v, c => { maxLen := castNat v } c)
        , Arg "--batch" (\v, c => { batch := castNat v } c) ]


----------------------------------------------------------------------
-- Main
----------------------------------------------------------------------

main : IO ()
main = do
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)

  srand cfg.seed

  putStrLn "=== NTM Copy ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr
           ++ " clip=" ++ show cfg.clipVal
           ++ " epochs=" ++ show cfg.epochs
           ++ " seed=" ++ show cfg.seed
           ++ " batch=" ++ show cfg.batch
           ++ " seqLen=" ++ show cfg.minLen ++ "-" ++ show cfg.maxLen
  putStrLn $ "Architecture: N=" ++ show N ++ " M=" ++ show M ++ " H=" ++ show H

  ntm <- ntmLayer {inputSize = InputW, outputSize = OutputW, n = N, m = M, h = H}
  let model = autoName $ OutputLayer ntm
  putStrLn $ "Model: " ++ show model
  putStrLn ""

  let opt = nativeRmsprop cfg.lr cfg.alpha cfg.eps cfg.clipVal cfg.momentum

  -- Data source: fresh batch each epoch (raw Doubles)
  let genBatch : IO (Vect (cfg.batch) (TwoPhaseDataPoint InputW OutputW Double))
      genBatch = copyTaskBinaryBatchVect {w = W} cfg.batch cfg.minLen cfg.maxLen

  -- Metrics: bit accuracy + memory (computed at each log step)
  let evalMetrics : Network InputW [] OutputW Variable -> IO (List (String, String))
      evalMetrics m = do
        let dblM = toDoubleNetwork (emap refreshValue m)
        evalBatch <- copyTaskBinaryBatchVect {w = W} 10 1 20
        let avgAcc = foldl (+) 0.0
              (toList (map (\dp => let (_, preds) = forwardTwoPhase dblM dp
                                   in bitAccuracy preds (targets dp)) evalBatch)) / 10.0
        pure [ ("acc", show (avgAcc * 100.0) ++ "%")
             , ("peak", show (getRssMB 0) ++ "MB")
             , ("cur", show (getCurrentRssMB 0) ++ "MB") ]

  let trainCfg = MkTrainConfig cfg.epochs 100
                   (WindowedAvg cfg.esThreshold cfg.esWindow cfg.esPatience) evalMetrics

  (trained, epochsDone, _) <- runTraining
    (\m, d => epochTwoPhaseTensor opt d m) genBatch trainCfg model

  -- Evaluation
  let dblModel = toDoubleNetwork (emap refreshValue trained)

  let evalOne : TwoPhaseDataPoint InputW OutputW Double -> Double
      evalOne dp =
        let (_, preds) = forwardTwoPhase dblModel dp
        in bitAccuracy preds (targets dp)

  shortBatch <- copyTaskBinaryBatchVect {w = W} TestSize 1 5
  fullBatch <- copyTaskBinaryBatchVect {w = W} TestSize 1 20
  let shortAcc = foldl (+) 0.0 (toList (map evalOne shortBatch)) / cast TestSize
  let fullAcc = foldl (+) 0.0 (toList (map evalOne fullBatch)) / cast TestSize

  putStrLn ""
  putStrLn "Eval:"
  sampleBatch <- copyTaskBinaryBatchVect {w = W} 2 3 5
  traverse_ (\dp =>
    let (_, preds) = forwardTwoPhase dblModel dp
    in do putStr "  Input:  "
          putStrLn $ unwords (map showBinaryVec (encodingInputs dp))
          putStr "  Target: "
          putStrLn $ unwords (map showBinaryVec (targets dp))
          putStr "  Output: "
          putStrLn $ unwords (map showBinaryLogits preds)
          putStrLn "") (toList sampleBatch)

  putStrLn $ "  Short (len 1-5):  " ++ show (shortAcc * 100.0) ++ "% bit accuracy"
  putStrLn $ "  Full  (len 1-20): " ++ show (fullAcc * 100.0) ++ "% bit accuracy"
  putStrLn ""
  putStrLn $ formatResult [("epochs", show epochsDone), ("acc_short", show shortAcc),
                            ("acc_full", show fullAcc), ("seed", show cfg.seed)]
