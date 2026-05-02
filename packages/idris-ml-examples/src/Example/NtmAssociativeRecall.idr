-- | NTM Associative Recall Task
-- |
-- | Binary vector recall task with LSTM controller, interpolation write,
-- | sigmoid output + BCE loss, and RMSprop optimizer.

module Example.NtmAssociativeRecall

import Data.List
import Data.String
import Data.Vect
import System
import System.Clock
import Compat.Random

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
import Device
import Variable


----------------------------------------------------------------------
-- Configuration
----------------------------------------------------------------------

W : Nat
W = 6

SeqLen : Nat
SeqLen = 3

InputW : Nat
InputW = S (S W)

OutputW : Nat
OutputW = W

N : Nat
N = 128

M : Nat
M = 20

H : Nat
H = 100

BatchSize : Nat
BatchSize = 16

TestSize : Nat
TestSize = 100


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
  minItems : Nat
  maxItems : Nat
  batch : Nat

defaultConfig : Config
defaultConfig = MkConfig 0.0001 10.0 0.95 1.0e-8 0.9 100000 0.01 1000 3 42 2 6 16

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
        , Arg "--min-items" (\v, c => { minItems := castNat v } c)
        , Arg "--max-items" (\v, c => { maxItems := castNat v } c)
        , Arg "--batch" (\v, c => { batch := castNat v } c) ]


----------------------------------------------------------------------
-- Main
----------------------------------------------------------------------

main : IO ()
main = do
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)

  srand cfg.seed

  putStrLn "=== NTM Associative Recall ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr
           ++ " clip=" ++ show cfg.clipVal
           ++ " epochs=" ++ show cfg.epochs
           ++ " seed=" ++ show cfg.seed
           ++ " batch=" ++ show cfg.batch
           ++ " items=" ++ show cfg.minItems ++ "-" ++ show cfg.maxItems
           ++ " seqLen=" ++ show SeqLen
  putStrLn $ "Architecture: N=" ++ show N ++ " M=" ++ show M ++ " H=" ++ show H

  ntm <- ntmLayer {inputSize = InputW, outputSize = OutputW, n = N, m = M, h = H}
  let model = autoName $ OutputLayer ntm
  putStrLn $ "Model: " ++ show model
  putStrLn ""

  let opt = nativeRmsprop cfg.lr cfg.alpha cfg.eps cfg.clipVal cfg.momentum

  -- Data source: fresh batch each epoch (raw Doubles)
  let genBatch : IO (Vect (cfg.batch) (TwoPhaseDataPoint InputW OutputW Double))
      genBatch = recallTaskBinaryBatchVect {w = W} cfg.batch cfg.minItems cfg.maxItems SeqLen

  -- Metrics: bit accuracy + memory
  let evalMetrics : Network InputW [] OutputW (Variable CPU) -> IO (List (String, String))
      evalMetrics m = do
        let dblM = toDoubleNetwork (emap refreshValue m)
        evalBatch <- recallTaskBinaryBatchVect {w = W} 10 cfg.minItems cfg.maxItems SeqLen
        let avgAcc = foldl (+) 0.0
              (toList (map (\dp => let (_, preds) = forwardTwoPhase dblM dp
                                   in bitAccuracy preds (targets dp)) evalBatch)) / 10.0
        pure [ ("acc", show (avgAcc * 100.0) ++ "%") ]

  let trainCfg = MkTrainConfig cfg.epochs 100
                   (WindowedAvg cfg.esThreshold cfg.esWindow cfg.esPatience) evalMetrics
                   (\_ => pure ())

  (trained, epochsDone, _) <- runTraining
    (\m, d => epochTwoPhaseTensor opt d m) genBatch trainCfg model

  -- Evaluation
  let dblModel = toDoubleNetwork (emap refreshValue trained)

  let evalOne : TwoPhaseDataPoint InputW OutputW Double -> Double
      evalOne dp =
        let (_, preds) = forwardTwoPhase dblModel dp
        in bitAccuracy preds (targets dp)

  k2Batch <- recallTaskBinaryBatchVect {w = W} TestSize 2 2 SeqLen
  k4Batch <- recallTaskBinaryBatchVect {w = W} TestSize 4 4 SeqLen
  k6Batch <- recallTaskBinaryBatchVect {w = W} TestSize 6 6 SeqLen
  let k2Acc = foldl (+) 0.0 (toList (map evalOne k2Batch)) / cast TestSize
  let k4Acc = foldl (+) 0.0 (toList (map evalOne k4Batch)) / cast TestSize
  let k6Acc = foldl (+) 0.0 (toList (map evalOne k6Batch)) / cast TestSize

  putStrLn ""
  putStrLn "Eval:"
  putStrLn $ "  K=2 items: " ++ show (k2Acc * 100.0) ++ "% bit accuracy"
  putStrLn $ "  K=4 items: " ++ show (k4Acc * 100.0) ++ "% bit accuracy"
  putStrLn $ "  K=6 items: " ++ show (k6Acc * 100.0) ++ "% bit accuracy"
  putStrLn ""
  putStrLn $ formatResult [("epochs", show epochsDone), ("acc_k2", show k2Acc),
                            ("acc_k4", show k4Acc), ("acc_k6", show k6Acc),
                            ("seed", show cfg.seed)]
