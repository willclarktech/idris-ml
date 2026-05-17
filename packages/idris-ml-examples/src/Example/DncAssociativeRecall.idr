-- | DNC Associative Recall Task
-- |
-- | Binary vector recall task with LSTM controller, DNC memory
-- | (usage allocation, temporal links, erase+add write),
-- | sigmoid output + BCE loss, and RMSprop optimizer.

module Example.DncAssociativeRecall

import Data.List
import Data.String
import Data.Vect
import System
import System.Clock
import Compat.Random

import Backprop
import DataPoint
import Floating
import Generate
import Hpo.LrFinder
import Layer.Core
import Layer.Dnc
import Math
import Array
import Train
import Util
import Device
import Tensor


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
N = 32  -- Reduced from 128 for faster link matrix ops (O(n^2))

M : Nat
M = 20

H : Nat
H = 100

R : Nat
R = 1

BatchSize : Nat
BatchSize = 16

TestSize : Nat
TestSize = 20


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
  lrFind : Bool

defaultConfig : Config
defaultConfig = MkConfig 0.0001 10.0 0.95 1.0e-8 0.9 30000 0.01 1000 3 42 2 6 1 False

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
        , Arg "--batch" (\v, c => { batch := castNat v } c)
        , Arg "--lr-find" (\v, c => { lrFind := (v == "1" || v == "true") } c) ]


----------------------------------------------------------------------
-- Main
----------------------------------------------------------------------

main : IO ()
main = do
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)

  srand cfg.seed

  putStrLn "=== DNC Associative Recall ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr
           ++ " clip=" ++ show cfg.clipVal
           ++ " epochs=" ++ show cfg.epochs
           ++ " seed=" ++ show cfg.seed
           ++ " batch=" ++ show cfg.batch
           ++ " items=" ++ show cfg.minItems ++ "-" ++ show cfg.maxItems
           ++ " seqLen=" ++ show SeqLen
  putStrLn $ "Architecture: N=" ++ show N ++ " M=" ++ show M ++ " H=" ++ show H ++ " R=" ++ show R

  dncAny <- dncLayerAny {r = R, n = N, m = M, h = H, i = InputW, o = OutputW} "dnc"
  let model : Network InputW [] OutputW CPU F64 WithGrad
      model = OutputLayer dncAny
  putStrLn ""

  let opt = nativeRmsprop cfg.lr cfg.alpha cfg.eps cfg.clipVal cfg.momentum

  -- Data source: fresh batch each epoch (raw Doubles)
  let genBatch : IO (Vect (cfg.batch) (TwoPhaseDataPoint InputW OutputW Double))
      genBatch = recallTaskBinaryBatchVect {w = W} cfg.batch cfg.minItems cfg.maxItems SeqLen

  -- Metrics: bit accuracy + memory
  let evalMetrics : Network InputW [] OutputW CPU F64 WithGrad -> IO (List (String, String))
      evalMetrics m = do
        evalBatch <- recallTaskBinaryBatchVect {w = W} 10 cfg.minItems cfg.maxItems SeqLen
        accs <- traverse (\dp => do
                  (_, preds) <- forwardTwoPhase m dp
                  pure (bitAccuracy preds (targets dp))) evalBatch
        let avgAcc = foldl (+) 0.0 (toList accs) / 10.0
        pure [ ("acc", show (avgAcc * 100.0) ++ "%") ]

  when cfg.lrFind $ do
    let lrCfg : LrFindConfig
        lrCfg = { numIters := 100 } defaultLrFindConfig
    _ <- lrFind lrCfg
      (\m, d => epochTwoPhaseVar opt d tbceLoss m)
      genBatch opt model
    putStrLn ""
    putStrLn "Done — re-run without --lr-find at the recommended LR."
    exitSuccess

  let trainCfg = MkTrainConfig cfg.epochs 100
                   (WindowedPercentile 0.10 cfg.esThreshold cfg.esWindow cfg.esPatience)
                   evalMetrics (\_ => pure ())

  (trained, epochsDone, _) <- runTraining
    (\m, d => epochTwoPhaseVar opt d tbceLoss m) genBatch trainCfg model

  let evalOne : TwoPhaseDataPoint InputW OutputW Double -> IO Double
      evalOne dp = do
        (_, preds) <- forwardTwoPhase trained dp
        pure (bitAccuracy preds (targets dp))

  k2Batch <- recallTaskBinaryBatchVect {w = W} TestSize 2 2 SeqLen
  k4Batch <- recallTaskBinaryBatchVect {w = W} TestSize 4 4 SeqLen
  k6Batch <- recallTaskBinaryBatchVect {w = W} TestSize 6 6 SeqLen
  k2Acc <- withNoGrad $ do
    accs <- traverse evalOne k2Batch
    pure (foldl (+) 0.0 (toList accs) / cast TestSize)
  k4Acc <- withNoGrad $ do
    accs <- traverse evalOne k4Batch
    pure (foldl (+) 0.0 (toList accs) / cast TestSize)
  k6Acc <- withNoGrad $ do
    accs <- traverse evalOne k6Batch
    pure (foldl (+) 0.0 (toList accs) / cast TestSize)

  putStrLn ""
  putStrLn "Eval:"
  putStrLn $ "  K=2 items: " ++ show (k2Acc * 100.0) ++ "% bit accuracy"
  putStrLn $ "  K=4 items: " ++ show (k4Acc * 100.0) ++ "% bit accuracy"
  putStrLn $ "  K=6 items: " ++ show (k6Acc * 100.0) ++ "% bit accuracy"
  putStrLn ""
  putStrLn $ formatResult [("epochs", show epochsDone), ("acc_k2", show k2Acc),
                            ("acc_k4", show k4Acc), ("acc_k6", show k6Acc),
                            ("seed", show cfg.seed)]
