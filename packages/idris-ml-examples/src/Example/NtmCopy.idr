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
import Compat.Random

import Backprop
import Checkpoint
import DataPoint
import Floating
import Generate
import Hpo.LrFinder
import Layer.Core
import Layer.Ntm
import Math
import Array
import Train
import Util
import Executor
import Tensor
import BuildConfig


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
TestSize = 100


----------------------------------------------------------------------
-- Display Helpers
----------------------------------------------------------------------

showBinaryVec : {w : Nat} -> Vector w Double -> String
showBinaryVec (VArray xs) = "[" ++ go xs ++ "]"
  where
    go : Vect k (Scalar Double) -> String
    go [] = ""
    go [SArray x] = if x >= 0.5 then "1" else "0"
    go (SArray x :: rest) = (if x >= 0.5 then "1" else "0") ++ "," ++ go rest

showBinaryLogits : {w : Nat} -> Vector w Double -> String
showBinaryLogits (VArray xs) = "[" ++ go xs ++ "]"
  where
    go : Vect k (Scalar Double) -> String
    go [] = ""
    go [SArray x] = if sigD x >= 0.5 then "1" else "0"
    go (SArray x :: rest) = (if sigD x >= 0.5 then "1" else "0") ++ "," ++ go rest


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
  lrFind : Bool
  checkpointDir : String
  checkpointEvery : Nat

defaultConfig : Config
defaultConfig = MkConfig 0.0001 10.0 0.95 1.0e-8 0.9 10000 0.01 1000 3 42 1 20 1 False "" 500

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
        , Arg "--batch" (\v, c => { batch := castNat v } c)
        , Arg "--lr-find" (\v, c => { lrFind := (v == "1" || v == "true") } c)
        , Arg "--checkpoint-dir" (\v, c => { checkpointDir := v } c)
        , Arg "--resume" (\v, c => { checkpointDir := v } c)
        , Arg "--checkpoint-every" (\v, c => { checkpointEvery := castNat v } c) ]


----------------------------------------------------------------------
-- Main
----------------------------------------------------------------------

main : IO ()
main = do
  requireMachine {m = ChosenMachine}
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

  ntmAny <- ntmLayerAny {n = N, m = M, h = H, i = InputW, o = OutputW} "ntm"
  let model : Network InputW [] OutputW ExampleExecutor ExampleDType WithGrad
      model = OutputLayer ntmAny
  putStrLn ""

  let opt = nativeRmsprop cfg.lr cfg.alpha cfg.eps cfg.clipVal cfg.momentum

  -- Data source: fresh batch each epoch (raw Doubles)
  let genBatch : IO (Vect (cfg.batch) (TwoPhaseDataPoint InputW OutputW Double))
      genBatch = copyTaskBinaryBatchVect {w = W} cfg.batch cfg.minLen cfg.maxLen

  -- Metrics: bit accuracy + memory (computed at each log step)
  let evalMetrics : Network InputW [] OutputW ExampleExecutor ExampleDType WithGrad -> IO (List (String, String))
      evalMetrics m = do
        evalBatch <- copyTaskBinaryBatchVect {w = W} 10 1 20
        accs <- traverse (\dp => do
                  (_, preds) <- forwardTwoPhase m dp
                  pure (bitAccuracy preds (targets dp))) evalBatch
        let avgAcc = foldl (+) 0.0 (toList accs) / 10.0
        pure [ ("acc", show (avgAcc * 100.0) ++ "%") ]

  -- HPO branch: --lr-find runs lr_find using one batch per iter (BCE).
  when cfg.lrFind $ do
    let lrCfg : LrFindConfig
        lrCfg = { numIters := 100 } defaultLrFindConfig
    _ <- lrFind lrCfg
      (\m, d => epochTwoPhaseVar opt d tbceLoss m)
      genBatch opt model
    putStrLn ""
    putStrLn "Done — re-run without --lr-find at the recommended LR."
    exitSuccess

  let trainCfgBase = mkTrainConfig cfg.epochs 100
                       (WindowedPercentile 0.10 cfg.esThreshold cfg.esWindow cfg.esPatience)
                       evalMetrics (\_ => pure ())
      trainCfg = case cfg.checkpointDir of
                   "" => trainCfgBase
                   dir => withCheckpoint
                            (fileCheckpoint dir cfg.checkpointEvery True opt)
                            trainCfgBase

  (trained, epochsDone, _) <- runTraining {ex=ExampleExecutor}
    (\m, d => epochTwoPhaseVar opt d tbceLoss m) genBatch trainCfg model

  -- Evaluation: forwardTwoPhase produces per-step Vector predictions
  -- directly from the trained model (no Double-network bridge).
  -- Eval doesn't need gradients; each evalOne runs in its own
  -- withNoGrad bracket so the exit drain fires per-sequence on mlx.
  let evalOne : TwoPhaseDataPoint InputW OutputW Double -> IO Double
      evalOne dp = withNoGrad {ex=ExampleExecutor} $ do
        (_, preds) <- forwardTwoPhase trained dp
        pure (bitAccuracy preds (targets dp))

  shortBatch <- copyTaskBinaryBatchVect {w = W} TestSize 1 5
  fullBatch <- copyTaskBinaryBatchVect {w = W} TestSize 1 20
  shortAccs <- traverse evalOne shortBatch
  fullAccs <- traverse evalOne fullBatch
  let shortAcc = foldl (+) 0.0 (toList shortAccs) / cast TestSize
      fullAcc  = foldl (+) 0.0 (toList fullAccs) / cast TestSize

  putStrLn ""
  putStrLn "Eval:"
  sampleBatch <- copyTaskBinaryBatchVect {w = W} 2 3 5
  withNoGrad {ex=ExampleExecutor} $ traverse_ (\dp => do
    (_, preds) <- forwardTwoPhase trained dp
    putStr "  Input:  "
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
