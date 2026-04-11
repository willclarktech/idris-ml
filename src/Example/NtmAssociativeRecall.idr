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

defaultConfig : Config
defaultConfig = MkConfig 0.0001 10.0 0.95 1.0e-8 0.9 100000 0.01 1000 3 42 2 6 1

parseConfig : List String -> Config
parseConfig args = go args defaultConfig
  where
    go : List String -> Config -> Config
    go [] c = c
    go ("--lr" :: v :: rest) c = go rest ({ lr := cast v } c)
    go ("--clip" :: v :: rest) c = go rest ({ clipVal := cast v } c)
    go ("--alpha" :: v :: rest) c = go rest ({ alpha := cast v } c)
    go ("--eps" :: v :: rest) c = go rest ({ eps := cast v } c)
    go ("--momentum" :: v :: rest) c = go rest ({ momentum := cast v } c)
    go ("--epochs" :: v :: rest) c = go rest ({ epochs := cast (cast {to=Integer} v) } c)
    go ("--es-threshold" :: v :: rest) c = go rest ({ esThreshold := cast v } c)
    go ("--es-window" :: v :: rest) c = go rest ({ esWindow := cast (cast {to=Integer} v) } c)
    go ("--es-patience" :: v :: rest) c = go rest ({ esPatience := cast (cast {to=Integer} v) } c)
    go ("--seed" :: v :: rest) c = go rest ({ seed := cast (cast {to=Integer} v) } c)
    go ("--min-items" :: v :: rest) c = go rest ({ minItems := cast (cast {to=Integer} v) } c)
    go ("--max-items" :: v :: rest) c = go rest ({ maxItems := cast (cast {to=Integer} v) } c)
    go ("--batch" :: v :: rest) c = go rest ({ batch := cast (cast {to=Integer} v) } c)
    go (_ :: rest) c = go rest c


----------------------------------------------------------------------
-- Main
----------------------------------------------------------------------

main : IO ()
main = do
  tStart <- clockTime Monotonic
  args <- getArgs
  let cfg = parseConfig (drop 1 args)

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

  -- Data source: generate fresh batch each epoch
  let genBatch : IO (Vect (cfg.batch) (TwoPhaseDataPoint InputW OutputW Variable))
      genBatch = map (map fromDouble) <$> recallTaskBinaryBatchVect {w = W} cfg.batch cfg.minItems cfg.maxItems SeqLen

  -- Metrics: bit accuracy + memory
  let evalMetrics : Network InputW [] OutputW Variable -> IO (List (String, String))
      evalMetrics m = do
        let dblM = toDoubleNetwork (emap refreshValue m)
        evalBatch <- recallTaskBinaryBatchVect {w = W} 10 cfg.minItems cfg.maxItems SeqLen
        let avgAcc = foldl (+) 0.0
              (toList (map (\dp => let (_, preds) = forwardTwoPhase dblM dp
                                   in bitAccuracy preds (targets dp)) evalBatch)) / 10.0
        pure [ ("acc", show avgAcc)
             , ("peak", show (getRssMB 0) ++ "MB")
             , ("cur", show (getCurrentRssMB 0) ++ "MB") ]

  let trainCfg = MkTrainConfig cfg.epochs 100
                   (WindowedAvg cfg.esThreshold cfg.esWindow cfg.esPatience) evalMetrics

  (trained, epochsDone, _) <- runTraining
    (\m, d => epochTwoPhaseBceNative opt d m) genBatch trainCfg model
  t1 <- clockTime Monotonic

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
  putStrLn $ formatTimingSummary tStart t1 epochsDone
  putStrLn $ "RESULT\tepochs=" ++ show epochsDone
           ++ "\tacc_k2=" ++ show k2Acc
           ++ "\tacc_k4=" ++ show k4Acc
           ++ "\tacc_k6=" ++ show k6Acc
           ++ "\ttime=" ++ show (seconds t1 - seconds tStart) ++ "s"
           ++ "\tseed=" ++ show cfg.seed
