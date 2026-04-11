-- | NTM Associative Recall Task
-- |
-- | Binary vector recall task with LSTM controller, interpolation write,
-- | sigmoid output + BCE loss, and RMSprop optimizer.
-- |
-- | Architecture: NtmLayer (LSTM controller, separate head FCs from
-- | cell state, output FC from hidden ++ read_output) -> sigmoid.
-- | Data: binary vectors with item/query delimiters (two-phase training).

module Example.NtmAssociativeRecall

import Data.List
import Data.String
import Data.Vect
import System
import System.Clock
import System.Random

import Backprop
import DataPoint
import Debug
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
-- Configuration
----------------------------------------------------------------------

||| Binary vector width (data channels per timestep)
W : Nat
W = 6

||| Timesteps per item
SeqLen : Nat
SeqLen = 3

||| Input width = data + item_delim + query_delim
InputW : Nat
InputW = S (S W)

||| Output width = data channels only
OutputW : Nat
OutputW = W

||| Number of memory slots
N : Nat
N = 128

||| Memory width
M : Nat
M = 20

||| Controller hidden size
H : Nat
H = 100

||| Training batch size (data points per chunk)
BatchSize : Nat
BatchSize = 16

||| Evaluation batch size
TestSize : Nat
TestSize = 20


----------------------------------------------------------------------
-- Training Loop
----------------------------------------------------------------------

||| Simple training loop with periodic data regeneration and
||| windowed-average convergence-based early stopping.
||| Returns (model, epochs completed).
trainLoop :
  NativeOptimizer ->
  Network InputW [] OutputW Variable ->
  (totalEpochs : Nat) -> (esThreshold : Double) -> (esWindow : Nat) -> (esPatience : Nat) ->
  (minItems, maxItems : Nat) ->
  Clock Monotonic ->
  IO (Network InputW [] OutputW Variable, Nat)
trainLoop opt model totalEpochs esThreshold esWindow esPatience minItems maxItems t0 =
  go 0 model 0.0 0 [] 0
  where
    wc : Nat
    wc = max 1 (div esWindow 100)

    go : Nat -> Network InputW [] OutputW Variable ->
         Double -> Nat -> List Double -> Nat ->
         IO (Network InputW [] OutputW Variable, Nat)
    go ep m iSum iCount avgs convCount =
      if ep >= totalEpochs then pure (m, ep)
      else do
        batch <- recallTaskBinaryBatchVect {w = W} BatchSize minItems maxItems SeqLen
        let dps = map (map fromDouble) batch
            (m', loss) = epochTwoPhaseBceNative opt dps m
        when (modNatNZ ep 100 ItIsSucc == 0) $ do
          now <- clockTime Monotonic
          let dblModel = toDoubleNetwork (emap refreshValue m')
          evalBatch <- recallTaskBinaryBatchVect {w = W} 10 minItems maxItems SeqLen
          let accs = map (\dp => let (_, preds) = forwardTwoPhase dblModel dp
                                 in bitAccuracy preds (targets dp)) evalBatch
          let avgAcc = foldl (+) 0.0 (toList accs) / 10.0
          putStrLn $ "  " ++ formatElapsed t0 now ++ " " ++ show ep ++ "\tloss=" ++ show loss
                   ++ "\tacc=" ++ show avgAcc
                   ++ "\tpeak=" ++ show (getRssMB ep) ++ "MB"
                   ++ "\tcur=" ++ show (getCurrentRssMB ep) ++ "MB"
        if loss /= loss
          then do
            now <- clockTime Monotonic
            putStrLn $ "  " ++ formatElapsed t0 now ++ " Diverged (NaN) at epoch " ++ show ep
            pure (m', ep)
          else do
            let iSum' = iSum + loss
                iCount' = iCount + 1
            if iCount' < 100
              then go (ep + 1) m' iSum' iCount' avgs convCount
              else do
                let avg = iSum' / 100.0
                    avgs' = avg :: avgs
                if length avgs' < wc
                  then go (ep + 1) m' 0.0 0 avgs' convCount
                  else do
                    let windowAvg = foldl (+) 0.0 (take wc avgs') / cast wc
                    if windowAvg >= esThreshold
                      then go (ep + 1) m' 0.0 0 avgs' 0
                      else do
                        let cc = convCount + 1
                        if cc >= esPatience
                          then do
                            now <- clockTime Monotonic
                            putStrLn $ "  " ++ formatElapsed t0 now
                                     ++ " Converged at epoch " ++ show (ep + 1)
                                     ++ " (window_avg=" ++ show windowAvg ++ ")"
                            pure (m', ep + 1)
                          else do
                            now <- clockTime Monotonic
                            putStrLn $ "    " ++ formatElapsed t0 now
                                     ++ " convergence " ++ show cc ++ "/" ++ show esPatience
                                     ++ " (window_avg=" ++ show windowAvg ++ ")"
                            go (ep + 1) m' 0.0 0 avgs' cc


||| Training loop with batch_size=1 (online learning) and
||| windowed-average convergence-based early stopping.
||| Generates 1 sequence per epoch for higher gradient noise.
||| Logs every 500 epochs (vs 100 for batched).
trainLoop1 :
  NativeOptimizer ->
  Network InputW [] OutputW Variable ->
  (totalEpochs : Nat) -> (esThreshold : Double) -> (esWindow : Nat) -> (esPatience : Nat) ->
  (minItems, maxItems : Nat) ->
  Clock Monotonic ->
  IO (Network InputW [] OutputW Variable, Nat)
trainLoop1 opt model totalEpochs esThreshold esWindow esPatience minItems maxItems t0 =
  go 0 model 0.0 0 [] 0
  where
    wc : Nat
    wc = max 1 (div esWindow 100)

    go : Nat -> Network InputW [] OutputW Variable ->
         Double -> Nat -> List Double -> Nat ->
         IO (Network InputW [] OutputW Variable, Nat)
    go ep m iSum iCount avgs convCount =
      if ep >= totalEpochs then pure (m, ep)
      else do
        batch <- recallTaskBinaryBatchVect {w = W} 1 minItems maxItems SeqLen
        let dps = map (map fromDouble) batch
            (m', loss) = epochTwoPhaseBceNative opt dps m
        when (modNatNZ ep 100 ItIsSucc == 0) $ do
          now <- clockTime Monotonic
          -- Quick eval for bit accuracy
          let dblModel = toDoubleNetwork (emap refreshValue m')
          evalBatch <- recallTaskBinaryBatchVect {w = W} 10 minItems maxItems SeqLen
          let accs = map (\dp => let (_, preds) = forwardTwoPhase dblModel dp
                                 in bitAccuracy preds (targets dp)) evalBatch
          let avgAcc = foldl (+) 0.0 (toList accs) / 10.0
          putStrLn $ "  " ++ formatElapsed t0 now ++ " " ++ show ep ++ "\tloss=" ++ show loss
                   ++ "\tacc=" ++ show avgAcc
                   ++ "\tpeak=" ++ show (getRssMB ep) ++ "MB"
                   ++ "\tcur=" ++ show (getCurrentRssMB ep) ++ "MB"
        if loss /= loss
          then do
            now <- clockTime Monotonic
            putStrLn $ "  " ++ formatElapsed t0 now ++ " Diverged (NaN) at epoch " ++ show ep
            pure (m', ep)
          else do
            let iSum' = iSum + loss
                iCount' = iCount + 1
            if iCount' < 100
              then go (ep + 1) m' iSum' iCount' avgs convCount
              else do
                let avg = iSum' / 100.0
                    avgs' = avg :: avgs
                if length avgs' < wc
                  then go (ep + 1) m' 0.0 0 avgs' convCount
                  else do
                    let windowAvg = foldl (+) 0.0 (take wc avgs') / cast wc
                    if windowAvg >= esThreshold
                      then go (ep + 1) m' 0.0 0 avgs' 0
                      else do
                        let cc = convCount + 1
                        if cc >= esPatience
                          then do
                            now <- clockTime Monotonic
                            putStrLn $ "  " ++ formatElapsed t0 now
                                     ++ " Converged at epoch " ++ show (ep + 1)
                                     ++ " (window_avg=" ++ show windowAvg ++ ")"
                            pure (m', ep + 1)
                          else do
                            now <- clockTime Monotonic
                            putStrLn $ "    " ++ formatElapsed t0 now
                                     ++ " convergence " ++ show cc ++ "/" ++ show esPatience
                                     ++ " (window_avg=" ++ show windowAvg ++ ")"
                            go (ep + 1) m' 0.0 0 avgs' cc


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
           ++ " alpha=" ++ show cfg.alpha
           ++ " momentum=" ++ show cfg.momentum
           ++ " epochs=" ++ show cfg.epochs
           ++ " seed=" ++ show cfg.seed
           ++ " batch=" ++ show cfg.batch
           ++ " items=" ++ show cfg.minItems ++ "-" ++ show cfg.maxItems
           ++ " seqLen=" ++ show SeqLen
  putStrLn $ "Early stopping: threshold=" ++ show cfg.esThreshold
           ++ " window=" ++ show cfg.esWindow
           ++ " patience=" ++ show cfg.esPatience
  putStrLn $ "Architecture: N=" ++ show N ++ " M=" ++ show M ++ " H=" ++ show H
  putStrLn ""

  -- Build NTM (no output activation; loss is BCEWithLogits)
  ntm <- ntmLayer {inputSize = InputW, outputSize = OutputW, n = N, m = M, h = H}
  let model = autoName $ OutputLayer ntm

  putStrLn $ "Model: " ++ show model
  putStrLn ""

  -- Training
  let opt = nativeRmsprop cfg.lr cfg.alpha cfg.eps cfg.clipVal cfg.momentum
  putStrLn "Training..."
  t0 <- clockTime Monotonic
  (trained, epochsDone) <-
    if cfg.batch == 1
      then trainLoop1 opt model
             cfg.epochs cfg.esThreshold cfg.esWindow cfg.esPatience cfg.minItems cfg.maxItems t0
      else trainLoop opt model
             cfg.epochs cfg.esThreshold cfg.esWindow cfg.esPatience cfg.minItems cfg.maxItems t0

  putStrLn $ "Training complete: " ++ show epochsDone ++ " epochs"
  putStrLn ""

  -- Evaluation (refresh Variable values from tensors after native optimizer)
  let dblModel = toDoubleNetwork (emap refreshValue trained)

  let evalOne : TwoPhaseDataPoint InputW OutputW Double -> Double
      evalOne dp =
        let (_, preds) = forwardTwoPhase dblModel dp
        in bitAccuracy preds (targets dp)

  k2Batch <- recallTaskBinaryBatchVect {w = W} TestSize 2 2 SeqLen
  k4Batch <- recallTaskBinaryBatchVect {w = W} TestSize 4 4 SeqLen
  k6Batch <- recallTaskBinaryBatchVect {w = W} TestSize 6 6 SeqLen
  let k2Accs = map evalOne k2Batch
  let k4Accs = map evalOne k4Batch
  let k6Accs = map evalOne k6Batch
  let k2Acc = foldl (+) 0.0 (toList k2Accs) / cast TestSize
  let k4Acc = foldl (+) 0.0 (toList k4Accs) / cast TestSize
  let k6Acc = foldl (+) 0.0 (toList k6Accs) / cast TestSize

  t1 <- clockTime Monotonic

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
