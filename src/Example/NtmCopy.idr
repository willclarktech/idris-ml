-- | NTM Copy Task (PyTorch-aligned)
-- |
-- | Binary vector copy task with LSTM controller, interpolation write,
-- | sigmoid output + BCE loss, and RMSprop optimizer. Matches the
-- | PyTorch reference in pytorch/torch_ref/.
-- |
-- | Architecture: NtmLayer (LSTM controller, separate head FCs from
-- | cell state, output FC from hidden ++ read_output) -> sigmoid.
-- | Data: binary vectors with delimiter channel (two-phase training).

module Example.NtmCopy

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
import Schedule
import Tensor
import Util
import Variable


----------------------------------------------------------------------
-- Configuration
----------------------------------------------------------------------

||| Binary vector width (data channels)
W : Nat
W = 8

||| Input width = data + delimiter channel
InputW : Nat
InputW = S W

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
-- Evaluation Helpers
----------------------------------------------------------------------

||| Apply sigmoid to a Double value.
sigD : Double -> Double
sigD x = 1.0 / (1.0 + exp (-x))

||| Count (correct, total) bits in one prediction-target pair.
countBits : {w : Nat} -> Vector w Double -> Vector w Double -> (Nat, Nat)
countBits (VTensor ps) (VTensor ts) = go ps ts 0 0
  where
    go : Vect k (Tensor [] Double) -> Vect k (Tensor [] Double) -> Nat -> Nat -> (Nat, Nat)
    go [] [] c t = (c, t)
    go (STensor p :: ps') (STensor tgt :: ts') c t =
      let predBit = if sigD p >= 0.5 then 1.0 else 0.0
          match : Nat
          match = if predBit == tgt then 1 else 0
      in go ps' ts' (c + match) (t + 1)

||| Compute bit accuracy: fraction of correctly predicted bits.
||| Predictions are thresholded at 0.5 after sigmoid.
bitAccuracy : {w : Nat} -> List (Vector w Double) -> List (Vector w Double) -> Double
bitAccuracy preds targets =
  let results = zipWith countBits preds targets
      totalCorrect = foldl (\acc, (c, _) => acc + c) (the Nat 0) results
      totalBits = foldl (\acc, (_, t) => acc + t) (the Nat 0) results
  in if totalBits == 0 then 0.0 else cast totalCorrect / cast totalBits


----------------------------------------------------------------------
-- Training Loop
----------------------------------------------------------------------

||| Training loop with native libtorch optimizer and early stopping.
trainLoop :
  NativeOptimizer ->
  Network InputW [] OutputW Variable ->
  (totalEpochs : Nat) -> (esThreshold : Double) -> (esWindow : Nat) -> (esPatience : Nat) ->
  (batchSize : Nat) ->
  (minLen, maxLen : Nat) ->
  Clock Monotonic ->
  IO (Network InputW [] OutputW Variable, Nat)
trainLoop opt model totalEpochs esThreshold esWindow esPatience batchSize minLen maxLen t0 =
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
        batch <- copyTaskBinaryBatchVect {w = W} batchSize minLen maxLen
        let dps = map (map fromDouble) batch
            (m', loss) = epochTwoPhaseBceNative opt dps m
        when (modNatNZ ep 100 ItIsSucc == 0) $ do
          now <- clockTime Monotonic
          putStrLn $ "  " ++ formatElapsed t0 now ++ " " ++ show ep ++ ":\tloss=" ++ show loss
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
  minLen : Nat
  maxLen : Nat
  batch : Nat

defaultConfig : Config
defaultConfig = MkConfig 0.0001 10.0 0.95 1.0e-8 0.9 50000 0.01 1000 3 42 1 20 1

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
    go ("--min-len" :: v :: rest) c = go rest ({ minLen := cast (cast {to=Integer} v) } c)
    go ("--max-len" :: v :: rest) c = go rest ({ maxLen := cast (cast {to=Integer} v) } c)
    go ("--batch" :: v :: rest) c = go rest ({ batch := cast (cast {to=Integer} v) } c)
    go (_ :: rest) c = go rest c


----------------------------------------------------------------------
-- Main
----------------------------------------------------------------------

main : IO ()
main = do
  args <- getArgs
  let cfg = parseConfig (drop 1 args)

  srand cfg.seed

  putStrLn "=== NTM Copy Task (PyTorch-aligned) ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr
           ++ " clip=" ++ show cfg.clipVal
           ++ " alpha=" ++ show cfg.alpha
           ++ " momentum=" ++ show cfg.momentum
           ++ " epochs=" ++ show cfg.epochs
           ++ " seed=" ++ show cfg.seed
           ++ " batch=" ++ show cfg.batch
           ++ " seqLen=" ++ show cfg.minLen ++ "-" ++ show cfg.maxLen
  putStrLn $ "Early stopping: threshold=" ++ show cfg.esThreshold
           ++ " window=" ++ show cfg.esWindow
           ++ " patience=" ++ show cfg.esPatience
  putStrLn $ "Architecture: N=" ++ show N ++ " M=" ++ show M ++ " H=" ++ show H
  putStrLn ""

  -- Build NTM (no output activation; loss is BCEWithLogits)
  ntm <- ntmLayer {inputSize = InputW, outputSize = OutputW, n = N, m = M, h = H}
  let model = autoName $ OutputLayer ntm

  putStr "Model:\t\t"
  printLn model
  putStrLn ""

  -- Training
  let opt = nativeRmsprop cfg.lr cfg.alpha cfg.eps cfg.clipVal cfg.momentum
  putStrLn $ "Training (batch=" ++ show cfg.batch ++ ")..."
  t0 <- clockTime Monotonic
  (trained, epochsDone) <-
    trainLoop opt model
      cfg.epochs cfg.esThreshold cfg.esWindow cfg.esPatience
      cfg.batch cfg.minLen cfg.maxLen t0

  putStrLn $ "Training complete: " ++ show epochsDone ++ " epochs"
  putStrLn ""

  -- Evaluation: refresh Variable values from tensors (optimizer mutates in-place)
  let dblModel = toDoubleNetwork (emap refreshValue trained)

  let evalOne : TwoPhaseDataPoint InputW OutputW Double -> Double
      evalOne dp =
        let (_, preds) = forwardTwoPhase dblModel dp
        in bitAccuracy preds (targets dp)

  shortBatch <- copyTaskBinaryBatchVect {w = W} TestSize 1 5
  fullBatch <- copyTaskBinaryBatchVect {w = W} TestSize 1 20
  let shortAccs = map evalOne shortBatch
  let fullAccs = map evalOne fullBatch
  let shortAcc = foldl (+) 0.0 (toList shortAccs) / cast TestSize
  let fullAcc = foldl (+) 0.0 (toList fullAccs) / cast TestSize

  putStrLn "Eval (random binary sequences):"
  putStr "  Short (len 1-5):\t"
  putStrLn $ show shortAcc
  putStr "  Full (len 1-20):\t"
  putStrLn $ show fullAcc

  -- Machine-readable result line
  putStrLn $ "RESULT\t"
           ++ show cfg.lr ++ "\t"
           ++ show cfg.clipVal ++ "\t"
           ++ show cfg.alpha ++ "\t"
           ++ show cfg.epochs ++ "\t"
           ++ show cfg.esThreshold ++ "\t"
           ++ show epochsDone ++ "\t"
           ++ show cfg.seed ++ "\t"
           ++ show shortAcc ++ "\t"
           ++ show fullAcc
