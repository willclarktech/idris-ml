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

||| Simple training loop with periodic data regeneration.
||| Returns (model, optimizer state, epochs completed).
trainLoop :
  (Double -> DenseOptimizer) -> Schedule ->
  Network InputW [] OutputW Variable ->
  (totalEpochs : Nat) -> (patience : Nat) -> (batchSize : Nat) ->
  (minLen, maxLen : Nat) ->
  DenseOptimizerState ->
  Clock Monotonic ->
  IO (Network InputW [] OutputW Variable, DenseOptimizerState, Nat)
trainLoop makeOpt schedule model totalEpochs patience batchSize minLen maxLen st t0 =
  go 0 model st (1.0/0.0) 0
  where
    go : Nat -> Network InputW [] OutputW Variable -> DenseOptimizerState ->
         Double -> Nat ->
         IO (Network InputW [] OutputW Variable, DenseOptimizerState, Nat)
    go ep m s bestLoss staleCount =
      if ep >= totalEpochs then pure (m, s, ep)
      else do
        batch <- copyTaskBinaryBatchVect {w = W} batchSize minLen maxLen
        let dps = map (map fromDouble) batch
            lr = schedule ep
            opt = makeOpt lr
            (m', s', loss) = epochTwoPhaseDenseBce opt dps m s
        when (modNatNZ ep 10 ItIsSucc == 0) forceGC
        when (modNatNZ ep 100 ItIsSucc == 0) $ do
          now <- clockTime Monotonic
          putStrLn $ "  " ++ formatElapsed t0 now ++ " " ++ show ep ++ ":\tloss=" ++ show loss
                   ++ "\tpeak=" ++ show (getRssMB ep) ++ "MB"
                   ++ "\tcur=" ++ show (getCurrentRssMB ep) ++ "MB"
        if loss /= loss
          then do
            now <- clockTime Monotonic
            putStrLn $ "  " ++ formatElapsed t0 now ++ " Diverged (NaN) at epoch " ++ show ep
            pure (m', s', ep)
          else do
            let improved = loss < bestLoss - 0.001
                bestLoss' = if improved then loss else bestLoss
                sc : Nat
                sc = if improved then 0 else staleCount + 1
            if patience > 0 && sc >= patience
              then do
                now <- clockTime Monotonic
                putStrLn $ "  " ++ formatElapsed t0 now ++ " Early stop at epoch " ++ show (ep + 1)
                         ++ " (patience=" ++ show patience ++ ")"
                pure (m', s', ep + 1)
              else go (ep + 1) m' s' bestLoss' sc


||| Training loop with batch_size=1 (online learning).
||| Generates 1 sequence per epoch for higher gradient noise.
trainLoop1 :
  (Double -> DenseOptimizer) -> Schedule ->
  Network InputW [] OutputW Variable ->
  (totalEpochs : Nat) -> (patience : Nat) ->
  (minLen, maxLen : Nat) ->
  DenseOptimizerState ->
  Clock Monotonic ->
  IO (Network InputW [] OutputW Variable, DenseOptimizerState, Nat)
trainLoop1 makeOpt schedule model totalEpochs patience minLen maxLen st t0 =
  go 0 model st (1.0/0.0) 0
  where
    go : Nat -> Network InputW [] OutputW Variable -> DenseOptimizerState ->
         Double -> Nat ->
         IO (Network InputW [] OutputW Variable, DenseOptimizerState, Nat)
    go ep m s bestLoss staleCount =
      if ep >= totalEpochs then pure (m, s, ep)
      else do
        batch <- copyTaskBinaryBatchVect {w = W} 1 minLen maxLen
        let dps = map (map fromDouble) batch
            lr = schedule ep
            opt = makeOpt lr
            (m', s', loss) = epochTwoPhaseDenseBce opt dps m s
        when (modNatNZ ep 10 ItIsSucc == 0) forceGC
        when (modNatNZ ep 500 ItIsSucc == 0) $ do
          now <- clockTime Monotonic
          putStrLn $ "  " ++ formatElapsed t0 now ++ " " ++ show ep ++ ":\tloss=" ++ show loss
                   ++ "\tpeak=" ++ show (getRssMB ep) ++ "MB"
                   ++ "\tcur=" ++ show (getCurrentRssMB ep) ++ "MB"
        if loss /= loss
          then do
            now <- clockTime Monotonic
            putStrLn $ "  " ++ formatElapsed t0 now ++ " Diverged (NaN) at epoch " ++ show ep
            pure (m', s', ep)
          else do
            let improved = loss < bestLoss - 0.001
                bestLoss' = if improved then loss else bestLoss
                sc : Nat
                sc = if improved then 0 else staleCount + 1
            if patience > 0 && sc >= patience
              then do
                now <- clockTime Monotonic
                putStrLn $ "  " ++ formatElapsed t0 now ++ " Early stop at epoch " ++ show (ep + 1)
                         ++ " (patience=" ++ show patience ++ ")"
                pure (m', s', ep + 1)
              else go (ep + 1) m' s' bestLoss' sc


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
  patience : Nat
  seed : Bits64
  minLen : Nat
  maxLen : Nat
  batch : Nat

defaultConfig : Config
defaultConfig = MkConfig 0.0001 10.0 0.95 1.0e-8 0.9 50000 1000 42 1 20 16

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
    go ("--patience" :: v :: rest) c = go rest ({ patience := cast (cast {to=Integer} v) } c)
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
           ++ " patience=" ++ show cfg.patience
           ++ " seed=" ++ show cfg.seed
           ++ " batch=" ++ show cfg.batch
           ++ " seqLen=" ++ show cfg.minLen ++ "-" ++ show cfg.maxLen
  putStrLn $ "Architecture: N=" ++ show N ++ " M=" ++ show M ++ " H=" ++ show H
  putStrLn ""

  -- Build NTM (no output activation; loss is BCEWithLogits)
  ntm <- ntmLayer {inputSize = InputW, outputSize = OutputW, n = N, m = M, h = H}
  let model = autoName $ OutputLayer ntm

  putStr "Model:\t\t"
  printLn model
  putStrLn ""

  -- Training
  let numPids = getNumPids 0
  let makeOpt = \lr => rmspropValueClipMomentumDense lr cfg.alpha cfg.eps cfg.clipVal cfg.momentum
  let schedule = constant cfg.lr
  let st0 = initDenseState numPids
  putStrLn $ "Training (batch=" ++ show cfg.batch ++ ")..."
  t0 <- clockTime Monotonic
  (trained, finalSt, epochsDone) <-
    if cfg.batch == 1
      then trainLoop1 makeOpt schedule model
             cfg.epochs cfg.patience cfg.minLen cfg.maxLen st0 t0
      else trainLoop makeOpt schedule model
             cfg.epochs cfg.patience cfg.batch cfg.minLen cfg.maxLen st0 t0

  putStrLn $ "Training complete: " ++ show epochsDone ++ " epochs"
  putStrLn ""

  -- Evaluation (sync C buffer values back to Variable records for toDoubleNetwork)
  let dblModel = toDoubleNetwork (readFromBuffersNetwork trained)

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
           ++ show cfg.patience ++ "\t"
           ++ show epochsDone ++ "\t"
           ++ show cfg.seed ++ "\t"
           ++ show shortAcc ++ "\t"
           ++ show fullAcc
