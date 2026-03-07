-- | NTM Associative Recall Task (PyTorch-aligned)
-- |
-- | Binary vector recall task with LSTM controller, interpolation write,
-- | sigmoid output + BCE loss, and RMSprop optimizer. Matches the
-- | PyTorch reference in pytorch/torch_ref/.
-- |
-- | Architecture: NtmLayer (LSTM controller, separate head FCs from
-- | cell state, output FC from hidden ++ read_output) -> sigmoid.
-- | Data: binary vectors with item/query delimiters (two-phase training).

module Example.NtmAssociativeRecall

import Data.List
import Data.String
import Data.Vect
import System
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
  (totalEpochs : Nat) -> (patience : Nat) ->
  (minItems, maxItems : Nat) ->
  DenseOptimizerState ->
  IO (Network InputW [] OutputW Variable, DenseOptimizerState, Nat)
trainLoop makeOpt schedule model totalEpochs patience minItems maxItems st =
  go 0 model st (1.0/0.0) 0
  where
    go : Nat -> Network InputW [] OutputW Variable -> DenseOptimizerState ->
         Double -> Nat ->
         IO (Network InputW [] OutputW Variable, DenseOptimizerState, Nat)
    go ep m s bestLoss staleCount =
      if ep >= totalEpochs then pure (m, s, ep)
      else do
        batch <- recallTaskBinaryBatchVect {w = W} BatchSize minItems maxItems SeqLen
        let dps = map (map fromDouble) batch
            lr = schedule ep
            opt = makeOpt lr
            (m', s', loss) = epochTwoPhaseDense opt dps binaryCrossEntropyWithLogits m s
        when (modNatNZ ep 10 ItIsSucc == 0) forceGC
        when (modNatNZ ep 100 ItIsSucc == 0) $
          putStrLn $ "  " ++ show ep ++ ":\tloss=" ++ show loss
        if loss /= loss
          then do
            putStrLn $ "  Diverged (NaN) at epoch " ++ show ep
            pure (m', s', ep)
          else do
            let improved = loss < bestLoss - 0.001
                bestLoss' = if improved then loss else bestLoss
                sc : Nat
                sc = if improved then 0 else staleCount + 1
            if patience > 0 && sc >= patience
              then do
                putStrLn $ "  Early stop at epoch " ++ show (ep + 1)
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
  epochs : Nat
  patience : Nat
  seed : Bits64
  minItems : Nat
  maxItems : Nat

defaultConfig : Config
defaultConfig = MkConfig 0.0001 10.0 0.95 1.0e-8 100000 5000 123456 2 6

parseConfig : List String -> Config
parseConfig args = go args defaultConfig
  where
    go : List String -> Config -> Config
    go [] c = c
    go ("--lr" :: v :: rest) c = go rest ({ lr := cast v } c)
    go ("--clip" :: v :: rest) c = go rest ({ clipVal := cast v } c)
    go ("--alpha" :: v :: rest) c = go rest ({ alpha := cast v } c)
    go ("--eps" :: v :: rest) c = go rest ({ eps := cast v } c)
    go ("--epochs" :: v :: rest) c = go rest ({ epochs := cast (cast {to=Integer} v) } c)
    go ("--patience" :: v :: rest) c = go rest ({ patience := cast (cast {to=Integer} v) } c)
    go ("--seed" :: v :: rest) c = go rest ({ seed := cast (cast {to=Integer} v) } c)
    go ("--min-items" :: v :: rest) c = go rest ({ minItems := cast (cast {to=Integer} v) } c)
    go ("--max-items" :: v :: rest) c = go rest ({ maxItems := cast (cast {to=Integer} v) } c)
    go (_ :: rest) c = go rest c


----------------------------------------------------------------------
-- Main
----------------------------------------------------------------------

main : IO ()
main = do
  args <- getArgs
  let cfg = parseConfig (drop 1 args)

  srand cfg.seed

  putStrLn "=== NTM Associative Recall (PyTorch-aligned) ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr
           ++ " clip=" ++ show cfg.clipVal
           ++ " alpha=" ++ show cfg.alpha
           ++ " epochs=" ++ show cfg.epochs
           ++ " patience=" ++ show cfg.patience
           ++ " seed=" ++ show cfg.seed
           ++ " items=" ++ show cfg.minItems ++ "-" ++ show cfg.maxItems
           ++ " seqLen=" ++ show SeqLen
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
  let makeOpt = \lr => rmspropValueClipDense lr cfg.alpha cfg.eps cfg.clipVal
  let schedule = constant cfg.lr
  let st0 = initDenseState numPids
  putStrLn "Training..."
  (trained, finalSt, epochsDone) <- trainLoop makeOpt schedule model
    cfg.epochs cfg.patience cfg.minItems cfg.maxItems st0

  putStrLn $ "Training complete: " ++ show epochsDone ++ " epochs"
  putStrLn ""

  -- Evaluation
  let dblModel = toDoubleNetwork trained

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

  putStrLn "Eval (random binary sequences):"
  putStr "  K=2 items:\t"
  putStrLn $ show k2Acc
  putStr "  K=4 items:\t"
  putStrLn $ show k4Acc
  putStr "  K=6 items:\t"
  putStrLn $ show k6Acc

  -- Machine-readable result line
  putStrLn $ "RESULT\t"
           ++ show cfg.lr ++ "\t"
           ++ show cfg.clipVal ++ "\t"
           ++ show cfg.alpha ++ "\t"
           ++ show cfg.epochs ++ "\t"
           ++ show cfg.patience ++ "\t"
           ++ show epochsDone ++ "\t"
           ++ show cfg.seed ++ "\t"
           ++ show k2Acc ++ "\t"
           ++ show k4Acc ++ "\t"
           ++ show k6Acc
