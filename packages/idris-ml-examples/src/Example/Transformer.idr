-- | Transformer Sequence Sorting Example
-- |
-- | Sort a sequence of digits using a multi-block transformer with
-- | learned embeddings, sinusoidal PE, multi-head causal self-attention,
-- | and layer normalization.
-- |
-- | Input (teacher-forced): [t0, t1, ..., t4, SEP, sorted_0, ..., sorted_4, EOS]
-- | Target: predict next token at each position.

module Example.Transformer

import Data.List
import Data.Vect
import Decidable.Equality
import System
import Compat.Random

import Backprop
import DataPoint
import Endofunctor
import Floating
import Generate
import Layer
import Layer.Core
import Layer.Transformer
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

VocabSize : Nat
VocabSize = 8          -- digits 0-5 + SEP + EOS

InputLen : Nat
InputLen = 5           -- tokens to sort

-- Full sequence: [t0..t4, SEP, sorted_0..sorted_4, EOS] = 12 tokens
-- Teacher forcing: input = first 11, target = last 11
SeqLen : Nat
SeqLen = 2 * InputLen + 1

DModel : Nat
DModel = 32

NumHeads : Nat
NumHeads = 4

HeadDim : Nat
HeadDim = 8

NumBlocks : Nat
NumBlocks = 2

SepToken : Nat
SepToken = 6

EosToken : Nat
EosToken = 7

InputDim : Nat
InputDim = SeqLen

OutputDim : Nat
OutputDim = SeqLen * VocabSize

BatchSize : Nat
BatchSize = 16


----------------------------------------------------------------------
-- Per-position categorical cross-entropy loss (reversal portion only)
----------------------------------------------------------------------

-- Number of positions in the reversal portion (SEP + reversed + EOS)
ReversalLen : Nat
ReversalLen = SeqLen `minus` InputLen

||| Categorical cross-entropy on the reversal portion only.
||| Positions 0..InputLen-1 are random prefix (unpredictable from left
||| context), so we mask them out of the loss. Only positions
||| InputLen..SeqLen-1 (separator, reversed tokens, EOS) contribute.
reversalCE : {seqLen, vocabSize : Nat} -> (skipPositions : Int) ->
             Vector (seqLen * vocabSize) (Variable CPU) -> Vector (seqLen * vocabSize) (Variable CPU) -> Variable CPU
reversalCE {seqLen} {vocabSize} skip (VTensor preds) (VTensor targets) =
  let vsI = cast {to=Int} vocabSize
      sI = cast {to=Int} seqLen
      revLen = sI - skip
      -- Reshape to [seqLen, vocabSize], then narrow to reversal rows
      logitsFull = prim__reshape2d (vecStackTensor preds) sI vsI
      targetFull = prim__reshape2d (vecStackTensor targets) sI vsI
      logits = prim__narrow logitsFull 0 (skip * vsI) (revLen * vsI)
      logitsR = prim__reshape2d logits revLen vsI
      logProbs = prim__logSoftmax2d logitsR
      tgts = prim__narrow targetFull 0 (skip * vsI) (revLen * vsI)
      tgtsR = prim__reshape2d tgts revLen vsI
      product = prim__mul logProbs tgtsR
      totalSum = prim__sum product
      loss = prim__mulScalar (prim__neg totalSum) (1.0 / cast {to=Double} revLen)
      val = prim__item loss
  in Var loss Nothing val

catCELoss : LossFunction (Variable CPU)
catCELoss {n} preds targets =
  case decEq n (SeqLen * VocabSize) of
    Yes Refl => reversalCE {seqLen=SeqLen, vocabSize=VocabSize} (cast InputLen) preds targets
    No _ => fromDouble 0.0  -- unreachable

||| Tensor-level loss: takes raw AnyPtr tensors (pred, target), both 1D [seqLen*vocabSize].
catCELossTensor : LossFnTensor CPU
catCELossTensor predT targetT =
  let vsI = cast {to=Int} VocabSize
      sI = cast {to=Int} SeqLen
      skip = cast {to=Int} InputLen
      revLen = sI - skip
      logitsFull = prim__reshape2d predT sI vsI
      targetFull = prim__reshape2d targetT sI vsI
      logits = prim__narrow logitsFull 0 (skip * vsI) (revLen * vsI)
      logitsR = prim__reshape2d logits revLen vsI
      logProbs = prim__logSoftmax2d logitsR
      tgts = prim__narrow targetFull 0 (skip * vsI) (revLen * vsI)
      tgtsR = prim__reshape2d tgts revLen vsI
      product = prim__mul logProbs tgtsR
      totalSum = prim__sum product
      loss = prim__mulScalar (prim__neg totalSum) (1.0 / cast {to=Double} revLen)
      val = prim__item loss
  in Var loss Nothing val


----------------------------------------------------------------------
-- Helpers
----------------------------------------------------------------------

tokenName : Nat -> String
tokenName n = if n < 6
  then show n                   -- digits 0-5
  else if n == SepToken then "|"
  else if n == EosToken then "$"
  else "?"

||| Extract Variable values in forward order (avoids Tensor toList reversal).
tensorVals : {n : Nat} -> Vector n (Variable CPU) -> List Double
tensorVals (VTensor xs) = go xs
  where
    go : Vect k (Scalar (Variable CPU)) -> List Double
    go [] = []
    go (STensor v :: rest) = prim__item v.tensorPtr :: go rest

||| Argmax over vocabSize logits at a given position.
argmaxAt : (vocabSize : Nat) -> List Double -> Nat -> Nat
argmaxAt vocabSize vals pos =
  let listAt : Nat -> List Double -> Double
      listAt _ [] = 0.0
      listAt Z (xx :: _) = xx
      listAt (S k) (_ :: xs) = listAt k xs
      probs = map (\j => listAt (pos * vocabSize + j) vals)
                  (map finToNat (toList (Data.Vect.Fin.range {len=vocabSize})))
      best = foldl (\(bi,bv), (i,v) => if v > bv then (i,v) else (bi,bv))
                   (the (Nat, Double) (0, -1.0e10))
                   (zip (map finToNat (toList (Data.Vect.Fin.range {len=vocabSize}))) probs)
  in fst best

||| Count matching elements in two lists.
countMatches : List Nat -> List Nat -> Nat
countMatches xs ys = foldl (\acc, (a,b) => if a == b then acc + 1 else acc) 0 (zip xs ys)


----------------------------------------------------------------------
-- CLI
----------------------------------------------------------------------

record Config where
  constructor MkConfig
  lr : Double
  epochs : Nat
  patience : Nat
  seed : Bits64

defaultConfig : Config
defaultConfig = MkConfig 0.001 1000 300 42

specs : List (ArgSpec Config)
specs = [ Arg "--lr" (\v, c => { lr := cast v } c)
        , Arg "--epochs" (\v, c => { epochs := castNat v } c)
        , Arg "--patience" (\v, c => { patience := castNat v } c)
        , Arg "--seed" (\v, c => { seed := castBits64 v } c) ]


----------------------------------------------------------------------
-- Main
----------------------------------------------------------------------

partial
main : IO ()
main = do
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)

  srand cfg.seed

  let opt = nativeAdamGlobalClip cfg.lr 0.9 0.999 1.0e-8 1.0
      positions = map finToNat (toList (Data.Vect.Fin.range {len=SeqLen}))

  putStrLn "=== Transformer: Sequence Sorting ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr ++ " epochs=" ++ show cfg.epochs
           ++ " patience=" ++ show cfg.patience ++ " seed=" ++ show cfg.seed
  putStrLn $ "Architecture: seqLen=" ++ show SeqLen ++ " dModel=" ++ show DModel
           ++ " heads=" ++ show NumHeads ++ " headDim=" ++ show HeadDim
           ++ " blocks=" ++ show NumBlocks ++ " vocab=" ++ show VocabSize

  tfm <- mkTransformer {seqLen=SeqLen, dModel=DModel, numHeads=NumHeads,
                         headDim=HeadDim, numBlocks=NumBlocks, vocabSize=VocabSize}
  -- Name once — use the same named state for both model and batch forward
  let namedTfm = nameLayer "tfm0" tfm
      model = OutputLayer (MkAnyLayer
        (TransformerState SeqLen DModel NumHeads HeadDim NumBlocks VocabSize) namedTfm)
  putStrLn $ "Model: " ++ show model
  putStrLn ""

  putStrLn ""

  -- Data source: fresh batch each epoch (pre-allocated C tensors, zero conversion)
  let genBatch : IO (Vect BatchSize (TensorDataPoint InputDim OutputDim))
      genBatch = sortingTensorBatchVect InputDim OutputDim VocabSize InputLen SeqLen SepToken EosToken BatchSize

  -- Metrics: accuracy on a fresh eval batch (uses per-sequence forward)
  let evalMetrics : Network InputDim [] OutputDim (Variable CPU) -> IO (List (String, String))
      evalMetrics m = do
        evalData <- sortingTensorBatchVect InputDim OutputDim VocabSize InputLen SeqLen SepToken EosToken BatchSize
        let results = map (\dp =>
              let (_, outT) = forwardVarTensor m (inputTensor dp)
                  predVals = tensorVals (VTensor (tensorToScalars outT 0 OutputDim))
                  targetVals = tensorVals (VTensor (tensorToScalars (targetTensor dp) 0 OutputDim))
                  predicted = map (argmaxAt VocabSize predVals) positions
                  expected = map (argmaxAt VocabSize targetVals) positions
                  sortPred = drop InputLen predicted
                  sortExp = drop InputLen expected
              in countMatches sortPred sortExp) evalData
            totalCorrect = foldl (+) 0 (toList results)
            totalPositions = BatchSize * (SeqLen `minus` InputLen)
        pure [("sort_acc", show totalCorrect ++ "/" ++ show totalPositions)]

  let trainCfg = MkTrainConfig cfg.epochs 100 (Patience cfg.patience 0.001) evalMetrics

  let batchFwd = transformerForwardBatch namedTfm
  (trained, epochsDone, finalLoss) <- runTraining
    (\m, d => epochNativeTensorBatch opt d batchFwd catCELossTensor m) genBatch trainCfg model

  -- Evaluate on a fresh example
  putStrLn ""
  putStrLn "Evaluation:"
  evalRaw <- sortingTensorBatchVect InputDim OutputDim VocabSize InputLen SeqLen SepToken EosToken 1
  let tdp = index FZ evalRaw
      (_, outT) = forwardVarTensor trained (inputTensor tdp)
      predVals = tensorVals (VTensor (tensorToScalars outT 0 OutputDim))
      targetVals = tensorVals (VTensor (tensorToScalars (targetTensor tdp) 0 OutputDim))
      -- Input is token indices — read directly from tensor
      inpT = inputTensor tdp
      inputDecoded = map (\p => cast {to=Nat} (cast {to=Integer} (prim__item1d inpT (cast p)))) positions
      targetDecoded = map (argmaxAt VocabSize targetVals) positions
      predicted = map (argmaxAt VocabSize predVals) positions
      sortCorrect = countMatches (drop InputLen predicted) (drop InputLen targetDecoded)
      sortTotal = SeqLen `minus` InputLen

  let inputTokens = Data.List.take InputLen inputDecoded
      sortTarget = drop InputLen targetDecoded
      sortPredicted = drop InputLen predicted
  putStr "  Input:      "
  putStrLn $ concatMap tokenName inputTokens
  putStr "  Target:     "
  putStrLn $ concatMap tokenName sortTarget
  putStr "  Predicted:  "
  putStrLn $ concatMap tokenName sortPredicted
  putStrLn $ "  Sort acc:   " ++ show sortCorrect ++ "/" ++ show sortTotal

  putStrLn ""
  putStrLn $ formatResult [("epochs", show epochsDone),
                            ("sort_acc", show sortCorrect ++ "/" ++ show sortTotal),
                            ("seed", show cfg.seed)]
