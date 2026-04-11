-- | Transformer Sequence Reversal Example
-- |
-- | Sequence reversal with learned embeddings, sinusoidal positional
-- | encoding, multi-head causal self-attention, and layer normalization.
-- |
-- | Input (teacher-forced): [t0, t1, ..., t4, SEP, t4, ..., t0, EOS]
-- | Target: predict next token at each position.

module Example.Transformer

import Data.List
import Data.Vect
import Decidable.Equality
import System
import System.Random

import Backprop
import DataPoint
import Endofunctor
import Floating
import Generate
import Layer
import Layer.Core
import Layer.MultiHeadTransformer
import Math
import Optimizer
import Tensor
import Train
import Util
import Variable


----------------------------------------------------------------------
-- Configuration
----------------------------------------------------------------------

VocabSize : Nat
VocabSize = 10         -- 8 content tokens (A-H) + SEP + EOS

InputLen : Nat
InputLen = 5           -- tokens before separator

-- Full sequence: [t0..t4, SEP, t4..t0, EOS] = 12 tokens
-- Teacher forcing: input = first 11, target = last 11
SeqLen : Nat
SeqLen = 2 * InputLen + 1

DModel : Nat
DModel = 32

NumHeads : Nat
NumHeads = 4

HeadDim : Nat
HeadDim = 8

SepToken : Nat
SepToken = 8

EosToken : Nat
EosToken = 9

InputDim : Nat
InputDim = SeqLen * VocabSize

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
             Vector (seqLen * vocabSize) Variable -> Vector (seqLen * vocabSize) Variable -> Variable
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

catCELoss : LossFunction Variable
catCELoss {n} preds targets =
  case decEq n (SeqLen * VocabSize) of
    Yes Refl => reversalCE {seqLen=SeqLen, vocabSize=VocabSize} (cast InputLen) preds targets
    No _ => fromDouble 0.0  -- unreachable


----------------------------------------------------------------------
-- Helpers
----------------------------------------------------------------------

tokenName : Nat -> String
tokenName n = if n < 8
  then strCons (cast (cast {to=Int} n + 65)) ""  -- A=0, B=1, ...
  else if n == SepToken then "|"
  else if n == EosToken then "$"
  else "?"

||| Extract Variable values in forward order (avoids Tensor toList reversal).
tensorVals : {n : Nat} -> Vector n Variable -> List Double
tensorVals (VTensor xs) = go xs
  where
    go : Vect k (Scalar Variable) -> List Double
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
defaultConfig = MkConfig 0.001 500 200 42

specs : List (ArgSpec Config)
specs = [ Arg "--lr" (\v, c => { lr := cast v } c)
        , Arg "--epochs" (\v, c => { epochs := castNat v } c)
        , Arg "--patience" (\v, c => { patience := castNat v } c)
        , Arg "--seed" (\v, c => { seed := castBits64 v } c) ]


----------------------------------------------------------------------
-- Main
----------------------------------------------------------------------

main : IO ()
main = do
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)

  srand cfg.seed

  let opt = nativeAdamGlobalClip cfg.lr 0.9 0.999 1.0e-8 1.0
      positions = map finToNat (toList (Data.Vect.Fin.range {len=SeqLen}))

  putStrLn "=== Transformer: Sequence Reversal ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr ++ " epochs=" ++ show cfg.epochs
           ++ " patience=" ++ show cfg.patience ++ " seed=" ++ show cfg.seed
  putStrLn $ "Architecture: seqLen=" ++ show SeqLen ++ " dModel=" ++ show DModel
           ++ " heads=" ++ show NumHeads ++ " headDim=" ++ show HeadDim
           ++ " vocab=" ++ show VocabSize

  mht <- transformerLayer {seqLen=SeqLen, dModel=DModel, numHeads=NumHeads,
                            headDim=HeadDim, vocabSize=VocabSize}
  let model = autoName $ OutputLayer mht
  putStrLn $ "Model: " ++ show model
  putStrLn ""

  -- Data source: fresh batch each epoch
  let genBatch : IO (Vect BatchSize (DataPoint InputDim OutputDim Variable))
      genBatch = map (map fromDouble) <$>
        reversalBatchVect VocabSize InputLen SeqLen SepToken EosToken BatchSize

  -- Metrics: accuracy on a fresh eval batch
  let evalMetrics : Network InputDim [] OutputDim Variable -> IO (List (String, String))
      evalMetrics m = do
        raw <- reversalBatchVect VocabSize InputLen SeqLen SepToken EosToken BatchSize
        let evalData : Vect BatchSize (DataPoint InputDim OutputDim Variable)
            evalData = map (map fromDouble) raw
            results = map (\dp =>
              let (_, pred) = forwardVar m (x dp)
                  predVals = tensorVals pred
                  targetVals = tensorVals (y dp)
                  predicted = map (argmaxAt VocabSize predVals) positions
                  expected = map (argmaxAt VocabSize targetVals) positions
                  revPred = drop InputLen predicted
                  revExp = drop InputLen expected
              in countMatches revPred revExp) evalData
            totalRevCorrect = foldl (+) 0 (toList results)
            totalRevPositions = BatchSize * (SeqLen `minus` InputLen)
        pure [("rev_acc", show totalRevCorrect ++ "/" ++ show totalRevPositions)]

  let trainCfg = MkTrainConfig cfg.epochs 100 (Patience cfg.patience 0.001) evalMetrics

  (trained, epochsDone, finalLoss) <- runTraining
    (\m, d => epochNative opt d catCELoss m) genBatch trainCfg model

  -- Evaluate on a fresh example
  putStrLn ""
  putStrLn "Evaluation:"
  evalRaw <- reversalBatchVect VocabSize InputLen SeqLen SepToken EosToken 1
  let dp = the (DataPoint InputDim OutputDim Variable) (map fromDouble (index FZ evalRaw))
      (_, pred) = forwardVar trained (x dp)
      predVals = tensorVals pred
      inputVals = tensorVals (x dp)
      targetVals = tensorVals (y dp)
      inputDecoded = map (argmaxAt VocabSize inputVals) positions
      targetDecoded = map (argmaxAt VocabSize targetVals) positions
      predicted = map (argmaxAt VocabSize predVals) positions
      revCorrect = countMatches (drop InputLen predicted) (drop InputLen targetDecoded)
      revTotal = SeqLen `minus` InputLen

  let inputTokens = Data.List.take InputLen inputDecoded
      revTarget = drop InputLen targetDecoded
      revPredicted = drop InputLen predicted
  putStr "  Input:      "
  putStrLn $ concatMap tokenName inputTokens
  putStr "  Target:     "
  putStrLn $ concatMap tokenName revTarget
  putStr "  Predicted:  "
  putStrLn $ concatMap tokenName revPredicted
  putStrLn $ "  Rev acc:    " ++ show revCorrect ++ "/" ++ show revTotal

  putStrLn ""
  putStrLn $ formatResult [("epochs", show epochsDone),
                            ("rev_acc", show revCorrect ++ "/" ++ show revTotal),
                            ("seed", show cfg.seed)]
