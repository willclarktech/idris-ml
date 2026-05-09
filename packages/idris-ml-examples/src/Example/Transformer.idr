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
import Floating
import Generate
import Hpo.LrFinder
import Layer.Core
import Layer.Transformer
import Math
import Array
import Train
import Util
import Device
import Tensor


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
DModel = 16

NumHeads : Nat
NumHeads = 4

HeadDim : Nat
HeadDim = 4

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

|||  typed-surface CE loss on per-sample logits + target [seqLen *
||| vocabSize]. Masks the random-prefix positions so only the reversal
||| portion contributes (V1 `reversalCE` parity, returning a Tensor [] CPU).
catCELossVar : TVec OutputDim CPU -> TVec OutputDim CPU -> Tensor [] CPU
catCELossVar predV targetV =
  let vsI = cast {to=Int} VocabSize
      sI = cast {to=Int} SeqLen
      skip = cast {to=Int} InputLen
      revLen = sI - skip
      logitsFull = prim__reshape2d predV.tensorPtr sI vsI
      targetFull = prim__reshape2d targetV.tensorPtr sI vsI
      logits = prim__narrow logitsFull 0 (skip * vsI) (revLen * vsI)
      logitsR = prim__reshape2d logits revLen vsI
      logProbs = prim__logSoftmax2d logitsR
      tgts = prim__narrow targetFull 0 (skip * vsI) (revLen * vsI)
      tgtsR = prim__reshape2d tgts revLen vsI
      product = prim__mul logProbs tgtsR
      totalSum = prim__sum product
      loss = prim__mulScalar (prim__neg totalSum) (1.0 / cast {to=Double} revLen)
  in MkTensor loss Nothing


----------------------------------------------------------------------
-- Helpers
----------------------------------------------------------------------

tokenName : Nat -> String
tokenName n = if n < 6
  then show n
  else if n == SepToken then "|"
  else if n == EosToken then "$"
  else "?"

||| Argmax over vocabSize logits at a given position, reading directly
||| from a tensor pointer.
argmaxAtPtr : (vocabSize : Nat) -> AnyPtr -> Nat -> Nat
argmaxAtPtr vocabSize t pos =
  let scan : Int -> Nat -> Double -> Nat
      scan k bestI bestV =
        if k >= cast {to=Int} vocabSize then bestI
        else let v = prim__item1d t (cast pos * cast vocabSize + k)
             in if v > bestV
                  then assert_total $ scan (k + 1) (cast k) v
                  else assert_total $ scan (k + 1) bestI bestV
  in scan 0 0 (-1.0e10)

countMatches : List Nat -> List Nat -> Nat
countMatches xs ys = foldl (\acc, (a, b) => if a == b then acc + 1 else acc) 0 (zip xs ys)


----------------------------------------------------------------------
-- CLI
----------------------------------------------------------------------

record Config where
  constructor MkConfig
  lr : Double
  epochs : Nat
  patience : Nat
  seed : Bits64
  lrFind : Bool

defaultConfig : Config
defaultConfig = MkConfig 0.001 1000 300 42 False

specs : List (ArgSpec Config)
specs = [ Arg "--lr" (\v, c => { lr := cast v } c)
        , Arg "--epochs" (\v, c => { epochs := castNat v } c)
        , Arg "--patience" (\v, c => { patience := castNat v } c)
        , Arg "--seed" (\v, c => { seed := castBits64 v } c)
        , Arg "--lr-find" (\v, c => { lrFind := (v == "1" || v == "true") } c) ]


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

  tfmAny <- transformerLayerAny
              {seqLen=SeqLen, dModel=DModel, numHeads=NumHeads,
               headDim=HeadDim, numBlocks=NumBlocks, vocabSize=VocabSize}
              "tfm0"
  let model : Network InputDim [] OutputDim CPU
      model = OutputLayer tfmAny
  putStrLn ""

  let genBatch : IO (Vect BatchSize (TensorDataPoint InputDim OutputDim))
      genBatch = sortingTensorBatchVect InputDim OutputDim VocabSize InputLen SeqLen SepToken EosToken BatchSize

  -- Per-epoch metrics: accuracy on a fresh eval batch via single-sample forwardVar.
  let evalMetrics : Network InputDim [] OutputDim CPU -> IO (List (String, String))
      evalMetrics m = do
        evalData <- sortingTensorBatchVect InputDim OutputDim VocabSize InputLen SeqLen SepToken EosToken BatchSize
        let results = map (\dp =>
              let inV = the (TVec InputDim CPU) (MkTensor (inputTensor dp) Nothing)
                  (_, predV) = forwardVar m inV
                  predicted = map (argmaxAtPtr VocabSize predV.tensorPtr) positions
                  expected = map (argmaxAtPtr VocabSize (targetTensor dp)) positions
                  sortPred = drop InputLen predicted
                  sortExp = drop InputLen expected
              in countMatches sortPred sortExp) evalData
            totalCorrect = foldl (+) 0 (toList results)
            totalPositions = BatchSize * (SeqLen `minus` InputLen)
        pure [("sort_acc", show totalCorrect ++ "/" ++ show totalPositions)]

  let trainCfg = MkTrainConfig cfg.epochs 100 (Patience cfg.patience 0.001) evalMetrics (\_ => pure ())

  when cfg.lrFind $ do
    let lrCfg : LrFindConfig
        lrCfg = { numIters := 100 } defaultLrFindConfig
    _ <- lrFind lrCfg
      (\m, d => let (m', loss) = epochVarTensorBatch opt d catCELossVar m
                in pure (m', loss))
      genBatch opt model
    putStrLn ""
    putStrLn "Done — re-run without --lr-find at the recommended LR."
    exitSuccess

  (trained, epochsDone, finalLoss) <- runTraining
    (\m, d => epochVarTensorBatch opt d catCELossVar m) genBatch trainCfg model

  -- Single-sample eval
  putStrLn ""
  putStrLn "Evaluation:"
  evalRaw <- sortingTensorBatchVect InputDim OutputDim VocabSize InputLen SeqLen SepToken EosToken 1
  let tdp = index FZ evalRaw
      inV = the (TVec InputDim CPU) (MkTensor (inputTensor tdp) Nothing)
      (_, predV) = forwardVar trained inV
      inpT = inputTensor tdp
      inputDecoded = map (\p => cast {to=Nat} (cast {to=Integer} (prim__item1d inpT (cast p)))) positions
      targetDecoded = map (argmaxAtPtr VocabSize (targetTensor tdp)) positions
      predicted = map (argmaxAtPtr VocabSize predV.tensorPtr) positions
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
