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
import Checkpoint
import Train
import Util
import Executor
import Tensor
import BuildConfig


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
catCELossVar : TVec OutputDim ExampleExecutor ExampleDType WithGrad -> TVec OutputDim ExampleExecutor ExampleDType WithGrad -> IO (Tensor [] ExampleExecutor ExampleDType WithGrad)
catCELossVar predV targetV = ioRerun (\_ =>
  let vsI = cast {to=Int} VocabSize
      sI = cast {to=Int} SeqLen
      skip = cast {to=Int} InputLen
      revLen = sI - skip
      -- Narrow at axis 0 with ROW indices: rows `skip..skip+revLen-1`
      -- of the [seqLen, vocab] reshape. The pre-bd61bef8 (2026-05-26)
      -- mlx/torch `primNarrow` flattened the tensor and treated start
      -- + length as 1D element counts — so `primNarrow logitsFull 0
      -- (skip * vsI) (revLen * vsI)` accidentally did the right thing
      -- (skip=5, vsI=8, revLen=6 → flat slice 40..87 = rows 5..10 of
      -- the [11, 8]). When bd61bef8 fixed `tensor_narrow` to honor
      -- the axis arg properly, this loss silently broke: row-axis
      -- narrow with start=40 length=48 on an 11-row tensor returns
      -- an empty array, then the downstream `primReshape2d ... revLen
      -- vsI` aborts with "Cannot reshape array of size 0 into shape
      -- (6, 8)". Fix: row indices instead of flat indices.
      logitsFull = primReshape2d {ex=ExampleExecutor} predV.tensorPtr sI vsI
      targetFull = primReshape2d {ex=ExampleExecutor} targetV.tensorPtr sI vsI
      logitsR = primNarrow {ex=ExampleExecutor} logitsFull 0 skip revLen
      logProbs = primLogSoftmax2d {ex=ExampleExecutor} logitsR
      tgtsR = primNarrow {ex=ExampleExecutor} targetFull 0 skip revLen
      product = primMul {ex=ExampleExecutor} logProbs tgtsR
      totalSum = primSum {ex=ExampleExecutor} product
      loss = primMulScalar {ex=ExampleExecutor} (primNeg {ex=ExampleExecutor} totalSum) (1.0 / cast {to=Double} revLen)
  in MkTensor loss Nothing)


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
        else let v = primItem1d {ex=ExampleExecutor} t (cast pos * cast vocabSize + k)
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
  checkpointDir : String
  checkpointEvery : Nat

defaultConfig : Config
defaultConfig = MkConfig 0.001 1000 300 42 False "" 50

specs : List (ArgSpec Config)
specs = [ Arg "--lr" (\v, c => { lr := cast v } c)
        , Arg "--epochs" (\v, c => { epochs := castNat v } c)
        , Arg "--patience" (\v, c => { patience := castNat v } c)
        , Arg "--seed" (\v, c => { seed := castBits64 v } c)
        , Arg "--lr-find" (\v, c => { lrFind := (v == "1" || v == "true") } c)
        , Arg "--checkpoint-dir" (\v, c => { checkpointDir := v } c)
        , Arg "--resume" (\v, c => { checkpointDir := v } c)
        , Arg "--checkpoint-every" (\v, c => { checkpointEvery := castNat v } c) ]


----------------------------------------------------------------------
-- Main
----------------------------------------------------------------------

partial
main : IO ()
main = do
  requireMachine {m = ChosenMachine}
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)

  srand cfg.seed
  tsetInitSeed {ex = ExampleExecutor} cfg.seed

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
  let model : Network InputDim [] OutputDim ExampleExecutor ExampleDType WithGrad
      model = OutputLayer tfmAny
  putStrLn ""

  let genBatch : IO (Vect BatchSize (TensorDataPoint InputDim OutputDim))
      genBatch = sortingTensorBatchVect InputDim OutputDim VocabSize InputLen SeqLen SepToken EosToken BatchSize

  -- Per-epoch metrics: accuracy on a fresh eval batch via single-sample forwardVar.
  let evalMetrics : Network InputDim [] OutputDim ExampleExecutor ExampleDType WithGrad -> IO (List (String, String))
      evalMetrics m = do
        evalData <- sortingTensorBatchVect InputDim OutputDim VocabSize InputLen SeqLen SepToken EosToken BatchSize
        results <- traverse (\dp => do
              let inV = the (TVec InputDim ExampleExecutor ExampleDType WithGrad) (MkTensor (inputTensor dp) Nothing)
              (_, predV) <- forwardVar m inV
              let predicted = map (argmaxAtPtr VocabSize predV.tensorPtr) positions
                  expected = map (argmaxAtPtr VocabSize (targetTensor dp)) positions
                  sortPred = drop InputLen predicted
                  sortExp = drop InputLen expected
              pure (countMatches sortPred sortExp)) evalData
        let totalCorrect = foldl (+) 0 (toList results)
            totalPositions = BatchSize * (SeqLen `minus` InputLen)
        pure [("sort_acc", show totalCorrect ++ "/" ++ show totalPositions)]

  let trainCfgBase = mkTrainConfig cfg.epochs 100 (Patience cfg.patience 0.001) evalMetrics (\_ => pure ())
      trainCfg = case cfg.checkpointDir of
                   "" => trainCfgBase
                   dir => withCheckpoint
                            (fileCheckpoint dir cfg.checkpointEvery True opt)
                            trainCfgBase

  when cfg.lrFind $ do
    let lrCfg : LrFindConfig
        lrCfg = { numIters := 100 } defaultLrFindConfig
    _ <- lrFind lrCfg
      (\m, d => epochVarTensorBatch opt d catCELossVar m)
      genBatch opt model
    putStrLn ""
    putStrLn "Done — re-run without --lr-find at the recommended LR."
    exitSuccess

  (trained, epochsDone, finalLoss) <- runTraining {ex=ExampleExecutor}
    (\m, d => epochVarTensorBatch opt d catCELossVar m) genBatch trainCfg model

  -- Single-sample eval
  putStrLn ""
  putStrLn "Evaluation:"
  evalRaw <- sortingTensorBatchVect InputDim OutputDim VocabSize InputLen SeqLen SepToken EosToken 1
  let tdp = index FZ evalRaw
      inV = the (TVec InputDim ExampleExecutor ExampleDType WithGrad) (MkTensor (inputTensor tdp) Nothing)
  (_, predV) <- forwardVar trained inV
  let inpT = inputTensor tdp
      inputDecoded = map (\p => cast {to=Nat} (cast {to=Integer} (primItem1d {ex=ExampleExecutor} inpT (cast p)))) positions
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
