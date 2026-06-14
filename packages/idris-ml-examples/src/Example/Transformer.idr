-- | Transformer Sequence Sorting Example
-- |
-- | Sort a sequence of digits using a multi-block pre-norm transformer with
-- | learned embeddings, sinusoidal PE, multi-head causal self-attention, and
-- | layer normalization — assembled on the `Nn` models-as-records surface
-- | (`Nn.Embedding` + `Nn.transformerBlock` stacked in a `Seq` + bias-free
-- | head) and trained via the `fit` driver.
-- |
-- | Input (teacher-forced): [t0, t1, ..., t4, SEP, sorted_0, ..., sorted_4, EOS]
-- | Target: predict next token at each position.

module Example.Transformer

import Data.List
import Data.Vect
import System

import BuildConfig
import Checkpoint
import Compat.Random
import DataStream
import Executor
import Fit
import Generate
import Hpo.LrFinder
import Nn
import Optimizer
import Tensor
import Train
import Util

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

BatchSize : Nat
BatchSize = 16

----------------------------------------------------------------------
-- Model (Nn surface): embedding + sinusoidal PE + transformer blocks +
-- final norm + bias-free output head.
----------------------------------------------------------------------

public export
record TfmModel (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkTfmModel
  embed : Embedding VocabSize DModel ex dt g
  body  : Seq DModel DModel ex dt g
  headW : TMat VocabSize DModel ex dt g
  pe    : Tensor [SeqLen, DModel] ex dt g

Model : Type
Model = TfmModel ExampleExecutor ExampleDType WithGrad

-- NumBlocks pre-norm transformer blocks followed by the final LayerNorm.
buildBody : (k : Nat) -> Init (Seq DModel DModel ExampleExecutor ExampleDType WithGrad)
buildBody Z = do
  ln <- layerNorm {ex=ExampleExecutor} {dt=ExampleDType} {n=DModel}
  pure (ln :: Nil)
buildBody (S j) = do
  blk  <- scopedChild "block"
            (transformerBlock {ex=ExampleExecutor} {dt=ExampleDType}
                              {dModel=DModel} {numHeads=NumHeads} {headDim=HeadDim})
  rest <- buildBody j
  pure (blk :: rest)

partial
mkModel : IO Model
mkModel = do
  (emb, bdy, hw) <- runInit $ do
    e <- scoped "embed" (embedding {ex=ExampleExecutor} {dt=ExampleDType}
                                   {vocab=VocabSize} {embedDim=DModel})
    b <- buildBody NumBlocks
    hn <- freshChild "head"
    hw <- liftIO $ tparam2dNormal {ex=ExampleExecutor} {dt=ExampleDType}
                                  {o=VocabSize} {i=DModel}
                                  (hn ++ ".weight") 0.0 (1.0 / sqrt (cast {to=Double} DModel))
    pure (e, b, hw)
  pe <- sinusoidalPE {ex=ExampleExecutor} {dt=ExampleDType} {seqLen=SeqLen} {dModel=DModel}
  pure (MkTfmModel emb bdy hw pe)

-- Forward one sequence `[SeqLen]` → per-position logits `[SeqLen, VocabSize]`.
partial
tfmForward : Model -> Tensor [SeqLen] ExampleExecutor ExampleDType WithGrad ->
             IO (Tensor [SeqLen, VocabSize] ExampleExecutor ExampleDType WithGrad)
tfmForward (MkTfmModel emb body headW pe) tokens = do
  embFlat <- embeddingForward {ex=ExampleExecutor} {seqLen=SeqLen} {embedDim=DModel}
                              {vocab=VocabSize} emb tokens
  let sI = cast {to=Int} SeqLen
      dI = cast {to=Int} DModel
  emb2d <- ioRerun (\_ =>
    the (Tensor [SeqLen, DModel] ExampleExecutor ExampleDType WithGrad)
        (MkTensor (primReshape2d {ex=ExampleExecutor} embFlat.tensorPtr sI dI) Nothing))
  h0 <- tadd emb2d pe
  hN <- forwardSeq {b=SeqLen} body h0
  ioRerun (\_ =>
    MkTensor (primMm {ex=ExampleExecutor} hN.tensorPtr
                     (primTranspose2d {ex=ExampleExecutor} headW.tensorPtr)) Nothing)

----------------------------------------------------------------------
-- Data generation (sorting task)
----------------------------------------------------------------------

packDoubleBuf : AnyPtr -> Int -> List Int -> AnyPtr
packDoubleBuf buf _ []          = buf
packDoubleBuf buf off (x :: xs) =
  packDoubleBuf (prim__setDouble buf off (cast x)) (off + 1) xs

packIntBuf : AnyPtr -> Int -> List Int -> AnyPtr
packIntBuf buf _ []          = buf
packIntBuf buf off (x :: xs) =
  packIntBuf (prim__setInt buf off x) (off + 1) xs

-- One (input ids, one-hot target) sorting pair:
--   input  = [t0..t4, SEP, sorted_0..sorted_4]  (first SeqLen of the full seq)
--   target = the same shifted by 1 (next token), one-hot [SeqLen, VocabSize].
TfmSample : Type
TfmSample = ( Tensor [SeqLen] ExampleExecutor ExampleDType WithGrad
            , Tensor [SeqLen, VocabSize] ExampleExecutor ExampleDType WithGrad )

sortingSample : IO TfmSample
sortingSample = do
  tokens <- sequence (replicate InputLen (randomInt 0 (minus VocabSize 3)))
  let sorted = Data.List.sort tokens
      fullSeq    = tokens ++ [SepToken] ++ sorted ++ [EosToken]
      inputToks  = map (cast {to=Int} . cast {to=Integer}) (Data.List.take SeqLen fullSeq)
      targetToks = map (cast {to=Int} . cast {to=Integer}) (Data.List.take SeqLen (drop 1 fullSeq))
      sI         = cast {to=Int} SeqLen
      vI         = cast {to=Int} VocabSize
      inT        = dtCreate1d {ex=ExampleExecutor} {t=ExampleDType} sI (packDoubleBuf (prim__allocDoubles sI) 0 inputToks) 0 (deviceStreamTag {ex=ExampleExecutor})
      tgtIdxBuf  = packIntBuf (prim__allocInts sI) 0 targetToks
      tgtFlat    = primOneHot {ex=ExampleExecutor} tgtIdxBuf sI vI (dtypeTag {t=ExampleDType})
      tgt2d      = primReshape2d {ex=ExampleExecutor} tgtFlat sI vI
  pure (MkTensor inT Nothing, MkTensor tgt2d Nothing)

sortingBatch : (n : Nat) -> IO (Vect n TfmSample)
sortingBatch Z     = pure []
sortingBatch (S k) = do
  s    <- sortingSample
  rest <- sortingBatch k
  pure (s :: rest)

----------------------------------------------------------------------
-- Per-position categorical cross-entropy (sorted portion only)
----------------------------------------------------------------------

-- CE on `[SeqLen, VocabSize]` logits + one-hot target, masked to the
-- sorted/output portion (rows InputLen..SeqLen-1) so the random-prefix
-- positions don't contribute. Mirrors the legacy `catCELossVar`.
tfmLoss : Tensor [SeqLen, VocabSize] ExampleExecutor ExampleDType WithGrad ->
          Tensor [SeqLen, VocabSize] ExampleExecutor ExampleDType WithGrad ->
          IO (Tensor [] ExampleExecutor ExampleDType WithGrad)
tfmLoss logits targets = ioRerun (\_ =>
  let sI     = cast {to=Int} SeqLen
      skip     = cast {to=Int} InputLen
      revLen   = sI - skip
      logitsR  = primNarrow {ex=ExampleExecutor} logits.tensorPtr 0 skip revLen
      logProbs = primLogSoftmax2d {ex=ExampleExecutor} logitsR
      tgtsR    = primNarrow {ex=ExampleExecutor} targets.tensorPtr 0 skip revLen
      product  = primMul {ex=ExampleExecutor} logProbs tgtsR
      totalSum = primSum {ex=ExampleExecutor} product
      loss     = primMulScalar {ex=ExampleExecutor} (primNeg {ex=ExampleExecutor} totalSum) (1.0 / cast {to=Double} revLen)
  in MkTensor loss Nothing)

-- Mean loss over a batch.
partial
batchLoss : Model -> Vect BatchSize TfmSample -> IO (Tensor [] ExampleExecutor ExampleDType WithGrad)
batchLoss model batch = do
  losses <- traverse (\(ids, tgt) => do logits <- tfmForward model ids; tfmLoss logits tgt) (toList batch)
  zero   <- tparamScalar {ex=ExampleExecutor} {dt=ExampleDType} "tfm.epoch_zero" 0.0
  summed <- foldlM (\acc, l => tadd acc l) zero losses
  tmulScalar summed (1.0 / cast {to=Double} BatchSize)

----------------------------------------------------------------------
-- Helpers (decoding / accuracy)
----------------------------------------------------------------------

tokenName : Nat -> String
tokenName n = if n < 6
  then show n
  else if n == SepToken then "|"
  else if n == EosToken then "$"
  else "?"

||| Argmax over vocabSize logits at a given position, reading directly
||| from a (row-major) tensor pointer.
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
  lr              : Double
  epochs          : Nat
  patience        : Nat
  seed            : Bits64
  lrFind          : Bool
  checkpointDir   : String
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

  let positions = map finToNat (toList (Data.Vect.Fin.range {len=SeqLen}))

  putStrLn "=== Transformer: Sequence Sorting ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr ++ " epochs=" ++ show cfg.epochs
           ++ " patience=" ++ show cfg.patience ++ " seed=" ++ show cfg.seed
  putStrLn $ "Architecture: seqLen=" ++ show SeqLen ++ " dModel=" ++ show DModel
           ++ " heads=" ++ show NumHeads ++ " headDim=" ++ show HeadDim
           ++ " blocks=" ++ show NumBlocks ++ " vocab=" ++ show VocabSize

  model <- mkModel
  -- Adam with global grad-clip norm 1.0 (was nativeAdamGlobalClip).
  opt <- adam {ex=ExampleExecutor} cfg.lr ({ clip := NormClip 1.0 } defaultOpts)
  putStrLn ""

  -- One fused training step (used by both fit and lrFind).
  let stepFn : Model -> Vect BatchSize TfmSample -> IO (Model, Double)
      stepFn m b = do
        loss <- batchLoss m b
        d    <- nativeTrainStep opt loss
        pure (m, d)

  -- Per-epoch metrics: sorted-portion accuracy on a fresh eval batch.
  let evalMetrics : Model -> IO (List (String, String))
      evalMetrics m = do
        evalData <- sortingBatch BatchSize
        results <- traverse (\(ids, tgt) => do
              predV <- tfmForward m ids
              let predicted = map (argmaxAtPtr VocabSize predV.tensorPtr) positions
                  expected = map (argmaxAtPtr VocabSize tgt.tensorPtr) positions
                  sortPred = drop InputLen predicted
                  sortExp  = drop InputLen expected
              pure (countMatches sortPred sortExp)) (toList evalData)
        let totalCorrect = foldl (+) 0 results
            totalPositions = BatchSize * (SeqLen `minus` InputLen)
        pure [("sort_acc", show totalCorrect ++ "/" ++ show totalPositions)]

  let trainCfgBase = mkTrainConfig cfg.epochs 100 (Patience cfg.patience 0.001) evalMetrics (\_ => pure ())
      trainCfg = case cfg.checkpointDir of
                   ""  => trainCfgBase
                   dir => withCheckpoint
                            (fileCheckpoint dir cfg.checkpointEvery True opt)
                            trainCfgBase

  when cfg.lrFind $ do
    let lrCfg : LrFindConfig
        lrCfg = { numIters := 100 } defaultLrFindConfig
    _ <- lrFind lrCfg stepFn (sortingBatch BatchSize) opt model
    putStrLn ""
    putStrLn "Done — re-run without --lr-find at the recommended LR."
    exitSuccess

  let stream = generate (sortingBatch BatchSize)

  (trained, epochsDone, finalLoss) <-
    fitSupervised {ex=ExampleExecutor} opt batchLoss stream trainCfg model

  -- Single-sample eval
  putStrLn ""
  putStrLn "Evaluation:"
  (inIds, tgt) <- sortingSample
  predV <- tfmForward trained inIds
  let inputDecoded  = map (\p => cast {to=Nat} (cast {to=Integer} (primItem1d {ex=ExampleExecutor} inIds.tensorPtr (cast p)))) positions
      targetDecoded = map (argmaxAtPtr VocabSize tgt.tensorPtr) positions
      predicted     = map (argmaxAtPtr VocabSize predV.tensorPtr) positions
      sortCorrect   = countMatches (drop InputLen predicted) (drop InputLen targetDecoded)
      sortTotal     = SeqLen `minus` InputLen

  let inputTokens = Data.List.take InputLen inputDecoded
      sortTarget    = drop InputLen targetDecoded
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
