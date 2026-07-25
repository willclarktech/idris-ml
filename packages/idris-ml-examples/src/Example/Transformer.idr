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

import Control.Linear.LIO
import Data.Linear.Notation
import Data.List
import Data.Vect
import System

import Ml.Checkpoint
import Ml.Compat.Random
import Ml.DataStream
import Ml.Executor
import Ml.Fit
import Ml.Hpo.LrFinder
import Ml.Nn
import Ml.Optimizer
import Ml.Tensor
import Ml.Train
import Ml.Util

import BuildConfig
import Generate

-- The transformer body is a linear `Seq`; hide the IO `Nn.Seq` constructors
-- (same `Nil`/`::` names) so the block-stack builder resolves to Seq.

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

-- Mixed field multiplicity by role: the `body` (the threaded sub-model) is a
-- **linear** `Seq` field; `embed`/`headW`/`pe` are read-only ω fields applied
-- once per forward. The body is stateless, but threading it linearly keeps the
-- single-owner discipline uniform across every forward (Option 3).
public export
record TfmModel (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkTfmModel
  embed  : Embedding VocabSize DModel ex dt g
  1 body : Seq DModel DModel ex dt g
  headW : TMat VocabSize DModel ex dt g
  pe    : Tensor [SeqLen, DModel] ex dt g

Model : Type
Model = TfmModel ExampleExecutor ExampleDType WithGrad

-- NumBlocks pre-norm transformer blocks followed by the final LayerNorm,
-- assembled as a linear `Seq`.
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

-- Single `Init Model` (PE folded in via `liftIO`) so the model can be born
-- linear with `runInitL` (and IO with `runInit` for the lrFind path).
partial
mkModelInit : Init Model
mkModelInit = do
  e <- scoped "embed" (embedding {ex=ExampleExecutor} {dt=ExampleDType}
                                 {vocab=VocabSize} {embedDim=DModel})
  b <- buildBody NumBlocks
  hn <- freshChild "head"
  hw <- liftIO $ tparam2dNormal {ex=ExampleExecutor} {dt=ExampleDType}
                                {o=VocabSize} {i=DModel}
                                (hn ++ ".weight") 0.0 (1.0 / sqrt (cast {to=Double} DModel))
  pe <- liftIO $ sinusoidalPE {ex=ExampleExecutor} {dt=ExampleDType} {seqLen=SeqLen} {dModel=DModel}
  pure (MkTfmModel e b hw pe)

-- Forward one sequence `[SeqLen]` → per-position logits `[SeqLen, VocabSize]`,
-- threading the linear `body` through `forwardSeq`. The ω `emb`/`pe`/`headW`
-- are read by projection. Returns the logits (banged) beside the rebuilt body.
partial
tfmForwardL : Embedding VocabSize DModel ExampleExecutor ExampleDType WithGrad ->
              Tensor [SeqLen, DModel] ExampleExecutor ExampleDType WithGrad ->
              TMat VocabSize DModel ExampleExecutor ExampleDType WithGrad ->
              (1 _ : Seq DModel DModel ExampleExecutor ExampleDType WithGrad) ->
              Tensor [SeqLen] ExampleExecutor ExampleDType WithGrad ->
              L IO {use = 1} (LPair (!* (Tensor [SeqLen, VocabSize] ExampleExecutor ExampleDType WithGrad))
                                    (Seq DModel DModel ExampleExecutor ExampleDType WithGrad))
tfmForwardL emb pe headW body tokens = do
  embFlat <- liftIO1 (embeddingForward {ex=ExampleExecutor} {seqLen=SeqLen} {embedDim=DModel}
                                       {vocab=VocabSize} emb tokens)
  let sI = cast {to=Int} SeqLen
      dI = cast {to=Int} DModel
  emb2d <- liftIO1 (ioRerun (\_ =>
    the (Tensor [SeqLen, DModel] ExampleExecutor ExampleDType WithGrad)
        (MkTensor (primReshape2d {ex=ExampleExecutor} embFlat.tensorPtr sI dI) Nothing)))
  h0 <- liftIO1 (tadd emb2d pe)
  (MkBang hN # body') <- forwardSeq {b=SeqLen} body h0
  out <- liftIO1 (ioRerun (\_ =>
    MkTensor (primMm {ex=ExampleExecutor} hN.tensorPtr
                     (primTranspose2d {ex=ExampleExecutor} headW.tensorPtr)) Nothing))
  pure1 (MkBang out # body')

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

-- Mean loss over a batch, fine-grained: match `MkTfmModel` (body linear, the
-- rest ω), thread the body through each sample's `tfmForwardL`, accumulate the
-- (ω) per-sample losses, mean-reduce, rebuild the model with the final body.
partial
batchLossL : (1 _ : Model) -> Vect BatchSize TfmSample ->
             L IO {use = 1} (LPair (!* (Tensor [] ExampleExecutor ExampleDType WithGrad)) Model)
batchLossL (MkTfmModel emb body headW pe) batch = do
  (MkBang losses # body') <- foldSamples emb headW pe body (toList batch) []
  loss <- liftIO1 $ do
            zero   <- tparamScalar {ex=ExampleExecutor} {dt=ExampleDType} "tfm.epoch_zero" 0.0
            summed <- foldlM (\acc, l => tadd acc l) zero losses
            tmulScalar summed (1.0 / cast {to=Double} BatchSize)
  pure1 (MkBang loss # MkTfmModel emb body' headW pe)
  where
    foldSamples : Embedding VocabSize DModel ExampleExecutor ExampleDType WithGrad ->
                  TMat VocabSize DModel ExampleExecutor ExampleDType WithGrad ->
                  Tensor [SeqLen, DModel] ExampleExecutor ExampleDType WithGrad ->
                  (1 _ : Seq DModel DModel ExampleExecutor ExampleDType WithGrad) ->
                  List TfmSample -> List (Tensor [] ExampleExecutor ExampleDType WithGrad) ->
                  L IO {use = 1} (LPair (!* (List (Tensor [] ExampleExecutor ExampleDType WithGrad)))
                                        (Seq DModel DModel ExampleExecutor ExampleDType WithGrad))
    foldSamples _ _  _ body []                acc    = pure1 (MkBang (reverse acc) # body)
    foldSamples e hw p body ((ids, tgt) :: rest) acc = do
      (MkBang logits # body') <- tfmForwardL e p hw body ids
      l <- liftIO1 (tfmLoss logits tgt)
      foldSamples e hw p body' rest (l :: acc)

-- One fused linear training step (loss + optimizer step), for lrFind.
partial
stepL : Optimizer ExampleExecutor -> (1 _ : Model) -> Vect BatchSize TfmSample ->
        L IO {use = 1} (LPair (!* Double) Model)
stepL opt m b = do
  (MkBang loss # m') <- batchLossL m b
  d <- liftIO1 (trainStep opt loss)
  pure1 (MkBang d # m')

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
-- Final eval (consumes the trained linear model)
----------------------------------------------------------------------

-- Consume the trained (linear) model: match `MkTfmModel …` (body linear, rest
-- ω), thread the body through one eval forward (tfmForwardL), discard it, then
-- decode + RESULT-report from the (ω) logits.
partial
evalReportL : List Nat -> Config -> Nat -> (1 _ : Model) -> L IO ()
evalReportL positions cfg epochsDone (MkTfmModel emb body headW pe) = do
  inIdsTgt <- liftIO1 sortingSample
  let (inIds, tgt) = inIdsTgt
  (MkBang predV # body') <- tfmForwardL emb pe headW body inIds
  discard body'
  liftIO1 $ do
    putStrLn ""
    putStrLn "Evaluation:"
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

-- Discard the (linear) model: its body is a linear field → discard; the ω
-- embed/headW/pe drop freely.
discardModel : (1 _ : Model) -> L IO ()
discardModel (MkTfmModel _ body _ _) = discard body

----------------------------------------------------------------------
-- Main
----------------------------------------------------------------------

lrFindCfg : LrFindConfig
lrFindCfg = { numIters := 100 } defaultLrFindConfig

-- Terminal linear consumer of the lrFind result. A named function with an
-- explicit `(1 _ : LPair ...)` signature so the bind continuation is linear
-- (the inline do-notation `<-` doesn't get recognised as linear for `lrFind`).
partial
finishLrFind : (1 _ : LPair (!* LrFindResult) Model) -> L IO ()
finishLrFind (MkBang _ # m') = do
  discardModel m'
  liftIO1 $ do
    putStrLn ""
    putStrLn "Done — re-run without --lr-find at the recommended LR."

partial
runLrFind : Config -> Optimizer ExampleExecutor -> IO ()
runLrFind cfg opt = Control.Linear.LIO.run $ do
  -- lrFind on the linear surface (lrFind): the model is threaded through
  -- the sweep by stepL, then discarded.
  model <- runInitL mkModelInit
  (LIO.(>>=))
    (lrFind {ex = ExampleExecutor} {model = Model} {dp = Vect BatchSize TfmSample} lrFindCfg
       (stepL opt) (sortingBatch BatchSize) opt model)
    finishLrFind

partial
runTrain : List Nat -> Config -> Optimizer ExampleExecutor ->
           TrainConfig Model -> IO ()
runTrain positions cfg opt trainCfg = Control.Linear.LIO.run $ do
  -- Linear surface end to end: model born linear (runInitL), threaded
  -- through fitSupervised (batchLossL consumes-and-returns it each step),
  -- final eval via evalReportL (which consumes the trained handle).
  model <- runInitL mkModelInit
  liftIO1 (maybeDumpInit {ex = ExampleExecutor})
  (MkBang (epochsDone, _) # trained) <-
    fitSupervised {ex=ExampleExecutor} opt batchLossL (generate (sortingBatch BatchSize)) trainCfg model
  evalReportL positions cfg epochsDone trained

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

  -- Adam with global grad-clip norm 1.0.
  opt <- adam {ex=ExampleExecutor} cfg.lr ({ clip := NormClip 1.0 } defaultOpts)
  putStrLn ""

  -- The per-epoch sorted-portion accuracy metric needs to forward the model,
  -- which the linear loop's model-free `metricsL` can't do (it reads the C
  -- registry only). So the linear path runs with no per-epoch metric; the
  -- final eval below still reports `sort_acc`.
  let trainCfgBase = mkTrainConfig cfg.epochs 100 (Patience cfg.patience 0.001) (const (pure [])) (\_ => pure ())
      trainCfg = case cfg.checkpointDir of
                   ""  => trainCfgBase
                   dir => withCheckpoint
                            (fileCheckpoint dir cfg.checkpointEvery True opt)
                            trainCfgBase

  if cfg.lrFind then runLrFind cfg opt else runTrain positions cfg opt trainCfg
