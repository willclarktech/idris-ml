-- | GPT: Character-Level Language Model
-- |
-- | Character-level language model on embedded Shakespeare text, following
-- | Karpathy's char-rnn/minGPT tradition. A multi-block pre-norm transformer
-- | with learned token embeddings, sinusoidal PE, and causal self-attention,
-- | assembled on the `Nn` models-as-records surface (`Nn.Embedding` +
-- | `Nn.transformerBlock` stacked in a `Seq` + bias-free output head) and
-- | trained via the `fit` driver.
-- |
-- | Input: sliding window of SeqLen characters from corpus (token ids)
-- | Target: shifted by 1 (next character at each position, one-hot)

module Example.Gpt

import Control.Linear.LIO
import Data.Linear.Notation
import Data.List
import Data.String
import Data.Vect
import System
import System.File

import Array
import BuildConfig
import Checkpoint
import Compat.Random
import DataStream
import Executor
import FitL
import Floating
import Generate
import Nn
import Optimizer
import Sampler
import Schedule
import Tensor
import Train
import Util

----------------------------------------------------------------------
-- Configuration
----------------------------------------------------------------------

VocabSize : Nat
VocabSize = 65

SeqLen : Nat
SeqLen = 64

DModel : Nat
DModel = 64

NumHeads : Nat
NumHeads = 4

HeadDim : Nat
HeadDim = 16

NumBlocks : Nat
NumBlocks = 2

BatchSize : Nat
BatchSize = 32

----------------------------------------------------------------------
-- Corpus & Tokenization
----------------------------------------------------------------------

embeddedCorpus : String
embeddedCorpus = "all the world's a stage, and all the men and women merely players; "
  ++ "they have their exits and their entrances, and one man in his time "
  ++ "plays many parts, his acts being seven ages. at first, the infant, "
  ++ "mewling and puking in the nurse's arms. then the whining schoolboy, "
  ++ "with his satchel and shining morning face, creeping like snail "
  ++ "unwillingly to school. and then the lover, sighing like a furnace, "
  ++ "with a woeful ballad made to his mistress' eyebrow. then a soldier, "
  ++ "full of strange oaths and bearded like the pard, jealous in honour, "
  ++ "sudden and quick in quarrel, seeking the bubble reputation even in "
  ++ "the cannon's mouth. and then the justice, in fair round belly with "
  ++ "good capon lined, with eyes severe and beard of formal cut, full of "
  ++ "wise saws and modern instances; and so he plays his part. "
  ++ "to be or not to be, that is the question; whether 'tis nobler in "
  ++ "the mind to suffer the slings and arrows of outrageous fortune, or "
  ++ "to take arms against a sea of troubles, and by opposing end them. "
  ++ "to die, to sleep; no more; and by a sleep to say we end the "
  ++ "heartache and the thousand natural shocks that flesh is heir to; "
  ++ "'tis a consummation devoutly to be wished. to die, to sleep; to "
  ++ "sleep, perchance to dream. ay, there's the rub, for in that sleep "
  ++ "of death what dreams may come, when we have shuffled off this mortal "
  ++ "coil, must give us pause."

vocabChars : String
vocabChars = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

charToIdx : Char -> Int
charToIdx c = go (unpack vocabChars) 0
  where
    go : List Char -> Int -> Int
    go [] _          = 1  -- unknown -> space
    go (h :: rest) i =
      if h == c then i else go rest (i + 1)

idxToChar : Int -> Char
idxToChar i =
  let n = the Nat (cast i)
      chars = unpack vocabChars
      go : List Char -> Nat -> Char
      go [] _              = ' '
      go (c :: _) Z        = c
      go (_ :: rest) (S k) = go rest k
  in go chars n

----------------------------------------------------------------------
-- Model (Nn surface): embedding + sinusoidal PE + transformer blocks +
-- final norm + bias-free output head.
----------------------------------------------------------------------

public export
record GptModel (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkGptModel
  embed : Embedding VocabSize DModel ex dt g
  -- NumBlocks pre-norm transformer blocks followed by the final LayerNorm,
  -- all `DModel -> DModel`, stacked in one `Seq`.
  body  : Seq DModel DModel ex dt g
  -- Bias-free vocab projection (matches the legacy `primMm`-only head).
  headW : TMat VocabSize DModel ex dt g
  -- Cached sinusoidal positional encoding (no paramId, optimizer-invisible).
  pe    : Tensor [SeqLen, DModel] ex dt g

Model : Type
Model = GptModel ExampleExecutor ExampleDType WithGrad

-- Build the block stack + final norm inside an `Init` derivation. Blocks
-- land at `block_0.*`..`block_{n-1}.*`; the final norm trails.
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
-- linear with `runInitL`.
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
  pure (MkGptModel e b hw pe)

-- Forward one sequence of token ids `[SeqLen]` → per-position logits
-- `[SeqLen, VocabSize]`.
partial
gptForward : Model -> Tensor [SeqLen] ExampleExecutor ExampleDType WithGrad ->
             IO (Tensor [SeqLen, VocabSize] ExampleExecutor ExampleDType WithGrad)
gptForward (MkGptModel emb body headW pe) tokens = do
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
-- Data generation
----------------------------------------------------------------------

listSlice : List a -> Nat -> Nat -> List a
listSlice xs start n = Data.List.take n (drop start xs)

packDoubleBuf : AnyPtr -> Int -> List Int -> AnyPtr
packDoubleBuf buf _ []          = buf
packDoubleBuf buf off (x :: xs) =
  packDoubleBuf (prim__setDouble buf off (cast x)) (off + 1) xs

packIntBuf : AnyPtr -> Int -> List Int -> AnyPtr
packIntBuf buf _ []          = buf
packIntBuf buf off (x :: xs) =
  packIntBuf (prim__setInt buf off x) (off + 1) xs

-- One (input ids, one-hot target) training pair for a random window.
GptSample : Type
GptSample = ( Tensor [SeqLen] ExampleExecutor ExampleDType WithGrad
            , Tensor [SeqLen, VocabSize] ExampleExecutor ExampleDType WithGrad )

gptSample : (corpus : List Int) -> (corpusLen : Nat) -> IO GptSample
gptSample corpus corpusLen = do
  let maxStart = minus corpusLen (SeqLen + 1)
  startN <- randomInt 0 maxStart
  let start      = startN
      window     = listSlice corpus start (SeqLen + 1)
      inputToks  = Data.List.take SeqLen window
      targetToks = Data.List.take SeqLen (drop 1 window)
      sI         = cast {to=Int} SeqLen
      vI         = cast {to=Int} VocabSize
      inT        = dtCreate1d {ex=ExampleExecutor} {t=ExampleDType} sI (packDoubleBuf (prim__allocDoubles sI) 0 inputToks) 0 (deviceStreamTag {ex=ExampleExecutor})
      tgtIdxBuf  = packIntBuf (prim__allocInts sI) 0 targetToks
      tgtFlat    = primOneHot {ex=ExampleExecutor} tgtIdxBuf sI vI (dtypeTag {t=ExampleDType})
      tgt2d      = primReshape2d {ex=ExampleExecutor} tgtFlat sI vI
  pure ( MkTensor inT Nothing, MkTensor tgt2d Nothing )

gptBatch : (corpus : List Int) -> (corpusLen : Nat) -> (n : Nat) -> IO (Vect n GptSample)
gptBatch _ _ Z                  = pure []
gptBatch corpus corpusLen (S k) = do
  s    <- gptSample corpus corpusLen
  rest <- gptBatch corpus corpusLen k
  pure (s :: rest)

----------------------------------------------------------------------
-- Loss: categorical cross-entropy on ALL positions (standard LM loss)
----------------------------------------------------------------------

-- Mean NLL across positions from `[SeqLen, VocabSize]` logits + one-hot.
lmLoss : Tensor [SeqLen, VocabSize] ExampleExecutor ExampleDType WithGrad ->
         Tensor [SeqLen, VocabSize] ExampleExecutor ExampleDType WithGrad ->
         IO (Tensor [] ExampleExecutor ExampleDType WithGrad)
lmLoss logits targets = ioRerun (\_ =>
  let logProbs = primLogSoftmax2d {ex=ExampleExecutor} logits.tensorPtr
      product  = primMul {ex=ExampleExecutor} logProbs targets.tensorPtr
      totalSum = primSum {ex=ExampleExecutor} product
      loss     = primMulScalar {ex=ExampleExecutor} (primNeg {ex=ExampleExecutor} totalSum) (1.0 / cast {to=Double} SeqLen)
  in MkTensor loss Nothing)

-- Mean loss over a batch: forward each sequence, sum the scalar losses,
-- divide by batch size. `fitSupervised` owns the fused optimizer step.
partial
batchLoss : Model -> Vect BatchSize GptSample -> IO (Tensor [] ExampleExecutor ExampleDType WithGrad)
batchLoss model batch = do
  losses <- traverse (\(ids, tgt) => do logits <- gptForward model ids; lmLoss logits tgt) (toList batch)
  zero   <- tparamScalar {ex=ExampleExecutor} {dt=ExampleDType} "gpt.epoch_zero" 0.0
  summed <- foldlM (\acc, l => tadd acc l) zero losses
  tmulScalar summed (1.0 / cast {to=Double} BatchSize)

-- Linear-resource loss (consume-match-rebuild-delegate): match `MkGptModel …`
-- to bind the fields at ω, delegate to the IO `batchLoss`, rebuild the record
-- beside the banged loss.
partial
batchLossL : (1 _ : Model) -> Vect BatchSize GptSample ->
             L IO {use = 1} (LPair (!* (Tensor [] ExampleExecutor ExampleDType WithGrad)) Model)
batchLossL (MkGptModel emb body headW pe) batch = do
  loss <- liftIO1 (batchLoss (MkGptModel emb body headW pe) batch)
  pure1 (MkBang loss # MkGptModel emb body headW pe)

----------------------------------------------------------------------
-- Autoregressive Generation (single-sample forward)
----------------------------------------------------------------------

partial
generateText : Model -> String -> Nat -> Double -> IO String
generateText model seed genLen temperature = do
  let seedIdxs = map charToIdx (unpack seed)
      padLen  = minus SeqLen (length seedIdxs)
      context = replicate padLen (the Int 1) ++ Data.List.take SeqLen seedIdxs
  chars <- go model context genLen []
  pure (seed ++ pack chars)
  where
    vocabIdxs : List Nat
    vocabIdxs = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
                ,10,11,12,13,14,15,16,17,18,19
                ,20,21,22,23,24,25,26,27,28,29
                ,30,31,32,33,34,35,36,37,38,39
                ,40,41,42,43,44,45,46,47,48,49
                ,50,51,52,53,54,55,56,57,58,59
                ,60,61,62,63,64
                ]

    -- Probabilities at position `pos` from `[SeqLen, VocabSize]` logits.
    sampleAt : Tensor [SeqLen, VocabSize] ExampleExecutor ExampleDType WithGrad -> Nat -> List Double
    sampleAt logits pos =
      let posI = cast {to=Int} (natToInteger pos)
      in map (\j => exp (primItem2d {ex=ExampleExecutor} logits.tensorPtr posI (cast j) / temperature))
             vocabIdxs

    argmax : List Double -> Int
    argmax probs =
      fst (foldl (\(bi, bv), (i, v) => if v > bv then (i, v) else (bi, bv))
           (the (Int, Double) (0, -1.0e10))
           (zip (map cast vocabIdxs) probs))

    go : Model -> List Int -> Nat -> List Char -> IO (List Char)
    go _ _ Z acc       = pure (reverse acc)
    go m ctx (S k) acc = do
      let sI = cast {to=Int} SeqLen
          inT = dtCreate1d {ex=ExampleExecutor} {t=ExampleDType} sI (packDoubleBuf (prim__allocDoubles sI) 0 ctx) 0 (deviceStreamTag {ex=ExampleExecutor})
          inV = the (Tensor [SeqLen] ExampleExecutor ExampleDType WithGrad) (MkTensor inT Nothing)
      logits <- gptForward m inV
      let unnorm = sampleAt logits (minus SeqLen 1)
          totSum  = foldl (+) 0.0 unnorm
          probs   = map (/ totSum) unnorm
          bestIdx = argmax probs
          ch      = idxToChar bestIdx
          ctx'    = drop 1 ctx ++ [bestIdx]
      go m ctx' k (ch :: acc)

----------------------------------------------------------------------
-- Evaluation: bits-per-character on a held-out corpus slice
----------------------------------------------------------------------

partial
evalBPC : Model -> (corpus : List Int) -> (corpusLen : Nat) -> (nSamples : Nat) -> IO Double
evalBPC model corpus corpusLen nSamples = go nSamples 0.0
  where
    singleBPC : Nat -> IO Double
    singleBPC start = do
      let window = listSlice corpus start (SeqLen + 1)
          inputToks  = Data.List.take SeqLen window
          targetToks = Data.List.take SeqLen (drop 1 window)
          sI         = cast {to=Int} SeqLen
          vI         = cast {to=Int} VocabSize
          inT        = dtCreate1d {ex=ExampleExecutor} {t=ExampleDType} sI (packDoubleBuf (prim__allocDoubles sI) 0 inputToks) 0 (deviceStreamTag {ex=ExampleExecutor})
          tgtIdxBuf  = packIntBuf (prim__allocInts sI) 0 targetToks
          tgtFlat    = primOneHot {ex=ExampleExecutor} tgtIdxBuf sI vI (dtypeTag {t=ExampleDType})
          tgt2d      = primReshape2d {ex=ExampleExecutor} tgtFlat sI vI
          inV        = the (Tensor [SeqLen] ExampleExecutor ExampleDType WithGrad) (MkTensor inT Nothing)
          tgtV       = the (Tensor [SeqLen, VocabSize] ExampleExecutor ExampleDType WithGrad) (MkTensor tgt2d Nothing)
      logits <- gptForward model inV
      lossT <- lmLoss logits tgtV
      pure (primItem {ex=ExampleExecutor} lossT.tensorPtr / log 2.0)

    go : Nat -> Double -> IO Double
    go Z acc     = pure acc
    go (S k) acc = do
      let maxStart = minus corpusLen (SeqLen + 1)
          pos = div (k * maxStart) nSamples
      bpc <- singleBPC pos
      go k (acc + bpc / cast {to=Double} (natToInteger nSamples))

----------------------------------------------------------------------
-- Corpus loading + train/val split
----------------------------------------------------------------------

tinyshakespearePath : String
tinyshakespearePath = "data/tinyshakespeare/input.txt"

loadCorpusText : String -> IO String
loadCorpusText "embedded"        = pure embeddedCorpus
loadCorpusText "tinyshakespeare" = do
  result <- readFile tinyshakespearePath
  case result of
    Right contents => pure contents
    Left err       => do
      putStrLn $ "WARNING: could not read " ++ tinyshakespearePath
              ++ " (" ++ show err ++ "); falling back to embedded corpus."
      putStrLn $ "         Run `make dataset-tinyshakespeare` from the repo root."
      pure embeddedCorpus
loadCorpusText other = do
  putStrLn $ "WARNING: unknown corpus '" ++ other ++ "'; using embedded."
  pure embeddedCorpus

trainValSplit : (valFrac : Double) -> List Int -> (List Int, List Int)
trainValSplit valFrac idx =
  let n = length idx
      nVal   = the Nat (cast (cast {to=Double} (natToInteger n) * valFrac))
      nTrain = minus n nVal
  in (Data.List.take nTrain idx, drop nTrain idx)

----------------------------------------------------------------------
-- Config & Main
----------------------------------------------------------------------

record Config where
  constructor MkConfig
  corpus          : String
  lr              : Double
  epochs          : Nat
  patience        : Nat
  seed            : Bits64
  lrFind          : Bool
  checkpointDir   : String
  checkpointEvery : Nat

defaultConfig : Config
defaultConfig = MkConfig "embedded" 0.001 30 0 42 False "" 10

specs : List (ArgSpec Config)
specs = [ Arg "--corpus" (\v, c => { corpus := v } c)
        , Arg "--lr" (\v, c => { lr := cast v } c)
        , Arg "--epochs" (\v, c => { epochs := castNat v } c)
        , Arg "--patience" (\v, c => { patience := castNat v } c)
        , Arg "--seed" (\v, c => { seed := castBits64 v } c)
        , Arg "--lr-find" (\v, c => { lrFind := (v == "1" || v == "true") } c)
        -- Checkpointing: save to / auto-resume from DIR. `--resume` is an
        -- alias for `--checkpoint-dir` (resumes if DIR/last.* is present).
        , Arg "--checkpoint-dir" (\v, c => { checkpointDir := v } c)
        , Arg "--resume" (\v, c => { checkpointDir := v } c)
        , Arg "--checkpoint-every" (\v, c => { checkpointEvery := castNat v } c) ]

----------------------------------------------------------------------
-- Final eval + generation (consumes the trained linear model)
----------------------------------------------------------------------

-- Consume the trained (linear) model: match `MkGptModel …` to bind the fields
-- at ω (discharging the single-owner obligation), then run the IO bpc eval +
-- generation + RESULT report on the rebuilt record.
partial
finalReportL : Config -> (valIdx : List Int) -> (valLen : Nat) ->
               (trainIdx : List Int) -> (trainLen : Nat) -> Nat -> (1 _ : Model) -> L IO ()
finalReportL cfg valIndices valLen trainIndices trainLen epochsDone (MkGptModel emb body headW pe) =
  liftIO1 $ do
    let trained = MkGptModel emb body headW pe
    putStrLn ""
    valBpc <- evalBPC trained valIndices valLen 50
    trainBpc <- evalBPC trained trainIndices trainLen 50
    putStrLn $ "Final val_bpc: " ++ show valBpc
            ++ "  (train_bpc: " ++ show trainBpc ++ ")"
    putStrLn ""
    putStrLn "Generation (seed='to be or '):"
    sample1 <- withNoGrad {ex=ExampleExecutor} (generateText trained "to be or " 200 1.0)
    putStrLn $ "  " ++ show sample1
    putStrLn ""
    putStrLn "Generation (seed='the '):"
    sample2 <- withNoGrad {ex=ExampleExecutor} (generateText trained "the " 200 1.0)
    putStrLn $ "  " ++ show sample2
    putStrLn ""
    let metricKey = if cfg.corpus == "embedded" then "bpc" else "val_bpc"
    putStrLn $ formatResult [(metricKey, show valBpc),
                              ("epochs", show epochsDone),
                              ("seed", show cfg.seed)]

partial
main : IO ()
main = do
  requireMachine {m = ChosenMachine}
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)

  srand cfg.seed
  tsetInitSeed {ex = ExampleExecutor} cfg.seed

  corpusText <- loadCorpusText cfg.corpus
  let allIndices = map charToIdx (unpack corpusText)
      (trainIndices, valIndices) =
        if cfg.corpus == "embedded"
          then (allIndices, allIndices)
          else trainValSplit 0.1 allIndices
      trainLen = length trainIndices
      valLen   = length valIndices

  putStrLn "=== GPT: Character-Level Language Model ==="
  putStrLn $ "Config: corpus=" ++ cfg.corpus
           ++ " lr=" ++ show cfg.lr ++ " epochs=" ++ show cfg.epochs
           ++ " patience=" ++ show cfg.patience ++ " seed=" ++ show cfg.seed
  putStrLn $ "Architecture: seqLen=" ++ show SeqLen ++ " dModel=" ++ show DModel
           ++ " heads=" ++ show NumHeads ++ " headDim=" ++ show HeadDim
           ++ " blocks=" ++ show NumBlocks ++ " vocab=" ++ show VocabSize
  putStrLn $ "Corpus: " ++ show (length allIndices) ++ " chars"
           ++ " (train=" ++ show trainLen ++ ", val=" ++ show valLen ++ ")"

  -- Cosine LR schedule with warmup, carried on the optimizer (fit ticks it
  -- per epoch). adamW betas (0.9, 0.99), weight decay 0.1, grad-clip norm 1.
  let warmupEpochs : Nat = min 100 (div cfg.epochs 10)
      minLR    : Double   = cfg.lr * 0.1
      schedule : Schedule = cosineWithWarmup cfg.lr minLR warmupEpochs cfg.epochs

  opt0 <- adamW {ex=ExampleExecutor} cfg.lr 0.1 ({ beta2 := 0.99, clip := NormClip 1.0 } defaultOpts)
  let opt = withSchedule schedule opt0
  putStrLn ""

  if cfg.lrFind
    then do
      putStrLn "lr_find skipped for GPT: cosine + warmup schedule conflicts with"
      putStrLn "lrFind's group-level setting; transformer-forward cost is also"
      putStrLn "prohibitive at 100 iters. See docs/develop/hyperparameter-tuning-2026.md."
    else do
      -- The per-epoch val_bpc metric needs to forward the model, which the
      -- linear loop's model-free `metricsL` can't do; the linear path runs
      -- with no per-epoch metric (the final report below still computes bpc).
      let noOpHook : Nat -> IO ()
          noOpHook _ = pure ()
      let trainCfgBase = mkTrainConfig cfg.epochs 100
                           (if cfg.patience == 0
                              then NoEarlyStop
                              else Patience cfg.patience 0.001)
                           (const (pure (the (List (String, String)) [])))
                           noOpHook
          trainCfg = case cfg.checkpointDir of
                       ""  => trainCfgBase
                       dir => withCheckpoint
                                (fileCheckpoint dir cfg.checkpointEvery True opt)
                                trainCfgBase
      -- Linear surface end to end: model born linear (runInitL), threaded
      -- through fitSupervisedL (batchLossL consumes-and-returns it each step),
      -- final eval + generation via finalReportL (consumes the trained handle).
      Control.Linear.LIO.run $ do
        model <- runInitL mkModelInit
        (MkBang (epochsDone, _) # trained) <-
          fitSupervisedL {ex=ExampleExecutor} opt batchLossL
                         (generate (gptBatch trainIndices trainLen BatchSize)) trainCfg model
        finalReportL cfg valIndices valLen trainIndices trainLen epochsDone trained
