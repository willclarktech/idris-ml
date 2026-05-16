-- | GptLarge: GPU-shaped Character-Level Language Model
-- |
-- | Same architecture as `Example.Gpt`, but with the dimensions cranked up
-- | so per-step compute is large enough that mlx GPU + compile is the
-- | unambiguously right call (the small `Gpt` config sits on the wrong
-- | side of mlx's kernel-launch wall — see `docs/develop/mlx-survey.md`).
-- |
-- | Config: dModel=256, heads=8, headDim=32, blocks=4, seq=128, batch=32
-- | (~1M params vs ~50K for the default GPT).
-- |
-- | Input: sliding window of SeqLen characters from corpus (one-hot)
-- | Target: shifted by 1 (next character at each position)

module Example.GptLarge

import Data.IORef
import Data.List
import Data.String
import Data.Vect
import Decidable.Equality
import System
import System.File
import Compat.Random

import Backprop
import DataPoint
import Floating
import Generate
import Layer.Core
import Layer.Transformer
import Sampler
import Schedule
import Array
import Train
import Util
import Device
import Tensor


----------------------------------------------------------------------
-- Configuration
----------------------------------------------------------------------

VocabSize : Nat
VocabSize = 65

SeqLen : Nat
SeqLen = 128

DModel : Nat
DModel = 256

NumHeads : Nat
NumHeads = 8

HeadDim : Nat
HeadDim = 32

NumBlocks : Nat
NumBlocks = 4

BatchSize : Nat
BatchSize = 32

InputDim : Nat
InputDim = SeqLen

OutputDim : Nat
OutputDim = SeqLen * VocabSize


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
    go [] _ = 1  -- unknown -> space
    go (h :: rest) i =
      if h == c then i else go rest (i + 1)

idxToChar : Int -> Char
idxToChar i =
  let n = the Nat (cast i)
      chars = unpack vocabChars
      go : List Char -> Nat -> Char
      go [] _ = ' '
      go (c :: _) Z = c
      go (_ :: rest) (S k) = go rest k
  in go chars n


----------------------------------------------------------------------
-- Data generation
----------------------------------------------------------------------

listSlice : List a -> Nat -> Nat -> List a
listSlice xs start n = Data.List.take n (drop start xs)

packDoubleBuf : AnyPtr -> Int -> List Int -> AnyPtr
packDoubleBuf buf _ [] = buf
packDoubleBuf buf off (x :: xs) =
  packDoubleBuf (prim__setDouble buf off (cast x)) (off + 1) xs

packIntBuf : AnyPtr -> Int -> List Int -> AnyPtr
packIntBuf buf _ [] = buf
packIntBuf buf off (x :: xs) =
  packIntBuf (prim__setInt buf off x) (off + 1) xs

gptTensorPoint : (corpus : List Int) -> (corpusLen : Nat) ->
                 IO (TensorDataPoint InputDim OutputDim)
gptTensorPoint corpus corpusLen = do
  let maxStart = minus corpusLen (SeqLen + 1)
  startN <- randomInt 0 (cast maxStart)
  let start = the Nat (cast startN)
      window = listSlice corpus start (SeqLen + 1)
      inputToks = Data.List.take SeqLen window
      targetToks = Data.List.take SeqLen (drop 1 window)
      sI = cast {to=Int} SeqLen
      vI = cast {to=Int} VocabSize
      inT = prim__create1d sI (packDoubleBuf (prim__allocDoubles sI) 0 inputToks) 0
      tgtIdxBuf = packIntBuf (prim__allocInts sI) 0 targetToks
  pure $ MkTensorDataPoint inT (prim__oneHot tgtIdxBuf sI vI)

gptBatchVect : (corpus : List Int) -> (corpusLen : Nat) -> (n : Nat) ->
               IO (Vect n (TensorDataPoint InputDim OutputDim))
gptBatchVect _ _ Z = pure []
gptBatchVect corpus corpusLen (S k) = do
  dp <- gptTensorPoint corpus corpusLen
  rest <- gptBatchVect corpus corpusLen k
  pure (dp :: rest)


----------------------------------------------------------------------
-- Loss: Cross-entropy on all positions ( typed-surface)
----------------------------------------------------------------------

||| Categorical cross-entropy on ALL positions (standard LM loss).
||| Operates on a flat [SeqLen * VocabSize] Tensor; reshapes to
||| [SeqLen, VocabSize] and computes mean NLL across positions.
allPositionsCELoss : TVec OutputDim CPU WithGrad -> TVec OutputDim CPU WithGrad -> Tensor [] CPU WithGrad
allPositionsCELoss predV targetV =
  let vsI = cast {to=Int} VocabSize
      sI = cast {to=Int} SeqLen
      logitsR = prim__reshape2d predV.tensorPtr sI vsI
      logProbs = prim__logSoftmax2d logitsR
      tgtsR = prim__reshape2d targetV.tensorPtr sI vsI
      product = prim__mul logProbs tgtsR
      totalSum = prim__sum product
      loss = prim__mulScalar (prim__neg totalSum) (1.0 / cast {to=Double} SeqLen)
  in MkTensor loss Nothing


----------------------------------------------------------------------
-- Autoregressive Generation (single-sample forward)
----------------------------------------------------------------------

generateText : Network InputDim [] OutputDim CPU WithGrad ->
               String -> Nat -> Double -> String
generateText model seed genLen temperature =
  let seedIdxs = map charToIdx (unpack seed)
      padLen = minus SeqLen (length seedIdxs)
      context = replicate padLen (the Int 1) ++ Data.List.take SeqLen seedIdxs
  in seed ++ pack (go model context genLen [])
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

    sampleAt : AnyPtr -> Nat -> List Double
    sampleAt outT pos =
      let vsI = cast {to=Int} VocabSize
          sI = cast {to=Int} SeqLen
          logitsR = prim__reshape2d outT sI vsI
          posI = cast {to=Int} (natToInteger pos)
      in map (\j => exp (prim__item2d logitsR posI (cast j) / temperature))
             vocabIdxs

    argmax : List Double -> Int
    argmax probs =
      fst (foldl (\(bi, bv), (i, v) => if v > bv then (i, v) else (bi, bv))
           (the (Int, Double) (0, -1.0e10))
           (zip (map cast vocabIdxs) probs))

    go : Network InputDim [] OutputDim CPU WithGrad ->
         List Int -> Nat -> List Char -> List Char
    go _ _ Z acc = reverse acc
    go m ctx (S k) acc =
      let sI = cast {to=Int} SeqLen
          inT = prim__create1d sI (packDoubleBuf (prim__allocDoubles sI) 0 ctx) 0
          inV = the (TVec InputDim CPU WithGrad) (MkTensor inT Nothing)
          (_, predV) = forwardVar m inV
          unnorm = sampleAt predV.tensorPtr (minus SeqLen 1)
          totSum = foldl (+) 0.0 unnorm
          probs = map (/ totSum) unnorm
          bestIdx = argmax probs
          ch = idxToChar bestIdx
          ctx' = drop 1 ctx ++ [bestIdx]
      in go m ctx' k (ch :: acc)


----------------------------------------------------------------------
-- Evaluation: bits-per-character on a held-out corpus slice
----------------------------------------------------------------------

evalBPC : Network InputDim [] OutputDim CPU WithGrad ->
          (corpus : List Int) -> (corpusLen : Nat) -> (nSamples : Nat) -> Double
evalBPC model corpus corpusLen nSamples = go nSamples 0.0
  where
    singleBPC : Nat -> Double
    singleBPC start =
      let window = listSlice corpus start (SeqLen + 1)
          inputToks = Data.List.take SeqLen window
          targetToks = Data.List.take SeqLen (drop 1 window)
          sI = cast {to=Int} SeqLen
          vI = cast {to=Int} VocabSize
          inT = prim__create1d sI (packDoubleBuf (prim__allocDoubles sI) 0 inputToks) 0
          tgtIdxBuf = packIntBuf (prim__allocInts sI) 0 targetToks
          tgtT = prim__oneHot tgtIdxBuf sI vI
          inV = the (TVec InputDim CPU WithGrad) (MkTensor inT Nothing)
          tgtV = the (TVec OutputDim CPU WithGrad) (MkTensor tgtT Nothing)
          (_, predV) = forwardVar model inV
          lossT = allPositionsCELoss predV tgtV
      in prim__item lossT.tensorPtr / log 2.0

    go : Nat -> Double -> Double
    go Z acc = acc
    go (S k) acc =
      let maxStart = minus corpusLen (SeqLen + 1)
          pos = div (k * maxStart) nSamples
          bpc = singleBPC pos
      in go k (acc + bpc / cast {to=Double} (natToInteger nSamples))


----------------------------------------------------------------------
-- Corpus loading + train/val split
----------------------------------------------------------------------

tinyshakespearePath : String
tinyshakespearePath = "data/tinyshakespeare/input.txt"

loadCorpusText : String -> IO String
loadCorpusText "embedded" = pure embeddedCorpus
loadCorpusText "tinyshakespeare" = do
  result <- readFile tinyshakespearePath
  case result of
    Right contents => pure contents
    Left err => do
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
      nVal = the Nat (cast (cast {to=Double} (natToInteger n) * valFrac))
      nTrain = minus n nVal
  in (Data.List.take nTrain idx, drop nTrain idx)


----------------------------------------------------------------------
-- LR-schedule helper: update all registered params each epoch.
----------------------------------------------------------------------

setLRAll : NativeOptimizer -> Double -> IO ()
setLRAll opt lr = do
  n <- getParamCount
  go 0 n
  where
    go : Int -> Int -> IO ()
    go i n =
      if i >= n
        then pure ()
        else do
          nm <- getParamName i
          pure (setParamLR opt nm lr)
          go (i + 1) n


----------------------------------------------------------------------
-- Config & Main
----------------------------------------------------------------------

record Config where
  constructor MkConfig
  corpus : String
  lr : Double
  epochs : Nat
  patience : Nat
  seed : Bits64
  lrFind : Bool

defaultConfig : Config
defaultConfig = MkConfig "embedded" 0.001 30 0 42 False

specs : List (ArgSpec Config)
specs = [ Arg "--corpus" (\v, c => { corpus := v } c)
        , Arg "--lr" (\v, c => { lr := cast v } c)
        , Arg "--epochs" (\v, c => { epochs := castNat v } c)
        , Arg "--patience" (\v, c => { patience := castNat v } c)
        , Arg "--seed" (\v, c => { seed := castBits64 v } c)
        , Arg "--lr-find" (\v, c => { lrFind := (v == "1" || v == "true") } c) ]


partial
main : IO ()
main = do
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)

  srand cfg.seed

  let opt = nativeAdamW cfg.lr 0.9 0.99 1.0e-8 0.1 1.0

  corpusText <- loadCorpusText cfg.corpus
  let allIndices = map charToIdx (unpack corpusText)
      (trainIndices, valIndices) =
        if cfg.corpus == "embedded"
          then (allIndices, allIndices)
          else trainValSplit 0.1 allIndices
      trainLen = length trainIndices
      valLen = length valIndices

  putStrLn "=== GPT: Character-Level Language Model ==="
  putStrLn $ "Config: corpus=" ++ cfg.corpus
           ++ " lr=" ++ show cfg.lr ++ " epochs=" ++ show cfg.epochs
           ++ " patience=" ++ show cfg.patience ++ " seed=" ++ show cfg.seed
  putStrLn $ "Architecture: seqLen=" ++ show SeqLen ++ " dModel=" ++ show DModel
           ++ " heads=" ++ show NumHeads ++ " headDim=" ++ show HeadDim
           ++ " blocks=" ++ show NumBlocks ++ " vocab=" ++ show VocabSize
  putStrLn $ "Corpus: " ++ show (length allIndices) ++ " chars"
           ++ " (train=" ++ show trainLen ++ ", val=" ++ show valLen ++ ")"

  tfmAny <- transformerLayerAny
              {seqLen=SeqLen, dModel=DModel, numHeads=NumHeads,
               headDim=HeadDim, numBlocks=NumBlocks, vocabSize=VocabSize}
              "tfm0"
  let model : Network InputDim [] OutputDim CPU WithGrad
      model = OutputLayer tfmAny
  putStrLn ""

  let warmupEpochs : Nat = min 100 (div cfg.epochs 10)
      minLR : Double = cfg.lr * 0.1
      schedule : Schedule = cosineWithWarmup cfg.lr minLR warmupEpochs cfg.epochs
  epochRef <- newIORef Z

  let genBatch : IO (Vect BatchSize (TensorDataPoint InputDim OutputDim))
      genBatch = gptBatchVect trainIndices trainLen BatchSize

  let evalMetrics : Network InputDim [] OutputDim CPU WithGrad -> IO (List (String, String))
      evalMetrics m = do
        let valBpc = evalBPC m valIndices valLen 20
        pure [("val_bpc", show valBpc)]

  let noOpHook : Nat -> IO ()
      noOpHook _ = pure ()

  when cfg.lrFind $ do
    putStrLn "lr_find skipped for GPT: per-param LR schedule (cosine + warmup)"
    putStrLn "conflicts with lrFind's group-level setting; transformer-forward"
    putStrLn "cost is also prohibitive at 100 iters."
    putStrLn "See docs/develop/hyperparameter-tuning-2026.md."
    exitSuccess

  let trainCfg = MkTrainConfig cfg.epochs 100
                   (if cfg.patience == 0
                      then NoEarlyStop
                      else Patience cfg.patience 0.001)
                   evalMetrics
                   noOpHook

  let stepFn : Network InputDim [] OutputDim CPU WithGrad ->
               Vect BatchSize (TensorDataPoint InputDim OutputDim) ->
               IO (Network InputDim [] OutputDim CPU WithGrad, Double)
      stepFn m d = do
        ep <- readIORef epochRef
        let lr = schedule ep
        setLRAll opt lr
        writeIORef epochRef (S ep)
        pure (epochVarTensorBatch opt d allPositionsCELoss m)

  (trained, epochsDone, finalLoss) <- runTrainingIO stepFn genBatch trainCfg model

  putStrLn ""
  let valBpc = evalBPC trained valIndices valLen 50
      trainBpc = evalBPC trained trainIndices trainLen 50
  putStrLn $ "Final val_bpc: " ++ show valBpc
          ++ "  (train_bpc: " ++ show trainBpc ++ ")"

  putStrLn ""
  putStrLn "Generation (seed='to be or '):"
  let sample1 = generateText trained "to be or " 200 1.0
  putStrLn $ "  " ++ show sample1

  putStrLn ""
  putStrLn "Generation (seed='the '):"
  let sample2 = generateText trained "the " 200 1.0
  putStrLn $ "  " ++ show sample2

  putStrLn ""
  let metricKey = if cfg.corpus == "embedded" then "bpc" else "val_bpc"
  putStrLn $ formatResult [(metricKey, show valBpc),
                            ("epochs", show epochsDone),
                            ("seed", show cfg.seed)]
