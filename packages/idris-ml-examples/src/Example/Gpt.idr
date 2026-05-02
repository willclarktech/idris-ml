-- | GPT: Character-Level Language Model
-- |
-- | Character-level language model on embedded Shakespeare text, following
-- | Karpathy's char-rnn/minGPT tradition. Reuses the multi-block transformer
-- | with learned embeddings, sinusoidal PE, and causal self-attention.
-- |
-- | Input: sliding window of SeqLen characters from corpus (one-hot)
-- | Target: shifted by 1 (next character at each position)

module Example.Gpt

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
import Endofunctor
import Floating
import Generate
import Layer
import Layer.Core
import Layer.Transformer
import Math
import Sampler
import Schedule
import Tensor
import Train
import Util
import Device
import Variable


----------------------------------------------------------------------
-- Configuration
----------------------------------------------------------------------

||| Vocabulary size — fixed at compile time at the size of the
||| tinyshakespeare 65-char vocab. Both the embedded smoke corpus and the
||| convergence-run tinyshakespeare corpus share this vocabulary; the
||| embedded corpus uses a subset of the same character indices.
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

InputDim : Nat
InputDim = SeqLen

OutputDim : Nat
OutputDim = SeqLen * VocabSize


----------------------------------------------------------------------
-- Corpus & Tokenization
----------------------------------------------------------------------

-- Shakespeare — "All the world's a stage" + Hamlet soliloquy
||| Embedded smoke-gate corpus: a 1342-char Shakespeare excerpt. Used by
||| `--corpus embedded` (the default for the smoke gate's quick wiring
||| test). Convergence runs use `--corpus tinyshakespeare` which loads
||| the canonical ~1.1 MB benchmark file from
||| data/tinyshakespeare/input.txt.
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

||| The 65 distinct characters in the tinyshakespeare corpus, in the same
||| order nanoGPT uses (sorted by codepoint). The smoke-gate embedded
||| corpus is a strict subset of these characters, so this single vocab
||| serves both paths.
||| Indices: \n=0, space=1, !=2, $=3, &=4, '=5, ,=6, -=7, .=8, 3=9,
||| :=10, ;=11, ?=12, A-Z=13..38, a-z=39..64.
vocabChars : String
vocabChars = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

||| Map character to token index. Unknown chars → space (id 1).
||| Linear scan over 65 chars; ~negligible cost vs the model forward.
charToIdx : Char -> Int
charToIdx c = go (unpack vocabChars) 0
  where
    go : List Char -> Int -> Int
    go [] _ = 1  -- unknown -> space
    go (x :: xs) i = if x == c then i else go xs (i + 1)

||| Map token index back to character. Out-of-range → space.
idxToChar : Int -> Char
idxToChar i =
  case strIndex vocabChars i of
    Just c => c
    Nothing => ' '
  where
    strIndex : String -> Int -> Maybe Char
    strIndex s n =
      let chars = unpack s
          k = integerToNat (cast n)
      in case drop k chars of
           [] => Nothing
           (c :: _) => Just c


----------------------------------------------------------------------
-- Helpers
----------------------------------------------------------------------

||| Pack a list of Ints into a C int buffer.
packIntBuf : AnyPtr -> Int -> List Int -> AnyPtr
packIntBuf buf _ [] = buf
packIntBuf buf off (tok :: rest) =
  let buf' = prim__setInt buf off tok
  in packIntBuf buf' (off + 1) rest

||| Pack a list of Ints as Doubles into a C double buffer.
packDoubleBuf : AnyPtr -> Int -> List Int -> AnyPtr
packDoubleBuf buf _ [] = buf
packDoubleBuf buf off (tok :: rest) =
  packDoubleBuf (prim__setDouble buf off (cast {to=Double} tok)) (off + 1) rest


----------------------------------------------------------------------
-- Data Generation
----------------------------------------------------------------------

||| List indexing (0-based).
listIdx : List a -> Nat -> Maybe a
listIdx [] _ = Nothing
listIdx (x :: _) Z = Just x
listIdx (_ :: xs) (S k) = listIdx xs k

||| Extract a slice [start, start+len) from a list.
listSlice : List a -> Nat -> Nat -> List a
listSlice xs start len = Data.List.take len (drop start xs)

||| Generate one GPT data point: random sliding window from `corpus`.
||| `corpus` is the (already-encoded) list of token indices; `corpusLen`
||| is its length. Both are passed in so that callers can feed train or
||| val sub-corpora rather than always sampling from the full text.
||| Input = tokens[0..SeqLen-1], Target = tokens[1..SeqLen], both one-hot.
gptTensorPoint :
  (corpus : List Int) -> (corpusLen : Nat) ->
  IO (TensorDataPoint InputDim OutputDim)
gptTensorPoint corpus corpusLen = do
  let maxStart = minus corpusLen (SeqLen + 1)
  start <- randomInt 0 maxStart
  let window = listSlice corpus start (SeqLen + 1)
      inputToks = Data.List.take SeqLen window
      targetToks = Data.List.take SeqLen (drop 1 window)
      sI = cast {to=Int} SeqLen
      vI = cast {to=Int} VocabSize
      -- Input: token indices as doubles [seqLen]
      inT = prim__create1d sI (packDoubleBuf (prim__allocDoubles sI) 0 inputToks) 0
      -- Target: still one-hot [seqLen * vocabSize] for cross-entropy
      tgtIdxBuf = packIntBuf (prim__allocInts sI) 0 targetToks
  pure $ MkTensorDataPoint inT (prim__oneHot tgtIdxBuf sI vI)

||| Generate a batch of GPT data points from `corpus`.
gptBatchVect :
  (corpus : List Int) -> (corpusLen : Nat) ->
  (n : Nat) -> IO (Vect n (TensorDataPoint InputDim OutputDim))
gptBatchVect _ _ Z = pure []
gptBatchVect corpus corpusLen (S k) = do
  dp <- gptTensorPoint corpus corpusLen
  rest <- gptBatchVect corpus corpusLen k
  pure (dp :: rest)


----------------------------------------------------------------------
-- Loss: Cross-entropy on all positions
----------------------------------------------------------------------

||| Categorical cross-entropy on ALL positions (standard LM loss).
allPositionsCE : LossFnTensor CPU
allPositionsCE predT targetT =
  let vsI = cast {to=Int} VocabSize
      sI = cast {to=Int} SeqLen
      logitsR = prim__reshape2d predT sI vsI
      logProbs = prim__logSoftmax2d logitsR
      tgtsR = prim__reshape2d targetT sI vsI
      product = prim__mul logProbs tgtsR
      totalSum = prim__sum product
      loss = prim__mulScalar (prim__neg totalSum) (1.0 / cast {to=Double} SeqLen)
      val = prim__item loss
  in Var loss Nothing val


----------------------------------------------------------------------
-- Autoregressive Generation
----------------------------------------------------------------------

||| Generate text autoregressively from a seed string.
||| Returns seed ++ generated characters.
generateText : {hs : List Nat} ->
               Network InputDim hs OutputDim (Variable CPU) ->
               String -> Nat -> Double -> String
generateText model seed genLen temperature =
  let seedIdxs = map charToIdx (unpack seed)
      padLen = minus SeqLen (length seedIdxs)
      -- Left-pad with space (id 1 in the 65-char vocab) to fill SeqLen
      context = replicate padLen (the Int 1) ++ Data.List.take SeqLen seedIdxs
  in seed ++ pack (go model context genLen [])
  where
    -- Enumerate all 65 vocab indices for argmax over logits.
    vocabIdxs : List Nat
    vocabIdxs = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
                ,10,11,12,13,14,15,16,17,18,19
                ,20,21,22,23,24,25,26,27,28,29
                ,30,31,32,33,34,35,36,37,38,39
                ,40,41,42,43,44,45,46,47,48,49
                ,50,51,52,53,54,55,56,57,58,59
                ,60,61,62,63,64
                ]

    ||| Extract logits at a given position, apply temperature, return unnormalized probs.
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
      fst (foldl (\(bi,bv), (i,v) => if v > bv then (i,v) else (bi,bv))
           (the (Int, Double) (0, -1.0e10))
           (zip (map cast vocabIdxs) probs))

    go : {hs' : List Nat} -> Network InputDim hs' OutputDim (Variable CPU) ->
         List Int -> Nat -> List Char -> List Char
    go _ _ Z acc = reverse acc
    go m ctx (S k) acc =
      let sI = cast {to=Int} SeqLen
          vI = cast {to=Int} VocabSize
          inT = prim__create1d sI (packDoubleBuf (prim__allocDoubles sI) 0 ctx) 0
          fwdPair = forwardVarTensor m inT
          outT = snd fwdPair
          unnorm = sampleAt outT (minus SeqLen 1)
          totSum = foldl (+) 0.0 unnorm
          probs = map (/ totSum) unnorm
          bestIdx = argmax probs
          ch = idxToChar bestIdx
          ctx' = drop 1 ctx ++ [bestIdx]
      in go m ctx' k (ch :: acc)


----------------------------------------------------------------------
-- Evaluation
----------------------------------------------------------------------

||| Evaluate bits per character over `corpus` at evenly-spaced windows.
||| Pass the (held-out) val sub-corpus to compute val_bpc, or the train
||| sub-corpus to compute train_bpc.
evalBPC : {hs : List Nat} ->
          Network InputDim hs OutputDim (Variable CPU) ->
          (corpus : List Int) -> (corpusLen : Nat) ->
          (nSamples : Nat) -> Double
evalBPC model corpus corpusLen nSamples = go model nSamples 0.0
  where
    singleBPC : {hs' : List Nat} ->
                Network InputDim hs' OutputDim (Variable CPU) -> Nat -> Double
    singleBPC m start =
      let window = listSlice corpus start (SeqLen + 1)
          inputToks = Data.List.take SeqLen window
          targetToks = Data.List.take SeqLen (drop 1 window)
          sI = cast {to=Int} SeqLen
          vI = cast {to=Int} VocabSize
          inT = prim__create1d sI (packDoubleBuf (prim__allocDoubles sI) 0 inputToks) 0
          tgtIdxBuf = packIntBuf (prim__allocInts sI) 0 targetToks
          tgtT = prim__oneHot tgtIdxBuf sI vI
          fwdPair = forwardVarTensor m inT
          outT = snd fwdPair
          loss = allPositionsCE outT tgtT
      in loss.value / log 2.0

    go : {hs' : List Nat} ->
         Network InputDim hs' OutputDim (Variable CPU) -> Nat -> Double -> Double
    go _ Z acc = acc
    go m (S k) acc =
      let maxStart = minus corpusLen (SeqLen + 1)
          -- Deterministic eval positions: evenly spaced
          pos = div (k * maxStart) nSamples
          bpc = singleBPC m pos
      in go m k (acc + bpc / cast {to=Double} (natToInteger nSamples))


----------------------------------------------------------------------
-- Corpus loading + train/val split
----------------------------------------------------------------------

||| Path to the tinyshakespeare benchmark file. Run `make
||| dataset-tinyshakespeare` to populate it.
tinyshakespearePath : String
tinyshakespearePath = "data/tinyshakespeare/input.txt"

||| Load a corpus by name. "embedded" returns the small smoke-gate
||| corpus; "tinyshakespeare" reads the canonical benchmark file.
||| Falls back to embedded with a warning if the file is missing.
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

||| Deterministic 90/10 split: last `valFrac` fraction of indices is val.
trainValSplit : (valFrac : Double) -> List Int ->
                (List Int, List Int)
trainValSplit valFrac idx =
  let n = length idx
      nVal = the Nat (cast (cast {to=Double} (natToInteger n) * valFrac))
      nTrain = minus n nVal
  in (Data.List.take nTrain idx, drop nTrain idx)


----------------------------------------------------------------------
-- LR-schedule helper: update all registered params each epoch.
----------------------------------------------------------------------

||| Update LR for every registered parameter — invoked once per epoch from
||| the training loop. Iterates the param registry and calls `setParamLR`
||| (which the optimizer reads on its next step). The wrapping `pure`
||| forces evaluation of the side-effecting setParamLR call inside IO,
||| matching the existing `polyakUpdate` pattern in Variable.idr.
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
  corpus : String       -- "tinyshakespeare" or "embedded"
  lr : Double
  epochs : Nat
  patience : Nat        -- 0 = disabled (rely on cosine LR for annealing)
  seed : Bits64

||| Defaults: tinyshakespeare corpus, 1000 epochs, no patience-based
||| stopping (cosine LR + warmup handles annealing). nanoGPT-aligned
||| optimizer params live in main() since they're not user-tunable.
defaultConfig : Config
defaultConfig = MkConfig "tinyshakespeare" 0.001 1000 0 42

specs : List (ArgSpec Config)
specs = [ Arg "--corpus" (\v, c => { corpus := v } c)
        , Arg "--lr" (\v, c => { lr := cast v } c)
        , Arg "--epochs" (\v, c => { epochs := castNat v } c)
        , Arg "--patience" (\v, c => { patience := castNat v } c)
        , Arg "--seed" (\v, c => { seed := castBits64 v } c) ]


partial
main : IO ()
main = do
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)

  srand cfg.seed

  -- nanoGPT-aligned optimizer recipe: AdamW β1=0.9 β2=0.99 wd=0.1
  -- + cosine LR with linear warmup (set per-epoch via setLRAll).
  -- Last param is the global grad-norm clip = 1.0.
  let opt = nativeAdamW cfg.lr 0.9 0.99 1.0e-8 0.1 1.0

  -- ---- Corpus + train/val split ----
  corpusText <- loadCorpusText cfg.corpus
  let allIndices = map charToIdx (unpack corpusText)
      (trainIndices, valIndices) =
        if cfg.corpus == "embedded"
          then (allIndices, allIndices)   -- smoke path: same set; val == train
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

  tfm <- mkTransformer {seqLen=SeqLen, dModel=DModel, numHeads=NumHeads,
                         headDim=HeadDim, numBlocks=NumBlocks, vocabSize=VocabSize}
  let namedTfm = nameLayer "tfm0" tfm
      model = OutputLayer (MkAnyLayer
        (TransformerState SeqLen DModel NumHeads HeadDim NumBlocks VocabSize) namedTfm)
  putStrLn $ "Model: " ++ show model
  putStrLn ""

  -- ---- LR schedule (cosineWithWarmup) + epoch counter ----
  -- 100-epoch linear warmup → cosine decay from cfg.lr to cfg.lr * 0.1
  -- Matches nanoGPT/train_shakespeare_char.py defaults.
  let warmupEpochs : Nat = 100
      minLR : Double = cfg.lr * 0.1
      schedule : Schedule = cosineWithWarmup cfg.lr minLR warmupEpochs cfg.epochs
  epochRef <- newIORef Z

  let genBatch : IO (Vect BatchSize (TensorDataPoint InputDim OutputDim))
      genBatch = gptBatchVect trainIndices trainLen BatchSize

  let evalMetrics : Network InputDim [] OutputDim (Variable CPU) -> IO (List (String, String))
      evalMetrics m = do
        let valBpc = evalBPC m valIndices valLen 20
            curEp = the Nat (cast (cast {to=Double} 0))  -- placeholder if needed
        pure [("val_bpc", show valBpc)]

  let noOpHook : Nat -> IO ()
      noOpHook _ = pure ()

  let trainCfg = MkTrainConfig cfg.epochs 100
                   (if cfg.patience == 0
                      then NoEarlyStop
                      else Patience cfg.patience 0.001)
                   evalMetrics
                   noOpHook

  let batchFwd = transformerForwardBatch namedTfm
  let stepFn : Network InputDim [] OutputDim (Variable CPU) ->
               Vect BatchSize (TensorDataPoint InputDim OutputDim) ->
               IO (Network InputDim [] OutputDim (Variable CPU), Double)
      stepFn m d = do
        ep <- readIORef epochRef
        let lr = schedule ep
        setLRAll opt lr
        writeIORef epochRef (S ep)
        pure (epochNativeTensorBatch opt d batchFwd allPositionsCE m)

  (trained, epochsDone, finalLoss) <- runTrainingIO stepFn genBatch trainCfg model

  -- ---- Final eval: held-out val_bpc plus train_bpc for diagnostics ----
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
  -- RESULT key: val_bpc on tinyshakespeare (held-out), bpc on embedded
  -- (train-corpus is the only set available for the smoke gate).
  let metricKey = if cfg.corpus == "embedded" then "bpc" else "val_bpc"
  putStrLn $ formatResult [(metricKey, show valBpc),
                            ("epochs", show epochsDone),
                            ("seed", show cfg.seed)]
