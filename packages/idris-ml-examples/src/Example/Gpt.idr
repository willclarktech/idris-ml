-- | GPT: Character-Level Language Model
-- |
-- | Character-level language model on embedded Shakespeare text, following
-- | Karpathy's char-rnn/minGPT tradition. Reuses the multi-block transformer
-- | with learned embeddings, sinusoidal PE, and causal self-attention.
-- |
-- | Input: sliding window of SeqLen characters from corpus (one-hot)
-- | Target: shifted by 1 (next character at each position)

module Example.Gpt

import Data.List
import Data.String
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
import Sampler
import Tensor
import Train
import Util
import Device
import Variable


----------------------------------------------------------------------
-- Configuration
----------------------------------------------------------------------

VocabSize : Nat
VocabSize = 36        -- a-z (0-25), space (26), newline (27), .,';:!?- (28-35)

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
Corpus : String
Corpus = "all the world's a stage, and all the men and women merely players; "
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

||| Map character to token index (0-35). Unknown chars become space (26).
charToIdx : Char -> Int
charToIdx c =
  if 'a' <= c && c <= 'z' then cast (ord c - ord 'a')
  else if c == ' ' then 26
  else if c == '\n' then 27
  else if c == '.' then 28
  else if c == ',' then 29
  else if c == '\'' then 30
  else if c == ';' then 31
  else if c == ':' then 32
  else if c == '!' then 33
  else if c == '?' then 34
  else if c == '-' then 35
  else 26  -- unknown -> space

||| Map token index back to character.
idxToChar : Int -> Char
idxToChar i =
  if 0 <= i && i <= 25 then chr (cast i + ord 'a')
  else if i == 26 then ' '
  else if i == 27 then '\n'
  else if i == 28 then '.'
  else if i == 29 then ','
  else if i == 30 then '\''
  else if i == 31 then ';'
  else if i == 32 then ':'
  else if i == 33 then '!'
  else if i == 34 then '?'
  else if i == 35 then '-'
  else ' '

||| Encode full corpus to list of token indices.
corpusIndices : List Int
corpusIndices = map charToIdx (unpack Corpus)

corpusLen : Nat
corpusLen = length corpusIndices


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

||| Generate one GPT data point: random sliding window from corpus.
||| Input = tokens[0..SeqLen-1], Target = tokens[1..SeqLen], both one-hot.
gptTensorPoint : IO (TensorDataPoint InputDim OutputDim)
gptTensorPoint = do
  let maxStart = minus corpusLen (SeqLen + 1)
  start <- randomInt 0 maxStart
  let window = listSlice corpusIndices start (SeqLen + 1)
      inputToks = Data.List.take SeqLen window
      targetToks = Data.List.take SeqLen (drop 1 window)
      sI = cast {to=Int} SeqLen
      vI = cast {to=Int} VocabSize
      -- Input: token indices as doubles [seqLen]
      inT = prim__create1d sI (packDoubleBuf (prim__allocDoubles sI) 0 inputToks) 0
      -- Target: still one-hot [seqLen * vocabSize] for cross-entropy
      tgtIdxBuf = packIntBuf (prim__allocInts sI) 0 targetToks
  pure $ MkTensorDataPoint inT (prim__oneHot tgtIdxBuf sI vI)

||| Generate a batch of GPT data points.
gptBatchVect : (n : Nat) -> IO (Vect n (TensorDataPoint InputDim OutputDim))
gptBatchVect Z = pure []
gptBatchVect (S k) = do
  dp <- gptTensorPoint
  rest <- gptBatchVect k
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
      -- Left-pad with spaces to fill SeqLen
      context = replicate padLen (the Int 26) ++ Data.List.take SeqLen seedIdxs
  in seed ++ pack (go model context genLen [])
  where
    vocabIdxs : List Nat
    vocabIdxs = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]

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

||| Evaluate bits per character on random windows.
evalBPC : {hs : List Nat} ->
          Network InputDim hs OutputDim (Variable CPU) -> Nat -> Double
evalBPC model nSamples = go model nSamples 0.0
  where
    singleBPC : {hs' : List Nat} ->
                Network InputDim hs' OutputDim (Variable CPU) -> Nat -> Double
    singleBPC m start =
      let window = listSlice corpusIndices start (SeqLen + 1)
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
-- Config & Main
----------------------------------------------------------------------

record Config where
  constructor MkConfig
  lr : Double
  epochs : Nat
  patience : Nat
  seed : Bits64

defaultConfig : Config
defaultConfig = MkConfig 0.001 2000 500 42

specs : List (ArgSpec Config)
specs = [ Arg "--lr" (\v, c => { lr := cast v } c)
        , Arg "--epochs" (\v, c => { epochs := castNat v } c)
        , Arg "--patience" (\v, c => { patience := castNat v } c)
        , Arg "--seed" (\v, c => { seed := castBits64 v } c) ]


partial
main : IO ()
main = do
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)

  srand cfg.seed

  let opt = nativeAdamW cfg.lr 0.9 0.999 1.0e-8 0.01 1.0

  putStrLn "=== GPT: Character-Level Language Model ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr ++ " epochs=" ++ show cfg.epochs
           ++ " patience=" ++ show cfg.patience ++ " seed=" ++ show cfg.seed
  putStrLn $ "Architecture: seqLen=" ++ show SeqLen ++ " dModel=" ++ show DModel
           ++ " heads=" ++ show NumHeads ++ " headDim=" ++ show HeadDim
           ++ " blocks=" ++ show NumBlocks ++ " vocab=" ++ show VocabSize
  putStrLn $ "Corpus: " ++ show corpusLen ++ " chars"

  tfm <- mkTransformer {seqLen=SeqLen, dModel=DModel, numHeads=NumHeads,
                         headDim=HeadDim, numBlocks=NumBlocks, vocabSize=VocabSize}
  let namedTfm = nameLayer "tfm0" tfm
      model = OutputLayer (MkAnyLayer
        (TransformerState SeqLen DModel NumHeads HeadDim NumBlocks VocabSize) namedTfm)
  putStrLn $ "Model: " ++ show model
  putStrLn ""

  let genBatch : IO (Vect BatchSize (TensorDataPoint InputDim OutputDim))
      genBatch = gptBatchVect BatchSize

  let evalMetrics : Network InputDim [] OutputDim (Variable CPU) -> IO (List (String, String))
      evalMetrics m = do
        let bpc = evalBPC m 20
        pure [("bpc", show bpc)]

  let trainCfg = MkTrainConfig cfg.epochs 100 (Patience cfg.patience 0.001) evalMetrics

  let batchFwd = transformerForwardBatch namedTfm
  (trained, epochsDone, finalLoss) <- runTraining
    (\m, d => epochNativeTensorBatch opt d batchFwd allPositionsCE m) genBatch trainCfg model

  putStrLn ""
  let bpc = evalBPC trained 50
  putStrLn $ "Final BPC: " ++ show bpc

  putStrLn ""
  putStrLn "Generation (seed='to be or '):"
  let sample1 = generateText trained "to be or " 100 1.0
  putStrLn $ "  " ++ show sample1

  putStrLn ""
  putStrLn "Generation (seed='the '):"
  let sample2 = generateText trained "the " 100 1.0
  putStrLn $ "  " ++ show sample2

  putStrLn ""
  putStrLn $ formatResult [("bpc", show bpc),
                            ("epochs", show epochsDone),
                            ("seed", show cfg.seed)]
