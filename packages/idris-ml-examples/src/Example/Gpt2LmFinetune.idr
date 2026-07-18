||| Gpt2LmFinetune — continued pretraining of distilgpt2 on Tiny
||| Shakespeare. Demonstrates the HF causal-LM fine-tune path:
||| backbone warm-start via `load … {allowCast := True}`, sliding-window
||| batches over a pre-tokenized corpus, per-position cross-entropy
||| against shifted-by-1 next-token targets.
|||
||| Pre-requisites (run once):
|||   make data-hf-distilgpt2     # fetches distilgpt2 weights
|||   make data-tinyshakespeare-distilgpt2  # tokenizes the corpus
|||                                          via distilgpt2's BPE
|||
||| The tokenized corpus lives at
||| `data/tinyshakespeare/input.distilgpt2.tokens` (single line of
||| comma-separated token IDs, ~338K tokens). The example reads it
||| via `readFile`, splits on `,`, samples random sliding windows of
||| (SeqLen+1) tokens per training example.
module Example.Gpt2LmFinetune

import Control.Linear.LIO
import Data.Linear.Notation
import Data.List
import Data.List1
import Data.Vect
import Data.String
import System
import System.File
import Ml.Compat.Random

import Ml.Array
import BuildConfig
import Ml.Executor
import Ml.Executor.Core
import Generate
import Ml.Tensor
import Ml.Train
import Ml.Util

import Ml.Checkpoint
import Transformers.Gpt2

----------------------------------------------------------------------
-- Config (matches distilgpt2)
----------------------------------------------------------------------

Vocab : Nat
Vocab = 50257

Hidden : Nat
Hidden = 768

NumLayers : Nat
NumLayers = 6

NumHeads : Nat
NumHeads = 12

HeadDim : Nat
HeadDim = 64

Intermediate : Nat
Intermediate = 3072

MaxPos : Nat
MaxPos = 1024

SeqLen : Nat
SeqLen = 32

record Config where
  constructor MkConfig
  lr       : Double
  steps    : Nat   -- number of train batches (1 batch = 1 example here, batched stack = future row)
  seed     : Bits64
  maxStart : Nat   -- corpus cap (0 = full)

defaultConfig : Config
defaultConfig = MkConfig 5.0e-5 100 42 0

specs : List (ArgSpec Config)
specs = [ Arg "--lr"        (\v, c => { lr := cast v } c)
        , Arg "--steps"     (\v, c => { steps := castNat v } c)
        , Arg "--seed"      (\v, c => { seed := castBits64 v } c)
        , Arg "--max-start" (\v, c => { maxStart := castNat v } c)
        ]

----------------------------------------------------------------------
-- Paths + corpus loader
----------------------------------------------------------------------

tokenPath : String
tokenPath = "data/tinyshakespeare/input.distilgpt2.tokens"

ckptPath : String
ckptPath = "models/distilgpt2/model.safetensors"

-- Parse a comma-separated token-id string into a List Nat. Drops any
-- chunk that doesn't parse cleanly (defensive — the tokenizer emits
-- pure integer text).
parseIds : String -> List Nat
parseIds s =
  let chunks = forget (split (== ',') (trim s))
  in mapMaybe parseOne chunks
  where
    parseOne : String -> Maybe Nat
    parseOne str =
      case parseInteger {a=Integer} (trim str) of
        Nothing => Nothing
        Just n  => if n < 0 then Nothing else Just (cast n)

loadTokens : (path : String) -> IO (List Nat)
loadTokens path = do
  res <- readFile path
  let body : String
      body = either (const "") id res
  pure (parseIds body)

----------------------------------------------------------------------
-- Sliding-window sampling
----------------------------------------------------------------------

Model : Type
Model = Gpt2ModelState Vocab Hidden NumLayers Intermediate MaxPos
                       ExampleExecutor ExampleDType WithGrad

-- Vect of token IDs (input) + the shifted-by-1 target IDs as a flat
-- Vect for the loss computation. Both are SeqLen-long Doubles.
WindowedSample : Type
WindowedSample = (Vect SeqLen Double, Vect SeqLen Double)

-- Build arange(SeqLen) as a Vect of Doubles.
arangeSeqLen : Vect SeqLen Double
arangeSeqLen = build SeqLen
  where
    build : (k : Nat) -> Vect k Double
    build Z     = []
    build (S k) =
      let here = cast {to=Double} (cast {to=Integer} (minus SeqLen (S k)))
      in here :: build k

-- Truncate a list to exactly `n`, padding with `pad` if short.
takePad : (n : Nat) -> a -> List a -> Vect n a
takePad Z     _   _         = []
takePad (S k) pad []        = pad :: takePad k pad []
takePad (S k) pad (x :: xs) = x :: takePad k pad xs

-- Sample one (input, target) pair. `corpus` is the full token list,
-- `corpusLen` is its length, `maxStart` caps the random start (0 =
-- use the whole corpus minus seqLen+1).
sampleWindow : (corpus : List Nat) -> (corpusLen : Nat)
            -> (maxStart : Nat)
            -> IO WindowedSample
sampleWindow corpus corpusLen capMaxStart = do
  let absMax = minus corpusLen (S SeqLen)
      cap    = if capMaxStart == 0 then absMax
               else minimum absMax capMaxStart
  startInt <- randomInt 0 (cast cap)
  let start  = the Nat (cast startInt)
      window    = take (S SeqLen) (drop start corpus)
      inputTok  = take SeqLen window
      targetTok = take SeqLen (drop 1 window)
  let inputVect : Vect SeqLen Double
      inputVect = takePad SeqLen 0.0 (map (\n => cast {to=Double} (cast {to=Integer} n)) inputTok)
      targetVect : Vect SeqLen Double
      targetVect = takePad SeqLen 0.0 (map (\n => cast {to=Double} (cast {to=Integer} n)) targetTok)
  pure (inputVect, targetVect)
  where
    minimum : Nat -> Nat -> Nat
    minimum a b = if a < b then a else b

----------------------------------------------------------------------
-- Tensor helpers
----------------------------------------------------------------------

mkIdsTensor : Vect SeqLen Double -> IO (Tensor [SeqLen] ExampleExecutor ExampleDType WithGrad)
mkIdsTensor xs = ioRerun (\_ =>
  let raw = bulkToTensor {ex=ExampleExecutor} {dt=ExampleDType}
                         (VArray (map SArray xs))
  in tinput1d {n=SeqLen} raw)

-- Build a [SeqLen, Vocab] one-hot target tensor from a Vect of target
-- token IDs. primOneHot returns the flat [seqLen * vocab] layout
-- (matches Example/Gpt's TVec OutputDim convention); we reshape to
-- 2D so the loss can multiply against [SeqLen, Vocab] logits directly.
mkTargetOneHot : Vect SeqLen Double -> IO (Tensor [SeqLen, Vocab] ExampleExecutor ExampleDType WithGrad)
mkTargetOneHot xs = do
  let sI = cast {to=Int} SeqLen
      vI     = cast {to=Int} Vocab
      idxBuf = packIdx (prim__allocInts sI) 0 (toList xs)
  ioRerun (\_ =>
    let flat   = primOneHot {ex=ExampleExecutor} idxBuf sI vI (dtypeTag {t=ExampleDType})
        as2d   = primReshape2d {ex=ExampleExecutor} flat sI vI
    in MkTensor as2d Nothing)
  where
    packIdx : AnyPtr -> Int -> List Double -> AnyPtr
    packIdx b _   []        = b
    packIdx b off (v :: rs) =
      packIdx (prim__setInt b off (cast {to=Int} (cast {to=Integer} v))) (off + 1) rs

-- Per-position CE loss between [seqLen, vocab] logits and [seqLen, vocab]
-- one-hot target. Mirrors Example/Gpt's `allPositionsCELoss` but starts
-- from a 2D tensor (no reshape needed).
gpt2LmLoss : Tensor [SeqLen, Vocab] ExampleExecutor ExampleDType WithGrad
          -> Tensor [SeqLen, Vocab] ExampleExecutor ExampleDType WithGrad
          -> IO (Tensor [] ExampleExecutor ExampleDType WithGrad)
gpt2LmLoss logits targets = ioRerun (\_ =>
  let logProbs = primLogSoftmax2d {ex=ExampleExecutor} logits.tensorPtr
      prod   = primMul {ex=ExampleExecutor} logProbs targets.tensorPtr
      summed = primSum {ex=ExampleExecutor} prod
      neg    = primNeg {ex=ExampleExecutor} summed
      loss   = primMulScalar {ex=ExampleExecutor} neg (1.0 / cast {to=Double} SeqLen)
  in MkTensor loss Nothing)

----------------------------------------------------------------------
-- Training loop
----------------------------------------------------------------------

-- Per-step LM train, threading the (linear) model through hfGpt2ForwardLmL.
trainStepL : NativeOptimizer ExampleExecutor
          -> (1 _ : Model)
          -> (Vect SeqLen Double, Vect SeqLen Double)
          -> L IO {use = 1} (LPair (!* Double) Model)
trainStepL opt model (inputTok, targetTok) = do
  (inputT, posT, targetT) <-
    liftIO1 (do inputT  <- mkIdsTensor inputTok
                posT    <- mkIdsTensor arangeSeqLen
                targetT <- mkTargetOneHot targetTok
                pure (inputT, posT, targetT))
  (MkBang logits # model') <-
    hfGpt2ForwardLmL {ex=ExampleExecutor} {dt=ExampleDType}
                     {seqLen=SeqLen}
                     {vocab=Vocab} {hidden=Hidden}
                     {numLayers=NumLayers} {numHeads=NumHeads}
                     {headDim=HeadDim} {intermediate=Intermediate}
                     {maxPos=MaxPos}
                     model inputT posT
  d <- liftIO1 (do loss <- gpt2LmLoss logits targetT; trainStep opt loss)
  pure1 (MkBang d # model')

-- Discard the (linear) model: its fields are ω registered-param records.
discardModel : (1 _ : Model) -> L IO ()
discardModel (MkGpt2Model _ _ _ _) = pure ()

----------------------------------------------------------------------
-- main
----------------------------------------------------------------------

%default partial

main : IO ()
main = do
  argv <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 argv)
  srand cfg.seed
  tsetInitSeed {ex = ExampleExecutor} cfg.seed

  putStrLn "=== Gpt2LmFinetune ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr
           ++ " steps=" ++ show cfg.steps
           ++ " seed=" ++ show cfg.seed
           ++ " max-start=" ++ show cfg.maxStart

  putStrLn ("Loading tokens from " ++ tokenPath ++ "...")
  tokens <- loadTokens tokenPath
  let nTokens = length tokens
  if nTokens < S SeqLen
    then do
      putStrLn $ "ERROR: corpus has only " ++ show nTokens
              ++ " tokens (need >= " ++ show (S SeqLen) ++ ")."
              ++ " Run `make data-tinyshakespeare-distilgpt2`."
      exitFailure
    else do
      putStrLn $ "Loaded " ++ show nTokens ++ " tokens."

      -- Linear surface: model born linear (bornL), warm-started by name, then
      -- threaded single-owner through the custom LM train loop, discarded.
      Control.Linear.LIO.run $ do
        model <- bornL (hfGpt2Model {ex=ExampleExecutor} {dt=ExampleDType}
                                    {vocab=Vocab} {hidden=Hidden}
                                    {numLayers=NumLayers} {numHeads=NumHeads}
                                    {headDim=HeadDim}
                                    {intermediate=Intermediate}
                                    {maxPos=MaxPos}
                                    "")
        liftIO1 (do ok <- (== Right ()) <$> load {ex=ExampleExecutor} ckptPath ({ allowCast := True } defaultLoadOpts)
                    if ok then putStrLn "distilgpt2 backbone warm-started."
                          else do putStrLn $ "ERROR: failed to load distilgpt2 from " ++ ckptPath
                                  exitFailure)
        opt <- liftIO1 (adamW {ex=ExampleExecutor} cfg.lr 0.01 ({ clip := NormClip 1.0 } defaultOpts))
        let trainLoopL : Nat -> Nat -> Double -> Double -> (1 _ : Model) ->
                         L IO {use = 1} (LPair (!* Double) Model)
            trainLoopL _    Z     _       lastLoss model = pure1 (MkBang lastLoss # model)
            trainLoopL step (S k) accLoss _        model = do
              sample <- liftIO1 (sampleWindow tokens nTokens cfg.maxStart)
              (MkBang loss # model') <- trainStepL opt model sample
              let acc'  = accLoss + loss
                  step' = step + 1
              liftIO1 (when (modNatNZ step' 10 SIsNonZero == 0) $
                         putStrLn $ "  step " ++ show step' ++ "/" ++ show cfg.steps
                                 ++ " loss=" ++ showFix 4 loss
                                 ++ "  ema=" ++ showFix 4 (acc' / cast {to=Double} step'))
              trainLoopL step' k acc' loss model'
        (MkBang finalLoss # trained) <- trainLoopL 0 cfg.steps 0.0 0.0 model
        discardModel trained
        liftIO1 $ do
          putStrLn ""
          putStrLn $ formatResult [ ("loss",  showFix 4 finalLoss)
                                  , ("steps", show cfg.steps)
                                  , ("seed",  show cfg.seed)
                                  ]
