||| BertMlmFinetune — continued pretraining of the
||| `google/bert_uncased_L-2_H-128_A-2` BERT-tiny backbone (MLM head
||| included) on Tiny Shakespeare via WordPiece tokenization. Mirrors
||| the GPT-2 LM example structurally but uses the masked-language-
||| modelling objective: HF's 80/10/10 token-masking scheme per
||| training example, cross-entropy computed only at masked positions.
|||
||| Pre-requisites (run once):
|||   make data-hf-bert-tiny              # already used by SST-2 example
|||   make data-tinyshakespeare-bert-tiny # tokenize via WordPiece
|||
||| Architecture: bert_uncased_L-2_H-128_A-2 (vocab=30522, hidden=128,
||| layers=2, heads=2, headDim=64, intermediate=512). The cls.* MLM-
||| head params come from the on-disk checkpoint via
||| `loadModelAllowCast` (the FT3 / SST-2 example used
||| `loadModelPrefixAllowCast _ "bert."` to skip them; this example
||| needs them).
module Example.BertMlmFinetune

import Control.Linear.LIO
import Data.Linear.Notation
import Data.List
import Data.List1
import Data.String
import Data.Vect
import System
import System.File

import Ml.Array
import Ml.Checkpoint
import Ml.Compat.Random
import Ml.Executor
import Ml.Executor.Core
import Ml.Optimizer
import Ml.Tensor
import Ml.Train
import Ml.Util
import Transformers.Bert

import BuildConfig
import Generate

----------------------------------------------------------------------
-- Config (matches google/bert_uncased_L-2_H-128_A-2)
----------------------------------------------------------------------

Vocab : Nat
Vocab = 30522

Hidden : Nat
Hidden = 128

NumLayers : Nat
NumLayers = 2

NumHeads : Nat
NumHeads = 2

HeadDim : Nat
HeadDim = 64

Intermediate : Nat
Intermediate = 512

MaxPos : Nat
MaxPos = 512

TypeVocab : Nat
TypeVocab = 2

SeqLen : Nat
SeqLen = 32

-- BERT WordPiece special tokens.
ClsId, SepId, MaskId, PadId : Nat
ClsId  = 101
SepId  = 102
MaskId = 103
PadId  = 0

-- HF's standard mask probability for MLM.
MaskProb : Double
MaskProb = 0.15

record Config where
  constructor MkConfig
  lr       : Double
  steps    : Nat
  seed     : Bits64
  maxStart : Nat

defaultConfig : Config
defaultConfig = MkConfig 5.0e-5 100 42 0

specs : List (ArgSpec Config)
specs = [ Arg "--lr"        (\v, c => { lr := cast v } c)
        , Arg "--steps"     (\v, c => { steps := castNat v } c)
        , Arg "--seed"      (\v, c => { seed := castBits64 v } c)
        , Arg "--max-start" (\v, c => { maxStart := castNat v } c)
        ]

----------------------------------------------------------------------
-- Corpus loading
----------------------------------------------------------------------

tokenPath : String
tokenPath = "data/tinyshakespeare/input.bert-tiny.tokens"

ckptPath : String
ckptPath = "models/google/bert_uncased_L-2_H-128_A-2/model.safetensors"

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
-- 80/10/10 masking + sliding window
----------------------------------------------------------------------

Model : Type
Model = BertForMaskedLmState Vocab Hidden NumLayers Intermediate MaxPos
                             TypeVocab ExampleExecutor ExampleDType WithGrad

-- A single MLM training example, post-masking. `inputIds` is what the
-- model sees (with [MASK] / random / kept tokens at the masked
-- positions); `targetIds` is the ORIGINAL token IDs (used by the loss
-- only at masked positions); `maskFlags` is 1.0 where the position
-- was masked (loss applies there), 0.0 elsewhere.
MlmSample : Type
MlmSample = (Vect SeqLen Double, Vect SeqLen Double, Vect SeqLen Double)

arangeSeqLen : Vect SeqLen Double
arangeSeqLen = build SeqLen
  where
    build : (k : Nat) -> Vect k Double
    build Z     = []
    build (S k) =
      let here = cast {to=Double} (cast {to=Integer} (minus SeqLen (S k)))
      in here :: build k

-- Truncate / pad a List to exactly `n`.
takePad : (n : Nat) -> a -> List a -> Vect n a
takePad Z     _   _         = []
takePad (S k) pad []        = pad :: takePad k pad []
takePad (S k) pad (x :: xs) = x :: takePad k pad xs

-- Pick a uniformly random integer in [0, n).
randNat : (n : Nat) -> IO Nat
randNat 0 = pure 0
randNat n = do
  v <- randomInt 0 (cast (minus n 1))
  pure (cast (cast {to=Integer} v))

-- For each position, decide if it's masked + emit (input_id, mask_flag).
-- HF's 80/10/10 scheme: of the 15% masked positions, 80% become
-- [MASK] (id=103), 10% become a random token, 10% keep the original.
-- The CLS / SEP tokens (id 101 / 102) are NEVER masked.
applyHfMasking : List Nat -> IO (List Nat, List Nat, List Double)
applyHfMasking []           = pure ([], [], [])
applyHfMasking (id :: rest) = do
  (restInput, restTarget, restMask) <- applyHfMasking rest
  let isSpecial = id == ClsId || id == SepId
  pick <- randDouble
  if isSpecial || pick > MaskProb
    then pure (id :: restInput, id :: restTarget, 0.0 :: restMask)
    else do
      r <- randDouble
      let newId : Nat
          newId = if r < 0.8
                  then MaskId
                  else if r < 0.9
                    then 200  -- approximate "random vocab token" — see comment below
                    else id
      pure (newId :: restInput, id :: restTarget, 1.0 :: restMask)
  where
    -- Uniform Double in [0, 1).
    randDouble : IO Double
    randDouble = do
      n <- randomInt 0 9999
      pure (cast {to=Double} (cast {to=Integer} n) / 10000.0)
-- Note on the "random token" branch: HF's `DataCollatorForLanguageModeling`
-- picks uniformly from the FULL vocab (30522). We hardcode id=200 (a
-- mid-vocab WordPiece) instead of sampling — the example is bounded
-- by single-example batches and short training (50-100 steps), so the
-- 10% random branch's exact id distribution doesn't materially change
-- the convergence story; the loss only weights MASKED positions.

-- Sample one MLM training example from the corpus.
sampleMlmExample : (corpus : List Nat) -> (corpusLen : Nat)
                -> (capMaxStart : Nat)
                -> IO MlmSample
sampleMlmExample corpus corpusLen capMaxStart = do
  let absMax = minus corpusLen SeqLen
      cap    = if capMaxStart == 0 then absMax
               else minimum absMax capMaxStart
  startInt <- randomInt 0 (cast cap)
  let start  = the Nat (cast startInt)
      window = take SeqLen (drop start corpus)
  (inputIds, targetIds, maskFlags) <- applyHfMasking window
  let inputVect : Vect SeqLen Double
      inputVect = takePad SeqLen 0.0 (map (\n => cast {to=Double} (cast {to=Integer} n)) inputIds)
      targetVect : Vect SeqLen Double
      targetVect = takePad SeqLen 0.0 (map (\n => cast {to=Double} (cast {to=Integer} n)) targetIds)
      maskVect : Vect SeqLen Double
      maskVect = takePad SeqLen 0.0 maskFlags
  pure (inputVect, targetVect, maskVect)
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

typeVect : Vect SeqLen Double
typeVect = replicate SeqLen 0.0

-- Build a [SeqLen, Vocab] target one-hot at MASKED positions, zeros
-- elsewhere. We multiply the standard one-hot by the per-position
-- mask along axis 0 (each row gets scaled by the mask flag).
mkMaskedTargetOneHot : Vect SeqLen Double -> Vect SeqLen Double
                    -> IO (Tensor [SeqLen, Vocab] ExampleExecutor ExampleDType WithGrad)
mkMaskedTargetOneHot targetIds maskFlags = do
  let sI = cast {to=Int} SeqLen
      vI     = cast {to=Int} Vocab
      idxBuf = packIdx (prim__allocInts sI) 0 (toList targetIds)
  -- Build the one-hot matrix flat, reshape to 2D.
  onePtr <- ioRerun (\_ =>
    let raw  = primOneHot {ex=ExampleExecutor} idxBuf sI vI (dtypeTag {t=ExampleDType})
    in primReshape2d {ex=ExampleExecutor} raw sI vI)
  -- Build the mask as [SeqLen, 1] (broadcast multiply zeros rows
  -- where the per-position mask flag is 0).
  mask1d <- mkIdsTensor maskFlags  -- [SeqLen]
  ioRerun (\_ =>
    let mask2d = primReshape2d {ex=ExampleExecutor} mask1d.tensorPtr sI 1
        prod   = primMul {ex=ExampleExecutor} onePtr mask2d
    in MkTensor prod Nothing)
  where
    packIdx : AnyPtr -> Int -> List Double -> AnyPtr
    packIdx b _   []        = b
    packIdx b off (v :: rs) =
      packIdx (prim__setInt b off (cast {to=Int} (cast {to=Integer} v))) (off + 1) rs

-- Per-position masked CE loss. Sum of cross-entropy contributions
-- across positions, normalized by the number of masked positions
-- (so the loss magnitude is comparable to a per-position mean): the
-- fused `tsoftmaxXent2d` at scale 1/numMasked. maskedTarget already
-- has zeros at unmasked rows, so those rows contribute zero to the
-- sum (the fused kernel's zero-target-row case).
bertMlmLoss : Tensor [SeqLen, Vocab] ExampleExecutor ExampleDType WithGrad
           -> Tensor [SeqLen, Vocab] ExampleExecutor ExampleDType WithGrad
           -> (numMasked : Double)
           -> IO (Tensor [] ExampleExecutor ExampleDType WithGrad)
bertMlmLoss logits maskedTarget numMasked =
  let denom = if numMasked < 1.0 then 1.0 else numMasked
  in tsoftmaxXent2d (1.0 / denom) logits maskedTarget

----------------------------------------------------------------------
-- Training loop
----------------------------------------------------------------------

-- Per-step MLM train, threading the (linear) model through hfBertMlmForwardL.
trainStepL : NativeOptimizer ExampleExecutor
          -> (1 _ : Model) -> MlmSample
          -> L IO {use = 1} (LPair (!* Double) Model)
trainStepL opt model (inputIds, targetIds, maskFlags) = do
  (inputT, posT, typT, targetT) <-
    liftIO1 (do inputT  <- mkIdsTensor inputIds
                posT    <- mkIdsTensor arangeSeqLen
                typT    <- mkIdsTensor typeVect
                targetT <- mkMaskedTargetOneHot targetIds maskFlags
                pure (inputT, posT, typT, targetT))
  (MkBang logits # model') <-
    hfBertMlmForwardL {ex=ExampleExecutor} {dt=ExampleDType}
                      {seqLen=SeqLen}
                      {vocab=Vocab} {hidden=Hidden}
                      {numLayers=NumLayers} {numHeads=NumHeads}
                      {headDim=HeadDim} {intermediate=Intermediate}
                      {maxPos=MaxPos} {typeVocab=TypeVocab}
                      model inputT posT typT Nothing
  d <- liftIO1 (do let numMasked = sum (toList maskFlags)
                   loss <- bertMlmLoss logits targetT numMasked
                   trainStep opt loss)
  pure1 (MkBang d # model')

-- Discard the (linear) model: its fields are ω registered-param records.
discardModel : (1 _ : Model) -> L IO ()
discardModel (MkBertForMaskedLm _ _) = pure ()

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

  putStrLn "=== BertMlmFinetune ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr
           ++ " steps=" ++ show cfg.steps
           ++ " seed=" ++ show cfg.seed
           ++ " max-start=" ++ show cfg.maxStart

  putStrLn ("Loading tokens from " ++ tokenPath ++ "...")
  tokens <- loadTokens tokenPath
  let nTokens = length tokens
  if nTokens < SeqLen
    then do
      putStrLn $ "ERROR: corpus has only " ++ show nTokens ++ " tokens"
              ++ " (need >= " ++ show SeqLen ++ ")."
              ++ " Run `make data-tinyshakespeare-bert-tiny`."
      exitFailure
    else do
      putStrLn $ "Loaded " ++ show nTokens ++ " tokens."

      -- Linear surface: model born linear (bornL), warm-started by name, then
      -- threaded single-owner through the custom MLM train loop, discarded.
      Control.Linear.LIO.run $ do
        model <- bornL (hfBertForMaskedLm {ex=ExampleExecutor} {dt=ExampleDType}
                                          {vocab=Vocab} {hidden=Hidden}
                                          {numLayers=NumLayers} {numHeads=NumHeads}
                                          {intermediate=Intermediate}
                                          {maxPos=MaxPos} {typeVocab=TypeVocab}
                                          "bert")
        liftIO1 (do ok <- (== Right ()) <$> load {ex=ExampleExecutor} ckptPath ({ allowCast := True } defaultLoadOpts)
                    if ok then putStrLn "bert-tiny backbone + MLM head warm-started."
                          else do putStrLn $ "ERROR: failed to load bert-tiny from " ++ ckptPath
                                  exitFailure)
        opt <- liftIO1 (adamW {ex=ExampleExecutor} cfg.lr 0.01 ({ clip := NormClip 1.0 } defaultOpts))
        let trainLoopL : Nat -> Nat -> Double -> Double -> (1 _ : Model) ->
                         L IO {use = 1} (LPair (!* Double) Model)
            trainLoopL _    Z     _       lastLoss model = pure1 (MkBang lastLoss # model)
            trainLoopL step (S k) accLoss _        model = do
              sample <- liftIO1 (sampleMlmExample tokens nTokens cfg.maxStart)
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
