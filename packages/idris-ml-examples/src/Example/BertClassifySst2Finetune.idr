||| BertClassifySst2Finetune — fine-tune a pretrained
||| `google/bert_uncased_L-2_H-128_A-2` backbone on a slice of the
||| GLUE SST-2 binary-sentiment classification task.
|||
||| Layered on RT1 (attention mask) + RT2 (HfDataset TSV loader). The
||| synthetic FT3 example demonstrates the from-scratch fine-tuning
||| code path; this one demonstrates the real-text path:
|||
|||   1. Backbone init at HF-canonical names; classifier head fresh-init
|||   2. `loadModelPrefix "models/.../model.safetensors" "bert."` warm-
|||      starts the backbone (head stays at its random init)
|||   3. Real SST-2 examples (variable length) padded to a fixed seqLen
|||      with an attention mask threading through `hfBertSeqClassify
|||      Forward`
|||   4. Standard cross-entropy + AdamW
|||
||| Pre-run: tokenized SST-2 train + validation must exist under
||| `data/hf-datasets/glue-sst2/{train,validation}.tsv`. The Makefile
||| target `data-sst2` invokes `scripts/hf-download-dataset.sh` for
||| both splits. Without those files the example exits non-zero with
||| a clear message rather than training on empty data.
module Example.BertClassifySst2Finetune

import Data.List
import Data.Vect
import System
import Compat.Random

import Array
import Backprop
import BuildConfig
import Executor
import Tensor
import Train
import Train.Freeze
import Util

import Checkpoint
import HfBert
import HfBertForClassification
import HfDataset

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

-- 2-class binary sentiment.
NumClasses : Nat
NumClasses = 2

-- Most SST-2 sentences fit in 32 WordPiece tokens (p95 ≈ 25). Pad/
-- truncate to this fixed length so the type-level seqLen is constant.
SeqLen : Nat
SeqLen = 32

-- BERT pad-token-id (canonical: 0 = [PAD]).
PadId : Nat
PadId = 0

record Config where
  constructor MkConfig
  lr             : Double
  epochs         : Nat
  seed           : Bits64
  freezeBackbone : Bool
  maxTrain       : Nat   -- subset of train examples to use (0 = use all)
  batchSize      : Nat
  maxDev         : Nat   -- subset of dev examples for held-out eval (0 = all)

defaultConfig : Config
-- lr=2e-5 matches HF's bert-base SST-2 tutorial; epochs=3 same.
defaultConfig = MkConfig 2.0e-5 3 42 False 256 8 256

boolFlag : String -> Bool
boolFlag v = v == "1" || v == "true" || v == "True" || v == "yes"

specs : List (ArgSpec Config)
specs = [ Arg "--lr"               (\v, c => { lr := cast v } c)
        , Arg "--epochs"           (\v, c => { epochs := castNat v } c)
        , Arg "--seed"             (\v, c => { seed := castBits64 v } c)
        , Arg "--freeze-backbone"  (\v, c => { freezeBackbone := boolFlag v } c)
        , Arg "--max-train"        (\v, c => { maxTrain := castNat v } c)
        , Arg "--batch-size"       (\v, c => { batchSize := castNat v } c)
        , Arg "--max-dev"          (\v, c => { maxDev := castNat v } c)
        ]

----------------------------------------------------------------------
-- Paths
----------------------------------------------------------------------

trainTsvPath : String
trainTsvPath = "data/hf-datasets/glue-sst2/train.tsv"

devTsvPath : String
devTsvPath = "data/hf-datasets/glue-sst2/validation.tsv"

ckptPath : String
ckptPath = "models/google/bert_uncased_L-2_H-128_A-2/model.safetensors"

----------------------------------------------------------------------
-- Model + helpers
----------------------------------------------------------------------

Model : Type
Model = BertForSequenceClassificationState
          Vocab Hidden NumLayers Intermediate MaxPos TypeVocab NumClasses
          ExampleExecutor ExampleDType WithGrad

PaddedExample : Type
PaddedExample = (Vect SeqLen Nat, Vect SeqLen Double, Nat)

-- arange(SeqLen) as a Vect of Doubles.
arangeSeqLen : Vect SeqLen Double
arangeSeqLen = build SeqLen
  where
    build : (k : Nat) -> Vect k Double
    build Z     = []
    build (S k) =
      let here = cast {to=Double} (cast {to=Integer} (minus SeqLen (S k)))
      in here :: build k

posVect : Vect SeqLen Double
posVect = arangeSeqLen

typeVect : Vect SeqLen Double
typeVect = replicate SeqLen 0.0

mkIdsTensor : Vect SeqLen Double -> IO (Tensor [SeqLen] ExampleExecutor ExampleDType WithGrad)
mkIdsTensor xs = ioRerun (\_ =>
  let raw = bulkToTensor {ex=ExampleExecutor} {dt=ExampleDType}
                         (VArray (map SArray xs))
  in tinput1d {n=SeqLen} raw)

-- Build the [SeqLen, SeqLen] attention-mask tensor from a 1D position
-- mask via toAttentionMask2d. Uses tparam2d as the available 2D
-- constructor; the param registry dedupes by name on repeated calls.
mkMaskTensor : Vect SeqLen Double
            -> IO (Tensor [SeqLen, SeqLen] ExampleExecutor ExampleDType WithGrad)
mkMaskTensor posMask = do
  let flat = toAttentionMask2d {seqLen=SeqLen} posMask
  let nElts = cast {to=Int} (SeqLen * SeqLen)
      buf   = prim__allocDoubles nElts
      buf'  = fillBuf buf 0 (toList flat)
  tparam2d {ex=ExampleExecutor} {dt=ExampleDType} {o=SeqLen} {i=SeqLen}
           "sst2.attn_mask" buf'
  where
    fillBuf : AnyPtr -> Int -> List Double -> AnyPtr
    fillBuf b _ [] = b
    fillBuf b off (v :: rest) =
      let b' = prim__setDouble b off v
      in fillBuf b' (off + 1) rest

-- One-hot Tensor [NumClasses] from a class label.
oneHotTensor : Nat -> IO (Tensor [NumClasses] ExampleExecutor ExampleDType WithGrad)
oneHotTensor lbl = ioRerun (\_ =>
  let oneHot : Vect NumClasses Double
      oneHot = case lbl of
                 0 => [1.0, 0.0]
                 _ => [0.0, 1.0]
      raw = bulkToTensor {ex=ExampleExecutor} {dt=ExampleDType}
                         (VArray (map SArray oneHot))
  in tinput1d {n=NumClasses} raw)

-- Run the full forward (with mask) on one padded example.
-- Returns the [NumClasses] logits tensor.
forwardOne : Model -> PaddedExample
          -> IO (Tensor [NumClasses] ExampleExecutor ExampleDType WithGrad)
forwardOne model (ids, mask, _) = do
  let idsDouble = map (\n => cast {to=Double} (cast {to=Integer} n)) ids
  idsT <- mkIdsTensor idsDouble
  posT <- mkIdsTensor posVect
  typT <- mkIdsTensor typeVect
  mskT <- mkMaskTensor mask
  hfBertSeqClassifyForward {ex=ExampleExecutor} {dt=ExampleDType}
                           {seqLen=SeqLen}
                           {vocab=Vocab} {hidden=Hidden}
                           {numLayers=NumLayers} {numHeads=NumHeads}
                           {headDim=HeadDim} {intermediate=Intermediate}
                           {maxPos=MaxPos} {typeVocab=TypeVocab}
                           {numClasses=NumClasses}
                           model idsT posT typT (Just mskT)

-- Per-example forward → cross-entropy scalar loss.
exampleLoss : Model -> PaddedExample
           -> IO (Tensor [] ExampleExecutor ExampleDType WithGrad)
exampleLoss model ex@(_, _, label) = do
  logits <- forwardOne model ex
  target <- oneHotTensor label
  tnllLoss logits target

-- Sum scalar Tensor losses.
sumScalars : Tensor [] ExampleExecutor ExampleDType WithGrad
          -> List (Tensor [] ExampleExecutor ExampleDType WithGrad)
          -> IO (Tensor [] ExampleExecutor ExampleDType WithGrad)
sumScalars acc []        = pure acc
sumScalars acc (x :: xs) = do
  acc' <- tadd acc x
  sumScalars acc' xs

-- Run one epoch over `items`, iterating in `batchSize` chunks. Each
-- chunk: forward each example, mean-reduce losses, one fused native
-- step. Returns the mean per-batch loss for logging.
epochSst2 : NativeOptimizer ExampleExecutor
         -> (batchSize : Nat) -> Model -> List PaddedExample
         -> IO (Model, Double)
epochSst2 opt batchSize model items = do
  finalLoss <- go model 0.0 0 items
  pure (model, finalLoss)
  where
    go : Model -> Double -> Nat -> List PaddedExample -> IO Double
    go _ accLoss nBatches [] =
      if nBatches == 0 then pure 0.0
      else pure (accLoss / cast {to=Double} (cast {to=Integer} nBatches))
    go m accLoss nBatches xs = do
      let (batch, rest) = splitAt batchSize xs
      case batch of
        [] => go m accLoss nBatches rest
        _  => do
          losses <- traverse (exampleLoss m) batch
          zero   <- tparamScalar {ex=ExampleExecutor} {dt=ExampleDType}
                                 "sst2.epoch_zero" 0.0
          summed <- sumScalars zero losses
          let denom = cast {to=Double} (cast {to=Integer} (length batch))
          meanLoss <- tmulScalar summed (1.0 / denom)
          v <- nativeTrainStep opt meanLoss
          go m (accLoss + v) (S nBatches) rest

-- Greedy-argmax classification on a single padded example.
predictClass : Model -> PaddedExample -> IO Nat
predictClass model ex = do
  logits <- forwardOne model ex
  let v0 = primItem1d {ex=ExampleExecutor} logits.tensorPtr 0
      v1 = primItem1d {ex=ExampleExecutor} logits.tensorPtr 1
  pure $ if v0 >= v1 then 0 else 1

heldOutAccuracy : Model -> List PaddedExample -> IO Double
heldOutAccuracy model items =
  withNoGrad {ex=ExampleExecutor} $ do
    let go : List PaddedExample -> Nat -> Nat -> IO (Nat, Nat)
        go [] n hits = pure (n, hits)
        go (ex@(_, _, label) :: rest) n hits = do
          p <- predictClass model ex
          go rest (S n) (if p == label then S hits else hits)
    (n, hits) <- go items 0 0
    let acc : Double
        acc = if n == 0
              then 0.0
              else cast {to=Double} (cast {to=Integer} hits)
                   / cast {to=Double} (cast {to=Integer} n)
    pure acc

-- Apply a Nat cap to a List (0 = no cap).
capAt : Nat -> List a -> List a
capAt 0 xs = xs
capAt n xs = take n xs

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

  putStrLn "=== BertClassifySst2Finetune ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr
           ++ " epochs=" ++ show cfg.epochs
           ++ " seed=" ++ show cfg.seed
           ++ " freeze-backbone=" ++ show cfg.freezeBackbone
  putStrLn $ "Subset: max-train=" ++ show cfg.maxTrain
           ++ " max-dev=" ++ show cfg.maxDev
           ++ " batch=" ++ show cfg.batchSize

  rawTrain <- loadHfDataset trainTsvPath
  rawDev   <- loadHfDataset devTsvPath
  let trainItems = capAt cfg.maxTrain rawTrain
  let devItems   = capAt cfg.maxDev rawDev
  let nTrain = length trainItems
  let nDev   = length devItems
  putStrLn $ "Loaded: train=" ++ show nTrain ++ " dev=" ++ show nDev

  if nTrain == 0 || nDev == 0
    then do
      putStrLn $ "ERROR: SST-2 not found at " ++ trainTsvPath
              ++ " — run `make data-sst2`."
      exitFailure
    else do
      let padTrain = map (padToSeqLen SeqLen PadId) trainItems
      let padDev   = map (padToSeqLen SeqLen PadId) devItems

      model <- hfBertForSequenceClassification
                 {ex=ExampleExecutor} {dt=ExampleDType}
                 {vocab=Vocab} {hidden=Hidden} {numLayers=NumLayers}
                 {numHeads=NumHeads} {intermediate=Intermediate}
                 {maxPos=MaxPos} {typeVocab=TypeVocab}
                 {numClasses=NumClasses}
                 "bert" "classifier"

      -- On-disk weights are F32; the tape default dtype is F64. Use
      -- the AllowCast variant so the load silently upcasts (mirrors
      -- HfBertInference's `loadModelAllowCast` call).
      True <- loadModelPrefixAllowCast {ex=ExampleExecutor} ckptPath "bert."
        | False => do
            putStrLn $ "ERROR: failed to load backbone from " ++ ckptPath
                    ++ " — run `make data-hf-bert-tiny`."
            exitFailure
      putStrLn "Backbone warm-started; head at fresh init."

      let opt = nativeAdamW {ex=ExampleExecutor} cfg.lr 0.9 0.999 1.0e-8 0.01 1.0
      when cfg.freezeBackbone $ do
        putStrLn "Freezing `bert.*` — head-only training."
        freezeByPrefix {ex=ExampleExecutor} opt "bert."

      -- Manual training loop (HF model isn't Network-shaped).
      let trainLoop : Model -> Nat -> Double -> IO (Model, Double)
          trainLoop m Z lastLoss = pure (m, lastLoss)
          trainLoop m (S k) _    = do
            (m', loss) <- epochSst2 opt cfg.batchSize m padTrain
            acc <- heldOutAccuracy m' padDev
            putStrLn $ "Epoch " ++ show (minus cfg.epochs k)
                    ++ ": loss=" ++ showFix 4 loss
                    ++ "  dev-acc=" ++ showFix 3 acc
            trainLoop m' k loss
      (trained, finalLoss) <- trainLoop model cfg.epochs 0.0
      finalAcc <- heldOutAccuracy trained padDev

      putStrLn ""
      putStrLn $ formatResult [ ("loss",     showFix 4 finalLoss)
                              , ("accuracy", showFix 3 finalAcc)
                              , ("epochs",   show cfg.epochs)
                              , ("seed",     show cfg.seed)
                              ]
