||| BertClassifySst2Lora — fine-tune a pretrained
||| `google/bert_uncased_L-2_H-128_A-2` backbone on SST-2 using
||| Low-Rank Adaptation (Hu et al. 2021) instead of full-weight
||| fine-tuning. Same task + same dataset slice as
||| BertClassifySst2Finetune; the only difference is what trains:
|||
|||   - Backbone: frozen (`freezeGroup opt =<< namesMatching (isPrefixOf "bert.")`).
|||   - LoRA adapters Q + V on every attention layer: TRAIN.
|||   - Classifier head (`classifier.*`): TRAIN.
|||
||| Trainable param count drops from ~4.4M (full) to ~6K for r=8
||| (~0.13% of the model). Adapter checkpoint is ~80 KB on disk vs
||| ~17 MB for the full safetensors.
|||
||| Pre-run: same as the non-LoRA variant.
|||   make data-sst2
|||   make data-hf-bert-tiny
|||
||| Usage:
|||   example-bert-classify-sst2-lora --lora-rank 8 --lora-alpha 16
|||   example-bert-classify-sst2-lora --save-adapter /tmp/lora-out
||| (then `make validate-lora-adapter ADAPTER_DIR=/tmp/lora-out`)
module Example.BertClassifySst2Lora

import Control.Linear.LIO
import Data.Linear.Notation
import Data.List
import Data.String
import Data.Vect
import System
import Compat.Random

import Array
import BuildConfig
import Executor
import Tensor
import Train
import Train.Freeze
import Util

import Checkpoint
import Transformers.Bert
import Transformers.BertForClassification
import Transformers.BertLora
import Transformers.Dataset
import Transformers.LoraIO

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

NumClasses : Nat
NumClasses = 2

SeqLen : Nat
SeqLen = 32

PadId : Nat
PadId = 0

record Config where
  constructor MkConfig
  lr          : Double
  epochs      : Nat
  seed        : Bits64
  maxTrain    : Nat
  batchSize   : Nat
  maxDev      : Nat
  loraRank    : Nat
  loraAlpha   : Double
  saveAdapter : String  -- "" means don't save

-- Defaults mirror peft tutorial recommendations for BERT-tiny:
--   lr = 1e-4 (LoRA tolerates higher LR than full FT because only the
--             adapters update; HF tutorial reports 1e-4..3e-4 sweet spot)
--   epochs = 3, rank=8, alpha=16 (canonical peft default)
defaultConfig : Config
defaultConfig =
  MkConfig 1.0e-4 3 42 256 8 256 8 16.0 ""

castNatStr : String -> Nat
castNatStr v = castNat v

castStrStr : String -> String
castStrStr v = v

specs : List (ArgSpec Config)
specs = [ Arg "--lr"            (\v, c => { lr := cast v } c)
        , Arg "--epochs"        (\v, c => { epochs := castNatStr v } c)
        , Arg "--seed"          (\v, c => { seed := castBits64 v } c)
        , Arg "--max-train"     (\v, c => { maxTrain := castNatStr v } c)
        , Arg "--batch-size"    (\v, c => { batchSize := castNatStr v } c)
        , Arg "--max-dev"       (\v, c => { maxDev := castNatStr v } c)
        , Arg "--lora-rank"     (\v, c => { loraRank := castNatStr v } c)
        , Arg "--lora-alpha"    (\v, c => { loraAlpha := cast v } c)
        , Arg "--save-adapter"  (\v, c => { saveAdapter := v } c)
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

-- LoRA adapters typed against the same backbone shape — rank=8 is
-- hard-pinned at the type level here for the worked example. A more
-- generic refactor would parameterise; for the demo we lock to 8.
LoraR : Nat
LoraR = 8

Adapters : Type
Adapters = BertLoraAdapters NumLayers Hidden LoraR
                            ExampleExecutor ExampleDType WithGrad

PaddedExample : Type
PaddedExample = (Vect SeqLen Nat, Vect SeqLen Double, Nat)

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

mkMaskTensor : Vect SeqLen Double
            -> IO (Tensor [SeqLen, SeqLen] ExampleExecutor ExampleDType WithGrad)
mkMaskTensor posMask = do
  let flat  = toAttentionMask2d {seqLen=SeqLen} posMask
  let nElts = cast {to=Int} (SeqLen * SeqLen)
      buf  = prim__allocDoubles nElts
      buf' = fillBuf buf 0 (toList flat)
  tparam2d {ex=ExampleExecutor} {dt=ExampleDType} {o=SeqLen} {i=SeqLen}
           "sst2lora.attn_mask" buf'
  where
    fillBuf : AnyPtr -> Int -> List Double -> AnyPtr
    fillBuf b _ []            = b
    fillBuf b off (v :: rest) =
      let b' = prim__setDouble b off v
      in fillBuf b' (off + 1) rest

oneHotTensor : Nat -> IO (Tensor [NumClasses] ExampleExecutor ExampleDType WithGrad)
oneHotTensor lbl = ioRerun (\_ =>
  let oneHot : Vect NumClasses Double
      oneHot = case lbl of
                 0 => [1.0, 0.0]
                 _ => [0.0, 1.0]
      raw = bulkToTensor {ex=ExampleExecutor} {dt=ExampleDType}
                         (VArray (map SArray oneHot))
  in tinput1d {n=NumClasses} raw)

-- Forward with the LoRA adapters threaded in. The frozen backbone `model` is
-- read-only (ω); the trained `adapters` are threaded single-owner via
-- hfBertSeqClassifyForwardWithLoraL.
forwardOneLoraL : Model -> (1 _ : Adapters) -> PaddedExample
               -> L IO {use = 1} (LPair (!* (Tensor [NumClasses] ExampleExecutor ExampleDType WithGrad)) Adapters)
forwardOneLoraL model adapters (ids, mask, _) = do
  (idsT, posT, typT, mskT) <-
    liftIO1 (do let idsDouble = map (\n => cast {to=Double} (cast {to=Integer} n)) ids
                idsT <- mkIdsTensor idsDouble
                posT <- mkIdsTensor posVect
                typT <- mkIdsTensor typeVect
                mskT <- mkMaskTensor mask
                pure (idsT, posT, typT, mskT))
  hfBertSeqClassifyForwardWithLoraL
        {ex=ExampleExecutor} {dt=ExampleDType}
        {seqLen=SeqLen}
        {vocab=Vocab} {hidden=Hidden}
        {numLayers=NumLayers} {numHeads=NumHeads}
        {headDim=HeadDim} {intermediate=Intermediate}
        {maxPos=MaxPos} {typeVocab=TypeVocab}
        {numClasses=NumClasses} {r=LoraR}
        model adapters idsT posT typT (Just mskT)

exampleLossL : Model -> (1 _ : Adapters) -> PaddedExample
            -> L IO {use = 1} (LPair (!* (Tensor [] ExampleExecutor ExampleDType WithGrad)) Adapters)
exampleLossL model adapters ex@(_, _, label) = do
  (MkBang logits # adapters') <- forwardOneLoraL model adapters ex
  loss <- liftIO1 (do target <- oneHotTensor label; tnllLoss logits target)
  pure1 (MkBang loss # adapters')

-- Discard the (linear) adapters: their fields are ω registered-param Vects.
discardAdapters : (1 _ : Adapters) -> L IO ()
discardAdapters (MkBertLoraAdapters _ _ _ _) = pure ()

sumScalars : Tensor [] ExampleExecutor ExampleDType WithGrad
          -> List (Tensor [] ExampleExecutor ExampleDType WithGrad)
          -> IO (Tensor [] ExampleExecutor ExampleDType WithGrad)
sumScalars acc []        = pure acc
sumScalars acc (x :: xs) = do
  acc' <- tadd acc x
  sumScalars acc' xs

epochLoraL : NativeOptimizer ExampleExecutor
          -> (batchSize : Nat) -> Model -> (1 _ : Adapters) -> List PaddedExample
          -> L IO {use = 1} (LPair (!* Double) Adapters)
epochLoraL opt batchSize model adapters items = go adapters 0.0 0 items
  where
    foldBatch : (1 _ : Adapters) -> List PaddedExample ->
                List (Tensor [] ExampleExecutor ExampleDType WithGrad) ->
                L IO {use = 1} (LPair (!* (List (Tensor [] ExampleExecutor ExampleDType WithGrad))) Adapters)
    foldBatch adapters []          acc = pure1 (MkBang (reverse acc) # adapters)
    foldBatch adapters (e :: rest) acc = do
      (MkBang l # adapters') <- exampleLossL model adapters e
      foldBatch adapters' rest (l :: acc)
    go : (1 _ : Adapters) -> Double -> Nat -> List PaddedExample -> L IO {use = 1} (LPair (!* Double) Adapters)
    go adapters accLoss nBatches [] =
      pure1 (MkBang (if nBatches == 0 then 0.0 else accLoss / cast {to=Double} (cast {to=Integer} nBatches)) # adapters)
    go adapters accLoss nBatches all@(_ :: _) = do
      let (batch, rest) = splitAt batchSize all
      (MkBang losses # adapters') <- foldBatch adapters batch []
      v <- liftIO1 (do zero   <- tparamScalar {ex=ExampleExecutor} {dt=ExampleDType} "sst2lora.epoch_zero" 0.0
                       summed <- sumScalars zero losses
                       let denom = cast {to=Double} (cast {to=Integer} (length batch))
                       meanLoss <- tmulScalar summed (1.0 / denom)
                       trainStep opt meanLoss)
      go adapters' (accLoss + v) (S nBatches) rest

predictClassL : Model -> (1 _ : Adapters) -> PaddedExample -> L IO {use = 1} (LPair (!* Nat) Adapters)
predictClassL model adapters ex = do
  (MkBang logits # adapters') <- forwardOneLoraL model adapters ex
  let v0 = primItem1d {ex=ExampleExecutor} logits.tensorPtr 0
      v1 = primItem1d {ex=ExampleExecutor} logits.tensorPtr 1
  pure1 (MkBang (if v0 >= v1 then the Nat 0 else 1) # adapters')

heldOutAccuracyL : Model -> (1 _ : Adapters) -> List PaddedExample -> L IO {use = 1} (LPair (!* Double) Adapters)
heldOutAccuracyL model adapters items = withNoGradL {ex=ExampleExecutor} $ do
  (MkBang nh # adapters') <- go adapters items 0 0
  let (n, hits) = nh
  pure1 (MkBang (if n == 0 then 0.0
                 else cast {to=Double} (cast {to=Integer} hits) / cast {to=Double} (cast {to=Integer} n)) # adapters')
  where
    go : (1 _ : Adapters) -> List PaddedExample -> Nat -> Nat -> L IO {use = 1} (LPair (!* (Nat, Nat)) Adapters)
    go adapters [] n hits                         = pure1 (MkBang (n, hits) # adapters)
    go adapters (ex@(_, _, label) :: rest) n hits = do
      (MkBang p # adapters') <- predictClassL model adapters ex
      go adapters' rest (S n) (if p == label then S hits else hits)

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

  putStrLn "=== BertClassifySst2Lora ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr
           ++ " epochs=" ++ show cfg.epochs
           ++ " seed=" ++ show cfg.seed
           ++ " lora-rank=" ++ show cfg.loraRank
           ++ " lora-alpha=" ++ show cfg.loraAlpha
  putStrLn $ "Subset: max-train=" ++ show cfg.maxTrain
           ++ " max-dev=" ++ show cfg.maxDev
           ++ " batch=" ++ show cfg.batchSize
  when (cfg.saveAdapter /= "") $
    putStrLn $ "Adapter save dir: " ++ cfg.saveAdapter

  rawTrain <- loadHfDataset trainTsvPath
  rawDev   <- loadHfDataset devTsvPath
  let trainItems = capAt cfg.maxTrain rawTrain
  let devItems   = capAt cfg.maxDev rawDev
  let nTrain     = length trainItems
  let nDev       = length devItems
  putStrLn $ "Loaded: train=" ++ show nTrain ++ " dev=" ++ show nDev

  if nTrain == 0 || nDev == 0
    then do
      putStrLn $ "ERROR: SST-2 not found at " ++ trainTsvPath
              ++ " — run `make data-sst2`."
      exitFailure
    else do
      let padTrain = map (padToSeqLen SeqLen PadId) trainItems
      let padDev   = map (padToSeqLen SeqLen PadId) devItems

      -- Linear surface: the frozen backbone `model` stays ω (read-only); the
      -- trained `adapters` are born linear (bornL) and threaded single-owner
      -- through the manual epoch loop + held-out eval, then discarded.
      Control.Linear.LIO.run $ do
        model <- liftIO1 (hfBertForSequenceClassification
                            {ex=ExampleExecutor} {dt=ExampleDType}
                            {vocab=Vocab} {hidden=Hidden} {numLayers=NumLayers}
                            {numHeads=NumHeads} {intermediate=Intermediate}
                            {maxPos=MaxPos} {typeVocab=TypeVocab}
                            {numClasses=NumClasses}
                            "bert" "classifier")
        liftIO1 (do ok <- (== Right ()) <$> load {ex=ExampleExecutor} ckptPath ({ allowCast := True, only := Just "bert." } defaultLoadOpts)
                    if ok then putStrLn "Backbone warm-started from HF safetensors."
                          else do putStrLn $ "ERROR: failed to load backbone from " ++ ckptPath
                                          ++ " — run `make data-hf-bert-tiny`."
                                  exitFailure)
        -- LoRA adapters under HF-aligned names (Q + V per layer, peft default).
        adapters <- bornL (loraInjectBert
                             {ex=ExampleExecutor} {dt=ExampleDType} {hidden=Hidden}
                             "bert" NumLayers LoraR cfg.loraAlpha)
        opt <- liftIO1 (do putStrLn $ "LoRA adapters injected (rank=" ++ show LoraR
                                    ++ ", alpha=" ++ show cfg.loraAlpha
                                    ++ ", target_modules=[\"query\",\"value\"])"
                           o <- adamW {ex=ExampleExecutor} cfg.lr 0.01 ({ clip := NormClip 1.0 } defaultOpts)
                           -- Canonical LoRA freeze: freeze `bert.` then unfreeze
                           -- the adapter suffixes; classifier.* stays trainable.
                           freezeGroup   {ex=ExampleExecutor} o
                             !(namesMatching {ex=ExampleExecutor} (isPrefixOf "bert."))
                           unfreezeGroup {ex=ExampleExecutor} o
                             !(namesMatching {ex=ExampleExecutor} (isSuffixOf "lora_A"))
                           unfreezeGroup {ex=ExampleExecutor} o
                             !(namesMatching {ex=ExampleExecutor} (isSuffixOf "lora_B"))
                           putStrLn "Backbone frozen; LoRA adapters + classifier head trainable."
                           pure o)
        let trainLoopL : (1 _ : Adapters) -> Nat -> Double -> L IO {use = 1} (LPair (!* Double) Adapters)
            trainLoopL adapters Z     lastLoss = pure1 (MkBang lastLoss # adapters)
            trainLoopL adapters (S k) _        = do
              (MkBang loss # adapters1) <- epochLoraL opt cfg.batchSize model adapters padTrain
              (MkBang acc # adapters2)  <- heldOutAccuracyL model adapters1 padDev
              liftIO1 (putStrLn $ "Epoch " ++ show (minus cfg.epochs k)
                                ++ ": loss=" ++ showFix 4 loss
                                ++ "  dev-acc=" ++ showFix 3 acc)
              trainLoopL adapters2 k loss
        (MkBang finalLoss # trainedAdp) <- trainLoopL adapters cfg.epochs 0.0
        (MkBang finalAcc # trainedAdp') <- heldOutAccuracyL model trainedAdp padDev
        discardAdapters trainedAdp'
        liftIO1 $ do
          putStrLn ""
          putStrLn $ formatResult [ ("loss",     showFix 4 finalLoss)
                                  , ("accuracy", showFix 3 finalAcc)
                                  , ("epochs",   show cfg.epochs)
                                  , ("seed",     show cfg.seed)
                                  ]
          -- Optional: write the trained adapter to a peft-compatible directory.
          when (cfg.saveAdapter /= "") $ do
            let adapterCfg = MkLoraAdapterConfig
                  cfg.loraRank
                  cfg.loraAlpha
                  (the (List String) ["query", "value"])
                  "SEQ_CLS"
            ok <- saveLoraAdapter {ex=ExampleExecutor} cfg.saveAdapter adapterCfg
            if ok
              then putStrLn $ "Saved LoRA adapter to " ++ cfg.saveAdapter
              else putStrLn $ "WARNING: failed to save adapter to " ++ cfg.saveAdapter
