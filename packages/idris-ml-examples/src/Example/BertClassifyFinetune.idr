||| BertClassifyFinetune — supervised fine-tune of a BERT-style
||| encoder + fresh classifier head on a synthetic 3-class task.
|||
||| Architecture: a tiny BERT (vocab=64, hidden=32, layers=1,
||| heads=2, headDim=16, intermediate=64, maxPos=8, typeVocab=2) +
||| `BertForSequenceClassification` head with `numClasses=3`. 32 base
||| params + 2 head params = 34 total.
|||
||| Task: a label-encoding token at position 1 (right after [CLS])
||| identifies the class. Positions 2-6 are random distractor tokens;
||| position 7 is the [SEP]-equivalent. The model has to learn to
||| attend from [CLS] to position 1 and read the label-token → class
||| mapping; convergence is fast (a few dozen epochs at lr=1e-3).
|||
||| Demonstrates the new API surface:
|||   - `hfBertForSequenceClassification` (backbone + fresh head)
|||   - optional `--freeze-backbone` calling `freezeByPrefix`
|||
||| The synthetic config is small enough that from-scratch training
||| converges in seconds. Pre-trained warm-start via `loadModelPrefix`
||| stays demonstrable in the FT1 unit test (`Test.CheckpointSubset`);
||| a real-text fine-tune on top of an actual HF checkpoint is a
||| medium-priority follow-up TODO row (needs attention masking +
||| tokenized dataset infra).
module Example.BertClassifyFinetune

import Data.List
import Data.Vect
import System
import Compat.Random

import Array
import Backprop
import BuildConfig
import Executor
import Generate
import Tensor
import Train
import Train.Freeze
import Util

import HfBert
import HfBertForClassification


----------------------------------------------------------------------
-- Config (tiny BERT for fast convergence)
----------------------------------------------------------------------

Vocab : Nat
Vocab = 64

Hidden : Nat
Hidden = 32

NumLayers : Nat
NumLayers = 1

NumHeads : Nat
NumHeads = 2

HeadDim : Nat
HeadDim = 16

Intermediate : Nat
Intermediate = 64

MaxPos : Nat
MaxPos = 8

TypeVocab : Nat
TypeVocab = 2

NumClasses : Nat
NumClasses = 3

SeqLen : Nat
SeqLen = 8

BatchSize : Nat
BatchSize = 16


-- The 3 "label" token IDs the model has to learn to read.
labelTokens : Vect 3 Double
labelTokens = [11.0, 13.0, 17.0]


record Config where
  constructor MkConfig
  lr            : Double
  epochs        : Nat
  patience      : Nat
  seed          : Bits64
  freezeBackbone : Bool

defaultConfig : Config
defaultConfig = MkConfig 0.001 2000 500 42 False

boolFlag : String -> Bool
boolFlag v = v == "1" || v == "true" || v == "True" || v == "yes"

specs : List (ArgSpec Config)
specs = [ Arg "--lr"               (\v, c => { lr := cast v } c)
        , Arg "--epochs"           (\v, c => { epochs := castNat v } c)
        , Arg "--patience"         (\v, c => { patience := castNat v } c)
        , Arg "--seed"             (\v, c => { seed := castBits64 v } c)
        , Arg "--freeze-backbone"  (\v, c => { freezeBackbone := boolFlag v } c) ]


----------------------------------------------------------------------
-- Synthetic dataset
----------------------------------------------------------------------

-- Random distractor token in [20, 60]. Avoids the labelTokens (11/13/17)
-- and the CLS / SEP IDs (0 / 1) so the model can't shortcut via
-- collisions.
randomDistractor : IO Double
randomDistractor = do
  k <- Generate.randomInt 20 60
  pure (cast {to=Double} (cast {to=Integer} k))

-- Build a length-SeqLen example with the label-token at position 1.
-- Position 0 = [CLS]-equivalent (62); position SeqLen-1 = [SEP]-equivalent (63).
-- CLS / SEP IDs are chosen within Vocab so the same data passes the
-- PyTorch-side embedding-table bounds check (vocab=64).
buildExample : (label : Nat) -> IO (Vect SeqLen Double)
buildExample label = do
  let labelTok = the Double (case label of
                              0 => 11.0
                              1 => 13.0
                              _ => 17.0)
  -- 5 distractor tokens for positions 2..6.
  d2 <- randomDistractor
  d3 <- randomDistractor
  d4 <- randomDistractor
  d5 <- randomDistractor
  d6 <- randomDistractor
  pure [0.0, labelTok, d2, d3, d4, d5, d6, 1.0]

-- Position IDs are arange(SeqLen) regardless of example.
posVect : Vect SeqLen Double
posVect = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]

-- Token-type IDs (all zero — segment A only).
typeVect : Vect SeqLen Double
typeVect = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


public export
record FtExample where
  constructor MkFtExample
  inputIds : Vect SeqLen Double
  label    : Nat

genExample : IO FtExample
genExample = do
  c <- Generate.randomInt 0 (minus NumClasses 1)
  ids <- buildExample c
  pure (MkFtExample ids c)

genBatch : (n : Nat) -> IO (Vect n FtExample)
genBatch Z     = pure []
genBatch (S k) = do
  e <- genExample
  rest <- genBatch k
  pure (e :: rest)


----------------------------------------------------------------------
-- Model alias + per-example forward + per-batch epoch fn
----------------------------------------------------------------------

Model : Type
Model = BertForSequenceClassificationState
          Vocab Hidden NumLayers Intermediate MaxPos TypeVocab NumClasses
          ExampleExecutor ExampleDType WithGrad

-- Build a single Tensor [SeqLen] from a Vect of token-ID Doubles.
mkIdsTensor : Vect SeqLen Double -> IO (Tensor [SeqLen] ExampleExecutor ExampleDType WithGrad)
mkIdsTensor xs = ioRerun (\_ =>
  let raw = bulkToTensor {ex=ExampleExecutor} {dt=ExampleDType}
                         (VArray (map SArray xs))
  in tinput1d {n=SeqLen} raw)

-- One-hot Tensor [NumClasses] from a class label.
oneHotTensor : Nat -> IO (Tensor [NumClasses] ExampleExecutor ExampleDType WithGrad)
oneHotTensor lbl = ioRerun (\_ =>
  let oneHot : Vect NumClasses Double
      oneHot = case lbl of
                 0 => [1.0, 0.0, 0.0]
                 1 => [0.0, 1.0, 0.0]
                 _ => [0.0, 0.0, 1.0]
      raw = bulkToTensor {ex=ExampleExecutor} {dt=ExampleDType}
                         (VArray (map SArray oneHot))
  in tinput1d {n=NumClasses} raw)

-- Per-example forward + cross-entropy → scalar Tensor loss.
exampleLoss : Model -> FtExample
           -> IO (Tensor [] ExampleExecutor ExampleDType WithGrad)
exampleLoss model ex = do
  ids   <- mkIdsTensor ex.inputIds
  pos   <- mkIdsTensor posVect
  typ   <- mkIdsTensor typeVect
  logits <- hfBertSeqClassifyForward {ex=ExampleExecutor} {dt=ExampleDType}
                                     {seqLen=SeqLen}
                                     {vocab=Vocab} {hidden=Hidden}
                                     {numLayers=NumLayers} {numHeads=NumHeads}
                                     {headDim=HeadDim} {intermediate=Intermediate}
                                     {maxPos=MaxPos} {typeVocab=TypeVocab}
                                     {numClasses=NumClasses}
                                     model ids pos typ Nothing
  target <- oneHotTensor ex.label
  tnllLoss logits target

-- Sum scalar losses (Tensor []-shape) over a List.
sumScalars : (Tensor [] ExampleExecutor ExampleDType WithGrad
              -> Tensor [] ExampleExecutor ExampleDType WithGrad
              -> IO (Tensor [] ExampleExecutor ExampleDType WithGrad))
          -> Tensor [] ExampleExecutor ExampleDType WithGrad
          -> List (Tensor [] ExampleExecutor ExampleDType WithGrad)
          -> IO (Tensor [] ExampleExecutor ExampleDType WithGrad)
sumScalars _ acc [] = pure acc
sumScalars op acc (x :: xs) = do
  acc' <- op acc x
  sumScalars op acc' xs

-- One epoch over the given batch: forward each example, mean the losses,
-- one fused native step.
epochBert : NativeOptimizer ExampleExecutor
         -> Model -> Vect BatchSize FtExample
         -> IO (Model, Double)
epochBert opt model batch = do
  losses <- traverse (exampleLoss model) (toList batch)
  -- Mean = sum / batchSize. tnllLoss returns a Tensor []; sumScalars
  -- accumulates via tadd.
  zero   <- tparamScalar {ex=ExampleExecutor} {dt=ExampleDType} "ftcls.epoch_zero" 0.0
  summed <- sumScalars tadd zero losses
  meanLoss <- tmulScalar summed (1.0 / cast {to=Double} BatchSize)
  v <- nativeTrainStep opt meanLoss
  pure (model, v)


-- Greedy-argmax classification: pick the index with max logit.
predictClass : Model -> Vect SeqLen Double -> IO Nat
predictClass model ids = do
  idsT <- mkIdsTensor ids
  pos  <- mkIdsTensor posVect
  typ  <- mkIdsTensor typeVect
  logits <- hfBertSeqClassifyForward {ex=ExampleExecutor} {dt=ExampleDType}
                                     {seqLen=SeqLen}
                                     {vocab=Vocab} {hidden=Hidden}
                                     {numLayers=NumLayers} {numHeads=NumHeads}
                                     {headDim=HeadDim} {intermediate=Intermediate}
                                     {maxPos=MaxPos} {typeVocab=TypeVocab}
                                     {numClasses=NumClasses}
                                     model idsT pos typ Nothing
  let v0 = primItem1d {ex=ExampleExecutor} logits.tensorPtr 0
      v1 = primItem1d {ex=ExampleExecutor} logits.tensorPtr 1
      v2 = primItem1d {ex=ExampleExecutor} logits.tensorPtr 2
  pure $ if v0 >= v1 && v0 >= v2 then 0
         else if v1 >= v2 then 1
         else 2

-- Held-out accuracy on a fresh batch.
heldOutAccuracy : Model -> IO Double
heldOutAccuracy model = do
  evalBatch <- genBatch 32
  let go : List FtExample -> Nat -> IO Nat
      go [] acc = pure acc
      go (e :: rest) acc = do
        p <- predictClass model e.inputIds
        go rest (if p == e.label then S acc else acc)
  hits <- withNoGrad {ex=ExampleExecutor} (go (toList evalBatch) 0)
  pure (cast {to=Double} (cast {to=Integer} hits) / 32.0)


----------------------------------------------------------------------
-- Main
----------------------------------------------------------------------

%default partial

main : IO ()
main = do
  requireMachine {m = ChosenMachine}
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)

  srand cfg.seed

  putStrLn "=== BertClassifyFinetune: synthetic 3-class fine-tune ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr ++ " epochs=" ++ show cfg.epochs
           ++ " patience=" ++ show cfg.patience ++ " seed=" ++ show cfg.seed
           ++ " freeze-backbone=" ++ show cfg.freezeBackbone
  putStrLn $ "Arch: vocab=" ++ show Vocab ++ " hidden=" ++ show Hidden
           ++ " layers=" ++ show NumLayers ++ " heads=" ++ show NumHeads
           ++ " classes=" ++ show NumClasses

  model <- hfBertForSequenceClassification {ex=ExampleExecutor} {dt=ExampleDType}
                                           {vocab=Vocab} {hidden=Hidden}
                                           {numLayers=NumLayers} {numHeads=NumHeads}
                                           {intermediate=Intermediate}
                                           {maxPos=MaxPos} {typeVocab=TypeVocab}
                                           {numClasses=NumClasses}
                                           "bert" "classifier"

  let opt = nativeAdamW cfg.lr 0.9 0.999 1.0e-8 0.01 1.0
  when cfg.freezeBackbone $ do
    putStrLn "Freezing backbone (`bert.*`); only classifier head trains."
    freezeByPrefix {ex=ExampleExecutor} opt "bert."

  let trainCfg = patienceConfig cfg.epochs cfg.patience

  (trained, epochsDone, finalLoss) <-
    runTrainingIO {ex=ExampleExecutor} {model=Model} {dp=Vect BatchSize FtExample}
      (\m, b => epochBert opt m b)
      (genBatch BatchSize) trainCfg model

  acc <- heldOutAccuracy trained
  putStrLn ""
  putStrLn $ formatResult [("loss",   showFix 4 finalLoss),
                            ("accuracy", showFix 3 acc),
                            ("epochs", show epochsDone),
                            ("seed",   show cfg.seed)]
