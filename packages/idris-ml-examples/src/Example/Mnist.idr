-- | MNIST: Convolutional Neural Network
-- |
-- | LeNet-style CNN for handwritten digit classification, on the v1 Nn/fit
-- | surface. Conv2d(1->16,k=5) -> ReLU -> MaxPool(2) ->
-- | Conv2d(16->32,k=5) -> ReLU -> MaxPool(2) -> Dropout(0.5) ->
-- | Linear(512->10). Raw logits; tnllLossMean applies log_softmax.
-- |
-- | Loads MNIST .idx files via the `idxDataset` Dataset adapter, streams
-- | shuffled mini-batches with `batched`, trains with `fitSupervised`.

module Example.Mnist

import Data.Fin
import Data.List
import Data.Vect
import System

import BuildConfig
import Compat.Random
import ML.Simple
import Train

----------------------------------------------------------------------
-- Architecture (flat dims)
----------------------------------------------------------------------

InC : Nat
InC = 1

OutC1 : Nat
OutC1 = 16

OutC2 : Nat
OutC2 = 32

KH : Nat
KH = 5

ImgH : Nat
ImgH = 28

Conv1Out : Nat
Conv1Out = ConvOutDim ImgH KH 0  -- 24

Pool1Out : Nat
Pool1Out = PoolOutDim Conv1Out 2 2  -- 12

Conv2Out : Nat
Conv2Out = ConvOutDim Pool1Out KH 0  -- 8

Pool2Out : Nat
Pool2Out = PoolOutDim Conv2Out 2 2  -- 4

InputDim : Nat
InputDim = InC * (ImgH * ImgH)  -- 784

AfterPool2 : Nat
AfterPool2 = OutC2 * (Pool2Out * Pool2Out)  -- 512

NumClasses : Nat
NumClasses = 10

BatchSize : Nat
BatchSize = 64

EvalSize : Nat
EvalSize = 1000

----------------------------------------------------------------------
-- Model + loss
----------------------------------------------------------------------

Model : Type
Model = Seq InputDim NumClasses Ex F WithGrad

buildModel : IO Model
buildModel = runInit $ do
  c1 <- conv2d {inC = InC}   {outC = OutC1} {h = ImgH}     {w = ImgH}     {kH = KH} {kW = KH} {padH = 0} {padW = 0}
  c2 <- conv2d {inC = OutC1} {outC = OutC2} {h = Pool1Out} {w = Pool1Out} {kH = KH} {kW = KH} {padH = 0} {padW = 0}
  l  <- linear {i = AfterPool2} {o = NumClasses}
  pure (c1 ~~> reluA
           ~~> maxPool2d {c = OutC1} {inH = Conv1Out} {inW = Conv1Out} {poolH = 2} {poolW = 2} {strH = 2} {strW = 2}
           ~~> c2 ~~> reluA
           ~~> maxPool2d {c = OutC2} {inH = Conv2Out} {inW = Conv2Out} {poolH = 2} {poolW = 2} {strH = 2} {strW = 2}
           ~~> dropout 0.5
           ~~> l ~~> Nil)

nllLoss : Model -> (Tensor [BatchSize, InputDim] Ex F NoGrad, Tensor [BatchSize, NumClasses] Ex F NoGrad) ->
          IO (Tensor [] Ex F WithGrad)
nllLoss model (x, tgt) = do
  out <- forwardSeq {b = BatchSize} model (retypeGrad x)
  tnllLossMean {b = BatchSize} {n = NumClasses} out (retypeGrad tgt)

----------------------------------------------------------------------
-- Evaluation: argmax accuracy over one EvalSize batch of test images
----------------------------------------------------------------------

-- argmax over the NumClasses entries of row r of a [b, NumClasses] tensor.
argmaxRow : AnyPtr -> Int -> Int
argmaxRow t r = go 1 0 (primItem2d {ex=Ex} t r 0)
  where
    go : Int -> Int -> Double -> Int
    go j best bestV =
      if j >= cast {to=Int} NumClasses then best
      else let v = primItem2d {ex=Ex} t r j
           in if v > bestV then assert_total (go (j + 1) j v)
                           else assert_total (go (j + 1) best bestV)

evalAccuracy : Seq InputDim NumClasses Ex F NoGrad ->
               (Tensor [EvalSize, InputDim] Ex F NoGrad, Tensor [EvalSize, NumClasses] Ex F NoGrad) ->
               IO Double
evalAccuracy model (x, tgt) = do
  pred <- forwardSeq {b = EvalSize} model x
  let correct = length $ filter id
        [ argmaxRow pred.tensorPtr r == argmaxRow tgt.tensorPtr r
        | r <- map (cast {to=Int}) [the Nat 0 .. EvalSize `minus` 1] ]
  pure (cast {to=Double} correct / cast {to=Double} EvalSize)

----------------------------------------------------------------------
-- Config & Main
----------------------------------------------------------------------

record Config where
  constructor MkConfig
  lr         : Double
  epochs     : Nat
  patience   : Nat
  seed       : Bits64
  dataDir    : String
  trainCount : Nat   -- 0 = full dataset

defaultConfig : Config
defaultConfig = MkConfig 0.001 5 3 42 "data/mnist" 0

specs : List (ArgSpec Config)
specs = [ Arg "--lr" (\v, c => { lr := cast v } c)
        , Arg "--epochs" (\v, c => { epochs := castNat v } c)
        , Arg "--patience" (\v, c => { patience := castNat v } c)
        , Arg "--seed" (\v, c => { seed := castBits64 v } c)
        , Arg "--data" (\v, c => { dataDir := v } c)
        , Arg "--train-count" (\v, c => { trainCount := castNat v } c) ]

-- Shrink a Dataset to the first `n` items (n <= size). Used by --train-count.
-- Can't field-update `size` (the dependent `item : Fin size -> _`), so rebuild
-- via `fromIndexed`, re-injecting the in-bounds index (Nothing unreachable).
limitDataset : Nat -> Dataset a -> Dataset a
limitDataset n ds = fromIndexed (min n ds.size) $ \k =>
  case natToFin k ds.size of
    Just fin => ds.item fin
    Nothing  => assert_total $ idris_crash "Mnist.limitDataset: index out of range"

%default partial

main : IO ()
main = do
  requireMachine {m = ChosenMachine}
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)

  srand cfg.seed
  tsetInitSeed {ex = Ex} cfg.seed

  putStrLn "=== MNIST: Convolutional Neural Network ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr ++ " epochs=" ++ show cfg.epochs
           ++ " patience=" ++ show cfg.patience ++ " seed=" ++ show cfg.seed
  putStrLn "Architecture: Conv2d(1->16,k=5) -> ReLU -> Pool(2) -> Conv2d(16->32,k=5) -> ReLU -> Pool(2) -> Dropout(0.5) -> Linear(512->10)"

  let trainDsFull = idxDataset {ex=Ex} {dt=F}
        (cfg.dataDir ++ "/train-images-idx3-ubyte") (cfg.dataDir ++ "/train-labels-idx1-ubyte")
        InputDim NumClasses
      trainDs = case cfg.trainCount of
                  Z => trainDsFull
                  n => limitDataset n trainDsFull
      testDs = idxDataset {ex=Ex} {dt=F}
        (cfg.dataDir ++ "/t10k-images-idx3-ubyte") (cfg.dataDir ++ "/t10k-labels-idx1-ubyte")
        InputDim NumClasses
  putStrLn $ "Train: " ++ show trainDs.size ++ " images, Test: " ++ show testDs.size ++ " images"

  opt <- adam cfg.lr ({ clip := NormClip 1.0 } defaultOpts)
  model <- buildModel
  trainStream <- stream (Shuffle cfg.seed) trainDs
  let bs = batched {b = BatchSize} {i = InputDim} {o = NumClasses} trainStream
  putStrLn ""

  (trained, epochsDone, finalLoss) <-
    fitSupervised opt nllLoss bs (patienceConfig cfg.epochs cfg.patience) model

  -- Accuracy on one EvalSize batch of (shuffled) test images.
  putStrLn ""
  infer <- eval trained
  testStream <- stream (Shuffle cfg.seed) testDs
  evalBatch <- (batched {b = EvalSize} {i = InputDim} {o = NumClasses} testStream).next
  acc <- evalAccuracy infer evalBatch
  putStrLn $ "Final accuracy (" ++ show EvalSize ++ " test samples): " ++ show (acc * 100.0) ++ "%"

  putStrLn ""
  putStrLn $ formatResult [("accuracy", show acc),
                           ("epochs", show epochsDone),
                           ("loss", show finalLoss),
                           ("seed", show cfg.seed)]
