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

import Control.Linear.LIO
import Data.Fin
import Data.Linear.Notation
import Data.List
import Data.Vect
import System

import Ml.Checkpoint
import Ml.Compat.Random
import Ml.DataStream
import Ml.Fit
import Ml.Simple
import Ml.Train

import BuildConfig

-- This example's model is a linear `Seq`; hide the IO `Nn.Seq` constructors
-- (same `Nil`/`::`/`~~>` names) so the chain builder resolves unambiguously.

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

-- The full 10k test set, in chunks: a single [10000, 784] batch would push
-- the first conv's [10000, 16, 24, 24] intermediate past 700 MB on F64, so
-- the eval loops instead. The reference walks its test DataLoader the same
-- way; evaluating a 1000-image slice (as this did until 2026-07-31) reported
-- a different measurement under the same `accuracy` key.
EvalChunk : Nat
EvalChunk = 1000

EvalChunks : Nat
EvalChunks = 10

-- Standard MNIST normalisation, `transforms.Normalize((0.1307,), (0.3081,))`
-- on the reference. `idxDataset` yields raw [0, 1] pixels; without this the
-- two sides trained on inputs of std 0.32 and 1.04 respectively — found by
-- `scripts/check-data-manifest.py`, invisible to every other gate.
MnistMean : Double
MnistMean = 0.1307

MnistStd : Double
MnistStd = 0.3081

-- Applied to an already-collated [b, InputDim] batch rather than per-sample:
-- wrapping the `Dataset` with a record update over its existential `size`
-- sent the elaborator into a >40 minute spin (gotchas.md).
normalizeBatch : {b : Nat} -> Tensor [b, InputDim] Ex F NoGrad ->
                 IO (Tensor [b, InputDim] Ex F NoGrad)
normalizeBatch x = do
  scaled <- tmulScalar x (1.0 / MnistStd)
  shift  <- tensor {dims = [b, InputDim]} (Const (MnistMean / MnistStd))
  scaled -. shift

-- Wrap a batched stream so every pull is normalised. Direct record
-- construction rather than a `Dataset` record update: the latter, over an
-- existentially-sized dataset, sent the elaborator into a >40 minute spin.
normalizedStream : {b : Nat} ->
                   DataStream (Tensor [b, InputDim] Ex F NoGrad, Tensor [b, NumClasses] Ex F NoGrad) ->
                   DataStream (Tensor [b, InputDim] Ex F NoGrad, Tensor [b, NumClasses] Ex F NoGrad)
normalizedStream st =
  MkDataStream (do (x, y) <- st.next
                   nx <- normalizeBatch {b} x
                   pure (nx, y))
               st.epochLen

----------------------------------------------------------------------
-- Model + loss
----------------------------------------------------------------------

Model : Type
Model = Seq InputDim NumClasses Ex F WithGrad

-- Top-level `Init` value (not inline under `runInitL`): a nested do-block under
-- the linear `run $ do …` trips the elaborator's ambiguity-depth limit. Built
-- as a linear `Seq`.
mkModel : Init Model
mkModel = do
  c1 <- conv2d {inC = InC}   {outC = OutC1} {h = ImgH}     {w = ImgH}     {kH = KH} {kW = KH} {padH = 0} {padW = 0}
  c2 <- conv2d {inC = OutC1} {outC = OutC2} {h = Pool1Out} {w = Pool1Out} {kH = KH} {kW = KH} {padH = 0} {padW = 0}
  l  <- linear {i = AfterPool2} {o = NumClasses}
  pure (c1 ~~> reluA
           ~~> maxPool2d {c = OutC1} {inH = Conv1Out} {inW = Conv1Out} {poolH = 2} {poolW = 2} {strH = 2} {strW = 2}
           ~~> c2 ~~> reluA
           ~~> maxPool2d {c = OutC2} {inH = Conv2Out} {inW = Conv2Out} {poolH = 2} {poolW = 2} {strH = 2} {strW = 2}
           ~~> dropout 0.5
           ~~> l ~~> Nil)

-- Linear-resource loss: consume the model, forward via forwardSeq, return the
-- banged scalar loss beside the rebuilt model.
nllLossL : (1 _ : Model) ->
           (Tensor [BatchSize, InputDim] Ex F NoGrad, Tensor [BatchSize, NumClasses] Ex F NoGrad) ->
           L IO {use = 1} (LPair (!* (Tensor [] Ex F WithGrad)) Model)
nllLossL model (x, tgt) = do
  (MkBang out # model') <- forwardSeq {b = BatchSize} model (retypeGrad x)
  loss <- tnllLossMeanL {b = BatchSize} {n = NumClasses} out (retypeGrad tgt)
  pure1 (MkBang loss # model')

----------------------------------------------------------------------
-- Evaluation: argmax accuracy over the full test set, EvalChunk at a time
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

-- Argmax hits in one already-forwarded [EvalChunk, NumClasses] chunk. Pure
-- (primItem2d reads are pure FFI); the forward happens in the linear block.
correctIn : Tensor [EvalChunk, NumClasses] Ex F NoGrad ->
            Tensor [EvalChunk, NumClasses] Ex F NoGrad -> Nat
correctIn pred tgt =
  length $ filter id
    [ argmaxRow pred.tensorPtr r == argmaxRow tgt.tensorPtr r
    | r <- map (cast {to=Int}) [the Nat 0 .. EvalChunk `minus` 1] ]

-- Fold argmax hits over `n` chunks, threading the inference model linearly.
evalChunks : (1 _ : Seq InputDim NumClasses Ex F NoGrad) ->
             DataStream (Tensor [EvalChunk, InputDim] Ex F NoGrad,
                         Tensor [EvalChunk, NumClasses] Ex F NoGrad) ->
             Nat -> Nat ->
             L IO {use = 1} (LPair (!* Nat) (Seq InputDim NumClasses Ex F NoGrad))
evalChunks m _  Z     hits = pure1 (MkBang hits # m)
evalChunks m st (S k) hits = do
  batch <- liftIO1 st.next
  (MkBang predB # m') <- forwardSeq {b = EvalChunk} m (fst batch)
  evalChunks m' st k (hits + correctIn predB (snd batch))

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
  trainStream <- stream (Shuffle cfg.seed) trainDs
  let bs = normalizedStream {b = BatchSize}
             (batched {b = BatchSize} {i = InputDim} {o = NumClasses} trainStream)
  maybeDumpBatch {ex = ExampleExecutor} bs
  testStream <- stream (Shuffle cfg.seed) testDs
  putStrLn ""

  -- Linear surface end to end: model born linear (runInitL), threaded through
  -- fitSupervised, converted to an inference model (eval), forwarded once on
  -- the eval batch (forwardSeq), then discarded. `run` is fully qualified —
  -- `import System` brings other `run`s that otherwise blow the ambiguity-depth
  -- limit in this do-block.
  --
  -- The eval batch is pulled *after* training (mirroring Transformer/Gpt's
  -- post-train eval): a collated batch is an intermediate arena tensor, so each
  -- training step's `optimizer_step` → `arena_reset` would dangle a batch
  -- pre-fetched before the loop (a use-after-free at eval).
  Control.Linear.LIO.run $ do
    model <- runInitL mkModel
    (MkBang (epochsDone, finalLoss) # trained) <-
      fitSupervised opt nllLossL bs (patienceConfig cfg.epochs cfg.patience) model
    liftIO1 (putStrLn "")
    infer <- eval trained
    (MkBang hits # infer') <-
      evalChunks infer
                 (normalizedStream {b = EvalChunk}
                    (batched {b = EvalChunk} {i = InputDim} {o = NumClasses} testStream))
                 EvalChunks 0
    discard infer'
    liftIO1 $ do
      let acc = cast {to=Double} hits / cast {to=Double} (EvalChunk * EvalChunks)
      putStrLn $ "Final accuracy (" ++ show (EvalChunk * EvalChunks)
               ++ " test samples): " ++ show (acc * 100.0) ++ "%"
      putStrLn ""
      putStrLn $ formatResult [("accuracy", show acc),
                               ("epochs", show epochsDone),
                               ("loss", show finalLoss),
                               ("seed", show cfg.seed)]
