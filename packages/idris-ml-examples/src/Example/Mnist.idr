-- | MNIST: Convolutional Neural Network
-- |
-- | LeNet-style CNN for handwritten digit classification.
-- | Conv2d(1->16,k=5) -> ReLU -> MaxPool(2) ->
-- | Conv2d(16->32,k=5) -> ReLU -> MaxPool(2) -> Dropout(0.5) ->
-- | Linear(512->10). Outputs raw logits; tnllLoss applies log_softmax.
-- |
-- | Loads MNIST .idx files via C FFI. Trains on random mini-batches.

module Example.Mnist

import Data.List
import Data.String
import Data.Vect
import Decidable.Equality
import System
import Compat.Random

import BackpropV2
import DataLoader
import DataPoint
import Floating
import Generate
import Hpo.LrFinder
import Layer.ActivationV2
import Layer.Conv  -- ConvOutDim / PoolOutDim type-level helpers
import Layer.ConvV2
import Layer.CoreV2
import Layer.DropoutV2
import Layer.LinearV2
import Tensor
import Train
import Util
import Device
import Variable


----------------------------------------------------------------------
-- Architecture
----------------------------------------------------------------------

InC : Nat
InC = 1

OutC1 : Nat
OutC1 = 16

OutC2 : Nat
OutC2 = 32

KH : Nat
KH = 5

KW : Nat
KW = 5

ImgH : Nat
ImgH = 28

ImgW : Nat
ImgW = 28

-- After Conv1: 28-5+1 = 24
Conv1OutH : Nat
Conv1OutH = ConvOutDim ImgH KH 0  -- 24

Conv1OutW : Nat
Conv1OutW = ConvOutDim ImgW KW 0  -- 24

-- After Pool1: 24/2 = 12
Pool1OutH : Nat
Pool1OutH = PoolOutDim Conv1OutH 2 2  -- 12

Pool1OutW : Nat
Pool1OutW = PoolOutDim Conv1OutW 2 2  -- 12

-- After Conv2: 12-5+1 = 8
Conv2OutH : Nat
Conv2OutH = ConvOutDim Pool1OutH KH 0  -- 8

Conv2OutW : Nat
Conv2OutW = ConvOutDim Pool1OutW KW 0  -- 8

-- After Pool2: 8/2 = 4
Pool2OutH : Nat
Pool2OutH = PoolOutDim Conv2OutH 2 2  -- 4

Pool2OutW : Nat
Pool2OutW = PoolOutDim Conv2OutW 2 2  -- 4

-- Flat dimensions for Network chain
InputDim : Nat
InputDim = InC * (ImgH * ImgW)  -- 784

AfterConv1 : Nat
AfterConv1 = OutC1 * (Conv1OutH * Conv1OutW)  -- 9216

AfterPool1 : Nat
AfterPool1 = OutC1 * (Pool1OutH * Pool1OutW)  -- 2304

AfterConv2 : Nat
AfterConv2 = OutC2 * (Conv2OutH * Conv2OutW)  -- 2048

AfterPool2 : Nat
AfterPool2 = OutC2 * (Pool2OutH * Pool2OutW)  -- 512

NumClasses : Nat
NumClasses = 10

BatchSize : Nat
BatchSize = 64


----------------------------------------------------------------------
-- Data Loading
----------------------------------------------------------------------

||| Fetch a single MNIST image as a TensorDataPoint (raw tensor pointer).
mnistItem : AnyPtr -> Nat -> IO (TensorDataPoint InputDim NumClasses)
mnistItem ds idx = do
  let imgT = prim__mnistGetImage ds (cast {to=Int} (natToInteger idx))
      lbl = prim__mnistGetLabel ds (cast {to=Int} (natToInteger idx))
      flatImg = prim__reshape1d imgT (cast {to=Int} InputDim)
      lblBuf = prim__setInt (prim__allocInts 1) 0 lbl
      tgtT = prim__oneHot lblBuf 1 (cast {to=Int} NumClasses)
  pure (MkTensorDataPoint flatImg tgtT)


----------------------------------------------------------------------
-- Evaluation
----------------------------------------------------------------------

||| Evaluate accuracy on nSamples random test images by forwarding each
||| image through the V2 model and arg-maxing the logits.
evalAccuracy : {hs : List Nat} ->
               NetworkV2 InputDim hs NumClasses CPU ->
               AnyPtr -> Int -> Nat -> (Double, Double)
evalAccuracy model ds numImages nSamples = go nSamples 0 0.0
  where
    argmax : AnyPtr -> Double -> Int -> Int -> Int
    argmax outT best bestI idx =
      if idx >= cast {to=Int} NumClasses then bestI
      else let v = prim__item1d outT idx
           in if v > best then assert_total $ argmax outT v idx (idx + 1)
                          else assert_total $ argmax outT best bestI (idx + 1)

    go : Nat -> Nat -> Double -> (Double, Double)
    go Z correct totalLoss =
      let n = cast {to=Double} (natToInteger nSamples)
      in (cast {to=Double} (natToInteger correct) / n, totalLoss / n)
    go (S k) correct totalLoss =
      let pos = cast {to=Int} (k * cast numImages `div` nSamples)
          imgT = prim__mnistGetImage ds pos
          lbl = prim__mnistGetLabel ds pos
          flatImg = prim__reshape1d imgT (cast {to=Int} InputDim)
          inV = the (TVec InputDim CPU) (MkTVar flatImg Nothing)
          (_, predV) = forwardTVar model inV
          outT = predV.tensorPtr
          pred = argmax outT (-1.0e30) 0 0
          correct' = if pred == lbl then S correct else correct
          lblBuf = prim__allocInts 1
          lblBuf' = prim__setInt lblBuf 0 lbl
          tgtT = prim__oneHot lblBuf' 1 (cast {to=Int} NumClasses)
          tgtV = the (TVec NumClasses CPU) (MkTVar tgtT Nothing)
          lossT = tnllLoss predV tgtV
          lossVal = prim__item lossT.tensorPtr
      in go k correct' (totalLoss + lossVal)


----------------------------------------------------------------------
-- Training helpers
----------------------------------------------------------------------

||| One epoch = one full pass over the training set (PyTorch semantics).
||| Threads the model and accumulates per-batch loss across all
||| `batchesPerEpoch` mini-batches drawn from the indexed loader.
||| Each mini-batch invokes `epochTVarTensor` (forward + loss + backward
||| + step). Returns the model and the mean per-batch loss.
trainOneFullPass : {hs : List Nat} ->
                   NativeOptimizer ->
                   IO (Vect BatchSize (TensorDataPoint InputDim NumClasses)) ->
                   (batchesPerEpoch : Nat) ->
                   NetworkV2 InputDim hs NumClasses CPU ->
                   IO (NetworkV2 InputDim hs NumClasses CPU, Double)
trainOneFullPass opt genBatch n m0 = go m0 n 0.0
  where
    go : NetworkV2 InputDim hs NumClasses CPU -> Nat -> Double ->
         IO (NetworkV2 InputDim hs NumClasses CPU, Double)
    go m Z     acc = pure (m, acc / cast (natToInteger n))
    go m (S k) acc = do
      batch <- genBatch
      let (m', loss) = epochTVarTensor opt batch tnllLoss m
      go m' k (acc + loss)

||| Per-epoch metrics: test accuracy and test loss over a small eval slice.
mnistMetrics : {hs : List Nat} ->
               AnyPtr -> Int ->
               NetworkV2 InputDim hs NumClasses CPU ->
               IO (List (String, String))
mnistMetrics testDs testCount m =
  let pair = evalAccuracy m testDs testCount 200
  in pure [("test_acc", show (fst pair)),
           ("test_loss", show (snd pair))]


----------------------------------------------------------------------
-- Config & Main
----------------------------------------------------------------------

record Config where
  constructor MkConfig
  lr : Double
  epochs : Nat
  patience : Nat
  seed : Bits64
  dataDir : String
  lrFind : Bool
  trainCount : Nat   -- 0 = use full dataset

defaultConfig : Config
defaultConfig = MkConfig 0.001 5 3 42 "data/mnist" False 0

specs : List (ArgSpec Config)
specs = [ Arg "--lr" (\v, c => { lr := cast v } c)
        , Arg "--epochs" (\v, c => { epochs := castNat v } c)
        , Arg "--patience" (\v, c => { patience := castNat v } c)
        , Arg "--seed" (\v, c => { seed := castBits64 v } c)
        , Arg "--data" (\v, c => { dataDir := v } c)
        , Arg "--lr-find" (\v, c => { lrFind := (v == "1" || v == "true") } c)
        , Arg "--train-count" (\v, c => { trainCount := castNat v } c) ]


partial
main : IO ()
main = do
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)

  srand cfg.seed

  putStrLn "=== MNIST: Convolutional Neural Network ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr ++ " epochs=" ++ show cfg.epochs
           ++ " patience=" ++ show cfg.patience ++ " seed=" ++ show cfg.seed
  putStrLn "Architecture: Conv2d(1->16,k=5) -> ReLU -> Pool(2) -> Conv2d(16->32,k=5) -> ReLU -> Pool(2) -> Dropout(0.5) -> Linear(512->10)"

  -- Load MNIST
  let trainImgPath = cfg.dataDir ++ "/train-images-idx3-ubyte"
      trainLblPath = cfg.dataDir ++ "/train-labels-idx1-ubyte"
      testImgPath  = cfg.dataDir ++ "/t10k-images-idx3-ubyte"
      testLblPath  = cfg.dataDir ++ "/t10k-labels-idx1-ubyte"
  let trainDs = prim__mnistLoad trainImgPath trainLblPath
  let testDs = prim__mnistLoad testImgPath testLblPath
  let trainCount = prim__mnistCount trainDs
      testCount = prim__mnistCount testDs
  putStrLn $ "Train: " ++ show trainCount ++ " images, Test: " ++ show testCount ++ " images"

  -- Build V2 model
  conv1Any <- conv2dLayerV2Any {inC=InC, outC=OutC1, h=ImgH, w=ImgW, kH=KH, kW=KW, padH=0, padW=0} "conv1"
  conv2Any <- conv2dLayerV2Any {inC=OutC1, outC=OutC2, h=Pool1OutH, w=Pool1OutW, kH=KH, kW=KW, padH=0, padW=0} "conv2"
  fcAny <- linearLayerV2Any {i=AfterPool2, o=NumClasses} "fc"

  let model = conv1Any
            ~~> reluLayerV2Any
            ~~> maxPool2dLayerV2 {c=OutC1, inH=Conv1OutH, inW=Conv1OutW, poolH=2, poolW=2, strH=2, strW=2}
            ~~> conv2Any
            ~~> reluLayerV2Any
            ~~> maxPool2dLayerV2 {c=OutC2, inH=Conv2OutH, inW=Conv2OutW, poolH=2, poolW=2, strH=2, strW=2}
            ~~> dropoutLayerV2Any 0.5
            ~~> OutputLayerV2 fcAny
  putStrLn ""

  let opt = nativeAdamGlobalClip cfg.lr 0.9 0.999 1.0e-8 1.0

  let effectiveCount : Int
      effectiveCount = case cfg.trainCount of
                         Z => trainCount
                         n@(S _) => min trainCount (cast n)
  when (cfg.trainCount > 0 && effectiveCount < trainCount) $
    putStrLn $ "Train subset: " ++ show effectiveCount ++ " images (--train-count)"

  genBatch <- mkIndexedLoader {batchSize=BatchSize} (cast effectiveCount) (mnistItem trainDs)

  let batchesPerEpoch : Nat
      batchesPerEpoch = cast {to=Nat} effectiveCount `div` BatchSize
  putStrLn $ "Batches/epoch: " ++ show batchesPerEpoch
           ++ " (batch_size=" ++ show BatchSize ++ ")"

  when cfg.lrFind $ do
    let lrCfg : LrFindConfig
        lrCfg = { numIters := 100 } defaultLrFindConfig
    _ <- lrFind lrCfg
      (\m, batch => let (m', loss) = epochTVarTensor opt batch tnllLoss m
                    in pure (m', loss))
      genBatch opt model
    putStrLn ""
    putStrLn "Done — re-run without --lr-find at the recommended LR."
    exitSuccess

  let trainCfg = MkTrainConfig cfg.epochs 1 (Patience cfg.patience 0.001)
                   (mnistMetrics testDs testCount) (\_ => pure ())

  (trained, epochsDone, finalLoss) <- runTrainingIO
    (\m, _ => trainOneFullPass opt genBatch batchesPerEpoch m)
    (pure ()) trainCfg model

  putStrLn ""
  let finalPair = evalAccuracy trained testDs testCount 1000
      finalAcc = fst finalPair
      finalTestLoss = snd finalPair
  putStrLn $ "Final accuracy (1000 test samples): " ++ show (finalAcc * 100.0) ++ "%"
  putStrLn $ "Final test loss: " ++ show finalTestLoss

  putStrLn ""
  putStrLn $ formatResult [("accuracy", show finalAcc),
                            ("epochs", show epochsDone),
                            ("seed", show cfg.seed)]
