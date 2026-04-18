-- | MNIST: Convolutional Neural Network
-- |
-- | LeNet-style CNN for handwritten digit classification.
-- | Conv2d(1->16,k=5) -> ReLU -> MaxPool(2) ->
-- | Conv2d(16->32,k=5) -> ReLU -> MaxPool(2) ->
-- | Linear(512->10) -> Softmax
-- |
-- | Loads MNIST .idx files via C FFI. Trains on random mini-batches.

module Example.Mnist

import Data.List
import Data.String
import Data.Vect
import Decidable.Equality
import System
import System.Random

import Backprop
import DataLoader
import DataPoint
import Endofunctor
import Floating
import Generate
import Layer
import Layer.Core
import Layer.Conv
import Layer.Dropout
import Math
import Tensor
import Train
import Util
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
BatchSize = 32


----------------------------------------------------------------------
-- Data Loading
----------------------------------------------------------------------

||| Fetch a single MNIST image as a TensorDataPoint.
mnistItem : AnyPtr -> Nat -> IO (TensorDataPoint InputDim NumClasses)
mnistItem ds idx = do
  let imgT = prim__mnistGetImage ds (cast {to=Int} (natToInteger idx))
      lbl = prim__mnistGetLabel ds (cast {to=Int} (natToInteger idx))
      flatImg = prim__reshape1d imgT (cast {to=Int} InputDim)
      lblBuf = prim__setInt (prim__allocInts 1) 0 lbl
      tgtT = prim__oneHot lblBuf 1 (cast {to=Int} NumClasses)
  pure (MkTensorDataPoint flatImg tgtT)


----------------------------------------------------------------------
-- Loss
----------------------------------------------------------------------

||| Cross-entropy loss: -sum(target * log_softmax(logits)) / batch
mnistCE : LossFnTensor
mnistCE predT targetT =
  let logProbs = prim__logSoftmax predT 0  -- dim=0 for 1D logits
      product = prim__mul logProbs targetT
      totalSum = prim__sum product
      loss = prim__neg totalSum
      val = prim__item loss
  in Var loss Nothing val


----------------------------------------------------------------------
-- Evaluation
----------------------------------------------------------------------

||| Evaluate accuracy on nSamples random test images.
evalAccuracy : {hs : List Nat} ->
               Network InputDim hs NumClasses Variable ->
               AnyPtr -> Int -> Nat -> (Double, Double)
evalAccuracy model ds numImages nSamples = go model nSamples 0 0.0
  where
    argmax : AnyPtr -> Double -> Int -> Int -> Int
    argmax outT best bestI idx =
      if idx >= cast {to=Int} NumClasses then bestI
      else let v = prim__item1d outT idx
           in if v > best then assert_total $ argmax outT v idx (idx + 1)
                          else assert_total $ argmax outT best bestI (idx + 1)

    go : {hs' : List Nat} -> Network InputDim hs' NumClasses Variable ->
         Nat -> Nat -> Double -> (Double, Double)
    go _ Z correct totalLoss =
      let n = cast {to=Double} (natToInteger nSamples)
      in (cast {to=Double} (natToInteger correct) / n, totalLoss / n)
    go m (S k) correct totalLoss =
      let pos = cast {to=Int} (k * cast numImages `div` nSamples)
          imgT = prim__mnistGetImage ds pos
          lbl = prim__mnistGetLabel ds pos
          flatImg = prim__reshape1d imgT (cast {to=Int} InputDim)
          fwdPair = forwardVarTensor m flatImg
          outT = snd fwdPair
          pred = argmax outT (-1.0e30) 0 0
          correct' = if pred == lbl then S correct else correct
          -- compute loss for this sample
          lblBuf = prim__allocInts 1
          lblBuf' = prim__setInt lblBuf 0 lbl
          tgtT = prim__oneHot lblBuf' 1 (cast {to=Int} NumClasses)
          lossVar = mnistCE outT tgtT
      in go m k correct' (totalLoss + lossVar.value)


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

defaultConfig : Config
defaultConfig = MkConfig 0.001 2000 500 42 "data/mnist"

specs : List (ArgSpec Config)
specs = [ Arg "--lr" (\v, c => { lr := cast v } c)
        , Arg "--epochs" (\v, c => { epochs := castNat v } c)
        , Arg "--patience" (\v, c => { patience := castNat v } c)
        , Arg "--seed" (\v, c => { seed := castBits64 v } c)
        , Arg "--data" (\v, c => { dataDir := v } c) ]


partial
main : IO ()
main = do
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)

  srand cfg.seed

  putStrLn "=== MNIST: Convolutional Neural Network ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr ++ " epochs=" ++ show cfg.epochs
           ++ " patience=" ++ show cfg.patience ++ " seed=" ++ show cfg.seed
  putStrLn "Architecture: Conv2d(1->16,k=5) -> ReLU -> Pool(2) -> Conv2d(16->32,k=5) -> ReLU -> Pool(2) -> Linear(512->10)"

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

  -- Build model with dropout
  -- (Batch norm omitted: Idris type-checker hangs on large Nat reduction
  -- for channels*spatialDim proofs. BatchNormState works for smaller dims.)
  conv1 <- conv2dLayer {inC=InC, outC=OutC1, h=ImgH, w=ImgW, kH=KH, kW=KW, padH=0, padW=0}
  let relu1 : AnyLayer AfterConv1 AfterConv1 Variable
      relu1 = reluLayer
  let pool1 : AnyLayer AfterConv1 AfterPool1 Variable
      pool1 = maxPool2dLayer {c=OutC1, inH=Conv1OutH, inW=Conv1OutW, poolH=2, poolW=2, strH=2, strW=2}
  let drop1 : AnyLayer AfterPool1 AfterPool1 Variable
      drop1 = dropoutLayer 0.25
  conv2 <- conv2dLayer {inC=OutC1, outC=OutC2, h=Pool1OutH, w=Pool1OutW, kH=KH, kW=KW, padH=0, padW=0}
  let relu2 : AnyLayer AfterConv2 AfterConv2 Variable
      relu2 = reluLayer
  let pool2 : AnyLayer AfterConv2 AfterPool2 Variable
      pool2 = maxPool2dLayer {c=OutC2, inH=Conv2OutH, inW=Conv2OutW, poolH=2, poolW=2, strH=2, strW=2}
  let drop2 : AnyLayer AfterPool2 AfterPool2 Variable
      drop2 = dropoutLayer 0.5
  fc <- linearLayer {i=AfterPool2, o=NumClasses}

  let model = autoName $
        conv1 ~> relu1 ~> pool1
        ~> conv2 ~> relu2 ~> pool2 ~> drop2
        ~> fc ~> OutputLayer softmaxLayer
  putStrLn $ "Model: " ++ show model
  putStrLn ""

  let opt = nativeAdamGlobalClip cfg.lr 0.9 0.999 1.0e-8 1.0

  genBatch <- mkIndexedLoader {batchSize=BatchSize} (cast trainCount) (mnistItem trainDs)

  let trainCfg = patienceConfig cfg.epochs cfg.patience

  (trained, epochsDone, finalLoss) <- runTraining
    (\m, d => epochNativeTensorPre opt d mnistCE m) genBatch trainCfg model

  putStrLn ""
  let evalModel = setNetworkTraining False trained
  let finalPair = evalAccuracy evalModel testDs testCount 1000
      finalAcc = fst finalPair
      finalTestLoss = snd finalPair
  putStrLn $ "Final accuracy (1000 test samples): " ++ show finalAcc
  putStrLn $ "Final test loss: " ++ show finalTestLoss

  putStrLn ""
  putStrLn $ formatResult [("accuracy", show finalAcc),
                            ("epochs", show epochsDone),
                            ("seed", show cfg.seed)]
