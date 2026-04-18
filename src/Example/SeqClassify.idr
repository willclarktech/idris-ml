-- | SeqClassify: 1D Waveform Classification
-- |
-- | Classify synthetic waveforms (sine, square, triangle) using Conv1D.
-- | All flat dims ≤ 120 to stay under the Idris 2 Peano Nat ceiling.
-- |
-- | Input: [1, 32] = 32 flat (single-channel, 32 timesteps)
-- | Conv1D(1->4, k=3) -> ReLU -> MaxPool1D(2) ->
-- | Conv1D(4->8, k=3) -> ReLU -> MaxPool1D(2) ->
-- | Dropout(0.5) -> Linear(48->3) -> Softmax

module Example.SeqClassify

import Data.List
import Data.String
import Data.Vect
import System
import System.Random

import Backprop
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
import Device
import Variable


----------------------------------------------------------------------
-- Architecture
----------------------------------------------------------------------

SeqLen : Nat
SeqLen = 32

InC : Nat
InC = 1

C1 : Nat
C1 = 4

C2 : Nat
C2 = 8

K : Nat
K = 3

NumClasses : Nat
NumClasses = 3

BatchSize : Nat
BatchSize = 32

-- After Conv1: 32 - 3 + 1 = 30
Conv1Out : Nat
Conv1Out = ConvOutDim SeqLen K 0  -- 30

-- After Pool1: (30 - 2) / 2 + 1 = 15
Pool1Out : Nat
Pool1Out = PoolOutDim Conv1Out 2 2  -- 15

-- After Conv2: 15 - 3 + 1 = 13
Conv2Out : Nat
Conv2Out = ConvOutDim Pool1Out K 0  -- 13

-- After Pool2: (13 - 2) / 2 + 1 = 6
Pool2Out : Nat
Pool2Out = PoolOutDim Conv2Out 2 2  -- 6

-- Flat dims
InputDim : Nat
InputDim = InC * SeqLen  -- 32

AfterConv1 : Nat
AfterConv1 = C1 * Conv1Out  -- 120

AfterPool1 : Nat
AfterPool1 = C1 * Pool1Out  -- 60

AfterConv2 : Nat
AfterConv2 = C2 * Conv2Out  -- 104

AfterPool2 : Nat
AfterPool2 = C2 * Pool2Out  -- 48


----------------------------------------------------------------------
-- Data Generation
----------------------------------------------------------------------

||| Generate a sine, square, or triangle wave with random freq/phase.
genSample : Double -> Double -> Int -> Int -> Double
genSample freq phase label i =
  let t = cast {to=Double} i / 32.0 * 2.0 * pi * freq + phase
  in if label == 0 then sin t
     else if label == 1 then (if sin t > 0.0 then 1.0 else -1.0)
     else 2.0 * abs (t / pi - 2.0 * cast {to=Double} (the Integer (cast (t / (2.0 * pi) + 0.5)))) - 1.0

genWaveform : Int -> IO (List Double)
genWaveform label = do
  freqN <- randomInt 10 30
  phaseN <- randomInt 0 100
  let freq = cast {to=Double} freqN / 10.0
      phase = cast {to=Double} phaseN / 100.0 * 2.0 * pi
      wave = map (genSample freq phase label) [0 .. cast {to=Int} (minus SeqLen 1)]
  pure wave

||| Generate one training data point.
seqPoint : IO (TensorDataPoint InputDim NumClasses)
seqPoint = do
  labelN <- randomInt 0 2
  let label = cast {to=Int} labelN
  wave <- genWaveform label
  let sI = cast {to=Int} SeqLen
      nI = cast {to=Int} NumClasses
      inBuf = prim__allocDoubles sI
      inBuf' = packWave inBuf 0 wave
      inT = prim__create1d sI inBuf' 0
      lblBuf = prim__allocInts 1
      lblBuf' = prim__setInt lblBuf 0 label
      tgtT = prim__oneHot lblBuf' 1 nI
  pure $ MkTensorDataPoint inT tgtT
  where
    packWave : AnyPtr -> Int -> List Double -> AnyPtr
    packWave buf _ [] = buf
    packWave buf i (x :: xs) = packWave (prim__setDouble buf i x) (i + 1) xs

seqBatch : (n : Nat) -> IO (Vect n (TensorDataPoint InputDim NumClasses))
seqBatch Z = pure []
seqBatch (S k) = do
  dp <- seqPoint
  rest <- seqBatch k
  pure (dp :: rest)


----------------------------------------------------------------------
-- Loss
----------------------------------------------------------------------

seqCE : LossFnTensor CPU
seqCE predT targetT =
  let logProbs = prim__logSoftmax predT 0
      product = prim__mul logProbs targetT
      totalSum = prim__sum product
      loss = prim__neg totalSum
      val = prim__item loss
  in Var loss Nothing val


----------------------------------------------------------------------
-- Config & Main
----------------------------------------------------------------------

record Config where
  constructor MkConfig
  lr : Double
  epochs : Nat
  patience : Nat
  seed : Bits64
  accumSteps : Nat

defaultConfig : Config
defaultConfig = MkConfig 0.001 1000 200 42 1

specs : List (ArgSpec Config)
specs = [ Arg "--lr" (\v, c => { lr := cast v } c)
        , Arg "--epochs" (\v, c => { epochs := castNat v } c)
        , Arg "--patience" (\v, c => { patience := castNat v } c)
        , Arg "--seed" (\v, c => { seed := castBits64 v } c)
        , Arg "--accum-steps" (\v, c => { accumSteps := castNat v } c) ]


partial
main : IO ()
main = do
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)

  srand cfg.seed

  putStrLn "=== SeqClassify: 1D Waveform Classification ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr ++ " epochs=" ++ show cfg.epochs
           ++ " patience=" ++ show cfg.patience ++ " seed=" ++ show cfg.seed
           ++ " accum=" ++ show cfg.accumSteps
  putStrLn "Architecture: Conv1d(1->4,k=3) -> ReLU -> Pool(2) -> Conv1d(4->8,k=3) -> ReLU -> Pool(2) -> Dropout(0.5) -> Linear(48->3)"

  conv1 <- conv1dLayer {inC=InC, outC=C1, len=SeqLen, kL=K, pad=0}
  let relu1 : AnyLayer AfterConv1 AfterConv1 (Variable CPU)
      relu1 = reluLayer
  let pool1 : AnyLayer AfterConv1 AfterPool1 (Variable CPU)
      pool1 = maxPool1dLayer {c=C1, len=Conv1Out, poolK=2, str=2}
  conv2 <- conv1dLayer {inC=C1, outC=C2, len=Pool1Out, kL=K, pad=0}
  let relu2 : AnyLayer AfterConv2 AfterConv2 (Variable CPU)
      relu2 = reluLayer
  let pool2 : AnyLayer AfterConv2 AfterPool2 (Variable CPU)
      pool2 = maxPool1dLayer {c=C2, len=Conv2Out, poolK=2, str=2}
  let drop : AnyLayer AfterPool2 AfterPool2 (Variable CPU)
      drop = dropoutLayer 0.5
  fc <- linearLayer {i=AfterPool2, o=NumClasses}

  let model = autoName $
        conv1 ~> relu1 ~> pool1 ~> conv2 ~> relu2 ~> pool2 ~> drop ~> fc ~> OutputLayer softmaxLayer
  putStrLn $ "Model: " ++ show model
  putStrLn ""

  let opt = nativeAdamGlobalClip cfg.lr 0.9 0.999 1.0e-8 1.0

  let genBatch : IO (Vect BatchSize (TensorDataPoint InputDim NumClasses))
      genBatch = seqBatch BatchSize

  let trainCfg = patienceConfig cfg.epochs cfg.patience

  let epochFn = if cfg.accumSteps > 1
        then \m, d => epochNativeTensorPreAccum opt cfg.accumSteps d seqCE m
        else \m, d => epochNativeTensorPre opt d seqCE m
  (trained, epochsDone, finalLoss) <- runTraining epochFn genBatch trainCfg model

  putStrLn ""
  putStrLn $ formatResult [("loss", show finalLoss),
                            ("epochs", show epochsDone),
                            ("seed", show cfg.seed)]
