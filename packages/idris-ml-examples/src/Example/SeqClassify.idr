-- | SeqClassify: 1D Waveform Classification
-- |
-- | Classify synthetic waveforms (sine, square, triangle) using Conv1D.
-- | All flat dims ≤ 120 to stay under the Idris 2 Peano Nat ceiling.
-- |
-- | Input: [1, 32] = 32 flat (single-channel, 32 timesteps)
-- | Conv1D(1->4, k=3) -> ReLU -> MaxPool1D(2) ->
-- | Conv1D(4->8, k=3) -> ReLU -> MaxPool1D(2) ->
-- | Dropout(0.5) -> Linear(48->3). Outputs raw logits; tnllLoss applies log_softmax.

module Example.SeqClassify

import Data.List
import Data.String
import Data.Vect
import System
import Compat.Random

import Backprop
import DataPoint
import Floating
import Generate
import Hpo.LrFinder
import Layer.Activation
import Layer.Conv  -- ConvOutDim / PoolOutDim type-level helpers
import Layer.Core
import Layer.Dropout
import Layer.Linear
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

genWaveformV : Int -> IO (Vect SeqLen Double)
genWaveformV label = do
  freqN <- randomInt 10 30
  phaseN <- randomInt 0 100
  let freq = cast {to=Double} freqN / 10.0
      phase = cast {to=Double} phaseN / 100.0 * 2.0 * pi
  pure (Data.Vect.Fin.tabulate (\i =>
    genSample freq phase label (cast {to=Int} (finToInteger i))))

oneHotV : Int -> Vector NumClasses Double
oneHotV label =
  VTensor (Data.Vect.Fin.tabulate (\i => STensor
    (if cast {to=Int} (finToInteger i) == label then 1.0 else 0.0)))

waveToVector : Vect SeqLen Double -> Vector InputDim Double
waveToVector wave = VTensor (map STensor wave)

||| Generate one  training data point.
seqPoint : IO (DataPoint InputDim NumClasses Double)
seqPoint = do
  labelN <- randomInt 0 2
  let label = cast {to=Int} labelN
  wave <- genWaveformV label
  pure $ MkDataPoint (waveToVector wave) (oneHotV label)

seqBatch : (n : Nat) -> IO (Vect n (DataPoint InputDim NumClasses Double))
seqBatch Z = pure []
seqBatch (S k) = do
  dp <- seqPoint
  rest <- seqBatch k
  pure (dp :: rest)


----------------------------------------------------------------------
-- Config & Main
----------------------------------------------------------------------

record Config where
  constructor MkConfig
  lr : Double
  epochs : Nat
  patience : Nat
  seed : Bits64
  lrFind : Bool

defaultConfig : Config
defaultConfig = MkConfig 0.001 1000 200 42 False

specs : List (ArgSpec Config)
specs = [ Arg "--lr" (\v, c => { lr := cast v } c)
        , Arg "--epochs" (\v, c => { epochs := castNat v } c)
        , Arg "--patience" (\v, c => { patience := castNat v } c)
        , Arg "--seed" (\v, c => { seed := castBits64 v } c)
        , Arg "--lr-find" (\v, c => { lrFind := (v == "1" || v == "true") } c) ]


partial
main : IO ()
main = do
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)

  srand cfg.seed

  putStrLn "=== SeqClassify: 1D Waveform Classification ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr ++ " epochs=" ++ show cfg.epochs
           ++ " patience=" ++ show cfg.patience ++ " seed=" ++ show cfg.seed
  putStrLn "Architecture: Conv1d(1->4,k=3) -> ReLU -> Pool(2) -> Conv1d(4->8,k=3) -> ReLU -> Pool(2) -> Dropout(0.5) -> Linear(48->3)"

  conv1Any <- conv1dLayerAny {inC=InC, outC=C1, len=SeqLen, kL=K, pad=0} "conv1"
  conv2Any <- conv1dLayerAny {inC=C1, outC=C2, len=Pool1Out, kL=K, pad=0} "conv2"
  fcAny <- linearLayerAny {i=AfterPool2, o=NumClasses} "fc"
  let model = conv1Any
            ~~> reluLayerAny
            ~~> maxPool1dLayer {c=C1, len=Conv1Out, poolK=2, str=2}
            ~~> conv2Any
            ~~> reluLayerAny
            ~~> maxPool1dLayer {c=C2, len=Conv2Out, poolK=2, str=2}
            ~~> dropoutLayerAny 0.5
            ~~> OutputLayer fcAny
  putStrLn ""

  let opt = nativeAdamGlobalClip cfg.lr 0.9 0.999 1.0e-8 1.0

  let genBatch : IO (Vect BatchSize (DataPoint InputDim NumClasses Double))
      genBatch = seqBatch BatchSize

  let trainCfg = patienceConfig cfg.epochs cfg.patience

  when cfg.lrFind $ do
    let lrCfg : LrFindConfig
        lrCfg = { numIters := 100 } defaultLrFindConfig
    _ <- lrFind lrCfg
      (\m, d => let (m', loss) = epochVar opt d tnllLoss m
                in pure (m', loss))
      genBatch opt model
    putStrLn ""
    putStrLn "Done — re-run without --lr-find at the recommended LR."
    exitSuccess

  (trained, epochsDone, finalLoss) <- runTraining
    (\m, d => epochVar opt d tnllLoss m) genBatch trainCfg model

  putStrLn ""
  putStrLn $ formatResult [("loss", show finalLoss),
                            ("epochs", show epochsDone),
                            ("seed", show cfg.seed)]
