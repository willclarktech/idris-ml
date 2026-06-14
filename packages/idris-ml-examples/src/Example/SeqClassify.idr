-- | SeqClassify: 1D Waveform Classification
-- |
-- | Classify synthetic waveforms (sine, square, triangle) using Conv1D,
-- | on the v1 Nn/fit surface. All flat dims <= 120 to stay under the
-- | Idris 2 Peano Nat ceiling.
-- |
-- | Conv1D(1->4,k=3) -> ReLU -> MaxPool1D(2) ->
-- | Conv1D(4->8,k=3) -> ReLU -> MaxPool1D(2) ->
-- | Dropout(0.5) -> Linear(48->3). Raw logits; tnllLossMean applies log_softmax.

module Example.SeqClassify

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

Conv1Out : Nat
Conv1Out = ConvOutDim SeqLen K 0  -- 30

Pool1Out : Nat
Pool1Out = PoolOutDim Conv1Out 2 2  -- 15

Conv2Out : Nat
Conv2Out = ConvOutDim Pool1Out K 0  -- 13

Pool2Out : Nat
Pool2Out = PoolOutDim Conv2Out 2 2  -- 6

InputDim : Nat
InputDim = InC * SeqLen  -- 32

AfterPool2 : Nat
AfterPool2 = C2 * Pool2Out  -- 48

----------------------------------------------------------------------
-- Synthetic data
----------------------------------------------------------------------

randomInt : (lo, hi : Nat) -> IO Nat
randomInt lo hi = do
  n <- randomRIO (cast {to=Int32} (natToInteger lo), cast {to=Int32} (natToInteger hi))
  pure (fromInteger (cast {to=Integer} n))

-- sine (0), square (1), triangle (2) — random freq/phase.
genSample : Double -> Double -> Int -> Int -> Double
genSample freq phase label i =
  let t = cast {to=Double} i / 32.0 * 2.0 * pi * freq + phase
  in if label == 0 then sin t
     else if label == 1 then (if sin t > 0.0 then 1.0 else -1.0)
     else 2.0 * abs (t / pi - 2.0 * cast {to=Double} (the Integer (cast (t / (2.0 * pi) + 0.5)))) - 1.0

genWaveform : Int -> IO (Vect SeqLen Double)
genWaveform label = do
  freqN <- randomInt 10 30
  phaseN <- randomInt 0 100
  let freq = cast {to=Double} freqN / 10.0
      phase = cast {to=Double} phaseN / 100.0 * 2.0 * pi
  pure (Data.Vect.Fin.tabulate (\i =>
    genSample freq phase label (cast {to=Int} (finToInteger i))))

oneHot : Int -> Vect NumClasses Double
oneHot label = Data.Vect.Fin.tabulate (\i =>
  if cast {to=Int} (finToInteger i) == label then 1.0 else 0.0)

-- Fresh device tensors per pull (no persistent handles → no use-after-free).
mkSample : IO (Tensor [InputDim] Ex F NoGrad, Tensor [NumClasses] Ex F NoGrad)
mkSample = do
  labelN <- randomInt 0 2
  let label = cast {to=Int} labelN
  wave <- genWaveform label
  x <- tensor {dims = [InputDim]} (FromVect wave)
  y <- tensor {dims = [NumClasses]} (FromVect (oneHot label))
  pure (x, y)

----------------------------------------------------------------------
-- Model + loss
----------------------------------------------------------------------

Model : Type
Model = Seq InputDim NumClasses Ex F WithGrad

buildModel : IO Model
buildModel = runInit $ do
  c1 <- conv1d {inC = InC} {outC = C1} {len = SeqLen}   {kL = K} {pad = 0}
  c2 <- conv1d {inC = C1}  {outC = C2} {len = Pool1Out} {kL = K} {pad = 0}
  l  <- linear {i = AfterPool2} {o = NumClasses}
  pure (c1 ~~> reluA
           ~~> maxPool1d {c = C1} {len = Conv1Out} {poolK = 2} {str = 2}
           ~~> c2 ~~> reluA
           ~~> maxPool1d {c = C2} {len = Conv2Out} {poolK = 2} {str = 2}
           ~~> dropout 0.5
           ~~> l ~~> Nil)

nllLoss : Model -> (Tensor [BatchSize, InputDim] Ex F NoGrad, Tensor [BatchSize, NumClasses] Ex F NoGrad) ->
          IO (Tensor [] Ex F WithGrad)
nllLoss model (x, tgt) = do
  out <- forwardSeq {b = BatchSize} model (retypeGrad x)
  tnllLossMean {b = BatchSize} {n = NumClasses} out (retypeGrad tgt)

----------------------------------------------------------------------
-- Config & Main
----------------------------------------------------------------------

record Config where
  constructor MkConfig
  lr       : Double
  epochs   : Nat
  patience : Nat
  seed     : Bits64

defaultConfig : Config
defaultConfig = MkConfig 0.001 1000 200 42

specs : List (ArgSpec Config)
specs = [ Arg "--lr" (\v, c => { lr := cast v } c)
        , Arg "--epochs" (\v, c => { epochs := castNat v } c)
        , Arg "--patience" (\v, c => { patience := castNat v } c)
        , Arg "--seed" (\v, c => { seed := castBits64 v } c) ]

%default partial

main : IO ()
main = do
  requireMachine {m = ChosenMachine}
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)

  srand cfg.seed
  tsetInitSeed {ex = Ex} cfg.seed

  putStrLn "=== SeqClassify: 1D Waveform Classification ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr ++ " epochs=" ++ show cfg.epochs
           ++ " patience=" ++ show cfg.patience ++ " seed=" ++ show cfg.seed
  putStrLn "Architecture: Conv1d(1->4,k=3) -> ReLU -> Pool(2) -> Conv1d(4->8,k=3) -> ReLU -> Pool(2) -> Dropout(0.5) -> Linear(48->3)"

  opt <- adam cfg.lr ({ clip := NormClip 1.0 } defaultOpts)
  model <- buildModel
  let bs = batched {b = BatchSize} {i = InputDim} {o = NumClasses} (generate mkSample)
  putStrLn ""

  (_, epochsDone, finalLoss) <-
    fitSupervised opt nllLoss bs (patienceConfig cfg.epochs cfg.patience) model

  putStrLn ""
  putStrLn $ formatResult [("loss", show finalLoss),
                           ("epochs", show epochsDone),
                           ("seed", show cfg.seed)]
