-- | SeqClassify: 1D Waveform Classification
-- |
-- | Classify synthetic waveforms (sine, square, triangle) using Conv1D,
-- | on the v1 Nn/fit surface. All flat dims <= 120 to stay under the
-- | Idris 2 Peano Nat ceiling. Waveforms carry additive N(0, 0.1) noise;
-- | the run ends with an argmax accuracy eval over 500 fresh samples.
-- |
-- | Conv1D(1->4,k=3) -> ReLU -> MaxPool1D(2) ->
-- | Conv1D(4->8,k=3) -> ReLU -> MaxPool1D(2) ->
-- | Dropout(0.5) -> Linear(48->3). Raw logits; tnllLossMean applies log_softmax.

module Example.SeqClassify

import Control.Linear.LIO
import Data.Linear.Notation
import Data.List
import Data.Vect
import System

import Ml.Checkpoint
import Ml.Compat.Random
import Ml.Fit
import Ml.Rng
import Ml.Sampler
import Ml.Simple
import Ml.Train

import BuildConfig

-- This example's model is a linear `Seq`; hide the IO `Nn.Seq` constructors
-- (same `Nil`/`::`/`~~>` names) so the chain builder resolves unambiguously.

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

-- Held-out samples for the post-training accuracy eval (matches the
-- reference's `evaluate(model, 500)`).
EvalSize : Nat
EvalSize = 500

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

-- Additive observation noise, matching the reference's `random.gauss(0, 0.1)`
-- per timestep. Without it the three waveform classes are separable by a
-- handful of samples and the task measures nothing.
NoiseSd : Double
NoiseSd = 0.1

genWaveform : Int -> IO (Vect SeqLen Double)
genWaveform label = do
  freqN <- randomInt 10 30
  phaseN <- randomInt 0 100
  let freq = cast {to=Double} freqN / 10.0
      phase = cast {to=Double} phaseN / 100.0 * 2.0 * pi
      clean = Data.Vect.Fin.tabulate (\i =>
        genSample freq phase label (cast {to=Int} (finToInteger i)))
  traverse (\v => do z <- normalSample
                     pure (v + NoiseSd * z)) clean

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

-- Top-level Init value (not inline under runInitL — see the linear-types
-- migration recipe: a nested do-block under `run $ do …` trips the
-- ambiguity-depth limit). Built as a linear `Seq`.
mkModel : MaskSource -> Init Model
mkModel msrc = do
  c1 <- conv1d {inC = InC} {outC = C1} {len = SeqLen}   {kL = K} {pad = 0}
  c2 <- conv1d {inC = C1}  {outC = C2} {len = Pool1Out} {kL = K} {pad = 0}
  l  <- linear {i = AfterPool2} {o = NumClasses}
  pure (c1 ~~> reluA
           ~~> maxPool1d {c = C1} {len = Conv1Out} {poolK = 2} {str = 2}
           ~~> c2 ~~> reluA
           ~~> maxPool1d {c = C2} {len = Conv2Out} {poolK = 2} {str = 2}
           ~~> dropoutWith msrc 0.5
           ~~> l ~~> Nil)

-- Linear-resource loss: consume the (linear) Seq model, forward it via
-- forwardSeq, return the banged scalar loss beside the rebuilt model.
nllLossL : (1 _ : Model) ->
           (Tensor [BatchSize, InputDim] Ex F NoGrad, Tensor [BatchSize, NumClasses] Ex F NoGrad) ->
           L IO {use = 1} (LPair (!* (Tensor [] Ex F WithGrad)) Model)
nllLossL model (x, tgt) = do
  (MkBang out # model') <- forwardSeq {b = BatchSize} model (retypeGrad x)
  loss <- tnllLossMeanL {b = BatchSize} {n = NumClasses} out (retypeGrad tgt)
  pure1 (MkBang loss # model')

----------------------------------------------------------------------
-- Evaluation: argmax accuracy over one EvalSize batch of fresh waveforms
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

-- Argmax accuracy over the already-forwarded [EvalSize, NumClasses] logits.
-- Pure (primItem2d reads are pure FFI); the forward itself happens in the
-- linear block via forwardSeq.
accuracyFrom : Tensor [EvalSize, NumClasses] Ex F NoGrad ->
               Tensor [EvalSize, NumClasses] Ex F NoGrad -> Double
accuracyFrom pred tgt =
  let correct = length $ filter id
        [ argmaxRow pred.tensorPtr r == argmaxRow tgt.tensorPtr r
        | r <- map (cast {to=Int}) [the Nat 0 .. EvalSize `minus` 1] ]
  in cast {to=Double} correct / cast {to=Double} EvalSize

----------------------------------------------------------------------
-- Config & Main
----------------------------------------------------------------------

record Config where
  constructor MkConfig
  lr       : Double
  epochs   : Nat
  patience : Nat
  seed     : Bits64
  replay   : String  -- "" = live; else a recorded-draws file (mask channel)

defaultConfig : Config
defaultConfig = MkConfig 0.001 1000 200 42 ""

specs : List (ArgSpec Config)
specs = [ Arg "--lr" (\v, c => { lr := cast v } c)
        , Arg "--epochs" (\v, c => { epochs := castNat v } c)
        , Arg "--patience" (\v, c => { patience := castNat v } c)
        , Arg "--seed" (\v, c => { seed := castBits64 v } c)
        , Arg "--replay" (\v, c => { replay := v } c) ]

%default partial

main : IO ()
main = do
  requireMachine {m = ChosenMachine}
  args <- getArgs
  let cfg = parseArgs defaultConfig specs (drop 1 args)

  srand cfg.seed
  tsetInitSeed {ex = Ex} cfg.seed

  -- Live: fresh dropout seeds per forward. --replay: the recorded run's
  -- draws; the dropout layer consumes its mask channel.
  replay <- if cfg.replay == "" then liveReplay else loadReplay cfg.replay

  putStrLn "=== SeqClassify: 1D Waveform Classification ==="
  putStrLn $ "Config: lr=" ++ show cfg.lr ++ " epochs=" ++ show cfg.epochs
           ++ " patience=" ++ show cfg.patience ++ " seed=" ++ show cfg.seed
  putStrLn "Architecture: Conv1d(1->4,k=3) -> ReLU -> Pool(2) -> Conv1d(4->8,k=3) -> ReLU -> Pool(2) -> Dropout(0.5) -> Linear(48->3)"

  opt <- adam cfg.lr ({ clip := NormClip 1.0 } defaultOpts)
  let bs = batched {b = BatchSize} {i = InputDim} {o = NumClasses} (generate mkSample)
  maybeDumpBatch {ex = ExampleExecutor} bs

  -- Linear surface: model born linear (runInitL), threaded through
  -- fitSupervised, converted to an inference model (eval — this chain has a
  -- Dropout), forwarded once on the eval batch, then discarded.
  --
  -- The eval batch is pulled *after* training: a collated batch is an
  -- intermediate arena tensor, so each training step's optimizer_step →
  -- arena_reset would dangle a batch pre-fetched before the loop.
  Control.Linear.LIO.run $ do
    model <- runInitL (mkModel replay.masks)
    -- Replays the reference's weights and batch under IDRISML_ORACLE_LOAD;
    -- returns `bs` untouched otherwise. See scripts/check-step-oracle.py.
    bs2 <- liftIO1 (oracleBatchStream {ex = ExampleExecutor} bs)
    liftIO1 (putStrLn "")
    (MkBang (epochsDone, finalLoss) # trained) <-
      fitSupervised opt nllLossL bs2 (patienceConfig cfg.epochs cfg.patience) model
    liftIO1 (putStrLn "")
    infer <- eval trained
    evalBatch <- liftIO1
      (batched {b = EvalSize} {i = InputDim} {o = NumClasses} (generate mkSample)).next
    (MkBang predB # infer') <- forwardSeq {b = EvalSize} infer (fst evalBatch)
    discard infer'
    liftIO1 $ do
      let acc = accuracyFrom predB (snd evalBatch)
      putStrLn $ "Final accuracy (" ++ show EvalSize ++ " samples): " ++ show (acc * 100.0) ++ "%"
      putStrLn ""
      putStrLn $ formatResult [("accuracy", show acc),
                               ("epochs", show epochsDone),
                               ("loss", show finalLoss),
                               ("seed", show cfg.seed)]
