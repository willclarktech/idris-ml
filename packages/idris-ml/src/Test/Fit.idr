module Test.Fit

import Control.Linear.LIO
import Data.Linear.Notation
import Data.Vect

import DataStream
import Dataset
import Executor
import Fit
import GradScaler
import Optimizer
import Tensor
import Test.Config
import Test.Harness
import Train

-- Registered scalar param (mirrors Test.Optimizer / Test.TrainEngine).
mkW : String -> Double -> IO (Tensor [] TestExecutor TestDType WithGrad)
mkW name v = do
  wptr <- ioRerun (\_ => primCreateScalar {ex=TestExecutor} v 1)
  _ <- ioRerun (\_ => primParamRegister {ex=TestExecutor} name wptr)
  pure (MkTensor wptr (Just name))

units : DataStream ()
units = generate (pure ())

-- A linear box around a `Double` "model": pattern-matching it consumes the
-- linear value and rebinds the payload unrestricted (a bare `Double` model
-- has no constructor to match, so it can't be read out of the linear slot).
record LBox where
  constructor MkLBox
  unLBox : Double

-- fitSupervised: loss = w*w from w=2.0 at lr=0.1 → w *= 0.8 each epoch;
-- after 5 epochs w = 2*0.8^5 = 0.655 < 1.0.
fitSupervisedConverges : IO Bool
fitSupervisedConverges = do
  w <- mkW "fit_sup_w" 2.0
  opt <- sgd {ex=TestExecutor} 0.1 defaultOpts
  Control.Linear.LIO.run (do
    (MkBang _ # ()) <- fitSupervised {ex=TestExecutor} opt
      (\(), () => do loss <- liftIO1 (tmul w w); pure1 (MkBang loss # ()))
      units (simpleConfig 5) ()
    let v = tensorItem w
    liftIO1 (check ("fitSupervised converges (w 2.0 -> " ++ show v ++ ")") (v < 1.0)))

-- Recurrent/two-phase are Step folds, not driver variants: a Step that
-- folds two "timesteps" into one loss. loss = 2w², grad = 4w, lr 0.05 →
-- w *= 0.8 each epoch; after 5, w = 0.8^5 = 0.328 < 1.0.
recStep : Tensor [] TestExecutor TestDType WithGrad -> Optimizer TestExecutor -> EpochStep () ()
recStep w opt () () = do
  l1 <- liftIO1 (tmul w w)
  l2 <- liftIO1 (tmul w w)
  summed <- liftIO1 (tadd l1 l2)
  d <- liftIO1 (trainStep opt summed)
  pure1 (MkBang d # ())

recurrentFoldConverges : IO Bool
recurrentFoldConverges = do
  w <- mkW "fit_rec_w" 1.0
  opt <- sgd {ex=TestExecutor} 0.05 defaultOpts
  Control.Linear.LIO.run (do
    (MkBang _ # ()) <- fit {ex=TestExecutor} (recStep w opt) opt units (simpleConfig 5) ()
    let v = tensorItem w
    liftIO1 (check ("recurrent Step-fold converges (w=" ++ show v ++ ")") (v < 1.0)))

-- Mixed precision via fitSupervisedMixed. On tape F64 the scaler never
-- overflows (scale 1.0), so it trains identically to single precision —
-- this exercises the scaled path + nanHalts=False wiring end to end.
fitMixedTrains : IO Bool
fitMixedTrains = do
  w <- mkW "fit_mix_w" 2.0
  opt <- sgd {ex=TestExecutor} 0.1 defaultOpts
  gs <- gradScaler {ex=TestExecutor} {dt=TestDType} 1.0 2.0 0.5 1000
  Control.Linear.LIO.run (do
    (MkBang _ # ()) <- fitSupervisedMixed {ex=TestExecutor} opt gs
      (\(), () => do loss <- liftIO1 (tmul w w); pure1 (MkBang loss # ()))
      units (simpleConfig 5) ()
    let v = tensorItem w
    liftIO1 (check ("fitSupervisedMixed trains (w 2.0 -> " ++ show v ++ ")") (v < 1.0)))

-- Full-pass epochs: a stream advertising `epochLen = Just k` makes ONE
-- fit epoch do k steps (a full dataset pass), not 1. loss=w², lr 0.1 →
-- w *= 0.8 per step; one epoch over epochLen=3 → w = 2*0.8³ = 1.024,
-- whereas the old one-step-per-epoch behaviour would give 2*0.8 = 1.6.
fitFullPassMultiStep : IO Bool
fitFullPassMultiStep = do
  w <- mkW "fit_fp_w" 2.0
  opt <- sgd {ex=TestExecutor} 0.1 defaultOpts
  let s3 = the (DataStream ()) (MkDataStream (pure ()) (Just 3))
  Control.Linear.LIO.run (do
    (MkBang _ # ()) <- fitSupervised {ex=TestExecutor} opt
      (\(), () => do loss <- liftIO1 (tmul w w); pure1 (MkBang loss # ()))
      s3 (simpleConfig 1) ()
    let v = tensorItem w
    -- Tolerance 1e-6, not 1e-9: mlx's F64 accumulation lands at 1.02399993
    -- (~7e-8 off) where tape/torch hit 1.024 exactly. The assertion only
    -- needs to separate correct multi-step (1.024) from one-step (1.6).
    liftIO1 (check ("fit full-pass runs epochLen steps/epoch (w 2.0 -> " ++ show v ++ " ~ 1.024)")
          (abs (v - 1.024) < 1.0e-6)))

-- fitCustom: the optimizer-free driver for non-gradient training
-- (tabular RL). A pure Double "model" halved each epoch by the
-- EpochStep — no registered params, no optimizer, no trainStep.
-- Proves the model threads through epochs and the loop runs the
-- requested count: 2.0 * 0.5^5 = 0.0625 over 5 epochs.
fitCustomThreadsModel : IO Bool
fitCustomThreadsModel =
  Control.Linear.LIO.run (do
    (MkBang (epochs, _) # MkLBox vFin) <- fitCustom {ex=TestExecutor} {m=LBox} {batch=()}
      (\(MkLBox v), () => pure1 (MkBang (v * 0.5) # MkLBox (v * 0.5))) units (simpleConfig 5) (MkLBox 2.0)
    liftIO1 (check ("fitCustom threads pure model (2.0 -> " ++ show vFin ++ " ~ 0.0625, "
           ++ show epochs ++ " epochs)")
          (abs (vFin - 0.0625) < 1.0e-12 && epochs == 5)))

-- P1 regression: in-memory training data across epochs. `fromVect` of
-- device tensors caches one fixed handle per index; the backend frees
-- non-grad input tensors after each optimizer step (they're assumed
-- per-epoch-fresh, freed by the tape arena reset), so epoch 2's pull of a
-- `fromVect`-cached handle reads freed memory ("invalid memory reference",
-- reproduced). `fromVectIO` honours `item`'s fresh-per-access contract:
-- it holds host values and materialises a NEW tensor per access, so each
-- epoch gets live handles. The loss is data-only (no params), so the same
-- host data must give the SAME loss in epoch 2 as epoch 1.
fromVectIOMultiEpochSafe : IO Bool
fromVectIOMultiEpochSafe = do
  let rows : Vect 2 (Vect 2 Double, Vect 2 Double)
          := [([1.0, 2.0], [1.0, 0.0]), ([3.0, 4.0], [0.0, 1.0])]
  let ds : Dataset (Tensor [2] TestExecutor TestDType NoGrad, Tensor [2] TestExecutor TestDType NoGrad)
         := fromVectIO rows (\(xs, ys) => do
              x <- tensor {dims=[2]} (FromVect xs)
              y <- tensor {dims=[2]} (FromVect ys)
              pure (x, y))
  s <- stream NoShuffle ds
  let bs = batched {b=2} {i=2} {o=2} s
  opt <- sgd {ex=TestExecutor} 0.1 defaultOpts
  -- epoch 1: pull batch, record the data-only loss, then a train step whose
  -- backward + arena reset frees this epoch's input tensors.
  (xb1, yb1) <- bs.next
  l1 <- the (IO (Tensor [] TestExecutor TestDType WithGrad))
            (tnllLossMean {b=2} {n=2} (retypeGrad xb1) (retypeGrad yb1))
  let v1 = tensorItem l1
  _ <- trainStep opt l1
  -- epoch 2: the stream wraps and re-materialises FRESH tensors from the
  -- retained host rows (a `fromVect` cache would be freed memory here).
  (xb2, yb2) <- bs.next
  l2 <- the (IO (Tensor [] TestExecutor TestDType WithGrad))
            (tnllLossMean {b=2} {n=2} (retypeGrad xb2) (retypeGrad yb2))
  let v2 = tensorItem l2
  check ("fromVectIO survives 2 epochs (l1=" ++ show v1 ++ " l2=" ++ show v2 ++ ")")
        (v1 == v2)

export
tests : List (IO Bool)
tests = [ fitSupervisedConverges
        , recurrentFoldConverges, fitMixedTrains, fitFullPassMultiStep
        , fitCustomThreadsModel, fromVectIOMultiEpochSafe ]
