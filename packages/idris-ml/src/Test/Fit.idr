module Test.Fit

import Data.Vect

import DataStream
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

-- fitSupervised: loss = w*w from w=2.0 at lr=0.1 → w *= 0.8 each epoch;
-- after 5 epochs w = 2*0.8^5 = 0.655 < 1.0.
fitSupervisedConverges : IO Bool
fitSupervisedConverges = do
  w <- mkW "fit_sup_w" 2.0
  opt <- sgd {ex=TestExecutor} 0.1 defaultOpts
  _ <- fitSupervised {ex=TestExecutor} opt (\_, () => tmul w w) units (simpleConfig 5) ()
  let v = tensorItem w
  check ("fitSupervised converges (w 2.0 -> " ++ show v ++ ")") (v < 1.0)

-- Equivalence: fitSupervised drives the SAME engine as runTrainingIO on
-- the same step, so identical final loss (bitwise). Grad-gating keeps
-- the two registered params from cross-contaminating.
fitEqualsLegacy : IO Bool
fitEqualsLegacy = do
  wL <- mkW "fit_eqL_w" 1.0
  optL <- sgd {ex=TestExecutor} 0.1 defaultOpts
  (_, _, lossL) <- runTrainingIO {ex=TestExecutor}
    (\m, () => do loss <- tmul wL wL; d <- nativeTrainStep optL loss; pure (m, d))
    (pure ()) (simpleConfig 5) ()
  wF <- mkW "fit_eqF_w" 1.0
  optF <- sgd {ex=TestExecutor} 0.1 defaultOpts
  (_, _, lossF) <- fitSupervised {ex=TestExecutor} optF (\_, () => tmul wF wF)
    units (simpleConfig 5) ()
  check ("fit == runTrainingIO final loss (" ++ show lossF ++ " vs " ++ show lossL ++ ")")
        (abs (lossL - lossF) < 1.0e-12)

-- Recurrent/two-phase are Step folds, not driver variants: a Step that
-- folds two "timesteps" into one loss. loss = 2w², grad = 4w, lr 0.05 →
-- w *= 0.8 each epoch; after 5, w = 0.8^5 = 0.328 < 1.0.
recStep : Tensor [] TestExecutor TestDType WithGrad -> Optimizer TestExecutor -> EpochStep () ()
recStep w opt m () = do
  l1 <- tmul w w
  l2 <- tmul w w
  summed <- tadd l1 l2
  d <- nativeTrainStep opt summed
  pure (m, d)

recurrentFoldConverges : IO Bool
recurrentFoldConverges = do
  w <- mkW "fit_rec_w" 1.0
  opt <- sgd {ex=TestExecutor} 0.05 defaultOpts
  _ <- fit {ex=TestExecutor} (recStep w opt) opt units (simpleConfig 5) ()
  let v = tensorItem w
  check ("recurrent Step-fold converges (w=" ++ show v ++ ")") (v < 1.0)

-- Mixed precision via fitSupervisedMixed. On tape F64 the scaler never
-- overflows (scale 1.0), so it trains identically to single precision —
-- this exercises the scaled path + nanHalts=False wiring end to end.
fitMixedTrains : IO Bool
fitMixedTrains = do
  w <- mkW "fit_mix_w" 2.0
  opt <- sgd {ex=TestExecutor} 0.1 defaultOpts
  gs <- gradScaler {ex=TestExecutor} {dt=TestDType} 1.0 2.0 0.5 1000
  _ <- fitSupervisedMixed {ex=TestExecutor} opt gs (\_, () => tmul w w)
    units (simpleConfig 5) ()
  let v = tensorItem w
  check ("fitSupervisedMixed trains (w 2.0 -> " ++ show v ++ ")") (v < 1.0)

-- Full-pass epochs: a stream advertising `epochLen = Just k` makes ONE
-- fit epoch do k steps (a full dataset pass), not 1. loss=w², lr 0.1 →
-- w *= 0.8 per step; one epoch over epochLen=3 → w = 2*0.8³ = 1.024,
-- whereas the old one-step-per-epoch behaviour would give 2*0.8 = 1.6.
fitFullPassMultiStep : IO Bool
fitFullPassMultiStep = do
  w <- mkW "fit_fp_w" 2.0
  opt <- sgd {ex=TestExecutor} 0.1 defaultOpts
  let s3 = the (DataStream ()) (MkDataStream (pure ()) (Just 3))
  _ <- fitSupervised {ex=TestExecutor} opt (\_, () => tmul w w) s3 (simpleConfig 1) ()
  let v = tensorItem w
  -- Tolerance 1e-6, not 1e-9: mlx's F64 accumulation lands at 1.02399993
  -- (~7e-8 off) where tape/torch hit 1.024 exactly. The assertion only
  -- needs to separate correct multi-step (1.024) from one-step (1.6).
  check ("fit full-pass runs epochLen steps/epoch (w 2.0 -> " ++ show v ++ " ~ 1.024)")
        (abs (v - 1.024) < 1.0e-6)

-- fitCustom: the optimizer-free driver for non-gradient training
-- (tabular RL). A pure Double "model" halved each epoch by the
-- EpochStep — no registered params, no optimizer, no nativeTrainStep.
-- Proves the model threads through epochs and the loop runs the
-- requested count: 2.0 * 0.5^5 = 0.0625 over 5 epochs.
fitCustomThreadsModel : IO Bool
fitCustomThreadsModel = do
  (vFin, epochs, _) <- fitCustom {ex=TestExecutor} {m=Double} {batch=()}
    (\v, () => pure (v * 0.5, v)) units (simpleConfig 5) 2.0
  check ("fitCustom threads pure model (2.0 -> " ++ show vFin ++ " ~ 0.0625, "
         ++ show epochs ++ " epochs)")
        (abs (vFin - 0.0625) < 1.0e-12 && epochs == 5)

export
tests : List (IO Bool)
tests = [ fitSupervisedConverges, fitEqualsLegacy
        , recurrentFoldConverges, fitMixedTrains, fitFullPassMultiStep
        , fitCustomThreadsModel ]
