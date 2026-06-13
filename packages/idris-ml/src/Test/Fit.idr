module Test.Fit

import Data.Vect

import Test.Harness
import Executor
import Tensor
import Optimizer
import GradScaler
import Train
import Fit
import DataStream
import Test.Config


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

export
tests : List (IO Bool)
tests = [ fitSupervisedConverges, fitEqualsLegacy
        , recurrentFoldConverges, fitMixedTrains ]
