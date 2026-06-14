module Test.Nn.BatchNorm

import Data.List
import Data.Vect

import Test.Harness
import Executor
import Tensor
import Checkpoint
import Nn.Init
import Nn.Module
import Nn.BatchNorm
import Test.Config

-- Eval mode with running mean=0, var=1, gamma=1, beta=0:
--   out = (x - 0)/sqrt(1 + 1e-5) * 1 + 0 ≈ x.
-- channels=2, spatialDim=1 → i = o = 2; x = [3, 4].
evalUsesRunningStats : IO Bool
evalUsesRunningStats = do
  g <- param  {ex=TestExecutor} {dt=TestDType} {dims=[2]} "bn.g" (Const 1.0)
  b <- param  {ex=TestExecutor} {dt=TestDType} {dims=[2]} "bn.b" (Const 0.0)
  m <- tensor {ex=TestExecutor} {dt=TestDType} {dims=[2]} (Const 0.0)
  v <- tensor {ex=TestExecutor} {dt=TestDType} {dims=[2]} (Const 1.0)
  let bn = MkBatchNorm {channels=2} {spatialDim=1} g b m v False 0.1 1.0e-5
  x   <- tensor {ex=TestExecutor} {dt=TestDType} {dims=[2]} (FromVect [3.0, 4.0])
  out <- batchNormForward bn x
  let v0 = primItem1d {ex=TestExecutor} out.tensorPtr 0
  let v1 = primItem1d {ex=TestExecutor} out.tensorPtr 1
  check ("eval BatchNorm ≈ identity with default stats (got [" ++ show v0 ++ ", " ++ show v1 ++ "])")
        (abs (v0 - 3.0) < 1.0e-3 && abs (v1 - 4.0) < 1.0e-3)

paramsAreGammaBeta : IO Bool
paramsAreGammaBeta = do
  g <- param  {ex=TestExecutor} {dt=TestDType} {dims=[2]} "bp.g" (Const 1.0)
  b <- param  {ex=TestExecutor} {dt=TestDType} {dims=[2]} "bp.b" (Const 0.0)
  m <- tensor {ex=TestExecutor} {dt=TestDType} {dims=[2]} (Const 0.0)
  v <- tensor {ex=TestExecutor} {dt=TestDType} {dims=[2]} (Const 1.0)
  let bn = MkBatchNorm {channels=2} {spatialDim=1} g b m v True 0.1 1.0e-5
  check ("Params (BatchNorm) = gamma,beta only (got " ++ show (mapMaybe paramName (params bn)) ++ ")")
        (mapMaybe paramName (params bn) == ["bp.g", "bp.b"])

smartCtorNames : IO Bool
smartCtorNames = do
  _ <- runInit $ scoped "net" (batchNorm {ex=TestExecutor} {dt=TestDType} {channels=4} {spatialDim=1})
  cnt <- getParamCount {ex=TestExecutor}
  names <- traverse (\i => getParamName {ex=TestExecutor} i) [0 .. cnt - 1]
  check "batchNorm registers net.batch_norm_0.weight + .bias"
        (("net.batch_norm_0.weight" `elem` names) && ("net.batch_norm_0.bias" `elem` names))

-- Running mean/var are non-learnable BUFFERS: the optimizer must not step
-- them, but save/load MUST persist them (they hold trained statistics).
-- This roundtrip drives the running stats off their 0/1 init in training
-- mode, saves, then loads into a fresh model and checks the stats came back.
bufferRoundtrip : IO Bool
bufferRoundtrip = do
  let path = "/tmp/idris-ml-bn-buffer-roundtrip.safetensors"
  -- Trained model: run training-mode forwards so running stats diverge.
  bn <- runInit $ scoped "bnrt" (batchNorm {ex=TestExecutor} {dt=TestDType} {channels=2} {spatialDim=1})
  x  <- tensor {ex=TestExecutor} {dt=TestDType} {dims=[2]} (FromVect [3.0, 4.0])
  for_ [the Nat 1 .. 20] $ \_ => batchNormForward bn x
  let (meanP, varP) = runningStatPtrs bn
  let tMean = primItem1d {ex=TestExecutor} meanP 0
      tVar  = primItem1d {ex=TestExecutor} varP 0
  _  <- saveAll {ex=TestExecutor} path
  -- Fresh model re-registers the same names with reset 0/1 buffers.
  fresh <- runInit $ scoped "bnrt" (batchNorm {ex=TestExecutor} {dt=TestDType} {channels=2} {spatialDim=1})
  _  <- loadModel {ex=TestExecutor} path
  let (fmeanP, fvarP) = runningStatPtrs fresh
  let lMean = primItem1d {ex=TestExecutor} fmeanP 0
      lVar  = primItem1d {ex=TestExecutor} fvarP 0
  r0 <- check ("training moved running mean off 0 (got " ++ show tMean ++ ")") (abs tMean > 0.5)
  r1 <- check ("training moved running var off 1 (got " ++ show tVar ++ ")") (abs (tVar - 1.0) > 0.1)
  r2 <- checkClose ("running mean restored after load (trained " ++ show tMean ++ ")") tMean lMean 1.0e-6
  r3 <- checkClose ("running var restored after load (trained " ++ show tVar ++ ")") tVar lVar 1.0e-6
  pure (r0 && r1 && r2 && r3)

export
tests : List (IO Bool)
tests = [evalUsesRunningStats, paramsAreGammaBeta, smartCtorNames, bufferRoundtrip]
