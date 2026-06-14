module Test.Nn.LinearMixed

import Data.List
import Data.Vect

import Test.Harness
import Executor
import Tensor
import Optimizer
import Nn.Init
import Nn.Module
import Nn.LinearMixed
import Test.Config

-- The mixed-precision (master-weights) Linear on the `Nn` surface. The
-- distinguishing behaviour — a LOSSY paramDt → computeDt cast that still
-- propagates an F32 grad into the master — is covered at the C level
-- (`cast_grad_propagation`), and exercising it here would hardcode a
-- second concrete dtype that isn't `Compatible` on every backend (F64 is
-- absent on mlx-gpu / torch-mps). So these tests pin paramDt = computeDt =
-- TestDType (the cast is a dtype-level no-op) and verify the layer + the
-- `ModuleMixed`/`ParamsMixed` machinery compose, register, and keep the
-- autograd tape intact THROUGH the cast — the legacy `Test.MixedLayerLike`
-- precedent, minus the existential apparatus.

read4 : Tensor [2, 2] TestExecutor TestDType g -> List Double
read4 t = [ primItem2d {ex=TestExecutor} t.tensorPtr i j
          | (i, j) <- the (List (Int, Int)) [(0,0),(0,1),(1,0),(1,1)] ]

-- Deterministic forward: W[2,3] all 0.5, b[2] all 1.0, X[2,3] all 2.0.
-- out[k,j] = sum_i (0.5 * 2.0) + 1.0 = 3*1.0 + 1.0 = 4.0 everywhere —
-- same oracle as Test.Nn.Linear, now through the cast-cast-matmul path.
forwardMixedComputes : IO Bool
forwardMixedComputes = do
  w <- param {ex=TestExecutor} {dt=TestDType} {dims=[2, 3]} "lmt.w" (Const 0.5)
  b <- param {ex=TestExecutor} {dt=TestDType} {dims=[2]}    "lmt.b" (Const 1.0)
  let lyr = the (LinearMixed 3 2 TestExecutor TestDType TestDType WithGrad) (MkLinearMixed w b)
  x   <- tensor {ex=TestExecutor} {dt=TestDType} {dims=[2, 3]} (Const 2.0)
  out <- forwardMixed {b=2} lyr (retypeGrad x)
  check ("LinearMixed.forwardMixed computes cast(x·Wᵀ+b) (got " ++ show (read4 out) ++ ")")
        (read4 out == [4.0, 4.0, 4.0, 4.0])

-- ParamsMixed exposes both master leaves under their registered names.
paramsMixedExposed : IO Bool
paramsMixedExposed = do
  w <- param {ex=TestExecutor} {dt=TestDType} {dims=[2, 3]} "lmp.w" (Const 0.5)
  b <- param {ex=TestExecutor} {dt=TestDType} {dims=[2]}    "lmp.b" (Const 1.0)
  let lyr = the (LinearMixed 3 2 TestExecutor TestDType TestDType WithGrad) (MkLinearMixed w b)
  let names = mapMaybe paramName (paramsMixed lyr)
  check ("ParamsMixed (LinearMixed) exposes weight + bias (got " ++ show names ++ ")")
        (names == ["lmp.w", "lmp.b"])

-- The Init smart constructor registers PyTorch-style dotted master names.
smartCtorNames : IO Bool
smartCtorNames = do
  _ <- runInit $ scoped "mlp"
         (linearMixed {ex=TestExecutor} {paramDt=TestDType} {computeDt=TestDType} {i=3} {o=2})
  cnt <- getParamCount {ex=TestExecutor}
  names <- traverse (\i => getParamName {ex=TestExecutor} i) [0 .. cnt - 1]
  check "linearMixed registers mlp.linear_0.weight + .bias"
        (("mlp.linear_0.weight" `elem` names) && ("mlp.linear_0.bias" `elem` names))

-- A training step changes the master weight: backward writes a paramDt
-- gradient back THROUGH the cast into the registered master, and the
-- optimizer steps it. This is the master-weights guarantee (the cast
-- doesn't sever the tape). Loss = sum((x·Wᵀ+b))² with non-zero W, so the
-- gradient is non-zero and the step must move w[0,0].
masterGradFlows : IO Bool
masterGradFlows = do
  w <- param {ex=TestExecutor} {dt=TestDType} {dims=[1, 2]} "lmg.w" (Const 0.5)
  b <- param {ex=TestExecutor} {dt=TestDType} {dims=[1]}    "lmg.b" (Const 0.0)
  let lyr = the (LinearMixed 2 1 TestExecutor TestDType TestDType WithGrad) (MkLinearMixed w b)
  let before = primItem2d {ex=TestExecutor} w.tensorPtr 0 0
  x0   <- tensor {ex=TestExecutor} {dt=TestDType} {dims=[2, 2]} (Const 1.0)
  tgt0 <- tensor {ex=TestExecutor} {dt=TestDType} {dims=[2, 1]} (Const 0.0)
  let x   = the (Tensor [2, 2] TestExecutor TestDType WithGrad) (retypeGrad x0)
  let tgt = the (Tensor [2, 1] TestExecutor TestDType WithGrad) (retypeGrad tgt0)
  out <- forwardMixed {b=2} lyr x
  loss <- the (IO (Tensor [] TestExecutor TestDType WithGrad)) $ ioRerun (\_ =>
    let diff = primSub {ex=TestExecutor} out.tensorPtr tgt.tensorPtr
        sq   = primMul {ex=TestExecutor} diff diff
    in MkTensor (primSum {ex=TestExecutor} sq) Nothing)
  opt <- pure (nativeSgd {ex=TestExecutor} 0.1)
  _ <- nativeTrainStep opt loss
  let after = primItem2d {ex=TestExecutor} w.tensorPtr 0 0
  check ("masterGradFlows: w[0,0] moves after a step (" ++ show before ++ " -> " ++ show after ++ ")")
        (before /= after)

export
tests : List (IO Bool)
tests = [forwardMixedComputes, paramsMixedExposed, smartCtorNames, masterGradFlows]
