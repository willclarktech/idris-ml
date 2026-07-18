module Test.GradScaler

import Data.Vect

import Ml.Executor
import Ml.GradScaler
import Ml.Optimizer
import Ml.Tensor
import Test.Config
import Test.Harness

-- A3 of #410: GradScaler state-machine. Verifies that the growth /
-- backoff policy advances correctly across successful steps. We
-- don't directly simulate overflow (the NaN-return path requires a
-- backward that produces non-finite grads on a real tensor chain —
-- hard to reproduce cleanly on tape's F64 lingua franca), but we do
-- verify (a) initScale, (b) scale grows after one successful step
-- with growthInterval=1, and (c) scale stays put when growthInterval
-- is larger than the success count.

-- Build a tiny scalar autograd chain: param w, forward = w * x,
-- pre-scaled by the scaler. trainStepScaled drives backward + the
-- state-machine advance.
runOneStep : NativeOptimizer TestExecutor -> GradScaler TestExecutor TestDType ->
             IO Double
runOneStep opt gs = do
  let wptr = primCreateScalar {ex=TestExecutor} 0.5 1
  let wT   = the (Tensor [] TestExecutor TestDType WithGrad)
              (MkTensor wptr (Just "w_state_machine_test"))
  _ <- pure $ primParamRegister {ex=TestExecutor} "w_state_machine_test" wptr
  let xptr = primCreateScalar {ex=TestExecutor} 3.0 0
  let xT   = the (Tensor [] TestExecutor TestDType WithGrad)
              (MkTensor xptr Nothing)
  loss <- pure $ MkTensor (primMul {ex=TestExecutor} wT.tensorPtr xT.tensorPtr) Nothing
  scaled <- applyScale gs (the (Tensor [] TestExecutor TestDType WithGrad) loss)
  trainStepScaled opt gs scaled

-- (a) Initial scale matches what `gradScaler` was constructed with.
initialScaleIsExact : IO Bool
initialScaleIsExact = do
  gs <- gradScaler {ex=TestExecutor} {dt=TestDType} 100.0 2.0 0.5 5
  s <- currentScale gs
  check ("initial scale is 100.0 (got " ++ show s ++ ")") (s == 100.0)

-- (b) After one successful step with growthInterval=1, scale has
-- grown by growthFactor=2 to 200.
scaleGrowsAfterOneSuccessfulStep : IO Bool
scaleGrowsAfterOneSuccessfulStep = do
  gs <- gradScaler {ex=TestExecutor} {dt=TestDType} 100.0 2.0 0.5 1
  opt <- sgd {ex=TestExecutor} 0.001 defaultOpts
  _ <- runOneStep opt gs
  s <- currentScale gs
  check ("scale grew 100 → 200 after one step (got " ++ show s ++ ")") (s == 200.0)

-- (c) With growthInterval=5 and only one successful step, scale
-- should stay at the initial value (counter incremented but not
-- yet triggering growth).
scaleStaysBeforeIntervalReached : IO Bool
scaleStaysBeforeIntervalReached = do
  gs <- gradScaler {ex=TestExecutor} {dt=TestDType} 100.0 2.0 0.5 5
  opt <- sgd {ex=TestExecutor} 0.001 defaultOpts
  _ <- runOneStep opt gs
  s <- currentScale gs
  check ("scale stays at 100 before growth interval (got " ++ show s ++ ")") (s == 100.0)

export
tests : List (IO Bool)
tests =
  [ initialScaleIsExact
  , scaleGrowsAfterOneSuccessfulStep
  , scaleStaysBeforeIntervalReached
  ]
