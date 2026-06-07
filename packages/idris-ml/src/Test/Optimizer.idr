module Test.Optimizer

import Data.List
import Data.Vect

import Test.Harness
import Executor
import Tensor
import Optimizer
import Test.Config


-- Trajectory equivalence: the new `sgd` / `rmsprop` constructors wrap
-- the SAME C prims as `nativeSgd` / `nativeRmsprop`, so driving the
-- same scalar param through the same loss must produce bitwise-
-- identical weight trajectories (exact ==, no tolerance).
--
-- Each run uses its own param + its own freshly-constructed optimizer;
-- the optimizers step the whole registry, but a param outside the
-- active loss gets zero grad, so cross-talk can't perturb the
-- trajectory under comparison (rmsprop/momentum state lives in the
-- optimizer handle, fresh per construction).

mkW : String -> Double -> IO (Tensor [] TestExecutor TestDType WithGrad)
mkW name v = do
  wptr <- ioRerun (\_ => primCreateScalar {ex=TestExecutor} v 1)
  _ <- ioRerun (\_ => primParamRegister {ex=TestExecutor} name wptr)
  pure (MkTensor wptr (Just name))

-- One step of loss = w * w; returns w's post-step value.
stepQuadratic : NativeOptimizer TestExecutor ->
                Tensor [] TestExecutor TestDType WithGrad -> IO Double
stepQuadratic opt w = do
  loss <- tmul w w
  _ <- nativeTrainStep opt loss
  pure (tensorItem w)

trajectory : NativeOptimizer TestExecutor ->
             Tensor [] TestExecutor TestDType WithGrad -> Nat -> IO (List Double)
trajectory opt w Z = pure []
trajectory opt w (S k) = do
  v <- stepQuadratic opt w
  rest <- trajectory opt w k
  pure (v :: rest)

sgdMatchesNative : IO Bool
sgdMatchesNative = do
  wOld <- mkW "opt_sgd_old" 1.0
  let optOld = nativeSgd {ex=TestExecutor} 0.1
  trajOld <- trajectory optOld wOld 3
  wNew <- mkW "opt_sgd_new" 1.0
  optNew <- sgd {ex=TestExecutor} 0.1 defaultOpts
  trajNew <- trajectory optNew wNew 3
  check ("sgd == nativeSgd over 3 steps (" ++ show trajNew
         ++ " vs " ++ show trajOld ++ ")") (trajNew == trajOld)

rmspropMatchesNative : IO Bool
rmspropMatchesNative = do
  wOld <- mkW "opt_rms_old" 1.0
  let optOld = nativeRmsprop {ex=TestExecutor} 0.01 0.9 1.0e-7 0.75 0.5
  trajOld <- trajectory optOld wOld 3
  wNew <- mkW "opt_rms_new" 1.0
  optNew <- rmsprop {ex=TestExecutor} 0.01 {alpha=0.9} {momentum=0.5}
              ({ eps := 1.0e-7, clip := ValueClip 0.75 } defaultOpts)
  trajNew <- trajectory optNew wNew 3
  check ("rmsprop == nativeRmsprop over 3 steps (" ++ show trajNew
         ++ " vs " ++ show trajOld ++ ")") (trajNew == trajOld)

-- PyTorch-default knobs: rmsprop's implicit alpha/momentum must equal
-- torch.optim.RMSprop's (alpha=0.99, momentum=0).
rmspropDefaultsMatchPyTorch : IO Bool
rmspropDefaultsMatchPyTorch = do
  wOld <- mkW "opt_rmsd_old" 1.0
  let optOld = nativeRmsprop {ex=TestExecutor} 0.01 0.99 1.0e-8 0.5 0.0
  trajOld <- trajectory optOld wOld 3
  wNew <- mkW "opt_rmsd_new" 1.0
  optNew <- rmsprop {ex=TestExecutor} 0.01 ({ clip := ValueClip 0.5 } defaultOpts)
  trajNew <- trajectory optNew wNew 3
  check ("rmsprop implicit defaults = PyTorch's (" ++ show trajNew
         ++ " vs " ++ show trajOld ++ ")") (trajNew == trajOld)

export
tests : List (IO Bool)
tests = [sgdMatchesNative, rmspropMatchesNative, rmspropDefaultsMatchPyTorch]
