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

-- One step of loss = c * w * w; the varying scale makes the grad
-- magnitude jump between steps, so the beta1/beta2 moment EMAs lag
-- differently — on a smooth quadratic Adam's bias-corrected update is
-- ~lr*sign(grad) and a beta swap would be invisible.
stepScaled : NativeOptimizer TestExecutor ->
             Tensor [] TestExecutor TestDType WithGrad -> Double -> IO Double
stepScaled opt w c = do
  l <- tmul w w
  loss <- tmulScalar l c
  _ <- nativeTrainStep opt loss
  pure (tensorItem w)

scaledTrajectory : NativeOptimizer TestExecutor ->
                   Tensor [] TestExecutor TestDType WithGrad -> IO (List Double)
scaledTrajectory opt w = traverse (stepScaled opt w) [1.0, 5.0, 0.5]

adamMatchesNative : IO Bool
adamMatchesNative = do
  wOld <- mkW "opt_adam_old" 1.0
  let optOld = nativeAdamGlobalClip {ex=TestExecutor} 0.01 0.8 0.95 1.0e-6 100.0
  trajOld <- scaledTrajectory optOld wOld
  wNew <- mkW "opt_adam_new" 1.0
  optNew <- adam {ex=TestExecutor} 0.01
              ({ beta1 := 0.8, beta2 := 0.95, eps := 1.0e-6,
                 clip := NormClip 100.0 } defaultOpts)
  trajNew <- scaledTrajectory optNew wNew
  check ("adam == nativeAdamGlobalClip over 3 scaled steps (" ++ show trajNew
         ++ " vs " ++ show trajOld ++ ")") (trajNew == trajOld)

adamWMatchesNative : IO Bool
adamWMatchesNative = do
  wOld <- mkW "opt_adamw_old" 1.0
  let optOld = nativeAdamW {ex=TestExecutor} 0.01 0.9 0.999 1.0e-8 0.1 1.0
  trajOld <- trajectory optOld wOld 3
  wNew <- mkW "opt_adamw_new" 1.0
  optNew <- adamW {ex=TestExecutor} 0.01 0.1 ({ clip := NormClip 1.0 } defaultOpts)
  trajNew <- trajectory optNew wNew 3
  check ("adamW == nativeAdamW over 3 steps (" ++ show trajNew
         ++ " vs " ++ show trajOld ++ ")") (trajNew == trajOld)

-- One step of loss = w*w + b*b; returns both post-step values.
stepPair : NativeOptimizer TestExecutor ->
           Tensor [] TestExecutor TestDType WithGrad ->
           Tensor [] TestExecutor TestDType WithGrad -> IO (Double, Double)
stepPair opt w b = do
  l1 <- tmul w w
  l2 <- tmul b b
  loss <- tadd l1 l2
  _ <- nativeTrainStep opt loss
  pure (tensorItem w, tensorItem b)

-- Scope must route to the Group prim: the scoped optimizer steps only
-- params under the prefix (the bystander keeps its initial value even
-- though it carries a nonzero grad), and the scoped param's trajectory
-- is bitwise-equal to nativeAdamGroup's.
adamScopeRouting : IO Bool
adamScopeRouting = do
  wOld <- mkW "opt_og_w" 1.0
  bOld <- mkW "opt_ob_w" 1.0
  let optOld = nativeAdamGroup {ex=TestExecutor} "opt_og_" 0.01 0.9 0.999 1.0e-8 1.0
  (wo1, bo1) <- stepPair optOld wOld bOld
  (wo2, _)   <- stepPair optOld wOld bOld
  wNew <- mkW "opt_ng_w" 1.0
  bNew <- mkW "opt_nb_w" 1.0
  optNew <- adam {ex=TestExecutor} {scope="opt_ng_"} 0.01
              ({ clip := NormClip 1.0 } defaultOpts)
  (wn1, bn1) <- stepPair optNew wNew bNew
  (wn2, bn2) <- stepPair optNew wNew bNew
  check ("adam scope: matches nativeAdamGroup ([" ++ show wn1 ++ ", " ++ show wn2
         ++ "] vs [" ++ show wo1 ++ ", " ++ show wo2
         ++ "]), bystander frozen (" ++ show bn2 ++ ")")
        (wn1 == wo1 && wn2 == wo2 && bo1 == 1.0 && bn1 == 1.0 && bn2 == 1.0)

export
tests : List (IO Bool)
tests = [ sgdMatchesNative, rmspropMatchesNative, rmspropDefaultsMatchPyTorch
        , adamMatchesNative, adamWMatchesNative, adamScopeRouting ]
