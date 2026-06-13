module Test.Optimizer

import Data.List
import Data.Vect

import Test.Harness
import Executor
import Optimizer
import Schedule
import Tensor
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

-- withSchedule + tick: tick pushes schedule(epoch) into the C
-- optimizer's base LR. Schedule values below are exact binary
-- fractions so the assertions can stay exact ==.

scheduleFreezesAtZero : IO Bool
scheduleFreezesAtZero = do
  w <- mkW "opt_sched0_w" 1.0
  opt0 <- sgd {ex=TestExecutor} 0.1 defaultOpts
  let opt = withSchedule (constant 0.0) opt0
  tick opt 0
  v <- stepQuadratic opt w
  check ("withSchedule (constant 0) + tick freezes the step (w = "
         ++ show v ++ ")") (v == 1.0)

tickAppliesScheduleEpoch : IO Bool
tickAppliesScheduleEpoch = do
  w <- mkW "opt_sched1_w" 1.0
  opt0 <- sgd {ex=TestExecutor} 0.25 defaultOpts
  let opt = withSchedule (exponentialLR 0.25 0.5) opt0
  tick opt 1
  v <- stepQuadratic opt w
  check ("tick 1 applies lr 0.25 * 0.5^1 = 0.125 (w = " ++ show v
         ++ ", expect 0.75)") (v == 0.75)

tickWithoutScheduleIsNoOp : IO Bool
tickWithoutScheduleIsNoOp = do
  wRef <- mkW "opt_noop_ref" 1.0
  optRef <- sgd {ex=TestExecutor} 0.1 defaultOpts
  trajRef <- trajectory optRef wRef 2
  w <- mkW "opt_noop_w" 1.0
  opt <- sgd {ex=TestExecutor} 0.1 defaultOpts
  tick opt 9
  traj <- trajectory opt w 2
  check ("tick without a schedule is a no-op (" ++ show traj
         ++ " vs " ++ show trajRef ++ ")") (traj == trajRef)

-- groups: per-prefix LR overrides applied by walking the param
-- registry at construction. One step at base lr 0.25 on three params:
-- the frozen group stays put, the scaled group steps at lr 0.125,
-- the bystander steps at base. All values exact binary fractions.
--
-- torch: `optimizer_set_param_lr` is a documented no-op there
-- (libtorch needs per-param param-groups — TODO row filed
-- 2026-06-12), so groups/freezeByPrefix silently don't apply; this
-- test locks that documented behaviour (everything steps at base LR)
-- so the future implementation flips it deliberately.
groupsOverrideByPrefix : IO Bool
groupsOverrideByPrefix = do
  wf <- mkW "opt_g4f_w" 1.0
  ws <- mkW "opt_g4s_w" 1.0
  wn <- mkW "opt_g4n_w" 1.0
  opt <- sgd {ex=TestExecutor} 0.25
           ({ groups := [("opt_g4f_", 0.0), ("opt_g4s_", 0.125)] } defaultOpts)
  l1 <- tmul wf wf
  l2 <- tmul ws ws
  l3 <- tmul wn wn
  l12 <- tadd l1 l2
  loss <- tadd l12 l3
  _ <- nativeTrainStep opt loss
  let (vf, vs, vn) = (tensorItem wf, tensorItem ws, tensorItem wn)
  if TestPrimaryBackend == "torch"
    then check ("groups on torch: documented per-param-LR no-op (frozen " ++ show vf
                ++ ", scaled " ++ show vs ++ ", base " ++ show vn ++ ")")
               (vf == 0.5 && vs == 0.5 && vn == 0.5)
    else check ("groups freeze/scale by prefix (frozen " ++ show vf
                ++ ", scaled " ++ show vs ++ ", base " ++ show vn ++ ")")
               (vf == 1.0 && vs == 0.75 && vn == 0.5)

export
tests : List (IO Bool)
tests = [ sgdMatchesNative, rmspropMatchesNative, rmspropDefaultsMatchPyTorch
        , adamMatchesNative, adamWMatchesNative, adamScopeRouting
        , scheduleFreezesAtZero, tickAppliesScheduleEpoch
        , tickWithoutScheduleIsNoOp, groupsOverrideByPrefix ]
