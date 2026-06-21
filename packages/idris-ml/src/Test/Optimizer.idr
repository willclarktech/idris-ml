module Test.Optimizer

import Data.List
import Data.String
import Data.Vect

import Executor
import Optimizer
import Schedule
import Tensor
import Test.Config
import Test.Harness
import Train.Freeze

-- Pinned trajectory tests: driving a scalar param through loss = w*w
-- with each typed constructor (`sgd`/`rmsprop`/`adam`/`adamW`) must
-- reproduce a fixed weight trajectory (via `trajApprox`, F32-safe — see
-- its note). The literals were captured from the tape backend
-- (deterministic at a fixed seed, single-thread BLAS) and guard against a
-- wrong knob being threaded into the underlying C prim.
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
  _ <- trainStep opt loss
  pure (tensorItem w)

trajectory : NativeOptimizer TestExecutor ->
             Tensor [] TestExecutor TestDType WithGrad -> Nat -> IO (List Double)
trajectory opt w Z     = pure []
trajectory opt w (S k) = do
  v <- stepQuadratic opt w
  rest <- trajectory opt w k
  pure (v :: rest)

-- Element-wise tolerance compare. The reference literals were captured on
-- the tape backend (F64); the mlx lane runs the same suite at F32, where
-- e.g. 0.512 comes back as 0.51199996 (~4e-8 off). Exact `==` is wrong for
-- a value crossing the real backend tensor — clear the F32 ULP floor with
-- the same 1e-5 tolerance Test.TrainEngine uses for its hand-loop check.
trajApprox : List Double -> List Double -> Bool
trajApprox xs ys =
  length xs == length ys && all (\(x, y) => abs (x - y) < 1.0e-5) (zip xs ys)

sgdStepsQuadratic : IO Bool
sgdStepsQuadratic = do
  w <- mkW "opt_sgd_w" 1.0
  opt <- sgd {ex=TestExecutor} 0.1 defaultOpts
  traj <- trajectory opt w 3
  check ("sgd steps quadratic (" ++ show traj ++ ")")
        (trajApprox traj [0.8, 0.64, 0.512])

rmspropStepsQuadratic : IO Bool
rmspropStepsQuadratic = do
  w <- mkW "opt_rms_w" 1.0
  opt <- rmsprop {ex=TestExecutor} 0.01 {alpha=0.9} {momentum=0.5}
           ({ eps := 1.0e-7, clip := ValueClip 0.75 } defaultOpts)
  traj <- trajectory opt w 3
  check ("rmsprop steps quadratic (" ++ show traj ++ ")")
        (trajApprox traj [0.9683772367316439, 0.9296242887279513, 0.8910383508862706])

-- PyTorch-default knobs: rmsprop's implicit alpha/momentum must equal
-- torch.optim.RMSprop's (alpha=0.99, momentum=0). Proven by trajectory
-- equality between the implicit-default construction and the explicit one
-- — no native reference needed.
rmspropDefaultsMatchPyTorch : IO Bool
rmspropDefaultsMatchPyTorch = do
  wImp <- mkW "opt_rmsd_imp" 1.0
  optImp <- rmsprop {ex=TestExecutor} 0.01 defaultOpts
  trajImp <- trajectory optImp wImp 3
  wExp <- mkW "opt_rmsd_exp" 1.0
  optExp <- rmsprop {ex=TestExecutor} 0.01 {alpha=0.99} {momentum=0.0} defaultOpts
  trajExp <- trajectory optExp wExp 3
  check ("rmsprop implicit defaults = PyTorch's alpha=0.99 momentum=0 ("
         ++ show trajImp ++ " vs " ++ show trajExp ++ ")") (trajImp == trajExp)

-- One step of loss = c * w * w; the varying scale makes the grad
-- magnitude jump between steps, so the beta1/beta2 moment EMAs lag
-- differently — on a smooth quadratic Adam's bias-corrected update is
-- ~lr*sign(grad) and a beta swap would be invisible.
stepScaled : NativeOptimizer TestExecutor ->
             Tensor [] TestExecutor TestDType WithGrad -> Double -> IO Double
stepScaled opt w c = do
  l <- tmul w w
  loss <- tmulScalar l c
  _ <- trainStep opt loss
  pure (tensorItem w)

scaledTrajectory : NativeOptimizer TestExecutor ->
                   Tensor [] TestExecutor TestDType WithGrad -> IO (List Double)
scaledTrajectory opt w = traverse (stepScaled opt w) [1.0, 5.0, 0.5]

adamStepsScaled : IO Bool
adamStepsScaled = do
  w <- mkW "opt_adam_w" 1.0
  opt <- adam {ex=TestExecutor} 0.01
           ({ beta1 := 0.8, beta2 := 0.95, eps := 1.0e-6,
              clip := NormClip 100.0 } defaultOpts)
  traj <- scaledTrajectory opt w
  check ("adam steps scaled quadratic (" ++ show traj ++ ")")
        (trajApprox traj [0.9900000049999975, 0.9811580691053646, 0.974027692742881])

adamWStepsQuadratic : IO Bool
adamWStepsQuadratic = do
  w <- mkW "opt_adamw_w" 1.0
  opt <- adamW {ex=TestExecutor} 0.01 0.1 ({ clip := NormClip 1.0 } defaultOpts)
  traj <- trajectory opt w 3
  -- Oracle = torch.optim.AdamW (lr=0.01, wd=0.1, clip-norm 1.0) on loss=w²,
  -- w₀=1.0, 3 steps. AdamW applies decoupled weight decay to the PRE-step
  -- weight, then the Adam update; all three backends match this order.
  check ("adamW steps quadratic (" ++ show traj ++ ")")
        (trajApprox traj [0.9890000001, 0.9780110002013497, 0.9670329893050524])

-- restrictTo scopes to an EXACT name set: the leak-free guarantee a string
-- prefix can't give. "rt_keep" is owned (steps at base LR 0.5 → 1 - 0.5*2 = 0),
-- while the prefix-SIBLING "rt_keepX" — which `isPrefixOf "rt_keep"` would
-- wrongly capture — is frozen because it isn't in the exact keep set.
restrictToExactComplement : IO Bool
restrictToExactComplement = do
  wk <- mkW "rt_keep" 1.0
  ws <- mkW "rt_keepX" 1.0
  wd <- mkW "rt_drop" 1.0
  opt <- sgd {ex=TestExecutor} 0.5 defaultOpts
  restrictTo {ex=TestExecutor} opt ["rt_keep"]
  l1 <- tmul wk wk
  l2 <- tmul ws ws
  l3 <- tmul wd wd
  l12 <- tadd l1 l2
  loss <- tadd l12 l3
  _ <- trainStep opt loss
  let (vk, vs, vd) = (tensorItem wk, tensorItem ws, tensorItem wd)
  check ("restrictTo keeps exact set (kept " ++ show vk ++ ", prefix-sibling "
         ++ show vs ++ ", other " ++ show vd ++ ")")
        (vk == 0.0 && vs == 1.0 && vd == 1.0)

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

-- setGroupLR: per-group LR overrides set after construction over the exact
-- names from a registry filter. One step at base lr 0.25 on three params: the
-- frozen group (LR 0) stays put, the scaled group steps at lr 0.125, the
-- bystander steps at base. All values exact binary fractions. Applies uniformly
-- across tape / mlx / torch (torch via per-param LR buckets in optimizer_step —
-- see backend_torch/training/optimizer.cpp).
setGroupLROverrides : IO Bool
setGroupLROverrides = do
  wf <- mkW "opt_g4f_w" 1.0
  ws <- mkW "opt_g4s_w" 1.0
  wn <- mkW "opt_g4n_w" 1.0
  opt <- sgd {ex=TestExecutor} 0.25 defaultOpts
  setGroupLR {ex=TestExecutor} opt !(namesMatching {ex=TestExecutor} (isPrefixOf "opt_g4f_")) 0.0
  setGroupLR {ex=TestExecutor} opt !(namesMatching {ex=TestExecutor} (isPrefixOf "opt_g4s_")) 0.125
  l1 <- tmul wf wf
  l2 <- tmul ws ws
  l3 <- tmul wn wn
  l12 <- tadd l1 l2
  loss <- tadd l12 l3
  _ <- trainStep opt loss
  let (vf, vs, vn) = (tensorItem wf, tensorItem ws, tensorItem wn)
  check ("setGroupLR freeze/scale by group (frozen " ++ show vf
         ++ ", scaled " ++ show vs ++ ", base " ++ show vn ++ ")")
        (vf == 1.0 && vs == 0.75 && vn == 0.5)

-- polyakUpdatePaired: pairs online↔target params positionally by EXACT name,
-- not by string prefix. Hard sync (tau=1) copies each online value into its
-- paired target; a prefix-SIBLING target name that a "q1tgt_" prefix would
-- wrongly capture is left untouched (the leak-free guarantee). Two pairs ->
-- count 2.
polyakPairedHardSync : IO Bool
polyakPairedHardSync = do
  _   <- mkW "pk_q1_0w" 2.0
  _   <- mkW "pk_q1_1w" 3.0
  t1  <- mkW "pk_q1tgt_0w" 0.0
  t2  <- mkW "pk_q1tgt_1w" 0.0
  sib <- mkW "pk_q1tgt_0wX" 9.0
  n <- polyakUpdatePaired {ex=TestExecutor}
         ["pk_q1_0w", "pk_q1_1w"] ["pk_q1tgt_0w", "pk_q1tgt_1w"] 1.0
  let (v1, v2, vs) = (tensorItem t1, tensorItem t2, tensorItem sib)
  check ("polyakUpdatePaired hard-sync copies paired names, leaves sibling (n="
         ++ show n ++ ", t1=" ++ show v1 ++ ", t2=" ++ show v2 ++ ", sib="
         ++ show vs ++ ")")
        (n == 2 && v1 == 2.0 && v2 == 3.0 && vs == 9.0)

-- polyakUpdatePaired blend formula: target ← (1-tau)·target + tau·online.
polyakPairedBlend : IO Bool
polyakPairedBlend = do
  _ <- mkW "pk_b_on" 1.0
  t <- mkW "pk_b_tg" 0.0
  n <- polyakUpdatePaired {ex=TestExecutor} ["pk_b_on"] ["pk_b_tg"] 0.25
  check ("polyakUpdatePaired blends 0.25 (n=" ++ show n ++ ", t="
         ++ show (tensorItem t) ++ ")")
        (n == 1 && tensorItem t == 0.25)

export
tests : List (IO Bool)
tests = [ sgdStepsQuadratic, rmspropStepsQuadratic, rmspropDefaultsMatchPyTorch
        , adamStepsScaled, adamWStepsQuadratic
        , restrictToExactComplement
        , polyakPairedHardSync, polyakPairedBlend
        , scheduleFreezesAtZero, tickAppliesScheduleEpoch
        , tickWithoutScheduleIsNoOp, setGroupLROverrides ]
