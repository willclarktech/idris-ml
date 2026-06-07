||| Unified optimizer construction surface (v1 API).
|||
||| One handle type — `Optimizer ex`, an alias of the existing
||| `NativeOptimizer ex` (same record, same fused `nativeTrainStep`) —
||| with one constructor per algorithm: `sgd`, `rmsprop`, `adam`,
||| `adamW`. Shared knobs (Adam betas, eps, clipping) live in
||| `OptimOpts`; knobs that only one algorithm consumes are arguments
||| on the constructor that owns them (`rmsprop`'s alpha/momentum,
||| `adamW`'s weightDecay, `adam`'s scope) so nothing is ever
||| accepted-but-ignored.
|||
||| Constructors are IO: construction touches the C-side optimizer
||| registry, and per-group LR overrides are applied by walking the
||| param registry at construction time.
|||
||| Numeric defaults mirror PyTorch: beta1 0.9, beta2 0.999, eps 1e-8,
||| rmsprop alpha 0.99 / momentum 0. Build options from `defaultOpts`
||| with record-update syntax:
|||
|||   opt <- adam 1.0e-3 ({ clip := NormClip 1.0 } defaultOpts)
|||
||| The `native*` constructors in Tensor.idr remain until the example
||| migration sweep; both surfaces drive identical C prims.
module Optimizer

import Data.String

import Executor
import Schedule
import Tensor

public export
Optimizer : (0 ex : Executor) -> Type
Optimizer = NativeOptimizer

||| Shared optimizer options. Algorithm-specific knobs (rmsprop
||| alpha/momentum, adamW weightDecay, adam scope) are constructor
||| arguments, not fields — a field only some constructors read would
||| silently no-op on the others.
|||
||| `groups` sets per-prefix LR overrides: every registered param whose
||| paramId starts with the prefix gets that LR instead of the base
||| (0 freezes; same mechanism as `setParamLR` / `freezeByPrefix`).
||| The walk happens at construction, so params registered AFTER the
||| optimizer is built don't pick up the override — construct
||| optimizers after the networks, the same registry-order hazard
||| `freezeByPrefix` documents.
public export
record OptimOpts where
  constructor MkOptimOpts
  beta1  : Double
  beta2  : Double
  eps    : Double
  clip   : ClipMode
  groups : List (String, Double)

||| PyTorch-default options: beta1 0.9, beta2 0.999, eps 1e-8, no
||| clipping, no group overrides.
public export
defaultOpts : OptimOpts
defaultOpts = MkOptimOpts 0.9 0.999 1.0e-8 NoClip []

-- Per-prefix override walk: for every (pfx, lr) group, set the
-- per-param LR override on each registered param whose name starts
-- with pfx (same mechanism as freezeByPrefix; kept local because
-- Train.Freeze sits above this module in the import order).
applyPrefix : {0 ex : Executor} -> UserExecutorTraining ex =>
              NativeOptimizer ex -> String -> Double -> Nat -> IO ()
applyPrefix opt pfx lr Z = pure ()
applyPrefix opt pfx lr (S k) = do
  name <- getParamName {ex} (cast {to=Int} k)
  when (isPrefixOf pfx name) (setParamLR {ex} opt name lr)
  applyPrefix opt pfx lr k

applyGroups : {0 ex : Executor} -> UserExecutorTraining ex =>
              NativeOptimizer ex -> List (String, Double) -> IO ()
applyGroups opt [] = pure ()
applyGroups opt ((pfx, lr) :: rest) = do
  n <- getParamCount {ex}
  applyPrefix opt pfx lr (cast {to=Nat} n)
  applyGroups opt rest

||| Plain SGD. Reads only `clip` and `groups` from opts.
export
sgd : {0 ex : Executor} -> UserExecutorTraining ex =>
      (lr : Double) -> OptimOpts -> IO (Optimizer ex)
sgd lr opts = do
  opt <- ioRerun (\_ =>
    MkNativeOptimizer (primOptimizerCreateSgd {ex} lr) opts.clip Nothing)
  applyGroups opt opts.groups
  pure opt

||| RMSprop (PyTorch parameterisation). `alpha` is the squared-grad
||| moving-average coefficient, `momentum` the heavy-ball term; both
||| default to torch.optim.RMSprop's values. Reads `eps` and `clip`
||| from opts.
export
rmsprop : {0 ex : Executor} -> UserExecutorTraining ex =>
          (lr : Double) -> {default 0.99 alpha : Double} ->
          {default 0.0 momentum : Double} -> OptimOpts -> IO (Optimizer ex)
rmsprop lr {alpha} {momentum} opts = do
  opt <- ioRerun (\_ =>
    MkNativeOptimizer
      (primOptimizerCreateRmsprop {ex} lr alpha opts.eps 0.0 momentum)
      opts.clip Nothing)
  applyGroups opt opts.groups
  pure opt

||| Adam. `scope` restricts the optimizer to params whose registry
||| paramId starts with that prefix — the multi-network pattern (SAC
||| actor / q1 / q2), one optimizer per net so one network's loss can't
||| leak updates into another's weights. Empty scope (the default)
||| manages every param. Reads beta1/beta2/eps/clip from opts.
export
adam : {0 ex : Executor} -> UserExecutorTraining ex =>
       {default "" scope : String} -> (lr : Double) -> OptimOpts -> IO (Optimizer ex)
adam {scope} lr opts = do
  opt <- ioRerun (\_ =>
    MkNativeOptimizer
      (case scope of
         "" => primOptimizerCreateAdam {ex} lr opts.beta1 opts.beta2 opts.eps
         _  => primOptimizerCreateAdamGroup {ex} lr opts.beta1 opts.beta2 opts.eps scope)
      opts.clip Nothing)
  applyGroups opt opts.groups
  pure opt

||| AdamW (decoupled weight decay). `weightDecay` is positional — only
||| AdamW's C prim consumes it; a shared OptimOpts field would be
||| silently ignored by the other constructors. Reads
||| beta1/beta2/eps/clip from opts.
export
adamW : {0 ex : Executor} -> UserExecutorTraining ex =>
        (lr : Double) -> (weightDecay : Double) -> OptimOpts -> IO (Optimizer ex)
adamW lr wd opts = do
  opt <- ioRerun (\_ =>
    MkNativeOptimizer
      (primOptimizerCreateAdamW {ex} lr opts.beta1 opts.beta2 opts.eps wd)
      opts.clip Nothing)
  applyGroups opt opts.groups
  pure opt

||| Attach an LR schedule. The schedule only takes effect through
||| `tick` — call it once per epoch (the interim driver spelling is
||| `{ beforeEpoch := tick opt } cfg`; the `fit` driver will own the
||| tick when it lands).
export
withSchedule : Schedule -> Optimizer ex -> Optimizer ex
withSchedule s opt = { schedule := Just s } opt

||| Push schedule(epoch) into the optimizer's base LR. No-op when no
||| schedule is attached. Per-param LR overrides (groups, freezes)
||| are untouched — see `setLearningRate`.
export
tick : {0 ex : Executor} -> UserExecutorTraining ex =>
       Optimizer ex -> (epoch : Nat) -> IO ()
tick opt epoch = case opt.schedule of
  Nothing => pure ()
  Just s  => setLearningRate opt (s epoch)
