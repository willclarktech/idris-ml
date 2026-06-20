||| Unified optimizer construction surface (v1 API).
|||
||| One handle type — `Optimizer ex`, an alias of the existing
||| `NativeOptimizer ex` (same record, same fused `trainStep`) —
||| with one constructor per algorithm: `sgd`, `rmsprop`, `adam`,
||| `adamW`. Shared knobs (Adam betas, eps, clipping) live in
||| `OptimOpts`; knobs that only one algorithm consumes are arguments
||| on the constructor that owns them (`rmsprop`'s alpha/momentum,
||| `adamW`'s weightDecay) so nothing is ever accepted-but-ignored.
|||
||| Constructors are IO: construction touches the C-side optimizer
||| registry. Per-param LR overrides — multi-network ownership, freeze,
||| group LR — are applied *after* construction via the typed
||| `Train.Freeze` surface (`restrictTo` / `freezeGroup` / `setGroupLR`),
||| fed exact names from `Nn.Group.groupOf` / `reflectNames` or
||| `namesMatching`, not a registry-prefix string.
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

import Executor
import Schedule
import Tensor

public export
Optimizer : (0 ex : Executor) -> Type
Optimizer = NativeOptimizer

||| Shared optimizer options. Algorithm-specific knobs (rmsprop
||| alpha/momentum, adamW weightDecay) are constructor arguments, not
||| fields — a field only some constructors read would silently no-op
||| on the others.
|||
||| Per-param LR overrides (multi-network ownership, freeze, group LR)
||| are no longer a string-prefix field here: scope an optimizer after
||| construction with `Train.Freeze.restrictTo` / `freezeGroup` /
||| `setGroupLR`, fed the exact names from `Nn.Group.groupOf` /
||| `reflectNames` (structural, leak-free) or `namesMatching` (explicit
||| name-pattern). Same registry-order hazard: build optimizers after
||| the networks register.
public export
record OptimOpts where
  constructor MkOptimOpts
  beta1 : Double
  beta2 : Double
  eps   : Double
  clip  : ClipMode

||| PyTorch-default options: beta1 0.9, beta2 0.999, eps 1e-8, no
||| clipping.
public export
defaultOpts : OptimOpts
defaultOpts = MkOptimOpts 0.9 0.999 1.0e-8 NoClip

||| Plain SGD. Reads only `clip` from opts.
export
sgd : {0 ex : Executor} -> UserExecutorTraining ex =>
      (lr : Double) -> OptimOpts -> IO (Optimizer ex)
sgd lr opts = ioRerun (\_ =>
  MkNativeOptimizer (primOptimizerCreateSgd {ex} lr) opts.clip Nothing)

||| RMSprop (PyTorch parameterisation). `alpha` is the squared-grad
||| moving-average coefficient, `momentum` the heavy-ball term; both
||| default to torch.optim.RMSprop's values. Reads `eps` and `clip`
||| from opts.
export
rmsprop : {0 ex : Executor} -> UserExecutorTraining ex =>
          (lr : Double) -> {default 0.99 alpha : Double} ->
          {default 0.0 momentum : Double} -> OptimOpts -> IO (Optimizer ex)
rmsprop lr {alpha} {momentum} opts = ioRerun (\_ =>
  MkNativeOptimizer
    (primOptimizerCreateRmsprop {ex} lr alpha opts.eps 0.0 momentum)
    opts.clip Nothing)

||| Adam. Reads beta1/beta2/eps/clip from opts. For multi-network
||| ownership (SAC actor / q1 / q2 — one optimizer per net so one
||| network's loss can't leak updates into another's weights) scope it
||| after construction with `Train.Freeze.restrictTo opt (reflectNames
||| net)` rather than a registry-prefix string.
export
adam : {0 ex : Executor} -> UserExecutorTraining ex =>
       (lr : Double) -> OptimOpts -> IO (Optimizer ex)
adam lr opts = ioRerun (\_ =>
  MkNativeOptimizer
    (primOptimizerCreateAdam {ex} lr opts.beta1 opts.beta2 opts.eps)
    opts.clip Nothing)

||| AdamW (decoupled weight decay). `weightDecay` is positional — only
||| AdamW's C prim consumes it; a shared OptimOpts field would be
||| silently ignored by the other constructors. Reads
||| beta1/beta2/eps/clip from opts.
export
adamW : {0 ex : Executor} -> UserExecutorTraining ex =>
        (lr : Double) -> (weightDecay : Double) -> OptimOpts -> IO (Optimizer ex)
adamW lr wd opts = ioRerun (\_ =>
  MkNativeOptimizer
    (primOptimizerCreateAdamW {ex} lr opts.beta1 opts.beta2 opts.eps wd)
    opts.clip Nothing)

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
