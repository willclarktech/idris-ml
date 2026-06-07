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

import Executor
import Tensor

public export
Optimizer : (0 ex : Executor) -> Type
Optimizer = NativeOptimizer

||| Shared optimizer options. Algorithm-specific knobs (rmsprop
||| alpha/momentum, adamW weightDecay, adam scope) are constructor
||| arguments, not fields — a field only some constructors read would
||| silently no-op on the others.
public export
record OptimOpts where
  constructor MkOptimOpts
  beta1 : Double
  beta2 : Double
  eps   : Double
  clip  : ClipMode

||| PyTorch-default options: beta1 0.9, beta2 0.999, eps 1e-8, no clipping.
public export
defaultOpts : OptimOpts
defaultOpts = MkOptimOpts 0.9 0.999 1.0e-8 NoClip

||| Plain SGD. Reads only `clip` from opts.
export
sgd : {0 ex : Executor} -> UserExecutorTraining ex =>
      (lr : Double) -> OptimOpts -> IO (Optimizer ex)
sgd lr opts = ioRerun (\_ =>
  MkNativeOptimizer (primOptimizerCreateSgd {ex} lr) opts.clip)

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
    opts.clip)
