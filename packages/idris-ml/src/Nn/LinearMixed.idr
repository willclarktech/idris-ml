||| `LinearMixed` — the master-weights dense layer on the `Nn` surface, the
||| port of legacy `Layer/LinearMixed.idr` to models-as-records.
|||
|||   * Weights are stored in `paramDt` (the F32 master); the forward casts
|||     them down to `computeDt` (BF16 / F16) via the autograd-aware
|||     `tcastUnsafe`, then runs the same fused `tlinear2d` `Nn.Linear` uses.
|||     Backward writes a `paramDt` gradient back through the cast into the
|||     master — the F32-master / low-precision-compute recipe.
|||   * The cast is *explicit* at the layer boundary (visible in the forward),
|||     not silently runtime-injected the way PyTorch autocast does it; the
|||     lossy `paramDt → computeDt` edge is code-visible.
|||   * No `AsMixed`/`AnyLayerMixed`/`NetworkMixed`/`lift*` machinery: a mixed
|||     model is a plain record with a hand-written forward (or, when a
|||     multi-layer mixed chain is actually needed, a future `SeqMixed`).
module Nn.LinearMixed

import Data.Vect

import Executor
import Tensor
import Nn.Init
import Nn.Module

%default total

||| A dense layer with split param/compute dtypes: `y = x · cast(Wᵀ) +
||| cast(b)`, weights stored in `paramDt`. No `GradMode` index — params are
||| `WithGrad` by construction.
public export
record LinearMixed (i : Nat) (o : Nat) (0 ex : Executor)
                   (0 paramDt : DType) (0 computeDt : DType) where
  constructor MkLinearMixed
  weightT : Tensor [o, i] ex paramDt WithGrad
  biasT   : Tensor [o] ex paramDt WithGrad

public export
ModuleMixed LinearMixed where
  -- Cast the master weight + bias paramDt → computeDt (autograd-aware), then
  -- the fused matmul + bias-add in computeDt. `retypeGrad` aligns the params'
  -- phantom `g` to the activation's (handles unchanged — `g` is erased), so
  -- the cast's `from` tensor matches `tcastUnsafe`'s `g`-polymorphic input.
  forwardMixed {computeDt} (MkLinearMixed w b) x = do
    wc <- tcastUnsafe computeDt (retypeGrad w)
    bc <- tcastUnsafe computeDt (retypeGrad b)
    tlinear2d wc x bc

public export
ParamsMixed LinearMixed where
  paramsMixed (MkLinearMixed w b) = [toParam w, toParam b]

||| Construct a `LinearMixed i o ex paramDt computeDt` with PyTorch's
||| `nn.Linear` normal-approx default (weight ~ N(0, 1/√fan_in), zero bias),
||| both stored in `paramDt`. Registers `<scope>.linear_<n>.weight` / `.bias`
||| — same naming as `Nn.Linear.linear`, so a mixed checkpoint loads into a
||| plain-Linear model when `paramDt` matches the on-disk dtype. `computeDt`
||| is observed only at the type level; the cast happens per-call in the
||| forward.
export
linearMixed : {0 ex : Executor} -> Backend ex paramDt => {i, o : Nat} ->
              Init (LinearMixed i o ex paramDt computeDt)
linearMixed = do
  name <- freshChild "linear"
  w <- liftIO $ tparam2dNormal {ex} {dt=paramDt} {o} {i}
                  (name ++ ".weight") 0.0 (1.0 / sqrt (cast {to=Double} i))
  b <- liftIO $ tparam1dConst {ex} {dt=paramDt} {n=o} (name ++ ".bias") 0.0
  pure (MkLinearMixed w b)
