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
module Ml.Nn.LinearMixed

import Control.Linear.LIO
import Data.Linear
import Data.Vect

import Ml.Executor
import Ml.Nn.Init
import Ml.Nn.Module
import Ml.Tensor

%default total

||| A dense layer with split param/compute dtypes: `y = x · cast(Wᵀ) +
||| cast(b)`, weights stored in `paramDt`. The `g` index is the master
||| params' grad-mode (`WithGrad` by construction; `NoGrad` for inference).
public export
record LinearMixed (i : Nat) (o : Nat) (0 ex : Executor)
                   (0 paramDt : DType) (0 computeDt : DType) (0 g : GradMode) where
  constructor MkLinearMixed
  weightT : Tensor [o, i] ex paramDt g
  biasT   : Tensor [o] ex paramDt g

public export
ParamsMixed LinearMixed where
  paramsMixed (MkLinearMixed w b)   = [toParam w, toParam b]
  reflectMixed (MkLinearMixed w b)  = MkBang [toParam w, toParam b] # MkLinearMixed w b
  castGradMixed (MkLinearMixed w b) = MkLinearMixed (retypeGrad w) (retypeGrad b)
  discardMixed (MkLinearMixed _ _)  = pure ()

public export
ModuleMixed LinearMixed where
  -- Cast the master weight + bias paramDt → computeDt (autograd-aware), then
  -- the fused matmul + bias-add in computeDt. Master params share the
  -- activation's `g`, so no `retypeGrad` is needed before the cast. Linear:
  -- pattern-match binds `w`/`b` at ω so they feed the cast/matmul *and* the
  -- record rebuild; the output rides the `(!*)` bang beside the threaded model.
  forwardMixed {computeDt} (MkLinearMixed w b) x = do
    wc <- tcastUnsafeL computeDt w
    bc <- tcastUnsafeL computeDt b
    y  <- tlinear2dL wc x bc
    pure1 (MkBang y # MkLinearMixed w b)

||| Construct a `LinearMixed i o ex paramDt computeDt` with PyTorch's
||| `nn.Linear` normal-approx default (weight ~ N(0, 1/√fan_in), zero bias),
||| both stored in `paramDt`. Registers `<scope>.linear_<n>.weight` / `.bias`
||| — same naming as `Nn.Linear.linear`, so a mixed checkpoint loads into a
||| plain-Linear model when `paramDt` matches the on-disk dtype. `computeDt`
||| is observed only at the type level; the cast happens per-call in the
||| forward.
export
linearMixed : {0 ex : Executor} -> Backend ex paramDt => {i, o : Nat} ->
              Init (LinearMixed i o ex paramDt computeDt WithGrad)
linearMixed = do
  name <- freshChild "linear"
  w <- liftIO $ tparam2dNormal {ex} {dt=paramDt} {o} {i}
                  (name ++ ".weight") 0.0 (1.0 / sqrt (cast {to=Double} i))
  b <- liftIO $ tparam1dConst {ex} {dt=paramDt} {n=o} (name ++ ".bias") 0.0
  pure (MkLinearMixed w b)
