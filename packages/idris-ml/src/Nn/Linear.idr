||| `Linear` — the dense-layer port to the v1 `Nn` surface, and the
||| exemplar every other dense layer follows.
|||
|||   * The record drops the `GradMode` index the legacy `LinearState`
|||     carried: a layer owns its `WithGrad` params by construction, so the
|||     fields are pinned `WithGrad` and `g` lives only on the activation.
|||   * `Module`'s batched `forward` reuses the same fused C op
|||     (`tlinear2d` / `primLinear2d`) the legacy `LayerLike.applyVarBatch`
|||     called — forward path is perf-neutral by construction.
|||   * `Params` is the hand-written 2-liner (the spike's verdict).
|||   * `linear` is the `Init` smart constructor: it derives its module name
|||     with `freshChild` and registers `<name>.weight` / `<name>.bias`,
|||     matching the legacy `linearLayer` init (weight ~ N(0, 1/√fan_in),
|||     zero bias) so an `Nn.Seq` MLP and the old `Network` MLP are
|||     numerically identical given the same RNG stream.
module Nn.Linear

import Control.Linear.LIO
import Data.Linear
import Data.Vect

import Executor
import Nn.Init
import Nn.Module
import Tensor

%default total

||| A dense layer: `y = x · Wᵀ + b`. The `g` index is the params' grad-mode;
||| construction yields `WithGrad`, inference handles are `NoGrad`.
public export
record Linear (i : Nat) (o : Nat) (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkLinear
  weightT : Tensor [o, i] ex dt g
  biasT   : Tensor [o] ex dt g

public export
Module Linear where
  -- Params and activation share `g`, so no `retypeGrad`: a `WithGrad` model
  -- keeps the tape, a `NoGrad` model is genuinely tape-free.
  forward (MkLinear w b) x = tlinear2d w x b

public export
Params Linear where
  params (MkLinear w b)   = [toParam w, toParam b]
  castGrad (MkLinear w b) = MkLinear (retypeGrad w) (retypeGrad b)

||| Linear-resource params. Pattern-matching `MkLinear` binds `w`/`b` at
||| their ω constructor quantity, so they are free to be reflected *and*
||| rebuild the record (the whole-model linearity obligation is discharged by
||| the single match).
public export
ParamsL Linear where
  reflectL (MkLinear w b)  = MkBang [toParam w, toParam b] # MkLinear w b
  castGradL (MkLinear w b) = MkLinear (retypeGrad w) (retypeGrad b)
  discardL (MkLinear _ _)  = pure ()

||| Linear-resource `Module`. The output tensor is unrestricted, so it rides
||| the linear return pair under the `(!*)` bang.
public export
ModuleL Linear where
  forwardL (MkLinear w b) x = do
    y <- tlinear2dL w x b
    pure1 (MkBang y # MkLinear w b)

||| Construct a `Linear i o` with caller-chosen init: weight ~ N(0,
||| weightStd), bias ~ N(0, biasStd) (biasStd = 0 → zero bias). Registers
||| `<scope>.linear_<n>.weight` / `.bias`. The escape hatch for layers that
||| need a non-default init (e.g. NTM's xavier-1.4 heads).
export
linearWith : {0 ex : Executor} -> Backend ex dt => {i, o : Nat} ->
             (weightStd : Double) -> (biasStd : Double) -> Init (Linear i o ex dt WithGrad)
linearWith weightStd biasStd = do
  name <- freshChild "linear"
  w <- liftIO $ tparam2dNormal {ex} {dt} {o} {i} (name ++ ".weight") 0.0 weightStd
  b <- liftIO $ if biasStd == 0.0
                  then tparam1dConst  {ex} {dt} {n=o} (name ++ ".bias") 0.0
                  else tparam1dNormal {ex} {dt} {n=o} (name ++ ".bias") 0.0 biasStd
  pure (MkLinear w b)

||| Construct a `Linear i o` with PyTorch's `nn.Linear` normal-approx
||| default (weight ~ N(0, 1/√fan_in), zero bias) — matches the legacy
||| `linearLayer`. The common case of `linearWith`.
export
linear : {0 ex : Executor} -> Backend ex dt => {i, o : Nat} -> Init (Linear i o ex dt WithGrad)
linear = linearWith (1.0 / sqrt (cast {to=Double} i)) 0.0
