module Layer.LinearMixed

import Data.Vect

import Compat.Random
import Device
import Init
import Layer.Core
import Layer.MixedCore
import Sampler
import Tensor


----------------------------------------------------------------------
-- LinearMixed: explicit param/compute dtype split
----------------------------------------------------------------------
--
-- The mixed-precision counterpart to `LinearState`. Weight and bias
-- are stored in `paramDt` (typically F32, the "master" precision);
-- the forward pass casts them down to `computeDt` (typically BF16 /
-- F16) before the matmul, with the cast registered in the autograd
-- graph so backward writes an F32 gradient back into the master.
--
-- This is the first concrete user of the `LayerLikeMixed` interface
-- (Layer.MixedCore) and the autograd-aware `tcast` (A1 in the type-
-- safe mixed-precision plan #410). The cast is *explicit* in the
-- forward — the lossy edge from paramDt to computeDt is visible at
-- the layer boundary, not silently runtime-injected the way PyTorch's
-- autocast would do it.
--
-- For `paramDt == computeDt` (e.g. F32-master / F32-compute) the cast
-- is a no-op at the dtype level; tape's lingua-franca path makes it
-- effectively free, and torch/mlx handle it via their native cast.

public export
record LinearMixedState (i : Nat) (o : Nat) (0 d : Device)
                        (0 paramDt : DType) (0 computeDt : DType) (0 g : GradMode) where
  constructor MkLinearMixed
  weightT : Tensor [o, i] d paramDt g
  biasT   : Tensor [o] d paramDt g


----------------------------------------------------------------------
-- LayerLikeMixed instance — cast-cast-matmul forward
----------------------------------------------------------------------

%default partial

public export
LayerLikeMixed LinearMixedState where
  -- Forward: cast paramDt → computeDt for both weight and bias, then
  -- fused matmul + bias-add in computeDt. The casts go through
  -- `tcastUnsafe` because the typical direction (F32 → BF16) is lossy
  -- and the layer's type signature already advertises both dtypes —
  -- the lossy edge is code-visible, not silent.
  applyVarMixed {cDt} st input = do
    wCast <- tcastUnsafe cDt st.weightT
    bCast <- tcastUnsafe cDt st.biasT
    out <- tlinear wCast input bCast
    pure (st, out)

  layerPrefixMixed _ = "lin_mixed"

  freezeLayerMixed (MkLinearMixed w b) = do
    w' <- weakenGrad w
    b' <- weakenGrad b
    pure (MkLinearMixed w' b')

  unfreezeLayerMixed (MkLinearMixed w b) = do
    primIO (primSetRequiresGrad {d} w.tensorPtr 1)
    primIO (primSetRequiresGrad {d} b.tensorPtr 1)
    pure (MkLinearMixed (retypeGrad w) (retypeGrad b))


----------------------------------------------------------------------
-- Constructor
----------------------------------------------------------------------

||| Build a `LinearMixedState i o d paramDt computeDt` with PyTorch's
||| default `nn.Linear` init (normal approximation): weight ~ N(0,
||| 1/sqrt(fan_in)), zero bias. Both weight and bias are stored in
||| `paramDt` and registered under `<paramPrefix>_weights` /
||| `<paramPrefix>_biases` — same naming convention as `linearLayer`,
||| so a mixed-precision checkpoint can be loaded by a plain-Linear
||| network when paramDt matches the checkpoint's dtype.
|||
||| The `computeDt` parameter is observed only at the type level — it
||| doesn't influence construction. The forward cast happens per-call
||| via `tcastUnsafe`.
export
mixedLinearLayer : UserDeviceTraining d =>
                   RuntimeDType paramDt => RuntimeDType computeDt =>
                   Linked d =>
                   Compatible d paramDt => Compatible d computeDt =>
                   {i, o : Nat} -> (paramPrefix : String) ->
                   IO (LinearMixedState i o d paramDt computeDt WithGrad)
mixedLinearLayer pfx = do
  let wName = pfx ++ "_weights"
      bName = pfx ++ "_biases"
      wStd  = 1.0 / sqrt (cast {to=Double} i)
  w <- tparam2dNormal {d} {dt=paramDt} {o} {i} wName 0.0 wStd
  b <- tparam1dConst  {d} {dt=paramDt} {n=o} bName 0.0
  pure $ MkLinearMixed w b

||| Wrap a `LinearMixedState` in `AnyLayerMixed` for use in a
||| `NetworkMixed`.
export
mixedLinearLayerAny : UserDeviceTraining d =>
                      RuntimeDType paramDt => RuntimeDType computeDt =>
                      Linked d =>
                      Compatible d paramDt => Compatible d computeDt =>
                      {i, o : Nat} -> (paramPrefix : String) ->
                      IO (AnyLayerMixed i o d paramDt computeDt WithGrad)
mixedLinearLayerAny pid =
  map (MkAnyLayerMixed LinearMixedState) (mixedLinearLayer pid)
