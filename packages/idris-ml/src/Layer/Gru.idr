module Layer.Gru

import Data.Vect

import Executor
import Layer.Core
import Tensor

----------------------------------------------------------------------
-- Gru — typed-surface GRU cell (Path C)
----------------------------------------------------------------------
--
-- Mirrors `Layer/Gru.idr`'s `applyVarTensor` path with the simplified
-- GRU variant the C kernel implements (`tensor_gru_cell`):
--   combined = (W_ih · x + b_ih) + (W_hh · h + b_hh)
--   h_t = `primGruCell {ex}` combined h_{t-1} o
--
-- Three gate paths (z, r, n) are computed inside the C op; the
-- combined vector has shape [3 * o]. Static type safety via
-- `TMat (3 * o) ...` and `TVec (3 * o) ...` aliases.

public export
record GruState (i : Nat) (o : Nat) (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkGru
  iwT     : TMat (3 * o) i ex dt g         -- W_ih [3*o, i]
  ihB     : TVec (3 * o) ex dt g           -- b_ih [3*o]
  hwT     : TMat (3 * o) o ex dt g         -- W_hh [3*o, o]
  hhB     : TVec (3 * o) ex dt g           -- b_hh [3*o]
  hiddenT : Maybe (TVec o ex dt g)

----------------------------------------------------------------------
-- Forward
----------------------------------------------------------------------

%default partial

export
applyGru : {0 ex : Executor} -> Backend ex dt => {o : Nat} ->
             GruState i o ex dt g ->
             TVec i ex dt g ->
             IO (GruState i o ex dt g, TVec o ex dt g)
applyGru {o} st input = do
  h <- case st.hiddenT of
         Just h => pure h
         Nothing => tzeroState1d {n = o}
  ihPart <- tlinear st.iwT input st.ihB    -- W_ih @ x + b_ih
  hhPart <- tlinear st.hwT h st.hhB        -- W_hh @ h + b_hh
  newH <- tgruCell {n = o} ihPart hhPart h
  pure ({ hiddenT := Just newH } st, newH)

----------------------------------------------------------------------
-- Constructor
----------------------------------------------------------------------

||| Build a `GruState i o TapeExecutor` with Xavier-uniform weights and
||| zero biases. Params register under `<prefix>_iw`, `<prefix>_ih_b`,
||| `<prefix>_hw`, `<prefix>_hh_b`.
export
gruLayer : Backend ex dt => {i, o : Nat} -> (paramPrefix : String) ->
             IO (GruState i o ex dt WithGrad)
gruLayer paramPrefix = do
  -- GRU has 3 gates (reset, update, new); weights are stacked along
  -- axis=0 → [3*o, i] for W_ih and [3*o, o] for W_hh. Xavier-normal-
  -- via-uniform: std = sqrt(2/(fan_in + fan_out)) where fan_out = 3*o.
  -- Biases zero-init.
  let iwStd = sqrt (2.0 / cast {to=Double} (i + 3 * o))
      hwStd = sqrt (2.0 / cast {to=Double} (o + 3 * o))
      iwName  = paramPrefix ++ "_iw"
      hwName  = paramPrefix ++ "_hw"
      ihBName = paramPrefix ++ "_ih_b"
      hhBName = paramPrefix ++ "_hh_b"
  iw  <- tparam2dNormal {o = 3 * o} {i} iwName 0.0 iwStd
  hw  <- tparam2dNormal {o = 3 * o} {i = o} hwName 0.0 hwStd
  ihB <- tparam1dConst {n = 3 * o} ihBName 0.0
  hhB <- tparam1dConst {n = 3 * o} hhBName 0.0
  pure $ MkGru iw ihB hw hhB Nothing

||| Reset hidden state. Lazy-allocate on next applyVar call.
export
resetGruState : {o : Nat} -> {0 ex : Executor} -> {0 g : GradMode} -> GruState i o ex dt g -> GruState i o ex dt g
resetGruState st = { hiddenT := Nothing } st

----------------------------------------------------------------------
-- LayerLike instance
----------------------------------------------------------------------

public export
LayerLike GruState where
  applyVar = applyGru
  layerPrefix _ = "gru"

  resetState = resetGruState

  freezeLayer (MkGru iw ihB hw hhB hid) = do
    iw'  <- weakenGrad iw
    ihB' <- weakenGrad ihB
    hw'  <- weakenGrad hw
    hhB' <- weakenGrad hhB
    hid' <- case hid of
      Nothing => pure Nothing
      Just h  => Just <$> weakenGrad h
    pure (MkGru iw' ihB' hw' hhB' hid')

  unfreezeLayer (MkGru iw ihB hw hhB hid) = do
    primIO (primSetRequiresGrad {ex} iw.tensorPtr 1)
    primIO (primSetRequiresGrad {ex} ihB.tensorPtr 1)
    primIO (primSetRequiresGrad {ex} hw.tensorPtr 1)
    primIO (primSetRequiresGrad {ex} hhB.tensorPtr 1)
    case hid of
      Nothing => pure ()
      Just h  => primIO (primSetRequiresGrad {ex} h.tensorPtr 1)
    pure (MkGru (retypeGrad iw) (retypeGrad ihB)
                (retypeGrad hw) (retypeGrad hhB)
                (map retypeGrad hid))

||| Wrap a `GruState` in `AnyLayer`.
export
gruLayerAny : Backend ex dt => {i, o : Nat} -> (paramPrefix : String) -> IO (AnyLayer i o ex dt WithGrad)
gruLayerAny pid = map (MkAnyLayer GruState) (gruLayer pid)
