module Layer.Gru

import Data.Vect

import Compat.Random
import Device
import Init
import Layer.Core
import Sampler
import Tensor


----------------------------------------------------------------------
-- Gru — typed-surface GRU cell (Path C)
----------------------------------------------------------------------
--
-- Mirrors `Layer/Gru.idr`'s `applyVarTensor` path with the simplified
-- GRU variant the C kernel implements (`tensor_gru_cell`):
--   combined = (W_ih · x + b_ih) + (W_hh · h + b_hh)
--   h_t = `prim__gruCell` combined h_{t-1} o
--
-- Three gate paths (z, r, n) are computed inside the C op; the
-- combined vector has shape [3 * o]. Static type safety via
-- `TMat (3 * o) ...` and `TVec (3 * o) ...` aliases.

public export
record GruState (i : Nat) (o : Nat) (0 d : Device) (0 dt : DType) (0 g : GradMode) where
  constructor MkGru
  iwT : TMat (3 * o) i d dt g         -- W_ih [3*o, i]
  ihB : TVec (3 * o) d dt g           -- b_ih [3*o]
  hwT : TMat (3 * o) o d dt g         -- W_hh [3*o, o]
  hhB : TVec (3 * o) d dt g           -- b_hh [3*o]
  hiddenT : Maybe (TVec o d dt g)


----------------------------------------------------------------------
-- Forward
----------------------------------------------------------------------

%default partial

export
applyGru : {0 d : Device} -> UserDeviceTape d => RuntimeDType dt => {o : Nat} ->
             GruState i o d dt g ->
             TVec i d dt g ->
             IO (GruState i o d dt g, TVec o d dt g)
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

packDoubles : AnyPtr -> Int -> Vect k Double -> AnyPtr
packDoubles buf _ [] = buf
packDoubles buf off (x :: rest) =
  packDoubles (prim__setDouble buf off x) (off + 1) rest

zeroBuf : AnyPtr -> Int -> Int -> AnyPtr
zeroBuf buf _ 0 = buf
zeroBuf buf off n =
  zeroBuf (prim__setDouble buf off 0.0) (off + 1) (n - 1)

||| Build a `GruState i o CPU` with Xavier-uniform weights and
||| zero biases. Params register under `<prefix>_iw`, `<prefix>_ih_b`,
||| `<prefix>_hw`, `<prefix>_hh_b`.
export
gruLayer : RuntimeDType dt => {i, o : Nat} -> (paramPrefix : String) ->
             IO (GruState i o d dt WithGrad)
gruLayer paramPrefix = do
  let gI = cast {to=Int} (3 * o)
      iI = cast {to=Int} i
      oI = cast {to=Int} o
  iwVals <- traverse (\_ => xavier uniform i (3 * o)) (Vect.replicate (3 * o * i) ())
  hwVals <- traverse (\_ => xavier uniform o (3 * o)) (Vect.replicate (3 * o * o) ())
  let iwBuf = prim__allocDoubles (gI * iI)
      iwBuf' = packDoubles iwBuf 0 iwVals
      hwBuf = prim__allocDoubles (gI * oI)
      hwBuf' = packDoubles hwBuf 0 hwVals
      ihBBuf = prim__allocDoubles gI
      ihBBuf' = zeroBuf ihBBuf 0 gI
      hhBBuf = prim__allocDoubles gI
      hhBBuf' = zeroBuf hhBBuf 0 gI
      iwName  = paramPrefix ++ "_iw"
      hwName  = paramPrefix ++ "_hw"
      ihBName = paramPrefix ++ "_ih_b"
      hhBName = paramPrefix ++ "_hh_b"
      iwPtr  = prim__paramRegister iwName  (prim__createParam2d gI iI iwBuf')
      hwPtr  = prim__paramRegister hwName  (prim__createParam2d gI oI hwBuf')
      ihBPtr = prim__paramRegister ihBName (prim__createParam1d gI ihBBuf')
      hhBPtr = prim__paramRegister hhBName (prim__createParam1d gI hhBBuf')
      iwTV : TMat (3 * o) i d dt WithGrad
      iwTV = MkTensor iwPtr (Just iwName)
      hwTV : TMat (3 * o) o d dt WithGrad
      hwTV = MkTensor hwPtr (Just hwName)
      ihBTV : TVec (3 * o) d dt WithGrad
      ihBTV = MkTensor ihBPtr (Just ihBName)
      hhBTV : TVec (3 * o) d dt WithGrad
      hhBTV = MkTensor hhBPtr (Just hhBName)
  pure $ MkGru iwTV ihBTV hwTV hhBTV Nothing

||| Reset hidden state. Lazy-allocate on next applyVar call.
export
resetGruState : {o : Nat} -> {0 d : Device} -> {0 g : GradMode} -> GruState i o d dt g -> GruState i o d dt g
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
    primIO (prim__setRequiresGrad iw.tensorPtr 1)
    primIO (prim__setRequiresGrad ihB.tensorPtr 1)
    primIO (prim__setRequiresGrad hw.tensorPtr 1)
    primIO (prim__setRequiresGrad hhB.tensorPtr 1)
    case hid of
      Nothing => pure ()
      Just h  => primIO (prim__setRequiresGrad h.tensorPtr 1)
    pure (MkGru (retypeGrad iw) (retypeGrad ihB)
                (retypeGrad hw) (retypeGrad hhB)
                (map retypeGrad hid))

||| Wrap a `GruState` in `AnyLayer`.
export
gruLayerAny : RuntimeDType dt => {i, o : Nat} -> (paramPrefix : String) -> IO (AnyLayer i o d dt WithGrad)
gruLayerAny pid = map (MkAnyLayer GruState) (gruLayer pid)
