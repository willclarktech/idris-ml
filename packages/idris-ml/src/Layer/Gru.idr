module Layer.Gru

import Data.Vect

import Compat.Random
import Device
import Init
import Layer.Core
import Sampler
import Variable


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
record GruState (i : Nat) (o : Nat) (0 d : Device) where
  constructor MkGru
  iwT : TMat (3 * o) i d         -- W_ih [3*o, i]
  ihB : TVec (3 * o) d           -- b_ih [3*o]
  hwT : TMat (3 * o) o d         -- W_hh [3*o, o]
  hhB : TVec (3 * o) d           -- b_hh [3*o]
  hiddenT : Maybe (TVec o d)


----------------------------------------------------------------------
-- Forward
----------------------------------------------------------------------

%default partial

export
applyGru : {o : Nat} ->
             GruState i o d ->
             TVec i d ->
             (GruState i o d, TVec o d)
applyGru {o} st input =
  let h = case st.hiddenT of
            Just h => h
            Nothing => tzeroState1d {n = o}
      ihPart = tadd (tmv st.iwT input) st.ihB
      hhPart = tadd (tmv st.hwT h) st.hhB
      combined = tadd ihPart hhPart
      newH = tgruCell {n = o} combined h
  in ({ hiddenT := Just newH } st, newH)


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
gruLayer : {i, o : Nat} -> (paramPrefix : String) ->
             IO (GruState i o CPU)
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
      iwTV : TMat (3 * o) i CPU
      iwTV = MkVar iwPtr (Just iwName)
      hwTV : TMat (3 * o) o CPU
      hwTV = MkVar hwPtr (Just hwName)
      ihBTV : TVec (3 * o) CPU
      ihBTV = MkVar ihBPtr (Just ihBName)
      hhBTV : TVec (3 * o) CPU
      hhBTV = MkVar hhBPtr (Just hhBName)
  pure $ MkGru iwTV ihBTV hwTV hhBTV Nothing

||| Reset hidden state. Lazy-allocate on next applyVar call.
export
resetGruState : {o : Nat} -> {0 d : Device} -> GruState i o d -> GruState i o d
resetGruState st = { hiddenT := Nothing } st


----------------------------------------------------------------------
-- LayerLike instance
----------------------------------------------------------------------

public export
LayerLike GruState where
  applyVar = applyGru
  layerPrefix _ = "gru"

||| Wrap a `GruState` in `AnyLayer`.
export
gruLayerAny : {i, o : Nat} -> (paramPrefix : String) -> IO (AnyLayer i o CPU)
gruLayerAny pid = map (MkAnyLayer GruState) (gruLayer pid)
