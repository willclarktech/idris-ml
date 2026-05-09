module Layer.GruV2

import Data.Vect

import Compat.Random
import Device
import Init
import Layer.CoreV2
import Sampler
import Variable


----------------------------------------------------------------------
-- GruV2 — typed-surface GRU cell (Path C)
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
record GruStateV2 (i : Nat) (o : Nat) (0 d : Device) where
  constructor MkGruV2
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
applyGruV2 : {o : Nat} ->
             GruStateV2 i o d ->
             TVec i d ->
             (GruStateV2 i o d, TVec o d)
applyGruV2 {o} st input =
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

||| Build a `GruStateV2 i o CPU` with Xavier-uniform weights and
||| zero biases. Params register under `<prefix>_iw`, `<prefix>_ih_b`,
||| `<prefix>_hw`, `<prefix>_hh_b`.
export
gruLayerV2 : {i, o : Nat} -> (paramPrefix : String) ->
             IO (GruStateV2 i o CPU)
gruLayerV2 paramPrefix = do
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
      iwTV = MkTVar iwPtr (Just iwName)
      hwTV : TMat (3 * o) o CPU
      hwTV = MkTVar hwPtr (Just hwName)
      ihBTV : TVec (3 * o) CPU
      ihBTV = MkTVar ihBPtr (Just ihBName)
      hhBTV : TVec (3 * o) CPU
      hhBTV = MkTVar hhBPtr (Just hhBName)
  pure $ MkGruV2 iwTV ihBTV hwTV hhBTV Nothing

||| Reset hidden state to fresh zero-tensor.
export
resetGruStateV2 : GruStateV2 i o d -> GruStateV2 i o d
resetGruStateV2 = { hiddenT := Nothing }


----------------------------------------------------------------------
-- LayerLikeV2 instance
----------------------------------------------------------------------

public export
LayerLikeV2 GruStateV2 where
  applyTVar = applyGruV2
  layerPrefixV2 _ = "gruV2"

||| Wrap a `GruStateV2` in `AnyLayerV2`.
export
gruLayerV2Any : {i, o : Nat} -> (paramPrefix : String) -> IO (AnyLayerV2 i o CPU)
gruLayerV2Any pid = map (MkAnyLayerV2 GruStateV2) (gruLayerV2 pid)
