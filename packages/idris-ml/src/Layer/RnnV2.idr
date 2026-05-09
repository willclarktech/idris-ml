module Layer.RnnV2

import Data.Vect

import Compat.Random
import Device
import Init
import Layer.CoreV2
import Sampler
import Variable


----------------------------------------------------------------------
-- RnnV2 — typed-surface vanilla RNN cell (Path C)
----------------------------------------------------------------------
--
-- Mirrors `Layer/Rnn.idr`'s `applyVarTensor` path:
--   h_t = (W_ih · x_t) + (W_hh · h_{t-1}) + b
--
-- No activation in the cell; chain a `tanhLayerV2` (or other) after
-- if needed. Matches V1 behaviour.
--
-- Uses `TMat` / `TVec` aliases for consistency with `LstmV2` even
-- though shape arithmetic isn't needed here (no `4 *`).

public export
record RnnStateV2 (i : Nat) (o : Nat) (0 d : Device) where
  constructor MkRnnV2
  iwT : TMat o i d         -- W_ih [o, i]
  rwT : TMat o o d         -- W_hh [o, o]
  bT  : TVec o d           -- bias [o]
  prevOutT : Maybe (TVec o d)


----------------------------------------------------------------------
-- Forward
----------------------------------------------------------------------

%default partial

export
applyRnnV2 : {o : Nat} ->
             RnnStateV2 i o d ->
             TVec i d ->
             (RnnStateV2 i o d, TVec o d)
applyRnnV2 {o} st input =
  let p = case st.prevOutT of
            Just po => po
            Nothing => tzeroState1d {n = o}
      out = tadd (tadd (tmv st.iwT input) (tmv st.rwT p)) st.bT
  in ({ prevOutT := Just out } st, out)


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

||| Build an `RnnStateV2 i o CPU` with Xavier-uniform weights and
||| zero bias. State starts as Nothing; first `applyRnnV2` call
||| zero-initialises it. Params register under `<prefix>_iw`,
||| `<prefix>_rw`, `<prefix>_b`.
export
rnnLayerV2 : {i, o : Nat} -> (paramPrefix : String) ->
             IO (RnnStateV2 i o CPU)
rnnLayerV2 paramPrefix = do
  let oI = cast {to=Int} o
      iI = cast {to=Int} i
  iwVals <- traverse (\_ => xavier uniform i o) (Vect.replicate (o * i) ())
  rwVals <- traverse (\_ => xavier uniform o o) (Vect.replicate (o * o) ())
  let iwBuf = prim__allocDoubles (oI * iI)
      iwBuf' = packDoubles iwBuf 0 iwVals
      rwBuf = prim__allocDoubles (oI * oI)
      rwBuf' = packDoubles rwBuf 0 rwVals
      bBuf = prim__allocDoubles oI
      bBuf' = zeroBuf bBuf 0 oI
      iwName = paramPrefix ++ "_iw"
      rwName = paramPrefix ++ "_rw"
      bName  = paramPrefix ++ "_b"
      iwPtr = prim__paramRegister iwName (prim__createParam2d oI iI iwBuf')
      rwPtr = prim__paramRegister rwName (prim__createParam2d oI oI rwBuf')
      bPtr  = prim__paramRegister bName  (prim__createParam1d oI bBuf')
      iwTV : TMat o i CPU
      iwTV = MkTVar iwPtr (Just iwName)
      rwTV : TMat o o CPU
      rwTV = MkTVar rwPtr (Just rwName)
      bTV : TVec o CPU
      bTV = MkTVar bPtr (Just bName)
  pure $ MkRnnV2 iwTV rwTV bTV Nothing

||| Reset hidden state to fresh zero-tensor.
export
resetRnnStateV2 : RnnStateV2 i o d -> RnnStateV2 i o d
resetRnnStateV2 = { prevOutT := Nothing }


----------------------------------------------------------------------
-- LayerLikeV2 instance
----------------------------------------------------------------------

public export
LayerLikeV2 RnnStateV2 where
  applyTVar = applyRnnV2
  layerPrefixV2 _ = "rnnV2"

||| Wrap an `RnnStateV2` in `AnyLayerV2`.
export
rnnLayerV2Any : {i, o : Nat} -> (paramPrefix : String) -> IO (AnyLayerV2 i o CPU)
rnnLayerV2Any pid = map (MkAnyLayerV2 RnnStateV2) (rnnLayerV2 pid)
