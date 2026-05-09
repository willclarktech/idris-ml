module Layer.Rnn

import Data.Vect

import Compat.Random
import Device
import Init
import Layer.Core
import Sampler
import Variable


----------------------------------------------------------------------
-- Rnn — typed-surface vanilla RNN cell (Path C)
----------------------------------------------------------------------
--
-- Mirrors `Layer/Rnn.idr`'s `applyVarTensor` path:
--   h_t = (W_ih · x_t) + (W_hh · h_{t-1}) + b
--
-- No activation in the cell; chain a `tanhLayer` (or other) after
-- if needed. Matches V1 behaviour.
--
-- Uses `TMat` / `TVec` aliases for consistency with `Lstm` even
-- though shape arithmetic isn't needed here (no `4 *`).

public export
record RnnState (i : Nat) (o : Nat) (0 d : Device) where
  constructor MkRnn
  iwT : TMat o i d         -- W_ih [o, i]
  rwT : TMat o o d         -- W_hh [o, o]
  bT  : TVec o d           -- bias [o]
  prevOutT : Maybe (TVec o d)


----------------------------------------------------------------------
-- Forward
----------------------------------------------------------------------

%default partial

export
applyRnn : {o : Nat} ->
             RnnState i o d ->
             TVec i d ->
             (RnnState i o d, TVec o d)
applyRnn {o} st input =
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

||| Build an `RnnState i o CPU` with Xavier-uniform weights and
||| zero bias. State starts as Nothing; first `applyRnn` call
||| zero-initialises it. Params register under `<prefix>_iw`,
||| `<prefix>_rw`, `<prefix>_b`.
export
rnnLayer : {i, o : Nat} -> (paramPrefix : String) ->
             IO (RnnState i o CPU)
rnnLayer paramPrefix = do
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
      iwTV = MkVar iwPtr (Just iwName)
      rwTV : TMat o o CPU
      rwTV = MkVar rwPtr (Just rwName)
      bTV : TVec o CPU
      bTV = MkVar bPtr (Just bName)
  pure $ MkRnn iwTV rwTV bTV Nothing

||| Reset hidden state. Lazy-allocate on next applyVar call.
export
resetRnnState : {o : Nat} -> {0 d : Device} -> RnnState i o d -> RnnState i o d
resetRnnState st = { prevOutT := Nothing } st


----------------------------------------------------------------------
-- LayerLike instance
----------------------------------------------------------------------

public export
LayerLike RnnState where
  applyVar = applyRnn
  layerPrefix _ = "rnn"

||| Wrap an `RnnState` in `AnyLayer`.
export
rnnLayerAny : {i, o : Nat} -> (paramPrefix : String) -> IO (AnyLayer i o CPU)
rnnLayerAny pid = map (MkAnyLayer RnnState) (rnnLayer pid)
