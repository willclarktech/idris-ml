module Layer.LstmV2

import Data.Vect

import Compat.Random
import Device
import Init
import Sampler
import Variable


----------------------------------------------------------------------
-- LstmV2 — typed-surface LSTM cell (Path C)
----------------------------------------------------------------------
--
-- All shape-arithmetic flows through the `TVec` / `TMat` aliases in
-- `Variable.idr`. Direct `TVar [4 * o, ...] d` triggers an Idris 2
-- type-checker hang; `TMat (4 * o) i d` works fine because the
-- multiplication sits in a Nat-argument slot of the alias rather
-- than inside a Vect literal.

public export
record LstmStateV2 (i : Nat) (o : Nat) (0 d : Device) where
  constructor MkLstmV2
  iwT : TMat (4 * o) i d
  rwT : TMat (4 * o) o d
  bT  : TVec (4 * o) d
  hiddenT : Maybe (TVec o d)
  cellT   : Maybe (TVec o d)


----------------------------------------------------------------------
-- Forward
----------------------------------------------------------------------

%default partial

||| Tensor-level LSTM cell forward. Reads (or zero-initialises) the
||| hidden + cell state, runs the fused gate computation, returns the
||| updated layer state and the new hidden output.
export
applyLstmV2 : {o : Nat} ->
              LstmStateV2 i o d ->
              TVec i d ->
              (LstmStateV2 i o d, TVec o d)
applyLstmV2 {o} st input =
  let h = case st.hiddenT of
            Just h => h
            Nothing => tzeroState1d {n = o}
      c = case st.cellT of
            Just c => c
            Nothing => tzeroState1d {n = o}
      combined = tadd (tadd (tmv st.iwT input) (tmv st.rwT h)) st.bT
      (newH, newC) = tlstmGatesPair {n = o} combined c
      st' = { hiddenT := Just newH, cellT := Just newC } st
  in (st', newH)


----------------------------------------------------------------------
-- Constructor
----------------------------------------------------------------------

-- Pack a Vect of Doubles into a pre-allocated buffer at offset.
packDoubles : AnyPtr -> Int -> Vect k Double -> AnyPtr
packDoubles buf _ [] = buf
packDoubles buf off (x :: rest) =
  packDoubles (prim__setDouble buf off x) (off + 1) rest

-- Zero a buffer for `n` elements starting at offset.
zeroBuf : AnyPtr -> Int -> Int -> AnyPtr
zeroBuf buf _ 0 = buf
zeroBuf buf off n =
  zeroBuf (prim__setDouble buf off 0.0) (off + 1) (n - 1)

||| Build an `LstmStateV2 i o CPU` with Xavier-uniform weight init,
||| zero bias, and Nothing hidden/cell state. Weights register as C
||| params under `<prefix>_iw`, `<prefix>_rw`, `<prefix>_b`.
export
lstmLayerV2 : {i, o : Nat} -> (paramPrefix : String) ->
              IO (LstmStateV2 i o CPU)
lstmLayerV2 paramPrefix = do
  let gI = cast {to=Int} (4 * o)
      iI = cast {to=Int} i
      oI = cast {to=Int} o
  iwVals <- traverse (\_ => xavier uniform i (4 * o)) (Vect.replicate (4 * o * i) ())
  rwVals <- traverse (\_ => xavier uniform o (4 * o)) (Vect.replicate (4 * o * o) ())
  let iwBuf = prim__allocDoubles (gI * iI)
      iwBuf' = packDoubles iwBuf 0 iwVals
      rwBuf = prim__allocDoubles (gI * oI)
      rwBuf' = packDoubles rwBuf 0 rwVals
      bBuf = prim__allocDoubles gI
      bBuf' = zeroBuf bBuf 0 gI
      iwName = paramPrefix ++ "_iw"
      rwName = paramPrefix ++ "_rw"
      bName  = paramPrefix ++ "_b"
      iwPtr = prim__paramRegister iwName (prim__createParam2d gI iI iwBuf')
      rwPtr = prim__paramRegister rwName (prim__createParam2d gI oI rwBuf')
      bPtr  = prim__paramRegister bName  (prim__createParam1d gI bBuf')
      iwTV : TMat (4 * o) i CPU
      iwTV = MkTVar iwPtr (Just iwName)
      rwTV : TMat (4 * o) o CPU
      rwTV = MkTVar rwPtr (Just rwName)
      bTV : TVec (4 * o) CPU
      bTV = MkTVar bPtr (Just bName)
  pure $ MkLstmV2 iwTV rwTV bTV Nothing Nothing

||| Reset hidden/cell state to fresh zero-tensors.
export
resetLstmStateV2 : LstmStateV2 i o d -> LstmStateV2 i o d
resetLstmStateV2 = { hiddenT := Nothing, cellT := Nothing }
