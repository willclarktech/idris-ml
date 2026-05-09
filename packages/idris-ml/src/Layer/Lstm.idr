module Layer.Lstm

import Data.Vect

import Compat.Random
import Device
import Init
import Layer.Core
import Sampler
import Tensor


----------------------------------------------------------------------
-- Lstm — typed-surface LSTM cell (Path C)
----------------------------------------------------------------------
--
-- All shape-arithmetic flows through the `TVec` / `TMat` aliases in
-- `Tensor.idr`. Direct `Tensor [4 * o, ...] d` triggers an Idris 2
-- type-checker hang; `TMat (4 * o) i d` works fine because the
-- multiplication sits in a Nat-argument slot of the alias rather
-- than inside a Vect literal.

public export
record LstmState (i : Nat) (o : Nat) (0 d : Device) where
  constructor MkLstm
  iwT : TMat (4 * o) i d
  rwT : TMat (4 * o) o d
  ihB : TVec (4 * o) d        -- input-hidden bias [4*o] (b_ih)
  hhB : TVec (4 * o) d        -- hidden-hidden bias [4*o] (b_hh)
  h0T : TVec o d              -- learned initial hidden state (zero-init)
  c0T : TVec o d              -- learned initial cell state (zero-init)
  hiddenT : Maybe (TVec o d)
  cellT   : Maybe (TVec o d)


----------------------------------------------------------------------
-- Forward
----------------------------------------------------------------------

%default partial

||| Array-level LSTM cell forward. Reads (or zero-initialises) the
||| hidden + cell state, runs the fused gate computation, returns the
||| updated layer state and the new hidden output.
export
applyLstm : {o : Nat} ->
              LstmState i o d ->
              TVec i d ->
              (LstmState i o d, TVec o d)
applyLstm {o} st input =
  let h = case st.hiddenT of
            Just h => h
            Nothing => st.h0T
      c = case st.cellT of
            Just c => c
            Nothing => st.c0T
      -- Gates: nn.LSTMCell equation
      --   tanh-cell-input(W_ih @ x + b_ih + W_hh @ h + b_hh)
      -- folded into a chain of tlinear + tadd. Three FFI calls:
      --   inner    = tlinear iwT input ihB    -- W_ih @ x + b_ih
      --   combined = tlinear rwT h inner      -- W_hh @ h + (above)
      --   gates    = tadd combined hhB        -- + b_hh
      -- (Pre-2026-05-09 we had a single fused bias, saving one FFI;
      -- aligned to nn.LSTMCell's two-bias convention so the example
      -- matches what library users expect.)
      gates = tadd (tlinear st.rwT h (tlinear st.iwT input st.ihB)) st.hhB
      (newH, newC) = tlstmGatesPair {n = o} gates c
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

||| Build an `LstmState i o CPU` with Xavier-uniform weight init,
||| two zero biases (matching `nn.LSTMCell`), and learned `h0`/`c0`
||| (zero-init, learned). Weights register as C params under
||| `<prefix>_iw`, `<prefix>_rw`, `<prefix>_ib`, `<prefix>_hb`,
||| `<prefix>_h0`, `<prefix>_c0`.
export
lstmLayer : {i, o : Nat} -> (paramPrefix : String) ->
              IO (LstmState i o CPU)
lstmLayer paramPrefix = do
  let gI = cast {to=Int} (4 * o)
      iI = cast {to=Int} i
      oI = cast {to=Int} o
  iwVals <- traverse (\_ => xavier uniform i (4 * o)) (Vect.replicate (4 * o * i) ())
  rwVals <- traverse (\_ => xavier uniform o (4 * o)) (Vect.replicate (4 * o * o) ())
  let iwBuf = prim__allocDoubles (gI * iI)
      iwBuf' = packDoubles iwBuf 0 iwVals
      rwBuf = prim__allocDoubles (gI * oI)
      rwBuf' = packDoubles rwBuf 0 rwVals
      ibBuf = prim__allocDoubles gI
      ibBuf' = zeroBuf ibBuf 0 gI
      hbBuf = prim__allocDoubles gI
      hbBuf' = zeroBuf hbBuf 0 gI
      h0Buf = prim__allocDoubles oI
      h0Buf' = zeroBuf h0Buf 0 oI
      c0Buf = prim__allocDoubles oI
      c0Buf' = zeroBuf c0Buf 0 oI
      iwName = paramPrefix ++ "_iw"
      rwName = paramPrefix ++ "_rw"
      ibName = paramPrefix ++ "_ib"
      hbName = paramPrefix ++ "_hb"
      h0Name = paramPrefix ++ "_h0"
      c0Name = paramPrefix ++ "_c0"
      iwPtr = prim__paramRegister iwName (prim__createParam2d gI iI iwBuf')
      rwPtr = prim__paramRegister rwName (prim__createParam2d gI oI rwBuf')
      ibPtr = prim__paramRegister ibName (prim__createParam1d gI ibBuf')
      hbPtr = prim__paramRegister hbName (prim__createParam1d gI hbBuf')
      h0Ptr = prim__paramRegister h0Name (prim__createParam1d oI h0Buf')
      c0Ptr = prim__paramRegister c0Name (prim__createParam1d oI c0Buf')
      iwTV : TMat (4 * o) i CPU
      iwTV = MkTensor iwPtr (Just iwName)
      rwTV : TMat (4 * o) o CPU
      rwTV = MkTensor rwPtr (Just rwName)
      ibTV : TVec (4 * o) CPU
      ibTV = MkTensor ibPtr (Just ibName)
      hbTV : TVec (4 * o) CPU
      hbTV = MkTensor hbPtr (Just hbName)
      h0TV : TVec o CPU
      h0TV = MkTensor h0Ptr (Just h0Name)
      c0TV : TVec o CPU
      c0TV = MkTensor c0Ptr (Just c0Name)
  pure $ MkLstm iwTV rwTV ibTV hbTV h0TV c0TV Nothing Nothing

||| Reset hidden/cell state. Setting to `Nothing` lets `applyLstm`'s
||| first call lazy-allocate fresh persistent zero buffers — mirrors
||| V1's `resetState`, where MLX trains correctly via this lazy path.
export
resetLstmState : {o : Nat} -> {0 d : Device} -> LstmState i o d -> LstmState i o d
resetLstmState st = { hiddenT := Nothing, cellT := Nothing } st


----------------------------------------------------------------------
-- LayerLike instance — lets Lstm chain in `Network` via `~~>`
----------------------------------------------------------------------

public export
LayerLike LstmState where
  applyVar = applyLstm
  layerPrefix _ = "lstm"
  resetState = resetLstmState

||| Wrap an `LstmState` in `AnyLayer`.
export
lstmLayerAny : {i, o : Nat} -> (paramPrefix : String) -> IO (AnyLayer i o CPU)
lstmLayerAny pid = map (MkAnyLayer LstmState) (lstmLayer pid)
