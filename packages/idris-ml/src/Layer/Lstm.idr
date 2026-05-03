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
record LstmState (i : Nat) (o : Nat) (0 d : Device) (0 dt : DType) (0 g : GradMode) where
  constructor MkLstm
  iwT : TMat (4 * o) i d dt g
  rwT : TMat (4 * o) o d dt g
  ihB : TVec (4 * o) d dt g        -- input-hidden bias [4*o] (b_ih)
  hhB : TVec (4 * o) d dt g        -- hidden-hidden bias [4*o] (b_hh)
  h0T : TVec o d dt g              -- learned initial hidden state (zero-init)
  c0T : TVec o d dt g              -- learned initial cell state (zero-init)
  hiddenT : Maybe (TVec o d dt g)
  cellT   : Maybe (TVec o d dt g)


----------------------------------------------------------------------
-- Forward
----------------------------------------------------------------------

%default partial

||| Array-level LSTM cell forward. Reads (or zero-initialises) the
||| hidden + cell state, runs the fused gate computation, returns the
||| updated layer state and the new hidden output.
export
applyLstm : {0 d : Device} -> UserDeviceTape d => UserDeviceCore d => RuntimeDType dt => {o : Nat} ->
              LstmState i o d dt g ->
              TVec i d dt g ->
              IO (LstmState i o d dt g, TVec o d dt g)
applyLstm {o} st input = do
  let h = case st.hiddenT of
            Just h => h
            Nothing => st.h0T
  let c = case st.cellT of
            Just c => c
            Nothing => st.c0T
  inner    <- tlinear st.iwT input st.ihB
  combined <- tlinear st.rwT h inner
  gates    <- tadd combined st.hhB
  (newH, newC) <- tlstmGatesPair {n = o} gates c
  let st' = { hiddenT := Just newH, cellT := Just newC } st
  pure (st', newH)


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

||| Build an `LstmState i o TapeDev` with Xavier-uniform weight init,
||| two zero biases (matching `nn.LSTMCell`), and learned `h0`/`c0`
||| (zero-init, learned). Weights register as C params under
||| `<prefix>_iw`, `<prefix>_rw`, `<prefix>_ib`, `<prefix>_hb`,
||| `<prefix>_h0`, `<prefix>_c0`.
export
lstmLayer : UserDeviceTape d => RuntimeDType dt => {i, o : Nat} -> (paramPrefix : String) ->
              IO (LstmState i o d dt WithGrad)
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
      iwPtr = primParamRegister {d} iwName (dtCreateParam2d {t=dt} gI iI iwBuf' (deviceStreamTag {d}))
      rwPtr = primParamRegister {d} rwName (dtCreateParam2d {t=dt} gI oI rwBuf' (deviceStreamTag {d}))
      ibPtr = primParamRegister {d} ibName (dtCreateParam1d {t=dt} gI ibBuf' (deviceStreamTag {d}))
      hbPtr = primParamRegister {d} hbName (dtCreateParam1d {t=dt} gI hbBuf' (deviceStreamTag {d}))
      h0Ptr = primParamRegister {d} h0Name (dtCreateParam1d {t=dt} oI h0Buf' (deviceStreamTag {d}))
      c0Ptr = primParamRegister {d} c0Name (dtCreateParam1d {t=dt} oI c0Buf' (deviceStreamTag {d}))
      iwTV : TMat (4 * o) i d dt WithGrad
      iwTV = MkTensor iwPtr (Just iwName)
      rwTV : TMat (4 * o) o d dt WithGrad
      rwTV = MkTensor rwPtr (Just rwName)
      ibTV : TVec (4 * o) d dt WithGrad
      ibTV = MkTensor ibPtr (Just ibName)
      hbTV : TVec (4 * o) d dt WithGrad
      hbTV = MkTensor hbPtr (Just hbName)
      h0TV : TVec o d dt WithGrad
      h0TV = MkTensor h0Ptr (Just h0Name)
      c0TV : TVec o d dt WithGrad
      c0TV = MkTensor c0Ptr (Just c0Name)
  pure $ MkLstm iwTV rwTV ibTV hbTV h0TV c0TV Nothing Nothing

||| Reset hidden/cell state. Setting to `Nothing` lets `applyLstm`'s
||| first call lazy-allocate fresh persistent zero buffers — mirrors
||| V1's `resetState`, where MLX trains correctly via this lazy path.
export
resetLstmState : {o : Nat} -> {0 d : Device} -> {0 g : GradMode} -> LstmState i o d dt g -> LstmState i o d dt g
resetLstmState st = { hiddenT := Nothing, cellT := Nothing } st


----------------------------------------------------------------------
-- LayerLike instance — lets Lstm chain in `Network` via `~~>`
----------------------------------------------------------------------

public export
LayerLike LstmState where
  applyVar = applyLstm
  layerPrefix _ = "lstm"
  resetState = resetLstmState

  freezeLayer (MkLstm iw rw ihB hhB h0 c0 hid cell) = do
    iw'  <- weakenGrad iw
    rw'  <- weakenGrad rw
    ihB' <- weakenGrad ihB
    hhB' <- weakenGrad hhB
    h0'  <- weakenGrad h0
    c0'  <- weakenGrad c0
    hid' <- case hid of
      Nothing => pure Nothing
      Just h  => Just <$> weakenGrad h
    cell' <- case cell of
      Nothing => pure Nothing
      Just c  => Just <$> weakenGrad c
    pure (MkLstm iw' rw' ihB' hhB' h0' c0' hid' cell')

  unfreezeLayer (MkLstm iw rw ihB hhB h0 c0 hid cell) = do
    primIO (primSetRequiresGrad {d} iw.tensorPtr 1)
    primIO (primSetRequiresGrad {d} rw.tensorPtr 1)
    primIO (primSetRequiresGrad {d} ihB.tensorPtr 1)
    primIO (primSetRequiresGrad {d} hhB.tensorPtr 1)
    primIO (primSetRequiresGrad {d} h0.tensorPtr 1)
    primIO (primSetRequiresGrad {d} c0.tensorPtr 1)
    case hid of
      Nothing => pure ()
      Just h  => primIO (primSetRequiresGrad {d} h.tensorPtr 1)
    case cell of
      Nothing => pure ()
      Just c  => primIO (primSetRequiresGrad {d} c.tensorPtr 1)
    pure (MkLstm (retypeGrad iw) (retypeGrad rw)
                 (retypeGrad ihB) (retypeGrad hhB)
                 (retypeGrad h0) (retypeGrad c0)
                 (map retypeGrad hid) (map retypeGrad cell))

||| Wrap an `LstmState` in `AnyLayer`.
export
lstmLayerAny : UserDeviceTape d => RuntimeDType dt => {i, o : Nat} -> (paramPrefix : String) -> IO (AnyLayer i o d dt WithGrad)
lstmLayerAny pid = map (MkAnyLayer LstmState) (lstmLayer pid)
