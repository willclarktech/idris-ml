module Layer.Rnn

import Data.Vect

import Compat.Random
import Device
import Init
import Layer.Core
import Sampler
import Tensor


----------------------------------------------------------------------
-- Rnn — typed-surface vanilla RNN cell (Path C)
----------------------------------------------------------------------
--
-- Matches PyTorch `nn.RNNCell`'s equation:
--
--   h_t = activation( W_ih · x_t + b_ih + W_hh · h_{t-1} + b_hh )
--
-- The `activation` field is a generic `TVec o d -> TVec o d` so any
-- unary tensor function works — typically `ttanh` (default) or `trelu`,
-- but `id` for a linear-recurrence variant or any custom nonlinearity.
-- PyTorch's `nn.RNN` only takes `'tanh'`/`'relu'`; we're more flexible.
--
-- Two separate biases (`ihB`, `hhB`) match PyTorch's storage convention.
-- Mathematically a single bias would suffice, but keeping them separate
-- matches `nn.RNNCell`'s checkpoint format and lets users inspect
-- input-vs-recurrent contributions independently.
--
-- Uses `TMat` / `TVec` aliases for consistency with `Lstm` even
-- though shape arithmetic isn't needed here (no `4 *`).

public export
record RnnState (i : Nat) (o : Nat) (0 d : Device) (0 dt : DType) (0 g : GradMode) where
  constructor MkRnn
  iwT : TMat o i d dt g         -- W_ih [o, i]
  rwT : TMat o o d dt g         -- W_hh [o, o]
  ihB : TVec o d dt g           -- input-hidden bias [o]
  hhB : TVec o d dt g           -- hidden-hidden bias [o]
  activation : {0 g' : GradMode} -> TVec o d dt g' -> IO (TVec o d dt g')
  prevOutT : Maybe (TVec o d dt g)


----------------------------------------------------------------------
-- Forward
----------------------------------------------------------------------

%default partial

export
applyRnn : {0 d : Device} -> UserDeviceTape d => UserDeviceCore d => RuntimeDType dt => {o : Nat} ->
             RnnState i o d dt g ->
             TVec i d dt g ->
             IO (RnnState i o d dt g, TVec o d dt g)
applyRnn {o} st input = do
  p <- case st.prevOutT of
         Just po => pure po
         Nothing => tzeroState1d {n = o}
  -- nn.RNNCell equation: activation(W_ih @ x + b_ih + W_hh @ h + b_hh).
  inner    <- tlinear st.iwT input st.ihB
  combined <- tlinear st.rwT p inner
  preact   <- tadd combined st.hhB
  out      <- st.activation preact
  pure ({ prevOutT := Just out } st, out)


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

||| Build an `RnnState i o TapeDev` with Xavier-uniform weights, zero
||| biases, and the given activation function. State starts as
||| Nothing; first `applyRnn` call zero-initialises it. Params
||| register under `<prefix>_iw`, `<prefix>_rw`, `<prefix>_ib`,
||| `<prefix>_hb`.
|||
||| Common activations: `ttanh` (default for `nn.RNN`), `trelu`,
||| `id` for a linear-recurrence variant.
export
rnnLayer : UserDeviceTape d => RuntimeDType dt => {i, o : Nat} ->
             (paramPrefix : String) ->
             (activation : {0 g' : GradMode} -> TVec o d dt g' -> IO (TVec o d dt g')) ->
             IO (RnnState i o d dt WithGrad)
rnnLayer paramPrefix activation = do
  let oI = cast {to=Int} o
      iI = cast {to=Int} i
  iwVals <- traverse (\_ => xavier uniform i o) (Vect.replicate (o * i) ())
  rwVals <- traverse (\_ => xavier uniform o o) (Vect.replicate (o * o) ())
  let iwBuf = prim__allocDoubles (oI * iI)
      iwBuf' = packDoubles iwBuf 0 iwVals
      rwBuf = prim__allocDoubles (oI * oI)
      rwBuf' = packDoubles rwBuf 0 rwVals
      ibBuf = prim__allocDoubles oI
      ibBuf' = zeroBuf ibBuf 0 oI
      hbBuf = prim__allocDoubles oI
      hbBuf' = zeroBuf hbBuf 0 oI
      iwName = paramPrefix ++ "_iw"
      rwName = paramPrefix ++ "_rw"
      ibName = paramPrefix ++ "_ib"
      hbName = paramPrefix ++ "_hb"
      iwPtr = primParamRegister {d} iwName (dtCreateParam2d {d} {t=dt} oI iI iwBuf' (deviceStreamTag {d}))
      rwPtr = primParamRegister {d} rwName (dtCreateParam2d {d} {t=dt} oI oI rwBuf' (deviceStreamTag {d}))
      ibPtr = primParamRegister {d} ibName (dtCreateParam1d {d} {t=dt} oI ibBuf' (deviceStreamTag {d}))
      hbPtr = primParamRegister {d} hbName (dtCreateParam1d {d} {t=dt} oI hbBuf' (deviceStreamTag {d}))
      iwTV : TMat o i d dt WithGrad
      iwTV = MkTensor iwPtr (Just iwName)
      rwTV : TMat o o d dt WithGrad
      rwTV = MkTensor rwPtr (Just rwName)
      ibTV : TVec o d dt WithGrad
      ibTV = MkTensor ibPtr (Just ibName)
      hbTV : TVec o d dt WithGrad
      hbTV = MkTensor hbPtr (Just hbName)
  pure $ MkRnn iwTV rwTV ibTV hbTV activation Nothing

||| Reset hidden state. Lazy-allocate on next applyVar call.
export
resetRnnState : {o : Nat} -> {0 d : Device} -> {0 g : GradMode} -> RnnState i o d dt g -> RnnState i o d dt g
resetRnnState st = { prevOutT := Nothing } st


----------------------------------------------------------------------
-- LayerLike instance
----------------------------------------------------------------------

public export
LayerLike RnnState where
  applyVar = applyRnn
  layerPrefix _ = "rnn"

  resetState = resetRnnState

  freezeLayer (MkRnn iw rw ihB hhB act prev) = do
    iw'  <- weakenGrad iw
    rw'  <- weakenGrad rw
    ihB' <- weakenGrad ihB
    hhB' <- weakenGrad hhB
    prev' <- case prev of
      Nothing => pure Nothing
      Just p  => Just <$> weakenGrad p
    pure (MkRnn iw' rw' ihB' hhB' act prev')

  unfreezeLayer (MkRnn iw rw ihB hhB act prev) = do
    primIO (primSetRequiresGrad {d} iw.tensorPtr 1)
    primIO (primSetRequiresGrad {d} rw.tensorPtr 1)
    primIO (primSetRequiresGrad {d} ihB.tensorPtr 1)
    primIO (primSetRequiresGrad {d} hhB.tensorPtr 1)
    case prev of
      Nothing => pure ()
      Just p  => primIO (primSetRequiresGrad {d} p.tensorPtr 1)
    pure (MkRnn (retypeGrad iw) (retypeGrad rw)
                (retypeGrad ihB) (retypeGrad hhB)
                act
                (map retypeGrad prev))

||| Wrap an `RnnState` in `AnyLayer`. Defaults activation to `ttanh`
||| (matching PyTorch's `nn.RNN` default). Use `rnnLayer` directly
||| if you need a different activation.
export
rnnLayerAny : {0 d : Device} -> UserDeviceTape d => RuntimeDType dt =>
              {i, o : Nat} -> (paramPrefix : String) -> IO (AnyLayer i o d dt WithGrad)
rnnLayerAny pid = map (MkAnyLayer RnnState) (rnnLayer pid ttanh)
