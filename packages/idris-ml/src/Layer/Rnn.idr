module Layer.Rnn

import Data.Vect

import Executor
import Layer.Core
import Tensor


----------------------------------------------------------------------
-- Rnn — typed-surface vanilla RNN cell (Path C)
----------------------------------------------------------------------
--
-- Matches PyTorch `nn.RNNCell`'s equation:
--
--   h_t = activation( W_ih · x_t + b_ih + W_hh · h_{t-1} + b_hh )
--
-- The `activation` field is a generic `TVec o ex -> TVec o ex` so any
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
record RnnState (i : Nat) (o : Nat) (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkRnn
  iwT : TMat o i ex dt g         -- W_ih [o, i]
  rwT : TMat o o ex dt g         -- W_hh [o, o]
  ihB : TVec o ex dt g           -- input-hidden bias [o]
  hhB : TVec o ex dt g           -- hidden-hidden bias [o]
  activation : {0 g' : GradMode} -> TVec o ex dt g' -> IO (TVec o ex dt g')
  prevOutT : Maybe (TVec o ex dt g)


----------------------------------------------------------------------
-- Forward
----------------------------------------------------------------------

%default partial

export
applyRnn : {0 ex : Executor} -> UserExecutorTraining ex => UserExecutorCore ex => RuntimeDType dt => Linked ex => Compatible ex dt => {o : Nat} ->
             RnnState i o ex dt g ->
             TVec i ex dt g ->
             IO (RnnState i o ex dt g, TVec o ex dt g)
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

||| Build an `RnnState i o TapeExecutor` with Xavier-uniform weights, zero
||| biases, and the given activation function. State starts as
||| Nothing; first `applyRnn` call zero-initialises it. Params
||| register under `<prefix>_iw`, `<prefix>_rw`, `<prefix>_ib`,
||| `<prefix>_hb`.
|||
||| Common activations: `ttanh` (default for `nn.RNN`), `trelu`,
||| `id` for a linear-recurrence variant.
export
rnnLayer : UserExecutorTraining ex => RuntimeDType dt => Linked ex => Compatible ex dt => {i, o : Nat} ->
             (paramPrefix : String) ->
             (activation : {0 g' : GradMode} -> TVec o ex dt g' -> IO (TVec o ex dt g')) ->
             IO (RnnState i o ex dt WithGrad)
rnnLayer paramPrefix activation = do
  -- Xavier-normal-via-uniform for weights:
  --   input weight  W_ih: fan_in=i, fan_out=o → std = sqrt(2/(i+o))
  --   hidden weight W_hh: fan_in=o, fan_out=o → std = 1/sqrt(o)
  -- Zero bias init.
  let iwStd = sqrt (2.0 / cast {to=Double} (i + o))
      rwStd = 1.0 / sqrt (cast {to=Double} o)
      iwName = paramPrefix ++ "_iw"
      rwName = paramPrefix ++ "_rw"
      ibName = paramPrefix ++ "_ib"
      hbName = paramPrefix ++ "_hb"
  iw <- tparam2dNormal {o} {i} iwName 0.0 iwStd
  rw <- tparam2dNormal {o} {i=o} rwName 0.0 rwStd
  ib <- tparam1dConst {n=o} ibName 0.0
  hb <- tparam1dConst {n=o} hbName 0.0
  pure $ MkRnn iw rw ib hb activation Nothing

||| Reset hidden state. Lazy-allocate on next applyVar call.
export
resetRnnState : {o : Nat} -> {0 ex : Executor} -> {0 g : GradMode} -> RnnState i o ex dt g -> RnnState i o ex dt g
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
    primIO (primSetRequiresGrad {ex} iw.tensorPtr 1)
    primIO (primSetRequiresGrad {ex} rw.tensorPtr 1)
    primIO (primSetRequiresGrad {ex} ihB.tensorPtr 1)
    primIO (primSetRequiresGrad {ex} hhB.tensorPtr 1)
    case prev of
      Nothing => pure ()
      Just p  => primIO (primSetRequiresGrad {ex} p.tensorPtr 1)
    pure (MkRnn (retypeGrad iw) (retypeGrad rw)
                (retypeGrad ihB) (retypeGrad hhB)
                act
                (map retypeGrad prev))

||| Wrap an `RnnState` in `AnyLayer`. Defaults activation to `ttanh`
||| (matching PyTorch's `nn.RNN` default). Use `rnnLayer` directly
||| if you need a different activation.
export
rnnLayerAny : {0 ex : Executor} -> UserExecutorTraining ex => RuntimeDType dt => Linked ex => Compatible ex dt =>
              {i, o : Nat} -> (paramPrefix : String) -> IO (AnyLayer i o ex dt WithGrad)
rnnLayerAny pid = map (MkAnyLayer RnnState) (rnnLayer pid ttanh)
