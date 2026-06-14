||| `Recurrent` — the shared interface for stateful per-timestep layers
||| (the design choice for the recurrent set: a uniform `step`/`reset` so
||| RNN/LSTM/GRU/… dispatch the same way, rather than ad-hoc per-layer
||| forwards). Recurrent layers are NOT batched `Module`s: their forward is
||| a 1-D per-timestep step that carries hidden state across calls. The
||| state lives IN the layer record (as the legacy did), so `recurStep`
||| returns an updated layer alongside the output, and `recurReset` clears
||| it. `recurStep` is `WithGrad`-pinned — recurrent layers are trained via
||| BPTT; inference is the niche case (retype the input).
|||
||| This module also holds the RNN port (the exemplar); LSTM/GRU/NTM/DNC
||| follow the same shape.
module Nn.Recurrent

import Data.Vect

import Executor
import Tensor
import Nn.Init
import Nn.Module

%default total

||| Stateful per-timestep layer. `recurStep` advances one timestep
||| (threading hidden state through the returned layer); `recurReset`
||| clears the hidden state (next step lazily re-initialises to zeros).
public export
interface Recurrent (l : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type) where
  recurStep : {0 ex : Executor} -> Backend ex dt => {i, o : Nat} ->
              l i o ex dt WithGrad -> Tensor [i] ex dt WithGrad ->
              IO (l i o ex dt WithGrad, Tensor [o] ex dt WithGrad)
  recurReset : {0 ex : Executor} -> {0 dt : DType} -> {0 g : GradMode} -> {i, o : Nat} ->
               l i o ex dt g -> l i o ex dt g

----------------------------------------------------------------------
-- RNN (vanilla nn.RNNCell): h_t = act(W_ih·x + b_ih + W_hh·h_{t-1} + b_hh)
----------------------------------------------------------------------

||| Vanilla RNN cell. Weights are `WithGrad` params; `prevOutT` is the
||| carried hidden state (a `WithGrad` activation, `Nothing` until the
||| first step); `activation` is any unary tensor fn (typically `ttanh`).
public export
record Rnn (i : Nat) (o : Nat) (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkRnn
  iwT : TMat o i ex dt g
  rwT : TMat o o ex dt g
  ihB : TVec o ex dt g
  hhB : TVec o ex dt g
  activation : {0 g' : GradMode} -> TVec o ex dt g' -> IO (TVec o ex dt g')
  prevOutT : Maybe (TVec o ex dt g)

public export
Params Rnn where
  params (MkRnn iw rw ib hb _ _) = [toParam iw, toParam rw, toParam ib, toParam hb]
  castGrad (MkRnn iw rw ib hb act prev) =
    MkRnn (retypeGrad iw) (retypeGrad rw) (retypeGrad ib) (retypeGrad hb) act (map retypeGrad prev)

public export
Recurrent Rnn where
  recurStep {o} st input = do
    p <- case st.prevOutT of
           Just po => pure po
           Nothing => tzeroState1d {n = o}
    inner    <- tlinear st.iwT input st.ihB
    combined <- tlinear st.rwT p inner
    preact   <- tadd combined st.hhB
    out      <- st.activation preact
    pure ({ prevOutT := Just out } st, out)

  recurReset st = { prevOutT := Nothing } st

||| Construct an `Rnn i o` inside an `Init` derivation. Xavier-ish weight
||| init (W_ih std √(2/(i+o)), W_hh std 1/√o), zero biases, hidden state
||| empty. Registers PyTorch RNNCell names
||| `<scope>.rnn_<n>.{weight_ih,weight_hh,bias_ih,bias_hh}`.
export
rnn : {0 ex : Executor} -> Backend ex dt => {i, o : Nat} ->
      (activation : {0 g' : GradMode} -> TVec o ex dt g' -> IO (TVec o ex dt g')) ->
      Init (Rnn i o ex dt WithGrad)
rnn activation = do
  name <- freshChild "rnn"
  let iwStd = sqrt (2.0 / cast {to=Double} (i + o))
      rwStd = 1.0 / sqrt (cast {to=Double} o)
  iw <- liftIO $ tparam2dNormal {ex} {dt} {o} {i}     (name ++ ".weight_ih") 0.0 iwStd
  rw <- liftIO $ tparam2dNormal {ex} {dt} {o} {i=o}   (name ++ ".weight_hh") 0.0 rwStd
  ib <- liftIO $ tparam1dConst  {ex} {dt} {n=o}       (name ++ ".bias_ih")   0.0
  hb <- liftIO $ tparam1dConst  {ex} {dt} {n=o}       (name ++ ".bias_hh")   0.0
  pure (MkRnn iw rw ib hb activation Nothing)
