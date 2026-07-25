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
module Ml.Nn.Recurrent

import Control.Linear.LIO
import Data.Linear
import Data.Vect

import Ml.Executor
import Ml.Nn.Init
import Ml.Nn.Module
import Ml.Tensor

%default total

||| Stateful per-timestep layer — the per-timestep capability on top of
||| `Params`. `recurStep` advances one timestep: it consumes the layer `(1 _)`
||| and returns the rebuilt layer (carrying the new hidden state) beside the
||| output under the `(!*)` bang; `recurReset` clears the hidden state (next
||| step lazily re-initialises to zeros). A recurrent layer is a single-owner
||| resource threaded step by step — the natural fine-grained spot.
public export
interface Params l => Recurrent (l : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type) where
  recurStep : {0 ex : Executor} -> Backend ex dt => {i, o : Nat} ->
              (1 _ : l i o ex dt WithGrad) -> Tensor [i] ex dt WithGrad ->
              L IO {use=1} (LPair (!* (Tensor [o] ex dt WithGrad)) (l i o ex dt WithGrad))
  recurReset : {0 ex : Executor} -> {0 dt : DType} -> {0 g : GradMode} -> {i, o : Nat} ->
               (1 _ : l i o ex dt g) -> l i o ex dt g

----------------------------------------------------------------------
-- RNN (vanilla nn.RNNCell): h_t = act(W_ih·x + b_ih + W_hh·h_{t-1} + b_hh)
----------------------------------------------------------------------

||| Vanilla RNN cell. Weights are `WithGrad` params; `prevOutT` is the
||| carried hidden state (a `WithGrad` activation, `Nothing` until the
||| first step); `activation` is any unary tensor fn (typically `ttanh`).
public export
record Rnn (i : Nat) (o : Nat) (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkRnn
  iwT        : TMat o i ex dt g
  rwT        : TMat o o ex dt g
  ihB        : TVec o ex dt g
  hhB        : TVec o ex dt g
  activation : {0 g' : GradMode} -> TVec o ex dt g' -> IO (TVec o ex dt g')
  prevOutT   : Maybe (TVec o ex dt g)

||| Params for `Rnn`. Fields bind at ω, so the weight tensors feed both the
||| reflected param list and the rebuild.
public export
Params Rnn where
  params (MkRnn iw rw ib hb _ _)       = [toParam iw, toParam rw, toParam ib, toParam hb]
  reflect (MkRnn iw rw ib hb act prev) =
    MkBang [toParam iw, toParam rw, toParam ib, toParam hb] # MkRnn iw rw ib hb act prev
  castGrad (MkRnn iw rw ib hb act prev) =
    MkRnn (retypeGrad iw) (retypeGrad rw) (retypeGrad ib) (retypeGrad hb) act (map retypeGrad prev)
  discard (MkRnn _ _ _ _ _ _) = pure ()

||| Recurrent step: consume the cell, advance one timestep, return the
||| (unrestricted, banged) output beside the rebuilt cell carrying the new
||| hidden state. Body sequences the `L IO` tensor ops directly; only the
||| user-supplied IO activation (`act`) is lifted via `liftIO1`.
public export
Recurrent Rnn where
  recurStep {o} (MkRnn iw rw ib hb act prev) input = do
    p <- the (L IO (TVec o ex dt WithGrad)) $ case prev of
           Just po => pure po
           Nothing => tzeroState1dL {n = o}
    inner    <- tlinearL iw input ib
    combined <- tlinearL rw p inner
    preact   <- taddL combined hb
    out      <- liftIO1 (act preact)
    pure1 (MkBang out # MkRnn iw rw ib hb act (Just out))
  recurReset (MkRnn iw rw ib hb act _) = MkRnn iw rw ib hb act Nothing

||| Construct an `Rnn i o` inside an `Init` derivation. Xavier-uniform
||| weights, `U(±√(6/(fan_in+fan_out)))`, zero biases, hidden state empty.
||| Registers PyTorch RNNCell names
||| `<scope>.rnn_<n>.{weight_ih,weight_hh,bias_ih,bias_hh}`.
|||
||| Matches `nn.init.xavier_uniform_` on the paired reference's
||| `LinearRNNCell`. Until 2026-07-31 these were normals of the same
||| variance (`√(2/(fan_in+fan_out))` is exactly the uniform's std), so the
||| tails diverged even though the spreads agreed. `Uniform` also routes
||| through the host-buffer fill, making the init identical across tape /
||| torch / mlx by construction.
export
rnn : KnownGrad g => {0 ex : Executor} -> Backend ex dt => {i, o : Nat} ->
      (activation : {0 g' : GradMode} -> TVec o ex dt g' -> IO (TVec o ex dt g')) ->
      Init (Rnn i o ex dt g)
rnn activation = do
  name <- freshChild "rnn"
  let iwB = sqrt (6.0 / cast {to=Double} (i + o))
      rwB = sqrt (6.0 / cast {to=Double} (o + o))
  iw <- liftIO $ param {ex} {dt} {dims=[o, i]} (name ++ ".weight_ih") (Uniform (-iwB) iwB)
  rw <- liftIO $ param {ex} {dt} {dims=[o, o]} (name ++ ".weight_hh") (Uniform (-rwB) rwB)
  ib <- liftIO $ tparam1dConst  {ex} {dt} {n=o}       (name ++ ".bias_ih")   0.0
  hb <- liftIO $ tparam1dConst  {ex} {dt} {n=o}       (name ++ ".bias_hh")   0.0
  case sgrad {g} of
    SWithGrad => pure (MkRnn iw rw ib hb activation Nothing)
    SNoGrad   => do iw' <- liftIO (weakenGrad iw); rw' <- liftIO (weakenGrad rw)
                    ib' <- liftIO (weakenGrad ib); hb' <- liftIO (weakenGrad hb)
                    pure (MkRnn iw' rw' ib' hb' activation Nothing)
