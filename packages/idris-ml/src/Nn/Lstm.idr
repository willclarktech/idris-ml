||| `Lstm` — LSTM cell on the v1 `Nn` surface, implementing `Recurrent`.
||| Four gates stacked along axis 0 (weights `[4·o, …]`, routed through the
||| `TMat`/`TVec` aliases to dodge the multiplicative-Nat type-checker
||| hang). Learned initial states `h0`/`c0` (zero-init, trainable); carried
||| `hiddenT`/`cellT` state lives in the record. `params` lists the six
||| learnable tensors (weights, biases, h0, c0); the carried state is not a
||| param.
module Nn.Lstm

import Data.Vect

import Executor
import Nn.Init
import Nn.Module
import Nn.Recurrent
import Tensor

%default total

||| LSTM cell. Weights/biases/h0/c0 are `WithGrad` params; hiddenT/cellT
||| are the carried state (`Nothing` until the first step).
public export
record Lstm (i : Nat) (o : Nat) (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkLstm
  iwT : TMat (4 * o) i ex dt g
  rwT : TMat (4 * o) o ex dt g
  ihB : TVec (4 * o) ex dt g
  hhB : TVec (4 * o) ex dt g
  h0T : TVec o ex dt g
  c0T : TVec o ex dt g
  hiddenT : Maybe (TVec o ex dt g)
  cellT   : Maybe (TVec o ex dt g)

public export
Params Lstm where
  params (MkLstm iw rw ib hb h0 c0 _ _) =
    [toParam iw, toParam rw, toParam ib, toParam hb, toParam h0, toParam c0]
  castGrad (MkLstm iw rw ib hb h0 c0 hid cell) =
    MkLstm (retypeGrad iw) (retypeGrad rw) (retypeGrad ib) (retypeGrad hb)
           (retypeGrad h0) (retypeGrad c0) (map retypeGrad hid) (map retypeGrad cell)

public export
Recurrent Lstm where
  recurStep {o} st input = do
    let h = case st.hiddenT of Just h => h; Nothing => st.h0T
    let c = case st.cellT   of Just c => c; Nothing => st.c0T
    inner    <- tlinear st.iwT input st.ihB
    combined <- tlinear st.rwT h inner
    gates    <- tadd combined st.hhB
    (newH, newC) <- tlstmGatesPair {n = o} gates c
    pure ({ hiddenT := Just newH, cellT := Just newC } st, newH)

  recurReset st = { hiddenT := Nothing, cellT := Nothing } st

||| Construct an `Lstm i o` inside an `Init` derivation. Xavier-ish weight
||| init (4 stacked gates → fan_out 4·o), zero biases + learned h0/c0,
||| empty state. Registers `<scope>.lstm_<n>.{weight,bias}_{ih,hh}` +
||| `.h0` / `.c0`.
export
lstm : {0 ex : Executor} -> Backend ex dt => {i, o : Nat} -> Init (Lstm i o ex dt WithGrad)
lstm = do
  name <- freshChild "lstm"
  let iwStd = sqrt (2.0 / cast {to=Double} (i + 4 * o))
      rwStd = sqrt (2.0 / cast {to=Double} (o + 4 * o))
  iw <- liftIO $ tparam2dNormal {ex} {dt} {o = 4 * o} {i}     (name ++ ".weight_ih") 0.0 iwStd
  rw <- liftIO $ tparam2dNormal {ex} {dt} {o = 4 * o} {i = o} (name ++ ".weight_hh") 0.0 rwStd
  ib <- liftIO $ tparam1dConst  {ex} {dt} {n = 4 * o} (name ++ ".bias_ih") 0.0
  hb <- liftIO $ tparam1dConst  {ex} {dt} {n = 4 * o} (name ++ ".bias_hh") 0.0
  h0 <- liftIO $ tparam1dConst  {ex} {dt} {n = o}     (name ++ ".h0") 0.0
  c0 <- liftIO $ tparam1dConst  {ex} {dt} {n = o}     (name ++ ".c0") 0.0
  pure (MkLstm iw rw ib hb h0 c0 Nothing Nothing)
