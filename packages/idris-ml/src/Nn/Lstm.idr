||| `Lstm` — LSTM cell on the v1 `Nn` surface, implementing `Recurrent`.
||| Four gates stacked along axis 0 (weights `[4·o, …]`, routed through the
||| `TMat`/`TVec` aliases to dodge the multiplicative-Nat type-checker
||| hang). Learned initial states `h0`/`c0` (zero-init, trainable); carried
||| `hiddenT`/`cellT` state lives in the record. `params` lists the six
||| learnable tensors (weights, biases, h0, c0); the carried state is not a
||| param.
module Nn.Lstm

import Control.Linear.LIO
import Data.Linear
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
  iwT     : TMat (4 * o) i ex dt g
  rwT     : TMat (4 * o) o ex dt g
  ihB     : TVec (4 * o) ex dt g
  hhB     : TVec (4 * o) ex dt g
  h0T     : TVec o ex dt g
  c0T     : TVec o ex dt g
  hiddenT : Maybe (TVec o ex dt g)
  cellT   : Maybe (TVec o ex dt g)

||| Params: six learnable tensors; carried state is not a param. Fields bind
||| at ω, free to reflect *and* rebuild.
public export
Params Lstm where
  params (MkLstm iw rw ib hb h0 c0 hid cell) =
    [toParam iw, toParam rw, toParam ib, toParam hb, toParam h0, toParam c0]
  reflect (MkLstm iw rw ib hb h0 c0 hid cell) =
    MkBang [toParam iw, toParam rw, toParam ib, toParam hb, toParam h0, toParam c0]
      # MkLstm iw rw ib hb h0 c0 hid cell
  castGrad (MkLstm iw rw ib hb h0 c0 hid cell) =
    MkLstm (retypeGrad iw) (retypeGrad rw) (retypeGrad ib) (retypeGrad hb)
           (retypeGrad h0) (retypeGrad c0) (map retypeGrad hid) (map retypeGrad cell)
  discard (MkLstm _ _ _ _ _ _ _ _) = pure ()

||| Recurrent step. Sequences the `L IO` gate ops directly;
||| `tlstmGatesPairL` yields `(newH, newC)` (both unrestricted) — `newH` is the
||| output, both update the carried state in the rebuilt cell.
public export
Recurrent Lstm where
  recurStep {o} (MkLstm iw rw ib hb h0 c0 hid cell) input = do
    let h = case hid  of Just h => h; Nothing => h0
    let c = case cell of Just c => c; Nothing => c0
    inner        <- tlinearL iw input ib
    combined     <- tlinearL rw h inner
    gates        <- taddL combined hb
    (newH, newC) <- tlstmGatesPairL {n = o} gates c
    pure1 (MkBang newH # MkLstm iw rw ib hb h0 c0 (Just newH) (Just newC))
  recurReset (MkLstm iw rw ib hb h0 c0 _ _) = MkLstm iw rw ib hb h0 c0 Nothing Nothing

||| Step the LSTM in plain `IO` (ω in / ω out), bridging the linear `recurStep`
||| via `run` + a constructor match (matching the returned cell binds its fields
||| at ω, so it can be rebuilt as an unrestricted value). For composite layers
||| (NTM/DNC) that thread their controller internally at ω while keeping the
||| outer cell the single-owner linear resource.
export
lstmStepIO : {0 ex : Executor} -> Backend ex dt => {i, o : Nat} ->
             Lstm i o ex dt WithGrad -> TVec i ex dt WithGrad ->
             IO (Lstm i o ex dt WithGrad, TVec o ex dt WithGrad)
lstmStepIO st input = run (do
  (MkBang hv # MkLstm iw rw ib hb h0 c0 hid cell) <- recurStep st input
  pure (MkLstm iw rw ib hb h0 c0 hid cell, hv))

||| Construct an `Lstm i o` inside an `Init` derivation. Xavier-ish weight
||| init (4 stacked gates → fan_out 4·o), zero biases + learned h0/c0,
||| empty state. Registers `<scope>.lstm_<n>.{weight,bias}_{ih,hh}` +
||| `.h0` / `.c0`.
export
lstm : KnownGrad g => {0 ex : Executor} -> Backend ex dt => {i, o : Nat} -> Init (Lstm i o ex dt g)
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
  case sgrad {g} of
    SWithGrad => pure (MkLstm iw rw ib hb h0 c0 Nothing Nothing)
    SNoGrad   => do iw' <- liftIO (weakenGrad iw); rw' <- liftIO (weakenGrad rw)
                    ib' <- liftIO (weakenGrad ib); hb' <- liftIO (weakenGrad hb)
                    h0' <- liftIO (weakenGrad h0); c0' <- liftIO (weakenGrad c0)
                    pure (MkLstm iw' rw' ib' hb' h0' c0' Nothing Nothing)
