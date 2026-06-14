||| `Gru` — GRU cell on the v1 `Nn` surface, implementing `Recurrent`.
||| Three gates (reset/update/new) stacked along axis 0 (weights `[3·o, …]`,
||| via the `TMat`/`TVec` aliases); the gate mixing happens inside the fused
||| `tgruCell`. Carried `hiddenT` state lives in the record. `params` lists
||| the four learnable tensors.
module Nn.Gru

import Data.Vect

import Executor
import Nn.Init
import Nn.Module
import Nn.Recurrent
import Tensor

%default total

||| GRU cell. Weights/biases are `WithGrad` params; hiddenT is the carried
||| state (`Nothing` until the first step).
public export
record Gru (i : Nat) (o : Nat) (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkGru
  iwT     : TMat (3 * o) i ex dt g
  ihB     : TVec (3 * o) ex dt g
  hwT     : TMat (3 * o) o ex dt g
  hhB     : TVec (3 * o) ex dt g
  hiddenT : Maybe (TVec o ex dt g)

public export
Params Gru where
  params (MkGru iw ib hw hb _) = [toParam iw, toParam ib, toParam hw, toParam hb]
  castGrad (MkGru iw ib hw hb hid) =
    MkGru (retypeGrad iw) (retypeGrad ib) (retypeGrad hw) (retypeGrad hb) (map retypeGrad hid)

public export
Recurrent Gru where
  recurStep {o} st input = do
    h <- case st.hiddenT of
           Just h => pure h
           Nothing => tzeroState1d {n = o}
    ihPart <- tlinear st.iwT input st.ihB
    hhPart <- tlinear st.hwT h st.hhB
    newH   <- tgruCell {n = o} ihPart hhPart h
    pure ({ hiddenT := Just newH } st, newH)

  recurReset st = { hiddenT := Nothing } st

||| Construct a `Gru i o` inside an `Init` derivation. Xavier-ish weight
||| init (3 stacked gates → fan_out 3·o), zero biases, empty state.
||| Registers `<scope>.gru_<n>.{weight,bias}_{ih,hh}`.
export
gru : {0 ex : Executor} -> Backend ex dt => {i, o : Nat} -> Init (Gru i o ex dt WithGrad)
gru = do
  name <- freshChild "gru"
  let iwStd = sqrt (2.0 / cast {to=Double} (i + 3 * o))
      hwStd = sqrt (2.0 / cast {to=Double} (o + 3 * o))
  iw  <- liftIO $ tparam2dNormal {ex} {dt} {o = 3 * o} {i}     (name ++ ".weight_ih") 0.0 iwStd
  hw  <- liftIO $ tparam2dNormal {ex} {dt} {o = 3 * o} {i = o} (name ++ ".weight_hh") 0.0 hwStd
  ihB <- liftIO $ tparam1dConst  {ex} {dt} {n = 3 * o} (name ++ ".bias_ih") 0.0
  hhB <- liftIO $ tparam1dConst  {ex} {dt} {n = 3 * o} (name ++ ".bias_hh") 0.0
  pure (MkGru iw ihB hw hhB Nothing)
