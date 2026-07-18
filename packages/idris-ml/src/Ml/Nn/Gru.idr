||| `Gru` — GRU cell on the v1 `Nn` surface, implementing `Recurrent`.
||| Three gates (reset/update/new) stacked along axis 0 (weights `[3·o, …]`,
||| via the `TMat`/`TVec` aliases); the gate mixing happens inside the fused
||| `tgruCell`. Carried `hiddenT` state lives in the record. `params` lists
||| the four learnable tensors.
module Ml.Nn.Gru

import Control.Linear.LIO
import Data.Linear
import Data.Vect

import Ml.Executor
import Ml.Nn.Init
import Ml.Nn.Module
import Ml.Nn.Recurrent
import Ml.Tensor

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

||| Params: four learnable tensors; carried state is not a param.
public export
Params Gru where
  params (MkGru iw ib hw hb hid) =
    [toParam iw, toParam ib, toParam hw, toParam hb]
  reflect (MkGru iw ib hw hb hid) =
    MkBang [toParam iw, toParam ib, toParam hw, toParam hb] # MkGru iw ib hw hb hid
  castGrad (MkGru iw ib hw hb hid) =
    MkGru (retypeGrad iw) (retypeGrad ib) (retypeGrad hw) (retypeGrad hb) (map retypeGrad hid)
  discard (MkGru _ _ _ _ _) = pure ()

||| Recurrent step (sequences the `L IO` tensor ops directly).
public export
Recurrent Gru where
  recurStep {o} (MkGru iw ib hw hb hid) input = do
    h <- the (L IO (TVec o ex dt WithGrad)) $ case hid of
           Just h  => pure h
           Nothing => tzeroState1dL {n = o}
    ihPart <- tlinearL iw input ib
    hhPart <- tlinearL hw h hb
    newH   <- tgruCellL {n = o} ihPart hhPart h
    pure1 (MkBang newH # MkGru iw ib hw hb (Just newH))
  recurReset (MkGru iw ib hw hb _) = MkGru iw ib hw hb Nothing

||| Construct a `Gru i o` inside an `Init` derivation. Xavier-ish weight
||| init (3 stacked gates → fan_out 3·o), zero biases, empty state.
||| Registers `<scope>.gru_<n>.{weight,bias}_{ih,hh}`.
export
gru : KnownGrad g => {0 ex : Executor} -> Backend ex dt => {i, o : Nat} -> Init (Gru i o ex dt g)
gru = do
  name <- freshChild "gru"
  let iwStd = sqrt (2.0 / cast {to=Double} (i + 3 * o))
      hwStd = sqrt (2.0 / cast {to=Double} (o + 3 * o))
  iw  <- liftIO $ tparam2dNormal {ex} {dt} {o = 3 * o} {i}     (name ++ ".weight_ih") 0.0 iwStd
  hw  <- liftIO $ tparam2dNormal {ex} {dt} {o = 3 * o} {i = o} (name ++ ".weight_hh") 0.0 hwStd
  ihB <- liftIO $ tparam1dConst  {ex} {dt} {n = 3 * o} (name ++ ".bias_ih") 0.0
  hhB <- liftIO $ tparam1dConst  {ex} {dt} {n = 3 * o} (name ++ ".bias_hh") 0.0
  case sgrad {g} of
    SWithGrad => pure (MkGru iw ihB hw hhB Nothing)
    SNoGrad   => do iw'  <- liftIO (weakenGrad iw);  hw'  <- liftIO (weakenGrad hw)
                    ihB' <- liftIO (weakenGrad ihB); hhB' <- liftIO (weakenGrad hhB)
                    pure (MkGru iw' ihB' hw' hhB' Nothing)
