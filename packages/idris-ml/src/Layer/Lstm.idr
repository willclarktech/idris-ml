module Layer.Lstm

import Data.Vect

import Executor
import Layer.Core
import Tensor


----------------------------------------------------------------------
-- Lstm — typed-surface LSTM cell (Path C)
----------------------------------------------------------------------
--
-- All shape-arithmetic flows through the `TVec` / `TMat` aliases in
-- `Tensor.idr`. Direct `Tensor [4 * o, ...] ex` triggers an Idris 2
-- type-checker hang; `TMat (4 * o) i d` works fine because the
-- multiplication sits in a Nat-argument slot of the alias rather
-- than inside a Vect literal.

public export
record LstmState (i : Nat) (o : Nat) (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkLstm
  iwT : TMat (4 * o) i ex dt g
  rwT : TMat (4 * o) o ex dt g
  ihB : TVec (4 * o) ex dt g        -- input-hidden bias [4*o] (b_ih)
  hhB : TVec (4 * o) ex dt g        -- hidden-hidden bias [4*o] (b_hh)
  h0T : TVec o ex dt g              -- learned initial hidden state (zero-init)
  c0T : TVec o ex dt g              -- learned initial cell state (zero-init)
  hiddenT : Maybe (TVec o ex dt g)
  cellT   : Maybe (TVec o ex dt g)


----------------------------------------------------------------------
-- Forward
----------------------------------------------------------------------

%default partial

||| Array-level LSTM cell forward. Reads (or zero-initialises) the
||| hidden + cell state, runs the fused gate computation, returns the
||| updated layer state and the new hidden output.
export
applyLstm : {0 ex : Executor} -> Backend ex dt => {o : Nat} ->
              LstmState i o ex dt g ->
              TVec i ex dt g ->
              IO (LstmState i o ex dt g, TVec o ex dt g)
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


||| Build an `LstmState i o TapeExecutor` with Xavier-uniform weight init,
||| two zero biases (matching `nn.LSTMCell`), and learned `h0`/`c0`
||| (zero-init, learned). Weights register as C params under
||| `<prefix>_iw`, `<prefix>_rw`, `<prefix>_ib`, `<prefix>_hb`,
||| `<prefix>_h0`, `<prefix>_c0`.
export
lstmLayer : Backend ex dt => {i, o : Nat} -> (paramPrefix : String) ->
              IO (LstmState i o ex dt WithGrad)
lstmLayer paramPrefix = do
  -- 4 gates (input, forget, gate, output) stacked along axis=0 →
  -- weights are [4*o, i] / [4*o, o]. Xavier-normal-via-uniform std =
  -- sqrt(2/(fan_in + fan_out)) with fan_out = 4*o. Biases + learned
  -- (h0, c0) initial states zero-init.
  let iwStd = sqrt (2.0 / cast {to=Double} (i + 4 * o))
      rwStd = sqrt (2.0 / cast {to=Double} (o + 4 * o))
      iwName = paramPrefix ++ "_iw"
      rwName = paramPrefix ++ "_rw"
      ibName = paramPrefix ++ "_ib"
      hbName = paramPrefix ++ "_hb"
      h0Name = paramPrefix ++ "_h0"
      c0Name = paramPrefix ++ "_c0"
  iw <- tparam2dNormal {o = 4 * o} {i} iwName 0.0 iwStd
  rw <- tparam2dNormal {o = 4 * o} {i = o} rwName 0.0 rwStd
  ib <- tparam1dConst {n = 4 * o} ibName 0.0
  hb <- tparam1dConst {n = 4 * o} hbName 0.0
  h0 <- tparam1dConst {n = o} h0Name 0.0
  c0 <- tparam1dConst {n = o} c0Name 0.0
  pure $ MkLstm iw rw ib hb h0 c0 Nothing Nothing

||| Reset hidden/cell state. Setting to `Nothing` lets `applyLstm`'s
||| first call lazy-allocate fresh persistent zero buffers — mirrors
||| V1's `resetState`, where MLX trains correctly via this lazy path.
export
resetLstmState : {o : Nat} -> {0 ex : Executor} -> {0 g : GradMode} -> LstmState i o ex dt g -> LstmState i o ex dt g
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
    primIO (primSetRequiresGrad {ex} iw.tensorPtr 1)
    primIO (primSetRequiresGrad {ex} rw.tensorPtr 1)
    primIO (primSetRequiresGrad {ex} ihB.tensorPtr 1)
    primIO (primSetRequiresGrad {ex} hhB.tensorPtr 1)
    primIO (primSetRequiresGrad {ex} h0.tensorPtr 1)
    primIO (primSetRequiresGrad {ex} c0.tensorPtr 1)
    case hid of
      Nothing => pure ()
      Just h  => primIO (primSetRequiresGrad {ex} h.tensorPtr 1)
    case cell of
      Nothing => pure ()
      Just c  => primIO (primSetRequiresGrad {ex} c.tensorPtr 1)
    pure (MkLstm (retypeGrad iw) (retypeGrad rw)
                 (retypeGrad ihB) (retypeGrad hhB)
                 (retypeGrad h0) (retypeGrad c0)
                 (map retypeGrad hid) (map retypeGrad cell))

||| Wrap an `LstmState` in `AnyLayer`.
export
lstmLayerAny : Backend ex dt => {i, o : Nat} -> (paramPrefix : String) -> IO (AnyLayer i o ex dt WithGrad)
lstmLayerAny pid = map (MkAnyLayer LstmState) (lstmLayer pid)
