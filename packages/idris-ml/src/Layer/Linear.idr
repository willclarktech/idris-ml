module Layer.Linear

import Data.Vect

import Compat.Random
import Device
import Init
import Layer.Core
import Sampler
import Tensor


----------------------------------------------------------------------
-- Linear State (Path C P3-1 spike)
----------------------------------------------------------------------
--
-- Direct shape-aware tensor weights — no scalar matrix, no view
-- Variables, no `Maybe AnyPtr` dual mode. The Tensor IS the weight.
-- Bias and weight are registered C params at construction time.

public export
record LinearState (i : Nat) (o : Nat) (0 d : Device) (0 dt : DType) (0 g : GradMode) where
  constructor MkLinear
  weightT : Tensor [o, i] d dt g
  biasT   : Tensor [o] d dt g


----------------------------------------------------------------------
-- LayerLike instance — two-line forward
----------------------------------------------------------------------

%default partial

public export
LayerLike LinearState where
  applyVar st input = do
    out <- tlinear st.weightT input st.biasT
    pure (st, out)

  applyVarBatch st input = do
    out <- tlinear2d st.weightT input st.biasT
    pure (st, out)

  layerPrefix _ = "llv2"

  freezeLayer (MkLinear w b) = do
    w' <- weakenGrad w
    b' <- weakenGrad b
    pure (MkLinear w' b')

  unfreezeLayer (MkLinear w b) = do
    primIO (primSetRequiresGrad {d} w.tensorPtr 1)
    primIO (primSetRequiresGrad {d} b.tensorPtr 1)
    pure (MkLinear (retypeGrad w) (retypeGrad b))


----------------------------------------------------------------------
-- Constructor
----------------------------------------------------------------------

||| Pack a Vect of Doubles into a pre-allocated buffer at offset.
||| Exported so other layer modules can reuse the same packing logic.
export
packDoubles : AnyPtr -> Int -> Vect k Double -> AnyPtr
packDoubles buf _ [] = buf
packDoubles buf off (x :: rest) =
  packDoubles (prim__setDouble buf off x) (off + 1) rest

-- Zero a buffer for `n` elements starting at offset.
zeroBuf : AnyPtr -> Int -> Int -> AnyPtr
zeroBuf buf _ 0 = buf
zeroBuf buf off n =
  zeroBuf (prim__setDouble buf off 0.0) (off + 1) (n - 1)

||| Build a `LinearState i o TapeDev` with custom weight + bias init
||| strategies. Mirrors PyTorch's per-FC init customization (e.g. NTM's
||| read FCs use Xavier(gain=1.4) + normal(std=0.01) biases). The default
||| `linearLayer` is `mkLinearWith ... (xavier uniform) (pure 0.0)`.
export
mkLinearWith : UserDeviceTraining d => RuntimeDType dt => Linked d => Compatible d dt => {i, o : Nat}
            -> (paramPrefix : String)
            -> (weightInit : InitStrategy)
            -> (biasInit : IO Double)
            -> IO (LinearState i o d dt WithGrad)
mkLinearWith pfx wInit bInit = do
  let oI = cast {to=Int} o
      iI = cast {to=Int} i
      wCount = o * i
  weightVals <- traverse (\_ => wInit i o) (Vect.replicate wCount ())
  biasVals <- traverse (\_ => bInit) (Vect.replicate o ())
  let wBuf = prim__allocDoubles (cast wCount)
      wBuf' = packDoubles wBuf 0 weightVals
      bBuf = prim__allocDoubles oI
      bBuf' = packDoubles bBuf 0 biasVals
  w <- tparam2d (pfx ++ "_weights") wBuf'
  b <- tparam1d (pfx ++ "_biases") bBuf'
  pure $ MkLinear w b

||| Build a `LinearState i o TapeDev` with Xavier-uniform weights and
||| zero bias. Weights and biases are allocated as registered C
||| params under `<paramPrefix>_weights` and `<paramPrefix>_biases` —
||| matching the existing `Layer/Linear.idr` naming so the optimizer
||| picks them up via the global registry.
export
linearLayer : UserDeviceTraining d => RuntimeDType dt => Linked d => Compatible d dt => {i, o : Nat} -> (paramPrefix : String) -> IO (LinearState i o d dt WithGrad)
linearLayer paramPrefix = do
  let oI = cast {to=Int} o
      iI = cast {to=Int} i
      wCount = o * i
  weightVals <- traverse (\_ => xavier uniform i o) (Vect.replicate wCount ())
  let wBuf = prim__allocDoubles (cast {to=Int} wCount)
      wBuf' = packDoubles wBuf 0 weightVals
      bBuf = prim__allocDoubles oI
      bBuf' = zeroBuf bBuf 0 oI
  w <- tparam2d (paramPrefix ++ "_weights") wBuf'
  b <- tparam1d (paramPrefix ++ "_biases") bBuf'
  pure $ MkLinear w b

||| Wrap a Linear in `AnyLayer` for use in a `Network`.
export
linearLayerAny : UserDeviceTraining d => RuntimeDType dt => Linked d => Compatible d dt => {i, o : Nat} -> (paramPrefix : String) -> IO (AnyLayer i o d dt WithGrad)
linearLayerAny pid = map (MkAnyLayer LinearState) (linearLayer pid)
