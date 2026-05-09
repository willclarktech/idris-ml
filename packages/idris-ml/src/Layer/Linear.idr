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
record LinearState (i : Nat) (o : Nat) (0 d : Device) where
  constructor MkLinear
  weightT : Tensor [o, i] d
  biasT   : Tensor [o] d


----------------------------------------------------------------------
-- LayerLike instance — two-line forward
----------------------------------------------------------------------

%default partial

public export
LayerLike LinearState where
  applyVar st input =
    (st, tlinear st.weightT input st.biasT)

  applyVarBatch st input =
    (st, tlinear2d st.weightT input st.biasT)

  layerPrefix _ = "llv2"


----------------------------------------------------------------------
-- Constructor
----------------------------------------------------------------------

-- Pack a Vect of Doubles into a pre-allocated buffer at offset.
packDoubles : AnyPtr -> Int -> Vect k Double -> AnyPtr
packDoubles buf _ [] = buf
packDoubles buf off (x :: rest) =
  packDoubles (prim__setDouble buf off x) (off + 1) rest

-- Zero a buffer for `n` elements starting at offset.
zeroBuf : AnyPtr -> Int -> Int -> AnyPtr
zeroBuf buf _ 0 = buf
zeroBuf buf off n =
  zeroBuf (prim__setDouble buf off 0.0) (off + 1) (n - 1)

||| Build a `LinearState i o CPU` with Xavier-uniform weights and
||| zero bias. Weights and biases are allocated as registered C
||| params under `<paramPrefix>_weights` and `<paramPrefix>_biases` —
||| matching the existing `Layer/Linear.idr` naming so the optimizer
||| picks them up via the global registry.
export
linearLayer : {i, o : Nat} -> (paramPrefix : String) -> IO (LinearState i o CPU)
linearLayer paramPrefix = do
  let oI = cast {to=Int} o
      iI = cast {to=Int} i
      wCount = o * i
  weightVals <- traverse (\_ => xavier uniform i o) (Vect.replicate wCount ())
  let wBuf = prim__allocDoubles (cast {to=Int} wCount)
      wBuf' = packDoubles wBuf 0 weightVals
      bBuf = prim__allocDoubles oI
      bBuf' = zeroBuf bBuf 0 oI
  pure $ MkLinear
    (tparam2d (paramPrefix ++ "_weights") wBuf')
    (tparam1d (paramPrefix ++ "_biases") bBuf')

||| Wrap a Linear in `AnyLayer` for use in a `Network`.
export
linearLayerAny : {i, o : Nat} -> (paramPrefix : String) -> IO (AnyLayer i o CPU)
linearLayerAny pid = map (MkAnyLayer LinearState) (linearLayer pid)
