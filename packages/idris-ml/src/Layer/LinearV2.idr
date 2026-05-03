module Layer.LinearV2

import Data.Vect

import Compat.Random
import Device
import Init
import Layer.CoreV2
import Sampler
import Variable


----------------------------------------------------------------------
-- LinearV2 State (Path C P3-1 spike)
----------------------------------------------------------------------
--
-- Direct shape-aware tensor weights — no scalar matrix, no view
-- Variables, no `Maybe AnyPtr` dual mode. The TVar IS the weight.
-- Bias and weight are registered C params at construction time.

public export
record LinearStateV2 (i : Nat) (o : Nat) (0 d : Device) where
  constructor MkLinearV2
  weightT : TVar [o, i] d
  biasT   : TVar [o] d


----------------------------------------------------------------------
-- LayerLikeV2 instance — two-line forward
----------------------------------------------------------------------

%default partial

public export
LayerLikeV2 LinearStateV2 where
  applyTVar st input =
    let pre = tmv st.weightT input
        out = tadd pre st.biasT
    in (st, out)

  layerPrefixV2 _ = "llv2"


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

||| Build a `LinearStateV2 i o CPU` with Xavier-uniform weights and
||| zero bias. Weights and biases are allocated as registered C
||| params under `<paramPrefix>_weights` and `<paramPrefix>_biases` —
||| matching the existing `Layer/Linear.idr` naming so the optimizer
||| picks them up via the global registry.
export
linearLayerV2 : {i, o : Nat} -> (paramPrefix : String) -> IO (LinearStateV2 i o CPU)
linearLayerV2 paramPrefix = do
  let oI = cast {to=Int} o
      iI = cast {to=Int} i
      wCount = o * i
  weightVals <- traverse (\_ => xavier uniform i o) (Vect.replicate wCount ())
  let wBuf = prim__allocDoubles (cast {to=Int} wCount)
      wBuf' = packDoubles wBuf 0 weightVals
      bBuf = prim__allocDoubles oI
      bBuf' = zeroBuf bBuf 0 oI
  pure $ MkLinearV2
    (tparam2d (paramPrefix ++ "_weights") wBuf')
    (tparam1d (paramPrefix ++ "_biases") bBuf')

||| Wrap a LinearV2 in `AnyLayerV2` for use in a `NetworkV2`.
export
linearLayerV2Any : {i, o : Nat} -> (paramPrefix : String) -> IO (AnyLayerV2 i o CPU)
linearLayerV2Any pid = map (MkAnyLayerV2 LinearStateV2) (linearLayerV2 pid)
