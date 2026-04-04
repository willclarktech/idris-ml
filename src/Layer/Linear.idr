module Layer.Linear

import Data.Vect
import Data.Zippable

import Endofunctor
import Floating
import Init
import Layer.Core
import Math
import Tensor
import Util
import Variable


----------------------------------------------------------------------
-- Linear State
----------------------------------------------------------------------

public export
record LinearState (inputSize : Nat) (outputSize : Nat) (ty : Type) where
  constructor MkLinear
  weights : Matrix outputSize inputSize ty
  bias : Vector outputSize ty


----------------------------------------------------------------------
-- LayerLike Instance
----------------------------------------------------------------------

export
LayerLike LinearState where
  applyGeneric (MkLinear w b) xs = (MkLinear w b, matrixVectorMultiply w xs + b)

  applyVar st@(MkLinear weights bias) xs = (st, matrixVectorMultiplyVar weights xs + bias)

  emapLayer f (MkLinear w b) = MkLinear (map f w) (map f b)

  showLayer {i} {o} _ = "Linear<" ++ show i ++ ":" ++ show o ++ ">"

  nameLayer prefx (MkLinear weights bias) =
    let np = nameParam . (prefx ++ "_" ++)
        namedWeights = zipWith (np "weight") enumerate weights
        namedBias = zipWith (np "bias") enumerate bias
    in MkLinear namedWeights namedBias

  layerPrefix _ = "ll"

  toDoubleLayer (MkLinear w b) = MkLinear (map value w) (map value b)

  debugApply {i} {o} st inp =
    let (updated, out) = applyGeneric st inp
    in (updated, out, MkDebugEntry ("Linear<" ++ show i ++ ":" ++ show o ++ ">") [])

  getParamIds (MkLinear w b) = tensorIds w ++ tensorIds b
    where
      tensorIds : {dims : Vect rank Nat} -> Tensor dims Variable -> List String
      tensorIds = mapMaybe paramId . toList


----------------------------------------------------------------------
-- Constructors
----------------------------------------------------------------------

||| Create a raw LinearState (for NTM internal use)
export
mkLinearWith : {i, o : Nat} -> (Num ty, FromDouble ty) => InitStrategy -> IO (LinearState i o ty)
mkLinearWith initFn = do
  weights <- traverse (\_ => map fromDouble (initFn i o)) (the (Matrix o i ty) zeros)
  pure $ MkLinear weights zeros

||| Create a raw LinearState with custom bias init (for NTM head FCs)
export
mkLinearWithBias : {i, o : Nat} -> (Num ty, FromDouble ty) =>
                   InitStrategy -> (biasStd : Double) -> IO (LinearState i o ty)
mkLinearWithBias initFn biasStd = do
  weights <- traverse (\_ => map fromDouble (initFn i o)) (the (Matrix o i ty) zeros)
  bias <- traverse (\_ => map fromDouble (normalSample >>= \s => pure (s * biasStd)))
                    (the (Vector o ty) zeros)
  pure $ MkLinear weights bias

||| Create a raw LinearState with default Xavier uniform init
export
mkLinear : {i, o : Nat} -> (Num ty, FromDouble ty) => IO (LinearState i o ty)
mkLinear = mkLinearWith (xavier uniform)

||| Create a LinearState wrapped in AnyLayer
export
linearLayerWith : {i, o : Nat} -> (Num ty, FromDouble ty) => InitStrategy -> IO (AnyLayer i o ty)
linearLayerWith initFn = map (MkAnyLayer LinearState) (mkLinearWith initFn)

||| Create a LinearState with custom bias, wrapped in AnyLayer
export
linearLayerWithBias : {i, o : Nat} -> (Num ty, FromDouble ty) =>
                      InitStrategy -> (biasStd : Double) -> IO (AnyLayer i o ty)
linearLayerWithBias initFn biasStd = map (MkAnyLayer LinearState) (mkLinearWithBias initFn biasStd)

||| Create a LinearState with Xavier uniform, wrapped in AnyLayer
export
linearLayer : {i, o : Nat} -> (Num ty, FromDouble ty) => IO (AnyLayer i o ty)
linearLayer = map (MkAnyLayer LinearState) mkLinear
