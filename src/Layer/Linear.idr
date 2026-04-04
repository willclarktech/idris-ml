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
  wBuf : Maybe AnyPtr
  bBuf : Maybe AnyPtr


----------------------------------------------------------------------
-- LayerLike Instance
----------------------------------------------------------------------

export
LayerLike LinearState where
  applyGeneric (MkLinear w b wb bb) xs = (MkLinear w b wb bb, matrixVectorMultiply w xs + b)

  applyVar st@(MkLinear weights bias wBuf bBuf) xs =
    case (wBuf, bBuf) of
      (Just wb, Just bb) => (st, matrixVectorMultiplyVarBufBias wb bb xs)
      (Just wb, Nothing) => (st, matrixVectorMultiplyVarBuf wb xs + bias)
      _ => (st, matrixVectorMultiplyVar weights xs + bias)

  emapLayer f (MkLinear w b wb bb) = MkLinear (map f w) (map f b) wb bb

  showLayer {i} {o} _ = "Linear<" ++ show i ++ ":" ++ show o ++ ">"

  nameLayer {i} {o} prefx (MkLinear weights bias _ _) =
    let np = nameParam . (prefx ++ "_" ++)
        namedWeights = zipWith (np "weight") enumerate weights
        namedBias = zipWith (np "bias") enumerate bias
    -- No buffers in libtorch backend — libtorch tensors ARE the weights
    in MkLinear namedWeights namedBias Nothing Nothing

  layerPrefix _ = "ll"

  toDoubleLayer (MkLinear w b _ _) = MkLinear (map value w) (map value b) Nothing Nothing

  debugApply {i} {o} st inp =
    let (updated, out) = applyGeneric st inp
    in (updated, out, MkDebugEntry ("Linear<" ++ show i ++ ":" ++ show o ++ ">") [])

  syncBuffers (MkLinear (VTensor wRows) (VTensor biasElems) (Just wb) (Just bb)) =
    let wb' = syncWeightBuf wb 0 wRows
        bb' = syncWeightBufRow bb 0 biasElems
    in MkLinear (VTensor wRows) (VTensor biasElems) (Just wb') (Just bb')
  syncBuffers (MkLinear (VTensor wRows) bias (Just wb) Nothing) =
    let wb' = syncWeightBuf wb 0 wRows
    in MkLinear (VTensor wRows) bias (Just wb') Nothing
  syncBuffers l = l

  applyDeltasAndSync deltas (MkLinear w b (Just wb) (Just bb)) =
    let wb' = prim__weightBufApplyDeltas wb deltas
        bb' = prim__weightBufApplyDeltas bb deltas
    in MkLinear w b (Just wb') (Just bb')
  applyDeltasAndSync deltas (MkLinear w b (Just wb) Nothing) =
    let wb' = prim__weightBufApplyDeltas wb deltas
    in MkLinear w b (Just wb') Nothing
  applyDeltasAndSync _ l = l

  readFromBuffers (MkLinear (VTensor wRows) (VTensor biasElems) (Just wb) (Just bb)) =
    MkLinear (VTensor (readWeightBuf wb 0 wRows)) (VTensor (readWeightBufRow bb 0 biasElems)) (Just wb) (Just bb)
  readFromBuffers (MkLinear (VTensor wRows) bias (Just wb) Nothing) =
    MkLinear (VTensor (readWeightBuf wb 0 wRows)) bias (Just wb) Nothing
  readFromBuffers l = l

  getParamIds (MkLinear w b _ _) = tensorIds w ++ tensorIds b
    where
      tensorIds : {dims : Vect rank Nat} -> Tensor dims Variable -> List String
      tensorIds = mapMaybe paramId . toList


----------------------------------------------------------------------
-- Public Helpers (for NTM buffer-passing)
----------------------------------------------------------------------

||| Extract weight and bias buffers from a LinearState.
export
getLinearBufs : LinearState i o Variable -> (Maybe AnyPtr, Maybe AnyPtr)
getLinearBufs st = (st.wBuf, st.bBuf)


----------------------------------------------------------------------
-- Constructors
----------------------------------------------------------------------

||| Create a raw LinearState (for NTM internal use)
export
mkLinearWith : {i, o : Nat} -> (Num ty, FromDouble ty) => InitStrategy -> IO (LinearState i o ty)
mkLinearWith initFn = do
  weights <- traverse (\_ => map fromDouble (initFn i o)) (the (Matrix o i ty) zeros)
  pure $ MkLinear weights zeros Nothing Nothing

||| Create a raw LinearState with custom bias init (for NTM head FCs)
export
mkLinearWithBias : {i, o : Nat} -> (Num ty, FromDouble ty) =>
                   InitStrategy -> (biasStd : Double) -> IO (LinearState i o ty)
mkLinearWithBias initFn biasStd = do
  weights <- traverse (\_ => map fromDouble (initFn i o)) (the (Matrix o i ty) zeros)
  bias <- traverse (\_ => map fromDouble (normalSample >>= \s => pure (s * biasStd)))
                    (the (Vector o ty) zeros)
  pure $ MkLinear weights bias Nothing Nothing

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
