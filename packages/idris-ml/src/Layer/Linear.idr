module Layer.Linear

import Data.Vect
import Data.Zippable

import Device
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
  weightTensor : Maybe AnyPtr  -- consolidated [o, i] tensor (for Variable forward)
  biasTensor : Maybe AnyPtr    -- consolidated [o] tensor


----------------------------------------------------------------------
-- Helpers: build Variable matrix/vector from tensor views
----------------------------------------------------------------------

-- Build a row of scalar Variables from views into a 2D tensor.
-- Each view Variable gets a paramId for identification (but is NOT individually registered).
-- linearIdx: offset for this row's start in the flat index space
export
buildViewRow : {d : Device} -> String -> AnyPtr -> Int -> Int -> Int -> (k : Nat) -> Vect k (Scalar (Variable d))
buildViewRow _ _ _ _ _ Z = []
buildViewRow name mat row col linearIdx (S k) =
  let ptr = prim__view2d mat row col
      val = prim__item2d mat row col
  in STensor (Var ptr (Just (name ++ show linearIdx)) val) :: buildViewRow name mat row (col + 1) (linearIdx + 1) k

-- Build a matrix of scalar Variables from views into a 2D tensor.
export
buildViewMatrix : {d : Device} -> String -> AnyPtr -> Int -> Int -> (rows : Nat) -> (cols : Nat) -> Vect rows (Vector cols (Variable d))
buildViewMatrix _ _ _ _ Z _ = []
buildViewMatrix name mat row linearIdx (S r) cols =
  let colsI = cast {to=Int} cols
  in VTensor (buildViewRow name mat row 0 linearIdx cols) :: buildViewMatrix name mat (row + 1) (linearIdx + colsI) r cols

-- Build a vector of scalar Variables from views into a 1D tensor.
export
buildViewVector : {d : Device} -> String -> AnyPtr -> Int -> (k : Nat) -> Vect k (Scalar (Variable d))
buildViewVector _ _ _ Z = []
buildViewVector name vec idx (S k) =
  let ptr = prim__view1d vec idx
      val = prim__item1d vec idx
  in STensor (Var ptr (Just (name ++ show idx)) val) :: buildViewVector name vec (idx + 1) k


----------------------------------------------------------------------
-- LayerLike Instance
----------------------------------------------------------------------

%default partial
export
LayerLike LinearState where
  applyGeneric (MkLinear w b _ _) xs = (MkLinear w b Nothing Nothing, matrixVectorMultiply w xs + b)

  applyVar {d} {i} {o} st xs =
    let (VTensor xElems) = xs
        inputT = vecStackTensor {n=i} xElems
        (st', outT) = applyVarTensor st inputT
    in (st', VTensor $ tensorToScalars outT 0 o)

  applyVarTensor {d} {i} {o} st@(MkLinear weights bias wt bt) inputT =
    case (wt, bt) of
      (Just weightT, Just biasT) =>
        (st, tensorAdd (tensorMv weightT inputT) biasT)
      _ => idris_crash "Linear: weight tensors not initialized (call autoName first)"

  emapLayer f (MkLinear w b wt bt) = MkLinear (map f w) (map f b) wt bt

  showLayer {i} {o} _ = "Linear<" ++ show i ++ ":" ++ show o ++ ">"

  nameLayer {d} {i} {o} prefx (MkLinear weights bias _ _) =
    if prim__backendSupportsTensorParams == 1
      then -- Tensor path: consolidated weight tensors + view Variables
        let (VTensor wRows) = weights
            (VTensor bElems) = bias
            oI = cast {to=Int} o
            iI = cast {to=Int} i
            wBuf = prim__allocDoubles (oI * iI)
            wBuf' = packMatrixValues wBuf 0 {n=i} wRows
            weightT = prim__paramRegister (prefx ++ "_weights") (prim__createParam2d oI iI wBuf')
            bBuf = prim__allocDoubles oI
            bBuf' = packScalarValues bBuf 0 bElems
            biasT = prim__paramRegister (prefx ++ "_biases") (prim__createParam1d oI bBuf')
        in MkLinear (VTensor $ buildViewMatrix (prefx ++ "_weight") weightT 0 0 o i)
                    (VTensor $ buildViewVector (prefx ++ "_bias") biasT 0 o)
                    (Just weightT) (Just biasT)
      else -- Scalar path: per-element named Variables (tape backend)
        let np = nameParam . (prefx ++ "_" ++)
            namedWeights = zipWith (np "weight") enumerate weights
            namedBias = zipWith (np "bias") enumerate bias
        in MkLinear namedWeights namedBias Nothing Nothing

  layerPrefix _ = "ll"

  toDoubleLayer {d} {i} {o} (MkLinear w b wt bt) =
    case (wt, bt) of
      (Just weightT, Just biasT) =>
        -- Read values from consolidated tensors
        let wRows = buildDoubleMatrix weightT 0 o i
            bElems = buildDoubleVector biasT 0 o
        in MkLinear (VTensor wRows) (VTensor bElems) Nothing Nothing
      _ => MkLinear (map value w) (map value b) Nothing Nothing
    where
      buildDoubleRow : AnyPtr -> Int -> Int -> (k : Nat) -> Vect k (Scalar Double)
      buildDoubleRow _ _ _ Z = []
      buildDoubleRow mat row col (S k) =
        STensor (prim__item2d mat row col) :: buildDoubleRow mat row (col + 1) k

      buildDoubleMatrix : AnyPtr -> Int -> (rows : Nat) -> (cols : Nat) -> Vect rows (Vector cols Double)
      buildDoubleMatrix _ _ Z _ = []
      buildDoubleMatrix mat row (S r) cols =
        VTensor (buildDoubleRow mat row 0 cols) :: buildDoubleMatrix mat (row + 1) r cols

      buildDoubleVector : AnyPtr -> Int -> (k : Nat) -> Vect k (Scalar Double)
      buildDoubleVector _ _ Z = []
      buildDoubleVector vec idx (S k) =
        STensor (prim__item1d vec idx) :: buildDoubleVector vec (idx + 1) k

  debugApply {i} {o} st inp =
    let (updated, out) = applyGeneric st inp
    in (updated, out, MkDebugEntry ("Linear<" ++ show i ++ ":" ++ show o ++ ">") [])

  getParamIds {d} (MkLinear w b _ _) = tensorIds w ++ tensorIds b
    where
      tensorIds : {dims : Vect rank Nat} -> Tensor dims (Variable d) -> List String
      tensorIds = mapMaybe paramId . toList


----------------------------------------------------------------------
-- Tensor Handle Accessors (for NTM direct tensor pipeline)
----------------------------------------------------------------------

export
extractWeightTensor : {d : Device} -> LinearState i o (Variable d) -> Maybe AnyPtr
extractWeightTensor st = st.weightTensor

export
extractBiasTensor : {d : Device} -> LinearState i o (Variable d) -> Maybe AnyPtr
extractBiasTensor st = st.biasTensor


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
