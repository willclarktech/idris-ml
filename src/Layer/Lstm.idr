module Layer.Lstm

import Data.Vect
import Data.Zippable

import Endofunctor
import Floating
import Init
import Layer.Core
import Layer.Linear
import Math
import Tensor
import Util
import Variable


----------------------------------------------------------------------
-- Tanh Memory Bounding
----------------------------------------------------------------------

||| Bounds values to [-1, 1] via tanh. Uses integer literals to avoid
||| requiring a FromDouble constraint.
export
tanhBound : (Neg ty, Fractional ty, Floating ty) => ty -> ty
tanhBound x = 2 * (1 / (1 + exp (negate (2 * x)))) - 1


----------------------------------------------------------------------
-- LSTM Gate Helpers
----------------------------------------------------------------------

||| Coerce `Vector (o + 0) ty` to `Vector o ty`.
coerceLastGate : {o : Nat} -> Vector (o + 0) ty -> Vector o ty
coerceLastGate {o} v = rewrite sym (plusZeroRightNeutral o) in v

||| Split the combined gate vector into (input, forget, cell, output) gates.
export
lstmSplitGates :
    {o : Nat} -> Vector (4 * o) ty
    -> (Vector o ty, Vector o ty, Vector o ty, Vector o ty)
lstmSplitGates {o} combined =
  let s1 = Tensor.splitAt o combined
      s2 = Tensor.splitAt o (snd s1)
      s3 = Tensor.splitAt o (snd s2)
  in (fst s1, fst s2, fst s3, coerceLastGate (snd s3))


----------------------------------------------------------------------
-- LSTM State
----------------------------------------------------------------------

public export
record LstmState (inputSize : Nat) (outputSize : Nat) (ty : Type) where
  constructor MkLstm
  inputWeights : Matrix (4 * outputSize) inputSize ty
  recurrentWeights : Matrix (4 * outputSize) outputSize ty
  bias : Vector (4 * outputSize) ty
  hiddenState : Vector outputSize ty
  cellState : Vector outputSize ty
  -- Consolidated tensors (Just after nameLayer, Nothing before)
  iwTensor : Maybe AnyPtr     -- [4*o, i]
  rwTensor : Maybe AnyPtr     -- [4*o, o]
  biasTensor : Maybe AnyPtr   -- [4*o]


----------------------------------------------------------------------
-- Cell State Extraction
----------------------------------------------------------------------

||| Extract the cell state from an LSTM layer (for NTM head FC input).
export
extractCellState : {o : Nat} -> LstmState i o ty -> Vector o ty
extractCellState st = st.cellState


----------------------------------------------------------------------
-- Forget Bias Helper
----------------------------------------------------------------------

||| Set forget gate bias to 1.0. Gate layout: input, forget, cell, output.
||| Forget gate occupies indices [o..2*o) in the bias vector.
setForgetBias : (FromDouble ty, Num ty) => {o : Nat} -> Vector (4 * o) ty -> Vector (4 * o) ty
setForgetBias {o} (VTensor elems) =
  VTensor (go 0 elems)
  where
    go : Nat -> Vect n (Scalar ty) -> Vect n (Scalar ty)
    go _ [] = []
    go idx (x :: xs) =
      if idx >= o && idx < 2 * o
        then STensor (fromDouble 1.0) :: go (idx + 1) xs
        else x :: go (idx + 1) xs


----------------------------------------------------------------------
-- Debug showVec helper
----------------------------------------------------------------------

showVecD : {n : Nat} -> Vector n Double -> String
showVecD (VTensor xs) = "[" ++ go xs ++ "]"
  where
    go : Vect k (Tensor [] Double) -> String
    go [] = ""
    go [STensor x] = show x
    go (STensor x :: rest) = show x ++ " " ++ go rest


----------------------------------------------------------------------
-- LayerLike Instance
----------------------------------------------------------------------

export
LayerLike LstmState where
  applyGeneric {i} {o} (MkLstm iw rw b hs cs _ _ _) xs =
    let combined = matrixVectorMultiply iw xs + matrixVectorMultiply rw hs + b
        gates = lstmSplitGates {o} combined
        iGate = fst gates
        fGate = fst (snd gates)
        gGate = fst (snd (snd gates))
        oGate = snd (snd (snd gates))
        newCell = map sig fGate * cs + map sig iGate * map tanhBound gGate
        newHidden = map sig oGate * map tanhBound newCell
    in (MkLstm iw rw b newHidden newCell Nothing Nothing Nothing, newHidden)
    where
      sig : ty -> ty
      sig x = 1 / (1 + exp (-x))

  applyVar {i} {o} st@(MkLstm iw rw b hs cs iwT rwT bT) xs =
    case (iwT, rwT, bT) of
      -- Tensor-level forward: 2 mv + 1 add + fused LSTM gates (all at tensor level)
      (Just iwTensor, Just rwTensor, Just biasTensor) =>
        let (VTensor hsElems) = hs
            (VTensor csElems) = cs
            (VTensor xElems) = xs
            oI = cast {to=Int} o
            -- Stack input and hidden into 1D tensors (only these, not weights)
            inputT = vecStackTensor {n=i} xElems
            hiddenT = vecStackTensor {n=o} hsElems
            cellT = vecStackTensor {n=o} csElems
            -- Tensor-level gate computation: mv + mv + bias
            combined = tensorAdd (tensorAdd (tensorMv iwTensor inputT) (tensorMv rwTensor hiddenT)) biasTensor
            -- Fused sigmoid/tanh gate application in C
            pair = prim__lstmGatesPair combined cellT oI
            newHiddenT = prim__pairFirst pair
            newCellT = prim__pairSecond pair
            -- Unpack back to Variable vectors
            newHidden = VTensor $ tensorToScalars newHiddenT 0 o
            newCell = VTensor $ tensorToScalars newCellT 0 o
        in (MkLstm iw rw b newHidden newCell iwT rwT bT, newHidden)
      -- Scalar fallback
      _ =>
        let gateSize : Nat
            gateSize = 4 * o
            mulIW = matrixVectorMultiplyVar {m=gateSize, n=i} iw xs
            mulRW = matrixVectorMultiplyVar {m=gateSize, n=o} rw hs
            cellResult = lstmCellVar mulIW mulRW b cs
            newCell = fst cellResult
            newHidden = snd cellResult
        in (MkLstm iw rw b newHidden newCell Nothing Nothing Nothing, newHidden)

  emapLayer f (MkLstm iw rw b hs cs iwT rwT bT) =
    MkLstm (map f iw) (map f rw) (map f b) (map f hs) (map f cs) iwT rwT bT

  showLayer {i} {o} _ = "Lstm<" ++ show i ++ ":" ++ show o ++ ">"

  nameLayer {i} {o} prefx (MkLstm iw rw b hs cs _ _ _) =
    if prim__backendSupportsTensorParams == 1
      then -- Tensor path
        let (VTensor iwRows) = iw
            (VTensor rwRows) = rw
            (VTensor bElems) = b
            (VTensor hsElems) = hs
            (VTensor csElems) = cs
            gsI = cast {to=Int} (4 * o)
            iI = cast {to=Int} i
            oI = cast {to=Int} o
            iwT = prim__paramRegister (prefx ++ "_inputWeights") (prim__createParam2d gsI iI (let buf = prim__allocDoubles (gsI * iI) in packMatrixValues buf 0 {n=i} iwRows))
            rwT = prim__paramRegister (prefx ++ "_recurrentWeights") (prim__createParam2d gsI oI (let buf = prim__allocDoubles (gsI * oI) in packMatrixValues buf 0 {n=o} rwRows))
            bT = prim__paramRegister (prefx ++ "_biases") (prim__createParam1d gsI (let buf = prim__allocDoubles gsI in packScalarValues buf 0 bElems))
            h0T = prim__paramRegister (prefx ++ "_h0") (prim__createParam1d oI (let buf = prim__allocDoubles oI in packScalarValues buf 0 hsElems))
            c0T = prim__paramRegister (prefx ++ "_c0") (prim__createParam1d oI (let buf = prim__allocDoubles oI in packScalarValues buf 0 csElems))
        in MkLstm (VTensor $ buildViewMatrix (prefx ++ "_inputWeight") iwT 0 0 (4 * o) i)
                  (VTensor $ buildViewMatrix (prefx ++ "_recurrentWeight") rwT 0 0 (4 * o) o)
                  (VTensor $ buildViewVector (prefx ++ "_bias") bT 0 (4 * o))
                  (VTensor $ buildViewVector (prefx ++ "_h0") h0T 0 o)
                  (VTensor $ buildViewVector (prefx ++ "_c0") c0T 0 o)
                  (Just iwT) (Just rwT) (Just bT)
      else -- Scalar path (tape backend)
        let np = nameParam . (prefx ++ "_" ++)
        in MkLstm (zipWith (np "inputWeight") enumerate iw)
                  (zipWith (np "recurrentWeight") enumerate rw)
                  (zipWith (np "bias") enumerate b)
                  (zipWith (np "h0") enumerate hs)
                  (zipWith (np "c0") enumerate cs)
                  Nothing Nothing Nothing

  layerPrefix _ = "lstm"

  toDoubleLayer {i} {o} (MkLstm iw rw b hs cs iwT rwT bT) =
    case (iwT, rwT, bT) of
      (Just iwTensor, Just rwTensor, Just biasTensor) =>
        let wIW = buildDoubleMatrix iwTensor 0 (4 * o) i
            wRW = buildDoubleMatrix rwTensor 0 (4 * o) o
            wBias = buildDoubleVector biasTensor 0 (4 * o)
        in MkLstm (VTensor wIW) (VTensor wRW) (VTensor wBias) (map value hs) (map value cs) Nothing Nothing Nothing
      _ => MkLstm (map value iw) (map value rw) (map value b) (map value hs) (map value cs) Nothing Nothing Nothing
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
    in (updated, out, MkDebugEntry ("Lstm<" ++ show i ++ ":" ++ show o ++ ">")
         [("hidden", showVecD st.hiddenState), ("cell", showVecD st.cellState)])

  getParamIds (MkLstm iw rw b hs cs _ _ _) =
    tensorIds iw ++ tensorIds rw ++ tensorIds b ++ tensorIds hs ++ tensorIds cs
    where
      tensorIds : {dims : Vect rank Nat} -> Tensor dims Variable -> List String
      tensorIds = mapMaybe paramId . toList


----------------------------------------------------------------------
-- Constructors
----------------------------------------------------------------------

||| Create a raw LstmState (for NTM internal use)
export
mkLstmWith : {i, o : Nat} -> (Num ty, FromDouble ty) => InitStrategy -> IO (LstmState i o ty)
mkLstmWith {i} {o} initFn = do
  iw <- traverse (\_ => map fromDouble (initFn i (4 * o))) (the (Matrix (4 * o) i ty) zeros)
  rw <- traverse (\_ => map fromDouble (initFn o (4 * o))) (the (Matrix (4 * o) o ty) zeros)
  let b = the (Vector (4 * o) ty) zeros
  h0 <- traverse (\_ => map fromDouble (xavier uniform o 1)) (the (Vector o ty) zeros)
  c0 <- traverse (\_ => map fromDouble (xavier uniform o 1)) (the (Vector o ty) zeros)
  pure $ MkLstm iw rw b h0 c0 Nothing Nothing Nothing

||| Create a raw LstmState with default Xavier uniform init
export
mkLstm : {i, o : Nat} -> (Num ty, FromDouble ty) => IO (LstmState i o ty)
mkLstm = mkLstmWith (xavier uniform)

||| Create a LstmState wrapped in AnyLayer
export
lstmLayerWith : {i, o : Nat} -> (Num ty, FromDouble ty) => InitStrategy -> IO (AnyLayer i o ty)
lstmLayerWith initFn = map (MkAnyLayer LstmState) (mkLstmWith initFn)

||| Create a LstmState with Xavier uniform, wrapped in AnyLayer
export
lstmLayer : {i, o : Nat} -> (Num ty, FromDouble ty) => IO (AnyLayer i o ty)
lstmLayer = map (MkAnyLayer LstmState) mkLstm
