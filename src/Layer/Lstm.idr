module Layer.Lstm

import Data.Vect
import Data.Zippable

import Floating
import Init
import Layer.Core
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
  applyGeneric {i} {o} (MkLstm iw rw b hs cs) xs =
    let combined = matrixVectorMultiply iw xs + matrixVectorMultiply rw hs + b
        gates = lstmSplitGates {o} combined
        iGate = fst gates
        fGate = fst (snd gates)
        gGate = fst (snd (snd gates))
        oGate = snd (snd (snd gates))
        newCell = map sig fGate * cs + map sig iGate * map tanhBound gGate
        newHidden = map sig oGate * map tanhBound newCell
    in (MkLstm iw rw b newHidden newCell, newHidden)
    where
      sig : ty -> ty
      sig x = 1 / (1 + exp (-x))

  applyVar {i} {o} (MkLstm iw rw b hs cs) xs =
    let gateSize : Nat
        gateSize = 4 * o
        mulIW = matrixVectorMultiplyVar {m=gateSize, n=i} iw xs
        mulRW = matrixVectorMultiplyVar {m=gateSize, n=o} rw hs
        cellResult = lstmCellVar mulIW mulRW b cs
        newCell = fst cellResult
        newHidden = snd cellResult
    in (MkLstm iw rw b newHidden newCell, newHidden)

  emapLayer f (MkLstm iw rw b hs cs) =
    MkLstm (map f iw) (map f rw) (map f b) (map f hs) (map f cs)

  showLayer {i} {o} _ = "Lstm<" ++ show i ++ ":" ++ show o ++ ">"

  nameLayer prefx (MkLstm iw rw b hs cs) =
    let np = nameParam . (prefx ++ "_" ++)
        namedIW = zipWith (np "inputWeight") enumerate iw
        namedRW = zipWith (np "recurrentWeight") enumerate rw
        namedBias = zipWith (np "bias") enumerate b
        namedH0 = zipWith (np "h0") enumerate hs
        namedC0 = zipWith (np "c0") enumerate cs
    in MkLstm namedIW namedRW namedBias namedH0 namedC0

  layerPrefix _ = "lstm"

  toDoubleLayer (MkLstm iw rw b hs cs) =
    MkLstm (map value iw) (map value rw) (map value b) (map value hs) (map value cs)

  debugApply {i} {o} st inp =
    let (updated, out) = applyGeneric st inp
    in (updated, out, MkDebugEntry ("Lstm<" ++ show i ++ ":" ++ show o ++ ">")
         [("hidden", showVecD st.hiddenState), ("cell", showVecD st.cellState)])

  getParamIds (MkLstm iw rw b hs cs) =
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
  pure $ MkLstm iw rw b h0 c0

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
