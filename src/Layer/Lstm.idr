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
  iwBuf : Maybe AnyPtr
  rwBuf : Maybe AnyPtr
  bBuf : Maybe AnyPtr
  h0Buf : Maybe AnyPtr
  c0Buf : Maybe AnyPtr


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
  applyGeneric {i} {o} (MkLstm iw rw b hs cs iwb rwb bb h0b c0b) xs =
    let combined = matrixVectorMultiply iw xs + matrixVectorMultiply rw hs + b
        gates = lstmSplitGates {o} combined
        iGate = fst gates
        fGate = fst (snd gates)
        gGate = fst (snd (snd gates))
        oGate = snd (snd (snd gates))
        newCell = map sig fGate * cs + map sig iGate * map tanhBound gGate
        newHidden = map sig oGate * map tanhBound newCell
    in (MkLstm iw rw b newHidden newCell iwb rwb Nothing h0b c0b, newHidden)
    where
      sig : ty -> ty
      sig x = 1 / (1 + exp (-x))

  applyVar {i} {o} st@(MkLstm iw rw b hs cs iwBuf rwBuf bBuf h0Buf c0Buf) xs =
    if i * o <= 4
      then applyGeneric st xs
      else case (iwBuf, rwBuf, bBuf) of
        -- Full buffer-passing: MatVec outputs feed directly into LstmCell (no Variables)
        (Just iwb, Just rwb, Just bb) =>
          let gateSize : Nat
              gateSize = 4 * o
              mulIWResult = matrixVectorMultiplyVarBufOut {m=gateSize, n=i} iwb xs
              mulRWResult = matrixVectorMultiplyVarBufOut {m=gateSize, n=o} rwb hs
              cellResult = lstmCellVarFromBufs
                             (fst mulIWResult) (snd mulIWResult)
                             (fst mulRWResult) (snd mulRWResult) bb cs
              newCell = fst cellResult
              newHidden = snd cellResult
          in ({ hiddenState := newHidden, cellState := newCell } st, newHidden)
        -- Fallback: materialize Variables
        _ =>
          let gateSize : Nat
              gateSize = 4 * o
              mulIW : Vector gateSize Variable
              mulIW = maybe (matrixVectorMultiplyVar {m=gateSize, n=i} iw xs)
                            (\wb => matrixVectorMultiplyVarBuf {m=gateSize, n=i} wb xs) iwBuf
              mulRW : Vector gateSize Variable
              mulRW = maybe (matrixVectorMultiplyVar {m=gateSize, n=o} rw hs)
                            (\wb => matrixVectorMultiplyVarBuf {m=gateSize, n=o} wb hs) rwBuf
              cellResult = maybe (lstmCellVar mulIW mulRW b cs)
                                 (\bb => lstmCellVarBuf mulIW mulRW bb cs) bBuf
              newCell = fst cellResult
              newHidden = snd cellResult
          in ({ hiddenState := newHidden, cellState := newCell } st, newHidden)

  emapLayer f (MkLstm iw rw b hs cs iwb rwb bb h0b c0b) =
    MkLstm (map f iw) (map f rw) (map f b) (map f hs) (map f cs) iwb rwb bb h0b c0b

  showLayer {i} {o} _ = "Lstm<" ++ show i ++ ":" ++ show o ++ ">"

  nameLayer {i} {o} prefx (MkLstm iw rw b hs cs _ _ _ _ _) =
    let np = nameParam . (prefx ++ "_" ++)
        gateSize : Nat
        gateSize = 4 * o
        namedIW = zipWith (np "inputWeight") enumerate iw
        namedRW = zipWith (np "recurrentWeight") enumerate rw
        namedBias = zipWith (np "bias") enumerate b
        namedH0 = zipWith (np "h0") enumerate hs
        namedC0 = zipWith (np "c0") enumerate cs
    in if i * o <= 4
      then MkLstm namedIW namedRW namedBias namedH0 namedC0 Nothing Nothing Nothing Nothing Nothing
      else let (VTensor iwRows) = namedIW
               (VTensor rwRows) = namedRW
               (VTensor biasElems) = namedBias
               (VTensor h0Elems) = namedH0
               (VTensor c0Elems) = namedC0
               iwb = prim__weightBufAlloc (cast (gateSize * i))
               iwb' = initWeightBuf iwb 0 iwRows
               rwb = prim__weightBufAlloc (cast (gateSize * o))
               rwb' = initWeightBuf rwb 0 rwRows
               bb = prim__weightBufAlloc (cast gateSize)
               bb' = initWeightBufRow bb 0 biasElems
               h0b = prim__weightBufAlloc (cast o)
               h0b' = initWeightBufRow h0b 0 h0Elems
               c0b = prim__weightBufAlloc (cast o)
               c0b' = initWeightBufRow c0b 0 c0Elems
           in MkLstm namedIW namedRW namedBias namedH0 namedC0 (Just iwb') (Just rwb') (Just bb') (Just h0b') (Just c0b')

  layerPrefix _ = "lstm"

  toDoubleLayer (MkLstm iw rw b hs cs _ _ _ _ _) =
    MkLstm (map value iw) (map value rw) (map value b) (map value hs) (map value cs)
           Nothing Nothing Nothing Nothing Nothing

  debugApply {i} {o} st inp =
    let (updated, out) = applyGeneric st inp
    in (updated, out, MkDebugEntry ("Lstm<" ++ show i ++ ":" ++ show o ++ ">")
         [("hidden", showVecD st.hiddenState), ("cell", showVecD st.cellState)])

  syncBuffers (MkLstm (VTensor iwRows) (VTensor rwRows) (VTensor biasElems) (VTensor h0Elems) (VTensor c0Elems) (Just iwb) (Just rwb) (Just bb) (Just h0b) (Just c0b)) =
    let iwb' = syncWeightBuf iwb 0 iwRows
        rwb' = syncWeightBuf rwb 0 rwRows
        bb' = syncWeightBufRow bb 0 biasElems
        h0b' = syncWeightBufRow h0b 0 h0Elems
        c0b' = syncWeightBufRow c0b 0 c0Elems
    in MkLstm (VTensor iwRows) (VTensor rwRows) (VTensor biasElems) (VTensor h0Elems) (VTensor c0Elems) (Just iwb') (Just rwb') (Just bb') (Just h0b') (Just c0b')
  syncBuffers (MkLstm (VTensor iwRows) (VTensor rwRows) b hs cs (Just iwb) (Just rwb) Nothing h0b c0b) =
    let iwb' = syncWeightBuf iwb 0 iwRows
        rwb' = syncWeightBuf rwb 0 rwRows
    in MkLstm (VTensor iwRows) (VTensor rwRows) b hs cs (Just iwb') (Just rwb') Nothing h0b c0b
  syncBuffers l = l

  applyDeltasAndSync deltas (MkLstm iw rw b hs cs (Just iwb) (Just rwb) (Just bb) (Just h0b) (Just c0b)) =
    let iwb' = prim__weightBufApplyDeltas iwb deltas
        rwb' = prim__weightBufApplyDeltas rwb deltas
        bb' = prim__weightBufApplyDeltas bb deltas
        h0b' = prim__weightBufApplyDeltas h0b deltas
        c0b' = prim__weightBufApplyDeltas c0b deltas
    in MkLstm iw rw b hs cs (Just iwb') (Just rwb') (Just bb') (Just h0b') (Just c0b')
  applyDeltasAndSync deltas (MkLstm iw rw b hs cs (Just iwb) (Just rwb) Nothing h0b c0b) =
    let iwb' = prim__weightBufApplyDeltas iwb deltas
        rwb' = prim__weightBufApplyDeltas rwb deltas
    in MkLstm iw rw b hs cs (Just iwb') (Just rwb') Nothing h0b c0b
  applyDeltasAndSync _ l = l

  readFromBuffers (MkLstm (VTensor iwRows) (VTensor rwRows) (VTensor biasElems) (VTensor h0Elems) (VTensor c0Elems) (Just iwb) (Just rwb) (Just bb) (Just h0b) (Just c0b)) =
    MkLstm (VTensor (readWeightBuf iwb 0 iwRows)) (VTensor (readWeightBuf rwb 0 rwRows))
              (VTensor (readWeightBufRow bb 0 biasElems))
              (VTensor (readWeightBufRow h0b 0 h0Elems)) (VTensor (readWeightBufRow c0b 0 c0Elems))
              (Just iwb) (Just rwb) (Just bb) (Just h0b) (Just c0b)
  readFromBuffers (MkLstm (VTensor iwRows) (VTensor rwRows) b hs cs (Just iwb) (Just rwb) Nothing h0b c0b) =
    MkLstm (VTensor (readWeightBuf iwb 0 iwRows)) (VTensor (readWeightBuf rwb 0 rwRows))
              b hs cs (Just iwb) (Just rwb) Nothing h0b c0b
  readFromBuffers l = l

  getParamIds (MkLstm iw rw b hs cs _ _ _ _ _) =
    tensorIds iw ++ tensorIds rw ++ tensorIds b ++ tensorIds hs ++ tensorIds cs
    where
      tensorIds : {dims : Vect rank Nat} -> Tensor dims Variable -> List String
      tensorIds = mapMaybe paramId . toList


----------------------------------------------------------------------
-- LSTM-specific buffer-passing forward (for NTM)
----------------------------------------------------------------------

||| LSTM forward that also returns the raw output buffer + cell const start.
||| Returns (updatedLayer, hidden, Just (outBuf, cellConstStart)) on the full-buffer path,
||| or (updatedLayer, hidden, Nothing) on fallback.
export
applyLstmGetBuf : {i, o : Nat} -> LstmState i o Variable -> Vector i Variable
    -> (LstmState i o Variable, Vector o Variable, Maybe (AnyPtr, Int))
applyLstmGetBuf {i} {o} st@(MkLstm iw rw b hs cs iwBuf rwBuf bBuf h0Buf c0Buf) xs =
  if i * o <= 4
    then let (st', out) = applyGeneric st xs
         in (st', out, Nothing)
    else case (iwBuf, rwBuf, bBuf) of
      (Just iwb, Just rwb, Just bb) =>
        let gateSize : Nat
            gateSize = 4 * o
            mulIWResult = matrixVectorMultiplyVarBufOut {m=gateSize, n=i} iwb xs
            mulRWResult = matrixVectorMultiplyVarBufOut {m=gateSize, n=o} rwb hs
        in case lstmCellVarFromBufsExt
                  (fst mulIWResult) (snd mulIWResult)
                  (fst mulRWResult) (snd mulRWResult) bb cs of
          (newCell, newHidden, outBuf, cellConstStart) =>
            ({ hiddenState := newHidden, cellState := newCell } st, newHidden, Just (outBuf, cellConstStart))
      _ =>
        let gateSize : Nat
            gateSize = 4 * o
            mulIW : Vector gateSize Variable
            mulIW = maybe (matrixVectorMultiplyVar {m=gateSize, n=i} iw xs)
                          (\wb => matrixVectorMultiplyVarBuf {m=gateSize, n=i} wb xs) iwBuf
            mulRW : Vector gateSize Variable
            mulRW = maybe (matrixVectorMultiplyVar {m=gateSize, n=o} rw hs)
                          (\wb => matrixVectorMultiplyVarBuf {m=gateSize, n=o} wb hs) rwBuf
            cellResult = maybe (lstmCellVar mulIW mulRW b cs)
                               (\bb => lstmCellVarBuf mulIW mulRW bb cs) bBuf
            newCell = fst cellResult
            newHidden = snd cellResult
        in ({ hiddenState := newHidden, cellState := newCell } st, newHidden, Nothing)


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
  pure $ MkLstm iw rw b h0 c0 Nothing Nothing Nothing Nothing Nothing

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
