module Layer.Rnn

import Data.Vect
import Data.Zippable

import Floating
import Init
import Layer.Core
import Layer.Linear
import Math
import Tensor
import Util
import Variable


----------------------------------------------------------------------
-- RNN State
----------------------------------------------------------------------

public export
record RnnState (inputSize : Nat) (outputSize : Nat) (ty : Type) where
  constructor MkRnn
  inputWeights : Matrix outputSize inputSize ty
  recurrentWeights : Matrix outputSize outputSize ty
  bias : Vector outputSize ty
  previousOutput : Vector outputSize ty
  -- Consolidated tensor handles (set by nameLayer)
  iwTensor : Maybe AnyPtr
  rwTensor : Maybe AnyPtr
  biasTensor : Maybe AnyPtr
  prevOutTensor : Maybe AnyPtr


----------------------------------------------------------------------
-- LayerLike Instance
----------------------------------------------------------------------

%default partial
export
LayerLike RnnState where
  applyGeneric (MkRnn iw rw b po _ _ _ _) xs =
    let output = matrixVectorMultiply iw xs + matrixVectorMultiply rw po + b
    in (MkRnn iw rw b output Nothing Nothing Nothing Nothing, output)

  applyVar st@(MkRnn iw rw b po _ _ _ _) xs =
    let output = matrixVectorMultiplyVar iw xs + matrixVectorMultiplyVar rw po + b
    in ({ previousOutput := output, prevOutTensor := Nothing } st, output)

  applyVarTensor {i} {o} st inputT =
    case (iwTensor st, rwTensor st, biasTensor st) of
      (Just iwT, Just rwT, Just bT) =>
        let poT = case prevOutTensor st of
              Just pt => pt
              Nothing => -- First call: previousOutput is zeros
                let buf = prim__allocDoubles (cast {to=Int} o)
                in prim__createState1d (cast {to=Int} o) buf
            resultT = tensorAdd (tensorAdd (tensorMv iwT inputT) (tensorMv rwT poT)) bT
        in ({ prevOutTensor := Just resultT } st, resultT)
      _ => idris_crash "Rnn: weight tensors not initialized (call autoName first)"

  emapLayer f (MkRnn iw rw b po iwt rwt bt pot) =
    MkRnn (map f iw) (map f rw) (map f b) (map f po) iwt rwt bt pot

  showLayer {i} {o} _ = "Rnn<" ++ show i ++ ":" ++ show o ++ ">"

  nameLayer {i} {o} prefx (MkRnn iw rw b po _ _ _ _) =
    if prim__backendSupportsTensorParams == 1
      then
        let oI = cast {to=Int} o
            iI = cast {to=Int} i
            -- Input weights [o, i]
            (VTensor iwRows) = iw
            iwBuf = prim__allocDoubles (oI * iI)
            iwBuf' = packMatrixValues iwBuf 0 {n=i} iwRows
            iwT = prim__paramRegister (prefx ++ "_iw") (prim__createParam2d oI iI iwBuf')
            -- Recurrent weights [o, o]
            (VTensor rwRows) = rw
            rwBuf = prim__allocDoubles (oI * oI)
            rwBuf' = packMatrixValues rwBuf 0 {n=o} rwRows
            rwT = prim__paramRegister (prefx ++ "_rw") (prim__createParam2d oI oI rwBuf')
            -- Bias [o]
            (VTensor bElems) = b
            bBuf = prim__allocDoubles oI
            bBuf' = packScalarValues bBuf 0 bElems
            bT = prim__paramRegister (prefx ++ "_bias") (prim__createParam1d oI bBuf')
        in MkRnn (VTensor $ buildViewMatrix (prefx ++ "_iw") iwT 0 0 o i)
                 (VTensor $ buildViewMatrix (prefx ++ "_rw") rwT 0 0 o o)
                 (VTensor $ buildViewVector (prefx ++ "_bias") bT 0 o)
                 po (Just iwT) (Just rwT) (Just bT) Nothing
      else
        let np = nameParam . (prefx ++ "_" ++)
            namedIW = zipWith (np "inputWeight") enumerate iw
            namedRW = zipWith (np "recurrentWeight") enumerate rw
            namedBias = zipWith (np "bias") enumerate b
        in MkRnn namedIW namedRW namedBias po Nothing Nothing Nothing Nothing

  layerPrefix _ = "rnn"

  toDoubleLayer (MkRnn iw rw b po _ _ _ _) =
    MkRnn (map value iw) (map value rw) (map value b) (map value po)
          Nothing Nothing Nothing Nothing

  resetState st = { previousOutput := zeros, prevOutTensor := Nothing } st

  debugApply {i} {o} st@(MkRnn _ _ _ po _ _ _ _) inp =
    let (updated, out) = applyGeneric st inp
    in (updated, out, MkDebugEntry ("Rnn<" ++ show i ++ ":" ++ show o ++ ">")
         [("hidden", showVec po)])
    where
      showVec : {n : Nat} -> Vector n Double -> String
      showVec (VTensor xs) = "[" ++ go xs ++ "]"
        where
          go : Vect k (Tensor [] Double) -> String
          go [] = ""
          go [STensor x] = show x
          go (STensor x :: rest) = show x ++ " " ++ go rest

  getParamIds (MkRnn iw rw b _ _ _ _ _) = tensorIds iw ++ tensorIds rw ++ tensorIds b
    where
      tensorIds : {dims : Vect rank Nat} -> Tensor dims Variable -> List String
      tensorIds = mapMaybe paramId . toList


----------------------------------------------------------------------
-- Constructors
----------------------------------------------------------------------

export
rnnLayerWith : {i, o : Nat} -> (Num ty, FromDouble ty) => InitStrategy -> IO (AnyLayer i o ty)
rnnLayerWith initFn = do
  iw <- traverse (\_ => map fromDouble (initFn i o)) (the (Matrix o i ty) zeros)
  rw <- traverse (\_ => map fromDouble (initFn o o)) (the (Matrix o o ty) zeros)
  pure $ MkAnyLayer RnnState $ MkRnn iw rw zeros zeros Nothing Nothing Nothing Nothing

export
rnnLayer : {i, o : Nat} -> (Num ty, FromDouble ty) => IO (AnyLayer i o ty)
rnnLayer = rnnLayerWith (xavier uniform)
