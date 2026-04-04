module Layer.Rnn

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
-- RNN State
----------------------------------------------------------------------

public export
record RnnState (inputSize : Nat) (outputSize : Nat) (ty : Type) where
  constructor MkRnn
  inputWeights : Matrix outputSize inputSize ty
  recurrentWeights : Matrix outputSize outputSize ty
  bias : Vector outputSize ty
  previousOutput : Vector outputSize ty


----------------------------------------------------------------------
-- LayerLike Instance
----------------------------------------------------------------------

export
LayerLike RnnState where
  applyGeneric (MkRnn iw rw b po) xs =
    let output = matrixVectorMultiply iw xs + matrixVectorMultiply rw po + b
    in (MkRnn iw rw b output, output)

  applyVar st@(MkRnn iw rw b po) xs =
    let output = matrixVectorMultiplyVar iw xs + matrixVectorMultiplyVar rw po + b
    in ({ previousOutput := output } st, output)

  emapLayer f (MkRnn iw rw b po) = MkRnn (map f iw) (map f rw) (map f b) (map f po)

  showLayer {i} {o} _ = "Rnn<" ++ show i ++ ":" ++ show o ++ ">"

  nameLayer prefx (MkRnn iw rw b po) =
    let np = nameParam . (prefx ++ "_" ++)
        namedIW = zipWith (np "inputWeight") enumerate iw
        namedRW = zipWith (np "recurrentWeight") enumerate rw
        namedBias = zipWith (np "bias") enumerate b
    in MkRnn namedIW namedRW namedBias po

  layerPrefix _ = "rnn"

  toDoubleLayer (MkRnn iw rw b po) =
    MkRnn (map value iw) (map value rw) (map value b) (map value po)

  debugApply {i} {o} st@(MkRnn _ _ _ po) inp =
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

  getParamIds (MkRnn iw rw b _) = tensorIds iw ++ tensorIds rw ++ tensorIds b
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
  pure $ MkAnyLayer RnnState $ MkRnn iw rw zeros zeros

export
rnnLayer : {i, o : Nat} -> (Num ty, FromDouble ty) => IO (AnyLayer i o ty)
rnnLayer = rnnLayerWith (xavier uniform)
