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
  iwBuf : Maybe AnyPtr
  rwBuf : Maybe AnyPtr


----------------------------------------------------------------------
-- LayerLike Instance
----------------------------------------------------------------------

export
LayerLike RnnState where
  applyGeneric (MkRnn iw rw b po iwb rwb) xs =
    let output = matrixVectorMultiply iw xs + matrixVectorMultiply rw po + b
    in (MkRnn iw rw b output iwb rwb, output)

  applyVar {i} {o} st@(MkRnn iw rw b po iwBuf rwBuf) xs =
    if i * o <= 4
      then applyGeneric st xs
      else let mulIW : Vector o Variable
               mulIW = maybe (matrixVectorMultiplyVar iw xs)
                             (\wb => matrixVectorMultiplyVarBuf {m=o, n=i} wb xs) iwBuf
               mulRW : Vector o Variable
               mulRW = maybe (matrixVectorMultiplyVar rw po)
                             (\wb => matrixVectorMultiplyVarBuf {m=o, n=o} wb po) rwBuf
               output = mulIW + mulRW + b
           in ({ previousOutput := output } st, output)

  emapLayer f (MkRnn iw rw b po iwb rwb) = MkRnn (map f iw) (map f rw) (map f b) (map f po) iwb rwb

  showLayer {i} {o} _ = "Rnn<" ++ show i ++ ":" ++ show o ++ ">"

  nameLayer {i} {o} prefx (MkRnn iw rw b po _ _) =
    let np = nameParam . (prefx ++ "_" ++)
        namedIW = zipWith (np "inputWeight") enumerate iw
        namedRW = zipWith (np "recurrentWeight") enumerate rw
        namedBias = zipWith (np "bias") enumerate b
    in if i * o <= 4
      then MkRnn namedIW namedRW namedBias po Nothing Nothing
      else let (VTensor iwRows) = namedIW
               (VTensor rwRows) = namedRW
               iwBuf = prim__weightBufAlloc (cast (o * i))
               iwBuf' = initWeightBuf iwBuf 0 iwRows
               rwBuf = prim__weightBufAlloc (cast (o * o))
               rwBuf' = initWeightBuf rwBuf 0 rwRows
           in MkRnn namedIW namedRW namedBias po (Just iwBuf') (Just rwBuf')

  layerPrefix _ = "rnn"

  toDoubleLayer (MkRnn iw rw b po _ _) =
    MkRnn (map value iw) (map value rw) (map value b) (map value po) Nothing Nothing

  debugApply {i} {o} st@(MkRnn _ _ _ po _ _) inp =
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

  syncBuffers (MkRnn (VTensor iwRows) (VTensor rwRows) b po (Just iwb) (Just rwb)) =
    let iwb' = syncWeightBuf iwb 0 iwRows
        rwb' = syncWeightBuf rwb 0 rwRows
    in MkRnn (VTensor iwRows) (VTensor rwRows) b po (Just iwb') (Just rwb')
  syncBuffers l = l

  getParamIds (MkRnn iw rw b _ _ _) = tensorIds iw ++ tensorIds rw ++ tensorIds b
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
  pure $ MkAnyLayer RnnState $ MkRnn iw rw zeros zeros Nothing Nothing

export
rnnLayer : {i, o : Nat} -> (Num ty, FromDouble ty) => IO (AnyLayer i o ty)
rnnLayer = rnnLayerWith (xavier uniform)
