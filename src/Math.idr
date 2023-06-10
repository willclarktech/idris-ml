module Math

import Data.Vect

import Tensor
import Variable


ActivationFunction ty = ty -> ty

export
sigmoid : ActivationFunction Variable
sigmoid x = 1.0 / (1.0 + exp (-x))

NormalizationFunction ty = {n : Nat} -> Vector n ty -> Vector n ty

export
softmax : NormalizationFunction Variable
softmax xs =
  let exps = map exp xs
  in map (/(sum exps)) exps

AggregateFunction f ty = f ty -> ty

export
mean : (Num ty, Fractional ty) => {n : Nat} -> AggregateFunction (Vector n) ty
mean {n} xs =
  let tot = fromInteger $ natToInteger $ length xs
  in sum xs / tot

LossFunction ty = {n : Nat} -> Vector n ty -> Vector n ty -> ty

export
meanSquaredError : LossFunction Variable
meanSquaredError {n} predictions ys = mean $ zipWith squaredError predictions ys
  where
    squaredError : Variable -> Variable -> Variable
    squaredError prediction y = pow (prediction - y) 2

export
binaryCrossEntropy : LossFunction Variable
binaryCrossEntropy predictions ys = mean $ zipWith bceError predictions ys
  where
    bceError : Variable -> Variable -> Variable
    bceError prediction y =
      -(y * log prediction + (1 - y) * log (1 - prediction))

export
crossEntropy : LossFunction Variable
crossEntropy predictions ys = ((-1) *) $ sum $ zipWith (\p, y => y * log p + (1 - y) * log (1 - p)) predictions ys

-- Tensor

export
dotProduct : Num a => {n : Nat} -> Vector n a -> Vector n a -> a
dotProduct {n} v1 v2 = sum $ v1 * v2

export
matrixVectorMultiply : Num a => {m, n : Nat} -> Matrix m n a -> Vector n a -> Vector m a
matrixVectorMultiply {n} (VTensor m) v = VTensor $ map (STensor . dotProduct v) m
