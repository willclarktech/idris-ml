module Math

import Data.Vect
import Tensor


ActivationFunction ty = ty -> ty

export
sigmoid : ActivationFunction Double
sigmoid x = 1.0 / (1.0 + exp (-x))

NormalizationFunction ty = {n : Nat} -> Vector n ty -> Vector n ty

export
softmax : NormalizationFunction Double
softmax xs =
  let exps = map exp xs
  in map (/(sum exps)) exps

AggregateFunction f ty = f ty -> ty

export
mean : {n : Nat} -> AggregateFunction (Vector n) Double
mean {n} xs = sum xs / cast (length xs)

LossFunction ty = {n : Nat} -> Vector n ty -> Vector n ty -> ty

export
meanSquaredError : LossFunction Double
meanSquaredError {n} predictions ys = mean $ zipWith squaredError predictions ys
  where
    squaredError : Double -> Double -> Double
    squaredError prediction y = pow (prediction - y) 2

export
binaryCrossEntropy : LossFunction Double
binaryCrossEntropy predictions ys = mean $ zipWith bceError predictions ys
  where
    bceError : Double -> Double -> Double
    bceError prediction y =
      -(y * log prediction + (1 - y) * log (1 - prediction))

export
crossEntropy : LossFunction Double
crossEntropy predictions ys = ((-1) *) $ sum $ zipWith (\p, y => y * log p + (1 - y) * log (1 - p)) predictions ys

-- Tensor

export
dotProduct : Num a => {n : Nat} -> Vector n a -> Vector n a -> a
dotProduct {n} v1 v2 = sum $ v1 * v2

export
matrixVectorMultiply : Num a => {m, n : Nat} -> Matrix m n a -> Vector n a -> Vector m a
matrixVectorMultiply {n} (VTensor m) v = VTensor $ map (STensor . dotProduct v) m
