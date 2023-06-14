module Math

import Data.Vect

import Floating
import Tensor


ActivationFunction ty = ty -> ty

export
sigmoid : (FromDouble ty, Neg ty, Fractional ty, Floating ty) => ActivationFunction ty
sigmoid x = 1.0 / (1.0 + exp (-x))

NormalizationFunction ty = {n : Nat} -> Vector n ty -> Vector n ty

export
softmax : (Fractional ty, Floating ty) => NormalizationFunction ty
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
meanSquaredError : (Neg ty, Fractional ty, Floating ty) => LossFunction ty
meanSquaredError {n} predictions ys = mean $ zipWith squaredError predictions ys
  where
    squaredError : ty -> ty -> ty
    squaredError prediction y = (prediction - y) ^ 2

export
binaryCrossEntropy : (Neg ty, Fractional ty, Floating ty) => LossFunction ty
binaryCrossEntropy predictions ys = mean $ zipWith bceError predictions ys
  where
    bceError : ty -> ty -> ty
    bceError prediction y = -(y * log prediction + (1 - y) * log (1 - prediction))

||| Equivalent to but more numerically stable than (BCE . sigmoid)
export
binaryCrossEntropyWithLogits : (FromDouble ty, Neg ty, Fractional ty, Floating ty) => LossFunction ty
binaryCrossEntropyWithLogits predictions ys = mean $ zipWith bceError predictions ys
  where
    bceError : ty -> ty -> ty
    bceError prediction y =
      let sigp = sigmoid prediction
      in -(y * log sigp + (1 - y) * log (1 - sigp))

export
crossEntropy : (Num ty, Neg ty, Floating ty) => LossFunction ty
crossEntropy predictions ys = ((-1) *) $ sum $ zipWith (\p, y => y * log p + (1 - y) * log (1 - p)) predictions ys

-- Tensor

export
oneHotEncode : {n : Nat} -> Fin n -> Vector n Nat
oneHotEncode i = VTensor $ replaceAt i 1 $ replicate n 0

export
oneHotDecode : {n : Nat} -> Vector n Nat -> Maybe (Fin n)
oneHotDecode (VTensor v) = findIndex (== STensor 1) v

export
dotProduct : Num a => {n : Nat} -> Vector n a -> Vector n a -> a
dotProduct {n} v1 v2 = sum $ v1 * v2

export
matrixVectorMultiply : Num a => {m, n : Nat} -> Matrix m n a -> Vector n a -> Vector m a
matrixVectorMultiply {n} (VTensor m) v = VTensor $ map (STensor . dotProduct v) m
