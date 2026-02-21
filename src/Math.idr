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
crossEntropy : (Num ty, Neg ty, Floating ty, Fractional ty, Ord ty) => LossFunction ty
crossEntropy {n} predictions ys =
  let losses = zipWith loss predictions ys
  in sum losses / fromInteger (natToInteger n)
  where
    epsilon : ty
    epsilon = pow 10 (-7)
    loss : ty -> ty -> ty
    loss prediction y =
      let p = max epsilon (min prediction (1 - epsilon))
      in - (y * log p) + -(1 - y) * log (1 - p)

export
oneHotEncode : {n : Nat} -> Fin n -> Vector n Nat
oneHotEncode i = VTensor $ replaceAt i 1 $ replicate n 0

export
oneHotDecode : {n : Nat} -> Vector n Nat -> Maybe (Fin n)
oneHotDecode (VTensor v) = findIndex (== STensor 1) v

-- TODO: Improve efficiency
export
argmax: Ord ty => {n : Nat} -> Vector (S n) ty -> Fin (S n)
argmax (VTensor v@(x::xs)) =
  foldl maxIndex FZ Data.Vect.Fin.range
  where
    -- current indexes v, next indexes xs
    maxIndex : Fin (S n) -> Fin n -> Fin (S n)
    maxIndex current next =
      let
        (STensor currentValue) = Data.Vect.index current v
        (STensor nextValue) = Data.Vect.index next xs
      -- Prioritise earlier value
      in if nextValue > currentValue
        -- Need to convert from index of xs to index of v
        then FS next
        else current

export
dotProduct : Num ty => {n : Nat} -> Vector n ty -> Vector n ty -> ty
dotProduct v1 v2 = sum $ v1 * v2

export
l2Norm : (Floating ty, Num ty, Ord ty) => {n : Nat} -> Vector n ty -> ty
l2Norm v =
  let
    norm = sqrt $ sum $ map (^ 2) v
    -- NOTE: Necessary to avoid division by 0
    epsilon = pow 10 (-7)
  in max norm epsilon

export
cosineSimilarity : (Floating ty, Fractional ty, Ord ty) => {n : Nat} -> Vector n ty -> Vector n ty -> ty
cosineSimilarity a b = dotProduct a b / (l2Norm a * l2Norm b)

export
matrixVectorMultiply : Num ty => {n : Nat} -> Matrix m n ty -> Vector n ty -> Vector m ty
matrixVectorMultiply (VTensor mat) vec = VTensor $ map (STensor . dotProduct vec) mat

export
vectorMatrixMultiply : (Num ty) => {n : Nat} -> Vector n ty -> Matrix m n ty -> Vector m ty
vectorMatrixMultiply = flip matrixVectorMultiply
