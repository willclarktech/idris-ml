module Math

import Data.Vect

import Floating
import Tensor


----------------------------------------------------------------------
-- Activation Functions
----------------------------------------------------------------------

public export
0 ActivationFunction : Type -> Type
ActivationFunction ty = ty -> ty

export
sigmoid : (FromDouble ty, Neg ty, Fractional ty, Floating ty) => ActivationFunction ty
sigmoid x = 1.0 / (1.0 + exp (-x))

----------------------------------------------------------------------
-- Normalization Functions
----------------------------------------------------------------------

public export
0 NormalizationFunction : Type -> Type
NormalizationFunction ty = {n : Nat} -> Vector n ty -> Vector n ty

export
softmax : (Fractional ty, Floating ty) => NormalizationFunction ty
softmax xs =
  let exps = map exp xs
  in map (/(sum exps)) exps

----------------------------------------------------------------------
-- Aggregate Functions
----------------------------------------------------------------------

public export
0 AggregateFunction : (Type -> Type) -> Type -> Type
AggregateFunction f ty = f ty -> ty

export
mean : (Num ty, Fractional ty) => {n : Nat} -> AggregateFunction (Vector n) ty
mean {n} xs =
  let tot = fromInteger $ natToInteger $ length xs
  in sum xs / tot

----------------------------------------------------------------------
-- Loss Functions
----------------------------------------------------------------------

public export
0 LossFunction : Type -> Type
LossFunction ty = {n : Nat} -> Vector n ty -> Vector n ty -> ty

reduceLoss : (Num ty, Fractional ty) => (ty -> ty -> ty) -> LossFunction ty
reduceLoss pointwise predictions targets = mean $ zipWith pointwise predictions targets

export
meanSquaredError : (Neg ty, Fractional ty, Floating ty) => LossFunction ty
meanSquaredError = reduceLoss (\p, y => (p - y) ^ 2)

export
binaryCrossEntropy : (Neg ty, Fractional ty, Floating ty) => LossFunction ty
binaryCrossEntropy = reduceLoss (\p, y => -(y * log p + (1 - y) * log (1 - p)))

||| Equivalent to but more numerically stable than (BCE . sigmoid)
export
binaryCrossEntropyWithLogits : (FromDouble ty, Neg ty, Fractional ty, Floating ty) => LossFunction ty
binaryCrossEntropyWithLogits = reduceLoss (\p, y =>
  let sigp = sigmoid p in -(y * log sigp + (1 - y) * log (1 - sigp)))

export
crossEntropy : (Num ty, Neg ty, Floating ty, Fractional ty, Ord ty) => LossFunction ty
crossEntropy = reduceLoss clampedLoss
  where
    clampedLoss : ty -> ty -> ty
    clampedLoss p y =
      let ep = pow 10 (-7)
          pp = max ep (min p (1 - ep))
      in -(y * log pp) + -(1 - y) * log (1 - pp)

----------------------------------------------------------------------
-- Encoding
----------------------------------------------------------------------

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

----------------------------------------------------------------------
-- Linear Algebra
----------------------------------------------------------------------

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
