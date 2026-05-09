module Math

import Data.Vect

import Floating
import Array


----------------------------------------------------------------------
-- Activation Functions
----------------------------------------------------------------------

public export
0 ActivationFunction : Type -> Type
ActivationFunction ty = ty -> ty

export
sigmoid : (FromDouble ty, Neg ty, Fractional ty, Floating ty) => ActivationFunction ty
sigmoid x = 1.0 / (1.0 + exp (-x))

export
tanh : (FromDouble ty, Neg ty, Fractional ty, Floating ty) => ActivationFunction ty
tanh x = 2.0 * sigmoid (2.0 * x) - 1.0

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

export
logSoftmax : (FromDouble ty, Cast ty Double, Neg ty, Floating ty, Fractional ty) => NormalizationFunction ty
logSoftmax xs =
  let shifted = map (\x => x - maxVal) xs
      logSumExp = log (sum (map exp shifted))
  in map (\x => x - maxVal - logSumExp) xs
  where
    -- Detach max from graph so gradient only flows through x - logSumExp
    maxVal : ty
    maxVal = fromDouble $ foldr max (-1.0e308) (map cast xs)

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

||| Numerically stable BCE: max(p,0) - p*y + log(1 + exp(-|p|))
export
binaryCrossEntropyWithLogits : (FromDouble ty, Neg ty, Fractional ty, Floating ty, Ord ty) => LossFunction ty
binaryCrossEntropyWithLogits = reduceLoss (\p, y =>
  let relu_p = max p (fromDouble 0.0)
      abs_p = if p >= fromDouble 0.0 then p else negate p
  in relu_p - p * y + log (1 + exp (negate abs_p)))

export
crossEntropy : (Num ty, Neg ty, Floating ty, Fractional ty, Ord ty) => LossFunction ty
crossEntropy = reduceLoss clampedLoss
  where
    clampedLoss : ty -> ty -> ty
    clampedLoss p y =
      let ep = pow 10 (-6)
          pp = max ep (min p (1 - ep))
      in -(y * log pp) + -(1 - y) * log (1 - pp)

||| Negative log-likelihood loss for use with logSoftmax outputs.
||| No log in the loss = no 1/pp gradient explosion.
export
nllLoss : (Neg ty, Fractional ty) => LossFunction ty
nllLoss = reduceLoss (\p, y => -(y * p))

||| L1 loss (mean absolute error): mean(|pred - target|).
export
l1Loss : (Neg ty, Fractional ty, Ord ty) => LossFunction ty
l1Loss = reduceLoss (\p, y => let d = p - y in max d (negate d))

||| Huber loss (Smooth L1): quadratic near zero, linear far away.
||| Robust to outliers. delta controls the transition point.
export
huberLoss : (FromDouble ty, Neg ty, Fractional ty, Ord ty) => (delta : Double) -> LossFunction ty
huberLoss delta = reduceLoss (\p, y =>
  let d = p - y
      absD = max d (negate d)
      deltaT = fromDouble delta
  in if absD <= deltaT then fromDouble 0.5 * absD * absD
     else deltaT * (absD - fromDouble 0.5 * deltaT))

||| KL divergence loss: sum(target * log(target / pred)).
||| Expects probability distributions (not log-space).
export
klDivLoss : (Neg ty, Floating ty, Fractional ty, Ord ty) => LossFunction ty
klDivLoss = reduceLoss (\p, y =>
  let ep = pow 10 (-10)
      safep = max p ep
  in y * log (y / safep))

||| KL divergence loss (log-space input): sum(target * (log(target) - logPred)).
||| Use with logSoftmax outputs.
export
klDivLossLog : (Neg ty, Floating ty, Fractional ty) => LossFunction ty
klDivLossLog = reduceLoss (\logP, y => y * (log y - logP))

----------------------------------------------------------------------
-- Encoding
----------------------------------------------------------------------

export
oneHotEncode : {n : Nat} -> Fin n -> Vector n Nat
oneHotEncode i = VArray $ replaceAt i 1 $ replicate n 0

export
oneHotDecode : {n : Nat} -> Vector n Nat -> Maybe (Fin n)
oneHotDecode (VArray v) = findIndex (== SArray 1) v

-- TODO: Improve efficiency
export
argmax: Ord ty => {n : Nat} -> Vector (S n) ty -> Fin (S n)
argmax (VArray v@(x::xs)) =
  foldl maxIndex FZ Data.Vect.Fin.range
  where
    -- current indexes v, next indexes xs
    maxIndex : Fin (S n) -> Fin n -> Fin (S n)
    maxIndex current next =
      let
        (SArray currentValue) = Data.Vect.index current v
        (SArray nextValue) = Data.Vect.index next xs
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
    epsilon = pow 10 (-6)
  in max norm epsilon

export
cosineSimilarity : (Floating ty, Fractional ty, Ord ty) => {n : Nat} -> Vector n ty -> Vector n ty -> ty
cosineSimilarity a b = dotProduct a b / (l2Norm a * l2Norm b)

export
matrixVectorMultiply : Num ty => {n : Nat} -> Matrix m n ty -> Vector n ty -> Vector m ty
matrixVectorMultiply (VArray mat) vec = VArray $ map (SArray . dotProduct vec) mat

export
vectorMatrixMultiply : (Num ty) => {n : Nat} -> Vector n ty -> Matrix m n ty -> Vector m ty
vectorMatrixMultiply = flip matrixVectorMultiply


----------------------------------------------------------------------
-- Matrix Operations (pure Idris, type-safe)
----------------------------------------------------------------------

||| Matrix multiply: [m, n] × [n, k] -> [m, k].
||| Each output[i,j] = dot(row_i of A, col_j of B).
export
matrixMultiply : Num ty => {m, n, k : Nat} -> Matrix m n ty -> Matrix n k ty -> Matrix m k ty
matrixMultiply (VArray aRows) b =
  let VArray bCols = transpose b  -- [k, n]
  in VArray $ map (\aRow => VArray $ map (\bCol => SArray (dotProduct aRow bCol)) bCols) aRows

----------------------------------------------------------------------
-- Infix Matrix Multiplication
----------------------------------------------------------------------

export infixl 9 <>

||| Matrix-matrix multiply: [m, n] <> [n, k] -> [m, k]
namespace MatMat
  export
  (<>) : Num ty => {m, n, k : Nat} -> Matrix m n ty -> Matrix n k ty -> Matrix m k ty
  (<>) = matrixMultiply

||| Matrix-vector multiply: [m, n] <> [n] -> [m]
namespace MatVec
  export
  (<>) : Num ty => {m, n : Nat} -> Matrix m n ty -> Vector n ty -> Vector m ty
  (<>) = matrixVectorMultiply

----------------------------------------------------------------------
-- Matrix Utilities
----------------------------------------------------------------------

||| Row-wise softmax on a matrix: each row independently normalized.
export
softmaxMatrix : (FromDouble ty, Floating ty, Fractional ty, Neg ty, Ord ty) =>
                {m, n : Nat} -> Matrix m n ty -> Matrix m n ty
softmaxMatrix (VArray rows) = VArray $ map (\row => softmax row) rows

||| Element-wise clamp minimum.
export
clampMinArray : (Num ty, Ord ty) => ty -> Array dims ty -> Array dims ty
clampMinArray minVal (SArray x) = SArray (max minVal x)
clampMinArray minVal (VArray xs) = VArray (map (clampMinArray minVal) xs)

||| Reshape flat vector to matrix: Vector (m * n) -> Matrix m n.
||| Type-safe: uses Array.splitAt which unifies (S k)*n = n + (k*n) via Refl.
export
reshapeToMatrix : {m, n : Nat} -> Vector (m * n) ty -> Matrix m n ty
reshapeToMatrix {m = Z} _ = VArray []
reshapeToMatrix {m = S k} {n} vec =
  let (row, rest) = Array.splitAt n vec  -- (S k)*n = n + (k*n) by definition
  in VArray (row :: case reshapeToMatrix {m=k} {n} rest of VArray rows => rows)

||| Flatten matrix to vector: Matrix m n -> Vector (m * n).
||| Type-safe: (S k)*n = n + (k*n) lets us concatenate Vects directly.
export
flattenMatrix : {m, n : Nat} -> Matrix m n ty -> Vector (m * n) ty
flattenMatrix {m = Z} _ = VArray []  -- 0 * n = 0 by definition
flattenMatrix {m = S k} {n} (VArray (VArray row :: rest)) =
  let VArray restFlat = flattenMatrix {m=k} {n} (VArray rest)
  in VArray (row ++ restFlat)  -- n + (k*n) = (S k)*n by definition

||| Scalar multiply each element of a matrix.
export
scaleMatrix : Num ty => ty -> Matrix m n ty -> Matrix m n ty
scaleMatrix s (VArray rows) = VArray $ map (\row => map (* s) row) rows

||| Apply causal mask: set upper triangle to a large negative value.
export
causalMaskMatrix : (FromDouble ty, Num ty) => {n : Nat} -> Matrix n n ty -> Matrix n n ty
causalMaskMatrix {n} mat =
  let maskVal : Fin n -> Fin n -> ty -> ty
      maskVal i j x = if finToNat j > finToNat i then fromDouble (-1.0e20) else x
      maskRow : Fin n -> Vector n ty -> Vector n ty
      maskRow i (VArray elems) = VArray $ zipWith (\j, e => case e of SArray x => SArray (maskVal i j x)) Data.Vect.Fin.range elems
  in VArray $ zipWith (\i, row => maskRow i row) Data.Vect.Fin.range (case mat of VArray rs => rs)


||| Row-wise layer normalization on a matrix.
||| Each row is independently normalized then scaled/shifted by gamma and beta.
||| y[i,j] = gamma[j] * (x[i,j] - mean_i) / sqrt(var_i + eps) + beta[j]
export
layerNormMatrix : (FromDouble ty, Floating ty, Fractional ty, Neg ty) =>
                  {m, n : Nat} -> Matrix m n ty -> Vector n ty -> Vector n ty -> ty
                  -> Matrix m n ty
layerNormMatrix {m} {n} (VArray rows) gamma beta eps =
  VArray $ map normRow rows
  where
    nf : ty
    nf = fromDouble (cast (natToInteger n))
    normRow : Vector n ty -> Vector n ty
    normRow row =
      let mu = sum row / nf
          centered = map (\x => x - mu) row
          var = sum (map (\x => x * x) centered) / nf
          invStd = fromDouble 1.0 / sqrt (var + eps)
          -- gamma * (x - mean) * invStd + beta
      in zipWith (*) gamma (map (* invStd) centered) + beta


----------------------------------------------------------------------
-- Evaluation Metrics
----------------------------------------------------------------------

||| Count correct/total bits between prediction and target vectors.
||| Predictions are thresholded at 0.5 after sigmoid.
export
countBits : {w : Nat} -> Vector w Double -> Vector w Double -> (Nat, Nat)
countBits (VArray preds) (VArray targets) = go preds targets 0 0
  where
    sigD : Double -> Double
    sigD x = 1.0 / (1.0 + exp (negate x))
    go : Vect k (Scalar Double) -> Vect k (Scalar Double) -> Nat -> Nat -> (Nat, Nat)
    go [] [] c t = (c, t)
    go (SArray p :: ps') (SArray tgt :: ts') c t =
      let predBit = if sigD p >= 0.5 then 1.0 else 0.0
          match : Nat
          match = if predBit == tgt then 1 else 0
      in go ps' ts' (c + match) (t + 1)

||| Fraction of correctly predicted bits (sigmoid threshold 0.5).
export
bitAccuracy : {w : Nat} -> List (Vector w Double) -> List (Vector w Double) -> Double
bitAccuracy preds targets = go preds targets 0 0
  where
    go : List (Vector w Double) -> List (Vector w Double) -> Nat -> Nat -> Double
    go [] _ c t = if t == 0 then 0.0 else cast c / cast t
    go _ [] c t = if t == 0 then 0.0 else cast c / cast t
    go (p :: ps) (tgt :: tgts) c t =
      let res = countBits p tgt
      in go ps tgts (c + fst res) (t + snd res)
