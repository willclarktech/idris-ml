module Test.Math

import Data.Fin
import Data.Vect

import Array
import Floating
import Math
import Test.Harness

tol : Double
tol = 1.0e-6

export
tests : List (IO Bool)
tests =
  [ -- Sigmoid
    checkClose "sigmoid 0 = 0.5" 0.5 (sigmoid {ty=Double} 0.0) tol
  , checkClose "sigmoid large" 1.0 (sigmoid {ty=Double} 100.0) tol
  , checkClose "sigmoid small" 0.0 (sigmoid {ty=Double} (-100.0)) tol

  -- Tanh
  , checkClose "tanh 0 = 0" 0.0 (Math.tanh {ty=Double} 0.0) tol
  , checkClose "tanh symmetry" (negate (Math.tanh {ty=Double} 1.5)) (Math.tanh {ty=Double} (-1.5)) tol

  -- Softmax sums to 1
  , let sm = softmax (the (Vector 3 Double) (VArray [SArray 1.0, SArray 2.0, SArray 3.0]))
    in checkClose "softmax sums to 1" 1.0 (sum sm) tol

  -- Softmax uniform
  , let sm = softmax (the (Vector 3 Double) (VArray [SArray 0.0, SArray 0.0, SArray 0.0]))
        (VArray [SArray v, _, _]) = sm
    in checkClose "softmax uniform" (1.0/3.0) v tol

  -- logSoftmax = log(softmax)
  , let xs = the (Vector 3 Double) (VArray [SArray 1.0, SArray 2.0, SArray 3.0])
        lsm                                        = logSoftmax xs
        sm                                         = softmax xs
        expected                                   = map Floating.log sm
        (VArray [SArray e0, SArray e1, SArray e2]) = expected
        (VArray [SArray a0, SArray a1, SArray a2]) = lsm
    in checkClose "logSoftmax matches log(softmax)" (abs (e0 - a0) + abs (e1 - a1) + abs (e2 - a2)) 0.0 1.0e-10

  -- dotProduct
  , checkClose "dotProduct" 11.0 (dotProduct (the (Vector 2 Double) (VArray [SArray 1.0, SArray 2.0])) (VArray [SArray 3.0, SArray 4.0])) tol

  -- matrixVectorMultiply
  , let mat = the (Matrix 2 2 Double) (VArray [VArray [SArray 1.0, SArray 2.0], VArray [SArray 3.0, SArray 4.0]])
        v                               = the (Vector 2 Double) (VArray [SArray 5.0, SArray 6.0])
        result                          = matrixVectorMultiply mat v
        (VArray [SArray r0, SArray r1]) = result
    in check "matrixVectorMultiply" (abs (r0 - 17.0) < tol && abs (r1 - 39.0) < tol)

  -- cosineSimilarity parallel
  , let v1 = the (Vector 2 Double) (VArray [SArray 1.0, SArray 0.0])
        v2 = the (Vector 2 Double) (VArray [SArray 2.0, SArray 0.0])
    in checkClose "cosineSimilarity parallel" 1.0 (cosineSimilarity v1 v2) tol

  -- cosineSimilarity orthogonal
  , let v1 = the (Vector 2 Double) (VArray [SArray 1.0, SArray 0.0])
        v2 = the (Vector 2 Double) (VArray [SArray 0.0, SArray 1.0])
    in checkClose "cosineSimilarity orthogonal" 0.0 (cosineSimilarity v1 v2) tol

  -- l2Norm
  , checkClose "l2Norm" 5.0 (l2Norm (the (Vector 2 Double) (VArray [SArray 3.0, SArray 4.0]))) tol

  -- nllLoss
  , let logProbs = the (Vector 3 Double) (VArray [SArray (-0.5), SArray (-1.0), SArray (-2.0)])
        targets = the (Vector 3 Double) (VArray [SArray 1.0, SArray 0.0, SArray 0.0])
    in checkClose "nllLoss" (0.5 / 3.0) (nllLoss logProbs targets) tol

  -- meanSquaredError
  , let preds = the (Vector 2 Double) (VArray [SArray 1.0, SArray 2.0])
        targets = the (Vector 2 Double) (VArray [SArray 3.0, SArray 4.0])
    in checkClose "meanSquaredError" 4.0 (meanSquaredError preds targets) tol

  -- argmax
  , check "argmax" (argmax (the (Vector 3 Double) (VArray [SArray 1.0, SArray 5.0, SArray 3.0])) == 1)

  -- oneHotEncode/decode roundtrip
  , check "oneHot roundtrip" (oneHotDecode (oneHotEncode {n=4} 2) == Just (the (Fin 4) 2))

  -- mean
  , checkClose "mean" 2.0 (mean (the (Vector 3 Double) (VArray [SArray 1.0, SArray 2.0, SArray 3.0]))) tol

  -- logSoftmax stability with large inputs
  , let xs = the (Vector 3 Double) (VArray [SArray 1000.0, SArray 1001.0, SArray 1002.0])
        lsm                        = logSoftmax xs
        (VArray [SArray a0, _, _]) = lsm
    in check "logSoftmax stable with large inputs" (a0 == a0)  -- not NaN

  -- l1Loss: mean(|pred - target|)
  , let preds = the (Vector 2 Double) (VArray [SArray 1.0, SArray 5.0])
        targets = the (Vector 2 Double) (VArray [SArray 3.0, SArray 2.0])
    in checkClose "l1Loss" 2.5 (l1Loss preds targets) tol  -- (2 + 3) / 2

  -- huberLoss: quadratic region (|d| <= delta)
  , let preds = the (Vector 2 Double) (VArray [SArray 1.0, SArray 2.0])
        targets = the (Vector 2 Double) (VArray [SArray 1.5, SArray 2.3])
        -- |d| = [0.5, 0.3], both <= 1.0, so 0.5*d^2 = [0.125, 0.045], mean = 0.085
    in checkClose "huberLoss quadratic" 0.085 (huberLoss 1.0 preds targets) tol

  -- huberLoss: linear region (|d| > delta)
  , let preds = the (Vector 1 Double) (VArray [SArray 0.0])
        targets = the (Vector 1 Double) (VArray [SArray 5.0])
        -- |d| = 5.0 > 1.0, so delta * (|d| - 0.5*delta) = 1.0 * (5.0 - 0.5) = 4.5
    in checkClose "huberLoss linear" 4.5 (huberLoss 1.0 preds targets) tol

  -- klDivLoss: identical distributions = 0
  , let p = the (Vector 3 Double) (VArray [SArray 0.2, SArray 0.3, SArray 0.5])
        q = the (Vector 3 Double) (VArray [SArray 0.2, SArray 0.3, SArray 0.5])
    in checkClose "klDivLoss identical = 0" 0.0 (klDivLoss p q) tol

  -- klDivLoss: non-negative for different distributions
  , let p = the (Vector 3 Double) (VArray [SArray 0.5, SArray 0.3, SArray 0.2])
        q = the (Vector 3 Double) (VArray [SArray 0.2, SArray 0.3, SArray 0.5])
    in check "klDivLoss non-negative" (klDivLoss p q >= 0.0)
  ]
