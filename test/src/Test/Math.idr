module Test.Math

import Data.Fin
import Data.Vect

import Floating
import Harness
import Math
import Tensor


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
  , let sm = softmax (the (Vector 3 Double) (VTensor [STensor 1.0, STensor 2.0, STensor 3.0]))
    in checkClose "softmax sums to 1" 1.0 (sum sm) tol

  -- Softmax uniform
  , let sm = softmax (the (Vector 3 Double) (VTensor [STensor 0.0, STensor 0.0, STensor 0.0]))
        (VTensor [STensor v, _, _]) = sm
    in checkClose "softmax uniform" (1.0/3.0) v tol

  -- logSoftmax = log(softmax)
  , let xs = the (Vector 3 Double) (VTensor [STensor 1.0, STensor 2.0, STensor 3.0])
        lsm = logSoftmax xs
        sm = softmax xs
        expected = map Floating.log sm
        (VTensor [STensor e0, STensor e1, STensor e2]) = expected
        (VTensor [STensor a0, STensor a1, STensor a2]) = lsm
    in checkClose "logSoftmax matches log(softmax)" (abs (e0 - a0) + abs (e1 - a1) + abs (e2 - a2)) 0.0 1.0e-10

  -- dotProduct
  , checkClose "dotProduct" 11.0 (dotProduct (the (Vector 2 Double) (VTensor [STensor 1.0, STensor 2.0])) (VTensor [STensor 3.0, STensor 4.0])) tol

  -- matrixVectorMultiply
  , let mat = the (Matrix 2 2 Double) (VTensor [VTensor [STensor 1.0, STensor 2.0], VTensor [STensor 3.0, STensor 4.0]])
        v = the (Vector 2 Double) (VTensor [STensor 5.0, STensor 6.0])
        result = matrixVectorMultiply mat v
        (VTensor [STensor r0, STensor r1]) = result
    in check "matrixVectorMultiply" (abs (r0 - 17.0) < tol && abs (r1 - 39.0) < tol)

  -- cosineSimilarity parallel
  , let v1 = the (Vector 2 Double) (VTensor [STensor 1.0, STensor 0.0])
        v2 = the (Vector 2 Double) (VTensor [STensor 2.0, STensor 0.0])
    in checkClose "cosineSimilarity parallel" 1.0 (cosineSimilarity v1 v2) tol

  -- cosineSimilarity orthogonal
  , let v1 = the (Vector 2 Double) (VTensor [STensor 1.0, STensor 0.0])
        v2 = the (Vector 2 Double) (VTensor [STensor 0.0, STensor 1.0])
    in checkClose "cosineSimilarity orthogonal" 0.0 (cosineSimilarity v1 v2) tol

  -- l2Norm
  , checkClose "l2Norm" 5.0 (l2Norm (the (Vector 2 Double) (VTensor [STensor 3.0, STensor 4.0]))) tol

  -- nllLoss
  , let logProbs = the (Vector 3 Double) (VTensor [STensor (-0.5), STensor (-1.0), STensor (-2.0)])
        targets = the (Vector 3 Double) (VTensor [STensor 1.0, STensor 0.0, STensor 0.0])
    in checkClose "nllLoss" (0.5 / 3.0) (nllLoss logProbs targets) tol

  -- meanSquaredError
  , let preds = the (Vector 2 Double) (VTensor [STensor 1.0, STensor 2.0])
        targets = the (Vector 2 Double) (VTensor [STensor 3.0, STensor 4.0])
    in checkClose "meanSquaredError" 4.0 (meanSquaredError preds targets) tol

  -- argmax
  , check "argmax" (argmax (the (Vector 3 Double) (VTensor [STensor 1.0, STensor 5.0, STensor 3.0])) == 1)

  -- oneHotEncode/decode roundtrip
  , check "oneHot roundtrip" (oneHotDecode (oneHotEncode {n=4} 2) == Just (the (Fin 4) 2))

  -- mean
  , checkClose "mean" 2.0 (mean (the (Vector 3 Double) (VTensor [STensor 1.0, STensor 2.0, STensor 3.0]))) tol

  -- logSoftmax stability with large inputs
  , let xs = the (Vector 3 Double) (VTensor [STensor 1000.0, STensor 1001.0, STensor 1002.0])
        lsm = logSoftmax xs
        (VTensor [STensor a0, _, _]) = lsm
    in check "logSoftmax stable with large inputs" (a0 == a0)  -- not NaN
  ]
