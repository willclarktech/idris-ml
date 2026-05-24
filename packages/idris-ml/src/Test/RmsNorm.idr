module Test.RmsNorm

import Data.Vect

import Test.Harness
import Device
import Tensor
import Array
import Layer.RmsNorm
import Test.Config


-- Tolerance for the forward-value check. The composed formulation
-- (sq → primSum → primMulScalar → primAddScalar → primSqrt →
-- primDiv → primMul) accumulates ~5 F64 ops; 1e-9 leaves headroom
-- for backend numerics drift while still catching real math bugs.
tol : Double
tol = 1.0e-9


-- PyTorch F64 reference for input=[1,2,3,4], eps=1e-5, weight=1s:
--   mean(x²) = (1+4+9+16)/4 = 7.5
--   rms      = sqrt(7.5 + 1e-5) = 2.7386146132670803
--   out[i]   = input[i] / rms
-- Generated via:
--   import torch
--   x = torch.tensor([1.,2.,3.,4.], dtype=torch.float64)
--   eps = 1e-5
--   out = x / torch.sqrt((x*x).mean() + eps)
expectedOut : Vect 4 Double
expectedOut =
  [ 0.3651481282381064
  , 0.7302962564762128
  , 1.0954443847143192
  , 1.4605925129524255
  ]


-- Lift a [n] tensor's raw pointer to a List Double by reading element-
-- by-element. Stops at `n` reads; reverses to put position 0 first.
readVec : (n : Nat) -> AnyPtr -> IO (List Double)
readVec n p = go (cast {to=Int} n) 0 []
  where
    go : Int -> Int -> List Double -> IO (List Double)
    go end i acc =
      if i >= end
        then pure (reverse acc)
        else let v = primItem1d {d=TestDevice} p i
             in go end (i + 1) (v :: acc)


-- Distance from `actual` to `expected`, element-wise. Returns the max
-- absolute diff; used to assert convergence within `tol`.
maxAbsDiff : List Double -> Vect n Double -> Double
maxAbsDiff actual expected = go actual (toList expected) 0.0
  where
    go : List Double -> List Double -> Double -> Double
    go []        _         m = m
    go _         []        m = m
    go (a :: as) (b :: bs) m =
      let d = abs (a - b)
      in go as bs (if d > m then d else m)


-- Build a [n] input tensor from a Vect of Doubles. Implicit `n` is
-- unrestricted (default for `{name : Type}` in Idris 2) so it's in
-- scope at runtime for the `tinput1d {n}` call below.
mkInput : {n : Nat} -> Vect n Double -> Tensor [n] TestDevice TestDType WithGrad
mkInput xs =
  let raw = bulkToTensor {d=TestDevice} {dt=TestDType}
                         (VArray (map SArray xs))
  in tinput1d {n} raw


testForwardValueAt1234 : IO Bool
testForwardValueAt1234 = do
  rms <- rmsNormLayer {d=TestDevice} {dt=TestDType} {n=4} "rms_test"
  let input = mkInput (the (Vect 4 Double) [1.0, 2.0, 3.0, 4.0])
  (_, out) <- applyRmsNormEps 1.0e-5 rms input
  vals <- readVec 4 out.tensorPtr
  let mdiff = maxAbsDiff vals expectedOut
  if mdiff < tol
    then check ("RmsNorm forward [1,2,3,4] matches PyTorch F64 reference " ++
                "(max-abs-diff " ++ show mdiff ++ ")") True
    else do
      putStrLn ("  FAIL: max-abs-diff " ++ show mdiff ++ " > tol " ++ show tol)
      putStrLn ("    got:      " ++ show vals)
      putStrLn ("    expected: " ++ show (toList expectedOut))
      pure False


testShapeAndFinite : IO Bool
testShapeAndFinite = do
  rms <- rmsNormLayer {d=TestDevice} {dt=TestDType} {n=8} "rms_shape"
  let input = mkInput (the (Vect 8 Double)
                           [0.5, -1.5, 2.0, -0.25, 3.0, 0.0, 1.0, -2.5])
  (_, out) <- applyRmsNormEps 1.0e-5 rms input
  vals <- readVec 8 out.tensorPtr
  let allFinite = all (\x => x == x && abs x < 1.0e100) vals
  check "RmsNorm n=8 with negatives + zero produces 8 finite values" allFinite


export
tests : List (IO Bool)
tests =
  [ testForwardValueAt1234
  , testShapeAndFinite
  ]
