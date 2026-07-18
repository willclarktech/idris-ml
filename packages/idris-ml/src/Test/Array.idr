module Test.Array

import Data.Vect

import Ml.Array
import Test.Harness

export
tests : List (IO Bool)
tests =
  [ -- Construction
    check "zeros scalar" (SArray 0.0 == the (Scalar Double) zeros)
  , check "zeros vector" (VArray [SArray 0.0, SArray 0.0, SArray 0.0] == the (Vector 3 Double) zeros)
  , check "ones scalar" (SArray 1.0 == the (Scalar Double) ones)
  , check "ones vector" (VArray [SArray 1.0, SArray 1.0] == the (Vector 2 Double) ones)
  , check "generate" (VArray [SArray 0, SArray 1, SArray 2] == the (Vector 3 Nat) (generate id))

  -- Functor
  , check "map (+1)" (VArray [SArray 2.0, SArray 3.0] == map (+1.0) (the (Vector 2 Double) (VArray [SArray 1.0, SArray 2.0])))

  -- Arithmetic
  , check "vector add" (VArray [SArray 5.0, SArray 7.0] == the (Vector 2 Double) (VArray [SArray 2.0, SArray 3.0]) + (VArray [SArray 3.0, SArray 4.0]))
  , check "vector mul" (VArray [SArray 6.0, SArray 12.0] == the (Vector 2 Double) (VArray [SArray 2.0, SArray 3.0]) * (VArray [SArray 3.0, SArray 4.0]))
  , check "sum" (the Double 6.0 == sum (the (Vector 3 Double) (VArray [SArray 1.0, SArray 2.0, SArray 3.0])))

  -- splitAt
  , let v = the (Vector 4 Double) (VArray [SArray 1.0, SArray 2.0, SArray 3.0, SArray 4.0])
        (a, b) = Array.splitAt 2 v
    in check "splitAt" (a == VArray [SArray 1.0, SArray 2.0] && b == VArray [SArray 3.0, SArray 4.0])

  -- concat
  , let a = the (Vector 2 Double) (VArray [SArray 1.0, SArray 2.0])
        b = the (Vector 2 Double) (VArray [SArray 3.0, SArray 4.0])
    in check "concat" (a ++ b == VArray [SArray 1.0, SArray 2.0, SArray 3.0, SArray 4.0])

  -- zipWith
  , check "zipWith" (VArray [SArray 4.0, SArray 10.0] == zipWith (*) (the (Vector 2 Double) (VArray [SArray 2.0, SArray 5.0])) (VArray [SArray 2.0, SArray 2.0]))

  -- enumerate length
  , check "enumerate length" (Array.length (the (Vector 5 Nat) enumerate) == 5)

  -- complement
  , check "complement" (VArray [SArray 0.0, SArray (-1.0)] == complement (the (Vector 2 Double) (VArray [SArray 1.0, SArray 2.0])))
  ]
