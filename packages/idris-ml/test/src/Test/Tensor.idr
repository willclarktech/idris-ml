module Test.Tensor

import Data.Vect

import Harness
import Tensor

export
tests : List (IO Bool)
tests =
  [ -- Construction
    check "zeros scalar" (STensor 0.0 == the (Scalar Double) zeros)
  , check "zeros vector" (VTensor [STensor 0.0, STensor 0.0, STensor 0.0] == the (Vector 3 Double) zeros)
  , check "ones scalar" (STensor 1.0 == the (Scalar Double) ones)
  , check "ones vector" (VTensor [STensor 1.0, STensor 1.0] == the (Vector 2 Double) ones)
  , check "generate" (VTensor [STensor 0, STensor 1, STensor 2] == the (Vector 3 Nat) (generate id))

  -- Functor
  , check "map (+1)" (VTensor [STensor 2.0, STensor 3.0] == map (+1.0) (the (Vector 2 Double) (VTensor [STensor 1.0, STensor 2.0])))

  -- Arithmetic
  , check "vector add" (VTensor [STensor 5.0, STensor 7.0] == the (Vector 2 Double) (VTensor [STensor 2.0, STensor 3.0]) + (VTensor [STensor 3.0, STensor 4.0]))
  , check "vector mul" (VTensor [STensor 6.0, STensor 12.0] == the (Vector 2 Double) (VTensor [STensor 2.0, STensor 3.0]) * (VTensor [STensor 3.0, STensor 4.0]))
  , check "sum" (the Double 6.0 == sum (the (Vector 3 Double) (VTensor [STensor 1.0, STensor 2.0, STensor 3.0])))

  -- splitAt
  , let v = the (Vector 4 Double) (VTensor [STensor 1.0, STensor 2.0, STensor 3.0, STensor 4.0])
        (a, b) = Tensor.splitAt 2 v
    in check "splitAt" (a == VTensor [STensor 1.0, STensor 2.0] && b == VTensor [STensor 3.0, STensor 4.0])

  -- concat
  , let a = the (Vector 2 Double) (VTensor [STensor 1.0, STensor 2.0])
        b = the (Vector 2 Double) (VTensor [STensor 3.0, STensor 4.0])
    in check "concat" (a ++ b == VTensor [STensor 1.0, STensor 2.0, STensor 3.0, STensor 4.0])

  -- zipWith
  , check "zipWith" (VTensor [STensor 4.0, STensor 10.0] == zipWith (*) (the (Vector 2 Double) (VTensor [STensor 2.0, STensor 5.0])) (VTensor [STensor 2.0, STensor 2.0]))

  -- enumerate length
  , check "enumerate length" (Tensor.length (the (Vector 5 Nat) enumerate) == 5)

  -- complement
  , check "complement" (VTensor [STensor 0.0, STensor (-1.0)] == complement (the (Vector 2 Double) (VTensor [STensor 1.0, STensor 2.0])))
  ]
