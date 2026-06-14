module Test.Nn.Pool

import Data.List
import Data.Vect

import Executor
import Nn.Module
import Nn.Pool
import Tensor
import Test.Config
import Test.Harness

-- MaxPool2D, c=1 inH=2 inW=2, 2x2 window stride 2 over [[1,2],[3,4]] ->
-- max = 4. PoolOutDim 2 2 2 = 1, so output is 1 value. Batched b=1:
-- [1,4] -> [1,1].
maxPool2dComputes : IO Bool
maxPool2dComputes = do
  x   <- tensor {ex=TestExecutor} {dt=TestDType} {dims=[1, 4]} (FromVect [1.0, 2.0, 3.0, 4.0])
  let mp = the (MaxPool2D 1 2 2 2 2 2 2 4 1 TestExecutor TestDType WithGrad) MkMaxPool2D
  out <- forward {b=1} mp (retypeGrad x)
  let v = primItem2d {ex=TestExecutor} out.tensorPtr 0 0
  check ("MaxPool2D 2x2 over [[1,2],[3,4]] (got " ++ show v ++ ")") (v == 4.0)

-- MaxPool1D, c=1 len=4, window 2 stride 2 over [1,3,2,5] -> [3,5].
-- PoolOutDim 4 2 2 = 2. Batched b=1: [1,4] -> [1,2].
maxPool1dComputes : IO Bool
maxPool1dComputes = do
  x   <- tensor {ex=TestExecutor} {dt=TestDType} {dims=[1, 4]} (FromVect [1.0, 3.0, 2.0, 5.0])
  let mp = the (MaxPool1D 1 4 2 2 4 2 TestExecutor TestDType WithGrad) MkMaxPool1D
  out <- forward {b=1} mp (retypeGrad x)
  let vs = [ primItem2d {ex=TestExecutor} out.tensorPtr 0 j | j <- the (List Int) [0,1] ]
  check ("MaxPool1D window 2 over [1,3,2,5] (got " ++ show vs ++ ")") (vs == [3.0, 5.0])

paramsEmpty : IO Bool
paramsEmpty = do
  let mp = the (MaxPool1D 1 4 2 2 4 2 TestExecutor TestDType WithGrad) MkMaxPool1D
  check ("Params (MaxPool1D) = [] (got " ++ show (length (params mp)) ++ ")")
        (length (params mp) == 0)

export
tests : List (IO Bool)
tests = [maxPool2dComputes, maxPool1dComputes, paramsEmpty]
