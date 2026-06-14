module Test.Nn.Activation

import Data.Vect

import Executor
import Nn.Activation
import Nn.Module
import Tensor
import Test.Config
import Test.Harness

read4 : Tensor [2, 2] TestExecutor TestDType g -> List Double
read4 t = [ primItem2d {ex=TestExecutor} t.tensorPtr i j
          | (i, j) <- the (List (Int, Int)) [(0,0),(0,1),(1,0),(1,1)] ]

-- relu zeroes negatives, passes positives — elementwise over the batch.
reluForward : IO Bool
reluForward = do
  x   <- tensor {ex=TestExecutor} {dt=TestDType} {dims=[2, 2]}
                (FromVect [-1.0, 2.0, -3.0, 4.0])
  out <- forward {b=2} (the (Activation 2 2 TestExecutor TestDType NoGrad) reluA) x
  check ("Activation relu forward (got " ++ show (read4 out) ++ ")")
        (read4 out == [0.0, 2.0, 0.0, 4.0])

-- Activation is stateless.
noParams : IO Bool
noParams =
  check "Params (Activation) is empty"
        (length (params (the (Activation 2 2 TestExecutor TestDType NoGrad) reluA)) == 0)

export
tests : List (IO Bool)
tests = [reluForward, noParams]
