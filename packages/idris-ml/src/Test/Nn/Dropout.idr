module Test.Nn.Dropout

import Data.Vect

import Test.Harness
import Executor
import Tensor
import Nn.Module
import Nn.Dropout
import Test.Config

read4 : Tensor [2, 2] TestExecutor TestDType g -> List Double
read4 t = [ primItem2d {ex=TestExecutor} t.tensorPtr i j
          | (i, j) <- the (List (Int, Int)) [(0,0),(0,1),(1,0),(1,1)] ]

-- Eval mode is identity (no masking).
evalIsIdentity : IO Bool
evalIsIdentity = do
  x   <- tensor {ex=TestExecutor} {dt=TestDType} {dims=[2, 2]} (FromVect [1.0, 2.0, 3.0, 4.0])
  out <- forward {b=2} (the (Dropout 2 2 TestExecutor TestDType) (setTraining False (dropout 0.5))) x
  check ("eval-mode dropout is identity (got " ++ show (read4 out) ++ ")")
        (read4 out == [1.0, 2.0, 3.0, 4.0])

-- p=0 training mode keeps everything (scale 1/(1-0) = 1).
zeroProbKeepsAll : IO Bool
zeroProbKeepsAll = do
  x   <- tensor {ex=TestExecutor} {dt=TestDType} {dims=[2, 2]} (FromVect [1.0, 2.0, 3.0, 4.0])
  out <- forward {b=2} (the (Dropout 2 2 TestExecutor TestDType) (dropout 0.0)) x
  check ("p=0 training dropout keeps all (got " ++ show (read4 out) ++ ")")
        (read4 out == [1.0, 2.0, 3.0, 4.0])

export
tests : List (IO Bool)
tests = [evalIsIdentity, zeroProbKeepsAll]
