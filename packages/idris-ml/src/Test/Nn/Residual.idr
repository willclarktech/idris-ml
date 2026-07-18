module Test.Nn.Residual

import Control.Linear.LIO
import Data.Linear.Notation
import Data.Vect

import Ml.Executor
import Ml.Nn.Activation
import Ml.Nn.Module
import Ml.Nn.Residual
import Ml.Tensor
import Test.Config
import Test.Harness

read4 : Tensor [2, 2] TestExecutor TestDType g -> List Double
read4 t = [ primItem2d {ex=TestExecutor} t.tensorPtr i j
          | (i, j) <- the (List (Int, Int)) [(0,0),(0,1),(1,0),(1,1)] ]

-- Residual around relu: out = x + relu(x).
-- x = [-1,2,-3,4] → relu = [0,2,0,4] → x + relu = [-1,4,-3,8].
residualAddsSkip : IO Bool
residualAddsSkip = do
  x   <- tensor {ex=TestExecutor} {dt=TestDType} {dims=[2, 2]} (FromVect [-1.0, 2.0, -3.0, 4.0])
  let blk = the (Residual 2 2 TestExecutor TestDType NoGrad)
              (residual (the (Activation 2 2 TestExecutor TestDType NoGrad) reluA))
  out <- Control.Linear.LIO.run (do
           (MkBang o # m') <- forward {b=2} blk x
           discard m'
           pure o)
  check ("Residual computes x + relu(x) (got " ++ show (read4 out) ++ ")")
        (read4 out == [-1.0, 4.0, -3.0, 8.0])

-- A residual around a param-free sublayer exposes no params.
noParams : IO Bool
noParams =
  check "Params (Residual relu) is empty"
        (length (params (the (Residual 2 2 TestExecutor TestDType NoGrad)
                             (residual (the (Activation 2 2 TestExecutor TestDType NoGrad) reluA)))) == 0)

export
tests : List (IO Bool)
tests = [residualAddsSkip, noParams]
