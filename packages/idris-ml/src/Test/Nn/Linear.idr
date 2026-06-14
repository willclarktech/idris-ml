module Test.Nn.Linear

import Control.Linear.LIO
import Data.Linear.Notation
import Data.List
import Data.Vect

import Executor
import Nn.Init
import Nn.Linear
import Nn.Module
import Tensor
import Test.Config
import Test.Harness

read4 : Tensor [2, 2] TestExecutor TestDType g -> List Double
read4 t = [ primItem2d {ex=TestExecutor} t.tensorPtr i j
          | (i, j) <- the (List (Int, Int)) [(0,0),(0,1),(1,0),(1,1)] ]

-- Deterministic forward: W[2,3] all 0.5, b[2] all 1.0, X[2,3] all 2.0.
-- out[k,j] = sum_i (0.5 * 2.0) + 1.0 = 3*1.0 + 1.0 = 4.0 everywhere.
forwardComputes : IO Bool
forwardComputes = do
  w <- param {ex=TestExecutor} {dt=TestDType} {dims=[2, 3]} "lt.w" (Const 0.5)
  b <- param {ex=TestExecutor} {dt=TestDType} {dims=[2]}    "lt.b" (Const 1.0)
  let lyr = the (Linear 3 2 TestExecutor TestDType WithGrad) (MkLinear w b)
  x   <- tensor {ex=TestExecutor} {dt=TestDType} {dims=[2, 3]} (Const 2.0)
  out <- Control.Linear.LIO.run (do
           (MkBang o # m') <- forward {b=2} lyr (retypeGrad x)
           discard m'
           pure o)
  check ("Linear.forward computes x·Wᵀ+b (got " ++ show (read4 out) ++ ")")
        (read4 out == [4.0, 4.0, 4.0, 4.0])

-- Params exposes both leaves under their registered names.
paramsExposed : IO Bool
paramsExposed = do
  w <- param {ex=TestExecutor} {dt=TestDType} {dims=[2, 3]} "lp.w" (Const 0.5)
  b <- param {ex=TestExecutor} {dt=TestDType} {dims=[2]}    "lp.b" (Const 1.0)
  let lyr   = the (Linear 3 2 TestExecutor TestDType WithGrad) (MkLinear w b)
  let names = mapMaybe paramName (params lyr)
  check ("Params (Linear) exposes weight + bias (got " ++ show names ++ ")")
        (names == ["lp.w", "lp.b"])

-- The Init smart constructor registers PyTorch-style dotted names.
smartCtorNames : IO Bool
smartCtorNames = do
  _ <- runInit $ scoped "mlp" (linear {ex=TestExecutor} {dt=TestDType} {i=3} {o=2})
  cnt <- getParamCount {ex=TestExecutor}
  names <- traverse (\i => getParamName {ex=TestExecutor} i) [0 .. cnt - 1]
  check "linear registers mlp.linear_0.weight + .bias"
        (("mlp.linear_0.weight" `elem` names) && ("mlp.linear_0.bias" `elem` names))

-- linearWith with biasStd=0 produces an exactly-zero bias (the path NTM's
-- heads and the default `linear` rely on).
linearWithZeroBias : IO Bool
linearWithZeroBias = do
  lyr <- runInit $ scoped "lw" (linearWith {ex=TestExecutor} {dt=TestDType} {i=3} {o=2} 0.5 0.0)
  let b0 = primItem1d {ex=TestExecutor} lyr.biasT.tensorPtr 0
  let b1 = primItem1d {ex=TestExecutor} lyr.biasT.tensorPtr 1
  check ("linearWith biasStd=0 → zero bias (got [" ++ show b0 ++ ", " ++ show b1 ++ "])")
        (b0 == 0.0 && b1 == 0.0)

export
tests : List (IO Bool)
tests = [forwardComputes, paramsExposed, smartCtorNames, linearWithZeroBias]
