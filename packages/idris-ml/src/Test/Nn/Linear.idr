module Test.Nn.Linear

import Control.Linear.LIO
import Data.Linear.Notation
import Data.List
import Data.Vect

import Ml.Executor
import Ml.Nn.Init
import Ml.Nn.Linear
import Ml.Nn.Module
import Ml.Tensor
import Test.Harness

import Test.Config

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
  _ <- runInit $ scoped "mlp" (linear {ex=TestExecutor} {dt=TestDType} {g=WithGrad} {i=3} {o=2})
  cnt <- getParamCount {ex=TestExecutor}
  names <- traverse (\i => getParamName {ex=TestExecutor} i) [0 .. cnt - 1]
  check "linear registers mlp.linear_0.weight + .bias"
        (("mlp.linear_0.weight" `elem` names) && ("mlp.linear_0.bias" `elem` names))

-- linearWith with biasStd=0 produces an exactly-zero bias (the path NTM's
-- heads and the default `linear` rely on).
linearWithZeroBias : IO Bool
linearWithZeroBias = do
  lyr <- runInit $ scoped "lw" (linearWith {ex=TestExecutor} {dt=TestDType} {g=WithGrad} {i=3} {o=2} 0.5 0.0)
  let b0 = primItem1d {ex=TestExecutor} lyr.biasT.tensorPtr 0
  let b1 = primItem1d {ex=TestExecutor} lyr.biasT.tensorPtr 1
  check ("linearWith biasStd=0 → zero bias (got [" ++ show b0 ++ ", " ++ show b1 ++ "])")
        (b0 == 0.0 && b1 == 0.0)

-- The default `linear` init contract, matched on the reference side by
-- `torch_ref.init.init_linear_`: weight ~ U(±1/√fan_in), bias exactly zero.
defaultWeightInRange : IO Bool
defaultWeightInRange = do
  lyr <- runInit $ scoped "dw" (linear {ex=TestExecutor} {dt=TestDType} {g=WithGrad} {i=3} {o=2})
  let bound = 1.0 / sqrt 3.0
      ws    = [ primItem2d {ex=TestExecutor} lyr.weightT.tensorPtr i j
              | (i, j) <- the (List (Int, Int)) [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)] ]
  -- Bound alone would admit an all-zero weight, so require a nonzero draw too.
  check ("linear weight ~ U(±1/√fan_in) (got " ++ show ws ++ ", bound " ++ show bound ++ ")")
        (all (\w => abs w <= bound) ws && any (\w => w /= 0.0) ws)

defaultBiasIsZero : IO Bool
defaultBiasIsZero = do
  lyr <- runInit $ scoped "db" (linear {ex=TestExecutor} {dt=TestDType} {g=WithGrad} {i=3} {o=2})
  let bs = [ primItem1d {ex=TestExecutor} lyr.biasT.tensorPtr k | k <- the (List Int) [0, 1] ]
  check ("linear bias is exactly zero (got " ++ show bs ++ ")") (all (== 0.0) bs)

-- NoGrad-from-birth: constructing `linear {g=NoGrad}` yields params that are
-- genuinely tape-free (C `requires_grad == 0`) with no post-construction
-- `eval` flip — the core-layer parity with the HF adapters' grad-poly
-- constructors. The WithGrad default still registers requires_grad==1.
noGradFromBirth : IO Bool
noGradFromBirth = do
  lyr <- runInit $ scoped "ng" (linear {ex=TestExecutor} {dt=TestDType} {g=NoGrad} {i=3} {o=2})
  let rgW = primRequiresGrad {ex=TestExecutor} lyr.weightT.tensorPtr
      rgB = primRequiresGrad {ex=TestExecutor} lyr.biasT.tensorPtr
  check ("linear {g=NoGrad} params are tape-free (requires_grad w=" ++ show rgW
         ++ " b=" ++ show rgB ++ ")")
        (rgW == 0 && rgB == 0)

withGradFromBirth : IO Bool
withGradFromBirth = do
  lyr <- runInit $ scoped "wg" (linear {ex=TestExecutor} {dt=TestDType} {g=WithGrad} {i=3} {o=2})
  let rgW = primRequiresGrad {ex=TestExecutor} lyr.weightT.tensorPtr
      rgB = primRequiresGrad {ex=TestExecutor} lyr.biasT.tensorPtr
  check ("linear {g=WithGrad} params are trainable (requires_grad w=" ++ show rgW
         ++ " b=" ++ show rgB ++ ")")
        (rgW == 1 && rgB == 1)

export
tests : List (IO Bool)
tests = [ forwardComputes, paramsExposed, smartCtorNames, linearWithZeroBias
        , defaultWeightInRange, defaultBiasIsZero
        , noGradFromBirth, withGradFromBirth ]
