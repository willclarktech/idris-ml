module Test.Nn.LayerNorm

import Data.List
import Data.Vect

import Executor
import Nn.Init
import Nn.LayerNorm
import Nn.Module
import Tensor
import Test.Config
import Test.Harness

-- Row [1, 3]: mean 2, population var 1 → normalised ≈ [-1, +1] (gamma=1,
-- beta=0). eps=1e-5 makes it ≈ ∓0.999995.
normalizesRow : IO Bool
normalizesRow = do
  g <- param {ex=TestExecutor} {dt=TestDType} {dims=[2]} "lnt.w" (Const 1.0)
  b <- param {ex=TestExecutor} {dt=TestDType} {dims=[2]} "lnt.b" (Const 0.0)
  let ln = the (LayerNorm 2 2 TestExecutor TestDType WithGrad) (MkLayerNorm g b)
  x   <- tensor {ex=TestExecutor} {dt=TestDType} {dims=[1, 2]} (FromVect [1.0, 3.0])
  out <- forward {b=1} ln (retypeGrad x)
  let v0 = primItem2d {ex=TestExecutor} out.tensorPtr 0 0
  let v1 = primItem2d {ex=TestExecutor} out.tensorPtr 0 1
  check ("LayerNorm normalises row to ≈[-1,1] (got [" ++ show v0 ++ ", " ++ show v1 ++ "])")
        (abs (v0 + 1.0) < 1.0e-3 && abs (v1 - 1.0) < 1.0e-3)

-- The Init smart constructor registers PyTorch-style weight/bias names.
smartCtorNames : IO Bool
smartCtorNames = do
  _ <- runInit $ scoped "enc" (layerNorm {ex=TestExecutor} {dt=TestDType} {n=4})
  cnt <- getParamCount {ex=TestExecutor}
  names <- traverse (\i => getParamName {ex=TestExecutor} i) [0 .. cnt - 1]
  check "layerNorm registers enc.layer_norm_0.weight + .bias"
        (("enc.layer_norm_0.weight" `elem` names) && ("enc.layer_norm_0.bias" `elem` names))

export
tests : List (IO Bool)
tests = [normalizesRow, smartCtorNames]
