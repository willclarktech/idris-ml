module Test.Nn.RmsNorm

import Data.List
import Data.Vect

import Executor
import Nn.Init
import Nn.Module
import Nn.RmsNorm
import Tensor
import Test.Config
import Test.Harness

-- input [3,4], weight 1: mean(x²) = (9+16)/2 = 12.5, rms = sqrt(12.5+eps)
-- ≈ 3.53553 → out ≈ [0.84853, 1.13137].
normalizesVector : IO Bool
normalizesVector = do
  wt <- param {ex=TestExecutor} {dt=TestDType} {dims=[2]} "rmt.w" (Const 1.0)
  let rn = the (RmsNorm 2 2 TestExecutor TestDType WithGrad) (MkRmsNorm wt)
  x   <- tensor {ex=TestExecutor} {dt=TestDType} {dims=[2]} (FromVect [3.0, 4.0])
  out <- rmsNormForward defaultRmsNormEps rn (retypeGrad x)
  let v0 = primItem1d {ex=TestExecutor} out.tensorPtr 0
  let v1 = primItem1d {ex=TestExecutor} out.tensorPtr 1
  check ("RmsNorm scales by 1/rms (got [" ++ show v0 ++ ", " ++ show v1 ++ "])")
        (abs (v0 - 0.848528) < 1.0e-4 && abs (v1 - 1.131370) < 1.0e-4)

paramExposed : IO Bool
paramExposed = do
  wt <- param {ex=TestExecutor} {dt=TestDType} {dims=[4]} "rmp.w" (Const 1.0)
  let rn = the (RmsNorm 4 4 TestExecutor TestDType WithGrad) (MkRmsNorm wt)
  check "Params (RmsNorm) exposes weight"
        (mapMaybe paramName (params rn) == ["rmp.w"])

smartCtorName : IO Bool
smartCtorName = do
  _ <- runInit $ scoped "dec" (rmsNorm {ex=TestExecutor} {dt=TestDType} {g=WithGrad} {n=8})
  cnt <- getParamCount {ex=TestExecutor}
  names <- traverse (\i => getParamName {ex=TestExecutor} i) [0 .. cnt - 1]
  check "rmsNorm registers dec.rms_norm_0.weight"
        ("dec.rms_norm_0.weight" `elem` names)

export
tests : List (IO Bool)
tests = [normalizesVector, paramExposed, smartCtorName]
