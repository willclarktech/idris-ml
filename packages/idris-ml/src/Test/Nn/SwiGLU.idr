module Test.Nn.SwiGLU

import Data.List
import Data.Vect

import Executor
import Nn.Init
import Nn.Module
import Nn.SwiGLU
import Tensor
import Test.Config
import Test.Harness

-- hidden=2, intermediate=3, all weights 0.5, x=1.0:
--   gate = up = [1,1,1] (sum_2 0.5*1 = 1 per row)
--   silu(1) = 1/(1+e^-1) = 0.731059; mid = silu(1)*1 = 0.731059
--   out[j] = sum_3 0.5*0.731059 = 1.096588
forwardComputes : IO Bool
forwardComputes = do
  gW <- param {ex=TestExecutor} {dt=TestDType} {dims=[3, 2]} "sg.g" (Const 0.5)
  uW <- param {ex=TestExecutor} {dt=TestDType} {dims=[3, 2]} "sg.u" (Const 0.5)
  dW <- param {ex=TestExecutor} {dt=TestDType} {dims=[2, 3]} "sg.d" (Const 0.5)
  let blk = the (SwiGLU 2 3 TestExecutor TestDType WithGrad) (MkSwiGLU gW uW dW)
  x   <- tensor {ex=TestExecutor} {dt=TestDType} {dims=[2]} (Const 1.0)
  out <- swigluForward blk (retypeGrad x)
  let v0 = primItem1d {ex=TestExecutor} out.tensorPtr 0
  let v1 = primItem1d {ex=TestExecutor} out.tensorPtr 1
  check ("SwiGLU forward (got [" ++ show v0 ++ ", " ++ show v1 ++ "])")
        (abs (v0 - 1.096588) < 1.0e-4 && abs (v1 - 1.096588) < 1.0e-4)

paramsExposed : IO Bool
paramsExposed = do
  gW <- param {ex=TestExecutor} {dt=TestDType} {dims=[3, 2]} "sp.g" (Const 0.5)
  uW <- param {ex=TestExecutor} {dt=TestDType} {dims=[3, 2]} "sp.u" (Const 0.5)
  dW <- param {ex=TestExecutor} {dt=TestDType} {dims=[2, 3]} "sp.d" (Const 0.5)
  let blk = the (SwiGLU 2 3 TestExecutor TestDType WithGrad) (MkSwiGLU gW uW dW)
  check ("Params (SwiGLU) = gate,up,down (got " ++ show (mapMaybe paramName (params blk)) ++ ")")
        (mapMaybe paramName (params blk) == ["sp.g", "sp.u", "sp.d"])

smartCtorNames : IO Bool
smartCtorNames = do
  _ <- runInit $ scoped "mlp" (swiglu {ex=TestExecutor} {dt=TestDType} {g=WithGrad} {hidden=4} {intermediate=8})
  cnt <- getParamCount {ex=TestExecutor}
  names <- traverse (\i => getParamName {ex=TestExecutor} i) [0 .. cnt - 1]
  check "swiglu registers mlp.swiglu_0.{gate,up,down}_proj.weight"
        (("mlp.swiglu_0.gate_proj.weight" `elem` names)
         && ("mlp.swiglu_0.up_proj.weight" `elem` names)
         && ("mlp.swiglu_0.down_proj.weight" `elem` names))

export
tests : List (IO Bool)
tests = [forwardComputes, paramsExposed, smartCtorNames]
