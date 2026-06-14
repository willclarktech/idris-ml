module Test.Nn.LoraLinear

import Data.List
import Data.Vect

import Test.Harness
import Executor
import Tensor
import Nn.Init
import Nn.Module
import Nn.Linear
import Nn.LoraLinear
import Test.Config

-- At init B = 0, so the LoRA delta is zero and loraForward == the bare
-- base linear (1-D tlinear). W[2,3]=0.5, b[2]=1.0, x[3]=2.0 →
-- W·x + b = 3*1.0 + 1.0 = 4.0 per output.
zeroBIsBase : IO Bool
zeroBIsBase = do
  bw <- param {ex=TestExecutor} {dt=TestDType} {dims=[2, 3]} "lr.w" (Const 0.5)
  bb <- param {ex=TestExecutor} {dt=TestDType} {dims=[2]}    "lr.b" (Const 1.0)
  la <- param {ex=TestExecutor} {dt=TestDType} {dims=[2, 3]} "lr.A" (Const 0.3)  -- rank=2
  lb <- param {ex=TestExecutor} {dt=TestDType} {dims=[2, 2]} "lr.B" (Const 0.0)  -- zero → delta 0
  let lora = the (LoraLinear 3 2 TestExecutor TestDType)
               (MkLoraLinear {rank=2} (MkLinear bw bb) la lb 16.0)
  x   <- tensor {ex=TestExecutor} {dt=TestDType} {dims=[3]} (Const 2.0)
  out <- loraForward lora x
  let vs = [ primItem1d {ex=TestExecutor} out.tensorPtr i | i <- the (List Int) [0, 1] ]
  check ("LoRA forward with B=0 equals base linear (got " ++ show vs ++ ")")
        (vs == [4.0, 4.0])

paramsCompose : IO Bool
paramsCompose = do
  bw <- param {ex=TestExecutor} {dt=TestDType} {dims=[2, 3]} "lp.w" (Const 0.5)
  bb <- param {ex=TestExecutor} {dt=TestDType} {dims=[2]}    "lp.b" (Const 1.0)
  la <- param {ex=TestExecutor} {dt=TestDType} {dims=[2, 3]} "lp.A" (Const 0.3)
  lb <- param {ex=TestExecutor} {dt=TestDType} {dims=[2, 2]} "lp.B" (Const 0.0)
  let lora = the (LoraLinear 3 2 TestExecutor TestDType)
               (MkLoraLinear {rank=2} (MkLinear bw bb) la lb 16.0)
  check ("Params (LoraLinear) = base + adapters (got "
         ++ show (mapMaybe paramName (params lora)) ++ ")")
        (mapMaybe paramName (params lora) == ["lp.w", "lp.b", "lp.A", "lp.B"])

export
tests : List (IO Bool)
tests = [zeroBIsBase, paramsCompose]
