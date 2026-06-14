module Test.Nn.Module

import Data.Vect

import Test.Harness
import Executor
import Tensor
import Nn.Module
import Test.Config

-- A trivial identity module (i = o, no params) — exercises the Module +
-- Params interfaces without depending on a real layer port.
data Id : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type where
  MkId : Id n n ex dt g

Module Id where
  forward MkId x = pure x

Params Id where
  params MkId = []
  castGrad MkId = MkId

read6 : Tensor [2, 3] TestExecutor TestDType g -> List Double
read6 t = [ primItem2d {ex=TestExecutor} t.tensorPtr i j
          | (i, j) <- the (List (Int, Int)) [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)] ]

forwardDispatches : IO Bool
forwardDispatches = do
  t <- tensor {ex=TestExecutor} {dt=TestDType} {dims=[2, 3]} (Const 4.0)
  out <- forward {ex=TestExecutor} {b=2} (the (Id 3 3 TestExecutor TestDType NoGrad) MkId) t
  check ("Module.forward dispatches (identity, got " ++ show (read6 out) ++ ")")
        (read6 out == [4.0, 4.0, 4.0, 4.0, 4.0, 4.0])

emptyParams : IO Bool
emptyParams =
  check "Params.params of a param-free module is empty"
        (length (params (the (Id 3 3 TestExecutor TestDType NoGrad) MkId)) == 0)

export
tests : List (IO Bool)
tests = [forwardDispatches, emptyParams]
