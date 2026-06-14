module Test.Nn.Module

import Control.Linear.LIO
import Data.Linear.Notation
import Data.Vect

import Executor
import Nn.Module
import Tensor
import Test.Config
import Test.Harness

-- A trivial identity module (i = o, no params) — exercises the Module +
-- Params interfaces without depending on a real layer port.
data Id : Nat -> Nat -> (0 _ : Executor) -> (0 _ : DType) -> (0 _ : GradMode) -> Type where
  MkId : Id n n ex dt g

Params Id where
  params MkId   = []
  reflect MkId  = MkBang [] # MkId
  castGrad MkId = MkId
  discard MkId  = pure ()

Module Id where
  forward MkId x = pure1 (MkBang x # MkId)

read6 : Tensor [2, 3] TestExecutor TestDType g -> List Double
read6 t = [ primItem2d {ex=TestExecutor} t.tensorPtr i j
          | (i, j) <- the (List (Int, Int)) [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)] ]

forwardDispatches : IO Bool
forwardDispatches = do
  t <- tensor {ex=TestExecutor} {dt=TestDType} {dims=[2, 3]} (Const 4.0)
  out <- Control.Linear.LIO.run (do
           (MkBang o # m') <- forward {ex=TestExecutor} {b=2} (the (Id 3 3 TestExecutor TestDType NoGrad) MkId) t
           discard m'
           pure o)
  check ("Module.forward dispatches (identity, got " ++ show (read6 out) ++ ")")
        (read6 out == [4.0, 4.0, 4.0, 4.0, 4.0, 4.0])

emptyParams : IO Bool
emptyParams =
  check "Params.params of a param-free module is empty"
        (length (params (the (Id 3 3 TestExecutor TestDType NoGrad) MkId)) == 0)

export
tests : List (IO Bool)
tests = [forwardDispatches, emptyParams]
