module Test.Nn.Dnc

import Control.Linear.LIO
import Data.Linear.Notation
import Data.List
import Data.Vect

import Executor
import Nn.Dnc
import Nn.Init
import Nn.Module
import Nn.Recurrent
import Tensor
import Test.Config
import Test.Harness

-- Small DNC: 2 read heads, 4 memory slots × width 3, hidden 8, in/out 2.
mkDnc : IO (Dnc 2 4 3 8 2 2 TestExecutor TestDType WithGrad)
mkDnc = runInit (dnc {r=2} {n=4} {m=3} {h=8} {i=2} {o=2})

inp2 : IO (Tensor [2] TestExecutor TestDType WithGrad)
inp2 = retypeGrad <$> tensor {ex=TestExecutor} {dt=TestDType} {dims=[2]} (Const 0.5)

stepCarriesState : IO Bool
stepCarriesState = do
  d0 <- mkDnc
  x  <- inp2
  (o1, o2) <- Control.Linear.LIO.run (do
     (MkBang a # d1) <- recurStep d0 x
     (MkBang b # d2) <- recurStep d1 x
     discard d2
     pure (a, b))
  let v1 = primItem1d {ex=TestExecutor} o1.tensorPtr 0
  let v2 = primItem1d {ex=TestExecutor} o2.tensorPtr 0
  check ("DNC forward finite + memory state carried (got " ++ show v1 ++ ", " ++ show v2 ++ ")")
        (v1 == v1 && v2 == v2 && abs (v1 - v2) > 1.0e-9)

resetRestores : IO Bool
resetRestores = do
  d0 <- mkDnc
  x  <- inp2
  -- One cell threaded: first step (from cleared init) → out a; reset clears
  -- the carried state (params unchanged) → stepping again reproduces a.
  (oA, oR) <- Control.Linear.LIO.run (do
     (MkBang a # d1) <- recurStep d0 x
     (MkBang r # d2) <- recurStep (recurReset d1) x
     discard d2
     pure (a, r))
  let vA = primItem1d {ex=TestExecutor} oA.tensorPtr 0
  let vR = primItem1d {ex=TestExecutor} oR.tensorPtr 0
  check ("DNC recurReset restores first-step output (got " ++ show vA ++ " vs " ++ show vR ++ ")")
        (abs (vA - vR) < 1.0e-9)

paramsCompose : IO Bool
paramsCompose = do
  d0 <- mkDnc
  -- controller LSTM (6) + 11 head FCs (2 each = 22) + memory_init (1) = 29.
  check ("Params (Dnc) composes 11 heads + controller + memInit (got " ++ show (length (params d0)) ++ ")")
        (length (params d0) == 29)

smartCtorNames : IO Bool
smartCtorNames = do
  _ <- mkDnc
  cnt <- getParamCount {ex=TestExecutor}
  names <- traverse (\i => getParamName {ex=TestExecutor} i) [0 .. cnt - 1]
  check "dnc nests controller/heads under dnc_0.*"
        (("dnc_0.controller.weight_ih" `elem` names) && ("dnc_0.read_keys.weight" `elem` names)
         && ("dnc_0.output.bias" `elem` names))

export
tests : List (IO Bool)
tests = [stepCarriesState, resetRestores, paramsCompose, smartCtorNames]
