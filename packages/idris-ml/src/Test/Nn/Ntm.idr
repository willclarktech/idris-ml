module Test.Nn.Ntm

import Data.List
import Data.Vect

import Test.Harness
import Executor
import Tensor
import Nn.Init
import Nn.Module
import Nn.Recurrent
import Nn.Ntm
import Test.Config

-- Small NTM: 4 memory slots × width 3, hidden 8, in/out 2.
mkNtm : IO (Ntm 4 3 8 2 2 TestExecutor TestDType WithGrad)
mkNtm = runInit (ntm {n=4} {m=3} {h=8} {i=2} {o=2})

inp2 : IO (Tensor [2] TestExecutor TestDType WithGrad)
inp2 = retypeGrad <$> tensor {ex=TestExecutor} {dt=TestDType} {dims=[2]} (Const 0.5)

stepCarriesState : IO Bool
stepCarriesState = do
  nt0 <- mkNtm
  x   <- inp2
  (nt1, o1) <- recurStep nt0 x
  (_,   o2) <- recurStep nt1 x
  let v1 = primItem1d {ex=TestExecutor} o1.tensorPtr 0
  let v2 = primItem1d {ex=TestExecutor} o2.tensorPtr 0
  check ("NTM forward finite + memory state carried (got " ++ show v1 ++ ", " ++ show v2 ++ ")")
        (v1 == v1 && v2 == v2 && abs (v1 - v2) > 1.0e-9)  -- v==v rejects NaN

resetRestores : IO Bool
resetRestores = do
  nt0 <- mkNtm
  x   <- inp2
  (_,   oA) <- recurStep nt0 x
  (nt1, _)  <- recurStep nt0 x
  (_,   oR) <- recurStep (recurReset nt1) x
  let vA = primItem1d {ex=TestExecutor} oA.tensorPtr 0
  let vR = primItem1d {ex=TestExecutor} oR.tensorPtr 0
  check ("NTM recurReset restores first-step output (got " ++ show vA ++ " vs " ++ show vR ++ ")")
        (abs (vA - vR) < 1.0e-9)

paramsCompose : IO Bool
paramsCompose = do
  nt0 <- mkNtm
  -- controller LSTM (6) + 3 heads (2 each) + memory_init (1) + read_out (1) = 14.
  check ("Params (Ntm) composes sub-layers (got " ++ show (length (params nt0)) ++ ")")
        (length (params nt0) == 14)

smartCtorNames : IO Bool
smartCtorNames = do
  _ <- mkNtm
  cnt <- getParamCount {ex=TestExecutor}
  names <- traverse (\i => getParamName {ex=TestExecutor} i) [0 .. cnt - 1]
  check "ntm nests controller/heads under ntm_0.*"
        (("ntm_0.controller.weight_ih" `elem` names) && ("ntm_0.read_fc.weight" `elem` names)
         && ("ntm_0.output_fc.bias" `elem` names))

export
tests : List (IO Bool)
tests = [stepCarriesState, resetRestores, paramsCompose, smartCtorNames]
