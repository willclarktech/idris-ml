module Test.Nn.Lstm

import Data.List
import Data.Vect

import Test.Harness
import Executor
import Tensor
import Nn.Init
import Nn.Module
import Nn.Recurrent
import Nn.Lstm
import Test.Config

-- o=i=1; W_ih=W_hh=1, biases 0, h0=c0=0, x=1. Exact gate values depend on
-- the C kernel's gate order, so assert deterministic *properties* instead:
-- hidden bounded in (-1,1), state carried (step2 ≠ step1), reset restores
-- step1 exactly.
mkLstm1 : IO (Lstm 1 1 TestExecutor TestDType WithGrad)
mkLstm1 = do
  iw <- param {ex=TestExecutor} {dt=TestDType} {dims=[4, 1]} "ls.iw" (Const 1.0)
  rw <- param {ex=TestExecutor} {dt=TestDType} {dims=[4, 1]} "ls.rw" (Const 1.0)
  ib <- param {ex=TestExecutor} {dt=TestDType} {dims=[4]}    "ls.ib" (Const 0.0)
  hb <- param {ex=TestExecutor} {dt=TestDType} {dims=[4]}    "ls.hb" (Const 0.0)
  h0 <- param {ex=TestExecutor} {dt=TestDType} {dims=[1]}    "ls.h0" (Const 0.0)
  c0 <- param {ex=TestExecutor} {dt=TestDType} {dims=[1]}    "ls.c0" (Const 0.0)
  pure (MkLstm iw rw ib hb h0 c0 Nothing Nothing)

inp1 : IO (Tensor [1] TestExecutor TestDType WithGrad)
inp1 = retypeGrad <$> tensor {ex=TestExecutor} {dt=TestDType} {dims=[1]} (Const 1.0)

stepCarriesState : IO Bool
stepCarriesState = do
  l0 <- mkLstm1
  x  <- inp1
  (l1, o1) <- recurStep l0 x
  (_,  o2) <- recurStep l1 x
  let v1 = primItem1d {ex=TestExecutor} o1.tensorPtr 0
  let v2 = primItem1d {ex=TestExecutor} o2.tensorPtr 0
  check ("LSTM hidden bounded + state carried (got " ++ show v1 ++ ", " ++ show v2 ++ ")")
        (abs v1 < 1.0 && abs v2 < 1.0 && abs (v2 - v1) > 1.0e-3)

resetRestores : IO Bool
resetRestores = do
  l0 <- mkLstm1
  x  <- inp1
  (_, oA)  <- recurStep l0 x
  (l1, _)  <- recurStep l0 x
  (_, oR)  <- recurStep (recurReset l1) x
  let vA = primItem1d {ex=TestExecutor} oA.tensorPtr 0
  let vR = primItem1d {ex=TestExecutor} oR.tensorPtr 0
  check ("LSTM recurReset restores step-1 output (got " ++ show vA ++ " vs " ++ show vR ++ ")")
        (abs (vA - vR) < 1.0e-9)

paramsExposed : IO Bool
paramsExposed = do
  l0 <- mkLstm1
  check ("Params (Lstm) lists 6 learnable tensors (got " ++ show (mapMaybe paramName (params l0)) ++ ")")
        (mapMaybe paramName (params l0) == ["ls.iw", "ls.rw", "ls.ib", "ls.hb", "ls.h0", "ls.c0"])

smartCtorNames : IO Bool
smartCtorNames = do
  _ <- runInit $ scoped "enc" (lstm {ex=TestExecutor} {dt=TestDType} {i=3} {o=4})
  cnt <- getParamCount {ex=TestExecutor}
  names <- traverse (\i => getParamName {ex=TestExecutor} i) [0 .. cnt - 1]
  check "lstm registers enc.lstm_0.{weight_ih,c0}"
        (("enc.lstm_0.weight_ih" `elem` names) && ("enc.lstm_0.c0" `elem` names))

export
tests : List (IO Bool)
tests = [stepCarriesState, resetRestores, paramsExposed, smartCtorNames]
