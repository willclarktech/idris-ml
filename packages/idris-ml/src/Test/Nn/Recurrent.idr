module Test.Nn.Recurrent

import Data.List
import Data.Vect

import Executor
import Nn.Init
import Nn.Module
import Nn.Recurrent
import Tensor
import Test.Config
import Test.Harness

-- o=i=1, W_ih=1, W_hh=0.5, biases 0, tanh, x=1, h0=0:
--   step1: tanh(1·1 + 0.5·0 + 0)          = tanh(1)       ≈ 0.761594
--   step2: tanh(1·1 + 0.5·0.761594 + 0)   = tanh(1.380797)≈ 0.881130
mkRnn1 : IO (Rnn 1 1 TestExecutor TestDType WithGrad)
mkRnn1 = do
  iw <- param {ex=TestExecutor} {dt=TestDType} {dims=[1, 1]} "rn.iw" (Const 1.0)
  rw <- param {ex=TestExecutor} {dt=TestDType} {dims=[1, 1]} "rn.rw" (Const 0.5)
  ib <- param {ex=TestExecutor} {dt=TestDType} {dims=[1]}    "rn.ib" (Const 0.0)
  hb <- param {ex=TestExecutor} {dt=TestDType} {dims=[1]}    "rn.hb" (Const 0.0)
  pure (MkRnn iw rw ib hb ttanh Nothing)

input1 : IO (Tensor [1] TestExecutor TestDType WithGrad)
input1 = do
  x <- tensor {ex=TestExecutor} {dt=TestDType} {dims=[1]} (Const 1.0)
  pure (retypeGrad x)

stepCarriesState : IO Bool
stepCarriesState = do
  r0 <- mkRnn1
  x  <- input1
  (r1, o1) <- recurStep r0 x
  (_,  o2) <- recurStep r1 x
  let v1 = primItem1d {ex=TestExecutor} o1.tensorPtr 0
  let v2 = primItem1d {ex=TestExecutor} o2.tensorPtr 0
  check ("recurStep carries hidden state (got " ++ show v1 ++ ", " ++ show v2 ++ ")")
        (abs (v1 - 0.761594) < 1.0e-4 && abs (v2 - 0.881130) < 1.0e-4)

resetClearsState : IO Bool
resetClearsState = do
  r0 <- mkRnn1
  x  <- input1
  (r1, _) <- recurStep r0 x
  (_, oR) <- recurStep (recurReset r1) x
  let vR = primItem1d {ex=TestExecutor} oR.tensorPtr 0
  check ("recurReset restarts from zero state (got " ++ show vR ++ ")")
        (abs (vR - 0.761594) < 1.0e-4)

paramsExposed : IO Bool
paramsExposed = do
  r0 <- mkRnn1
  check ("Params (Rnn) = ih/hh weights+biases (got " ++ show (mapMaybe paramName (params r0)) ++ ")")
        (mapMaybe paramName (params r0) == ["rn.iw", "rn.rw", "rn.ib", "rn.hb"])

smartCtorNames : IO Bool
smartCtorNames = do
  _ <- runInit $ scoped "enc" (rnn {ex=TestExecutor} {dt=TestDType} {i=3} {o=4} ttanh)
  cnt <- getParamCount {ex=TestExecutor}
  names <- traverse (\i => getParamName {ex=TestExecutor} i) [0 .. cnt - 1]
  check "rnn registers enc.rnn_0.{weight,bias}_{ih,hh}"
        (("enc.rnn_0.weight_ih" `elem` names) && ("enc.rnn_0.bias_hh" `elem` names))

export
tests : List (IO Bool)
tests = [stepCarriesState, resetClearsState, paramsExposed, smartCtorNames]
