module Test.Nn.Recurrent

import Control.Linear.LIO
import Data.Linear.Notation
import Data.List
import Data.Vect

import Ml.Executor
import Ml.Nn.Init
import Ml.Nn.Module
import Ml.Nn.Recurrent
import Ml.Tensor
import Test.Harness

import Test.Config

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
  (o1, o2) <- Control.Linear.LIO.run (do
     (MkBang a # r1) <- recurStep r0 x
     (MkBang b # r2) <- recurStep r1 x
     discard r2
     pure (a, b))
  let v1 = primItem1d {ex=TestExecutor} o1.tensorPtr 0
  let v2 = primItem1d {ex=TestExecutor} o2.tensorPtr 0
  check ("recurStep carries hidden state (got " ++ show v1 ++ ", " ++ show v2 ++ ")")
        (abs (v1 - 0.761594) < 1.0e-4 && abs (v2 - 0.881130) < 1.0e-4)

resetClearsState : IO Bool
resetClearsState = do
  r0 <- mkRnn1
  x  <- input1
  oR <- Control.Linear.LIO.run (do
     (MkBang _ # r1) <- recurStep r0 x
     (MkBang r # r2) <- recurStep (recurReset r1) x
     discard r2
     pure r)
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
  _ <- runInit $ scoped "enc" (rnn {ex=TestExecutor} {dt=TestDType} {g=WithGrad} {i=3} {o=4} ttanh)
  cnt <- getParamCount {ex=TestExecutor}
  names <- traverse (\i => getParamName {ex=TestExecutor} i) [0 .. cnt - 1]
  check "rnn registers enc.rnn_0.{weight,bias}_{ih,hh}"
        (("enc.rnn_0.weight_ih" `elem` names) && ("enc.rnn_0.bias_hh" `elem` names))

-- Init contract: every recurrent weight matrix is Xavier-*uniform*,
-- `U(±√(6/(fan_in+fan_out)))`, matching the reference's
-- `nn.init.xavier_uniform_`. The Idris side used a normal of the same
-- variance until 2026-07-31, which is why the check is on the tail rather
-- than the spread: at equal variance only a uniform keeps every draw inside
-- the bound. Over 1024 elements a normal clears it with probability
-- 0.9167^1024, i.e. never.
rnnInitIsXavierUniform : IO Bool
rnnInitIsXavierUniform = do
  m <- runInit $ scoped "ri" (rnn {ex=TestExecutor} {dt=TestDType} {g=WithGrad} {i=32} {o=32} ttanh)
  let bound = sqrt (6.0 / cast {to=Double} (32 + 32))
      ws    = case params m of
                (p :: _) => [ primItem1d {ex=TestExecutor} p.paramPtr k
                            | k <- map (cast {to=Int}) [the Nat 0 .. 1023] ]
                []       => []
  check ("rnn weight_ih ~ U(±√(6/(fan_in+fan_out))) (bound " ++ show bound
         ++ ", max " ++ show (foldl (\a, w => max a (abs w)) 0.0 ws) ++ ")")
        (all (\w => abs w <= bound) ws && any (\w => w /= 0.0) ws)

export
tests : List (IO Bool)
tests = [stepCarriesState, resetClearsState, paramsExposed, smartCtorNames, rnnInitIsXavierUniform]
