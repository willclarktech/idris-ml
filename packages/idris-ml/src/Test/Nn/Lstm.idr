module Test.Nn.Lstm

import Control.Linear.LIO
import Data.Linear.Notation
import Data.List
import Data.Vect

import Ml.Executor
import Ml.Nn.Init
import Ml.Nn.Lstm
import Ml.Nn.Module
import Ml.Nn.Recurrent
import Ml.Tensor
import Test.Harness

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
  (o1, o2) <- Control.Linear.LIO.run (do
     (MkBang a # l1) <- recurStep l0 x
     (MkBang b # l2) <- recurStep l1 x
     discard l2
     pure (a, b))
  let v1 = primItem1d {ex=TestExecutor} o1.tensorPtr 0
  let v2 = primItem1d {ex=TestExecutor} o2.tensorPtr 0
  check ("LSTM hidden bounded + state carried (got " ++ show v1 ++ ", " ++ show v2 ++ ")")
        (abs v1 < 1.0 && abs v2 < 1.0 && abs (v2 - v1) > 1.0e-3)

resetRestores : IO Bool
resetRestores = do
  lA <- mkLstm1
  lB <- mkLstm1
  x  <- inp1
  (oA, oR) <- Control.Linear.LIO.run (do
     (MkBang a # la1) <- recurStep lA x
     discard la1
     (MkBang _ # lb1) <- recurStep lB x
     (MkBang r # lb2) <- recurStep (recurReset lb1) x
     discard lb2
     pure (a, r))
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
  _ <- runInit $ scoped "enc" (lstm {ex=TestExecutor} {dt=TestDType} {g=WithGrad} {i=3} {o=4})
  cnt <- getParamCount {ex=TestExecutor}
  names <- traverse (\i => getParamName {ex=TestExecutor} i) [0 .. cnt - 1]
  check "lstm registers enc.lstm_0.{weight_ih,c0}"
        (("enc.lstm_0.weight_ih" `elem` names) && ("enc.lstm_0.c0" `elem` names))

-- Init contract: every recurrent weight matrix is Xavier-*uniform*,
-- `U(±√(6/(fan_in+fan_out)))`, matching the reference's
-- `nn.init.xavier_uniform_`. The Idris side used a normal of the same
-- variance until 2026-07-31, which is why the check is on the tail rather
-- than the spread: at equal variance only a uniform keeps every draw inside
-- the bound. Over 2048 elements a normal clears it with probability
-- 0.9167^2048, i.e. never.
lstmInitIsXavierUniform : IO Bool
lstmInitIsXavierUniform = do
  m <- runInit $ scoped "li" (lstm {ex=TestExecutor} {dt=TestDType} {g=WithGrad} {i=32} {o=16})
  let bound = sqrt (6.0 / cast {to=Double} (32 + 64))
      ws    = case params m of
                (p :: _) => [ primItem1d {ex=TestExecutor} p.paramPtr k
                            | k <- map (cast {to=Int}) [the Nat 0 .. 2047] ]
                []       => []
  check ("lstm weight_ih ~ U(±√(6/(fan_in+fan_out))) (bound " ++ show bound
         ++ ", max " ++ show (foldl (\a, w => max a (abs w)) 0.0 ws) ++ ")")
        (all (\w => abs w <= bound) ws && any (\w => w /= 0.0) ws)

export
tests : List (IO Bool)
tests = [stepCarriesState, resetRestores, paramsExposed, smartCtorNames, lstmInitIsXavierUniform]
