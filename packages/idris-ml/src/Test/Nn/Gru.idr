module Test.Nn.Gru

import Control.Linear.LIO
import Data.Linear.Notation
import Data.List
import Data.Vect

import Ml.Executor
import Ml.Nn.Gru
import Ml.Nn.Init
import Ml.Nn.Module
import Ml.Nn.Recurrent
import Ml.Tensor
import Test.Harness

import Test.Config

-- Deterministic property checks (gate order is a C-kernel detail): hidden
-- bounded, state carried, reset restores step-1 output exactly.
mkGru1 : IO (Gru 1 1 TestExecutor TestDType WithGrad)
mkGru1 = do
  iw <- param {ex=TestExecutor} {dt=TestDType} {dims=[3, 1]} "gr.iw" (Const 1.0)
  ib <- param {ex=TestExecutor} {dt=TestDType} {dims=[3]}    "gr.ib" (Const 0.0)
  hw <- param {ex=TestExecutor} {dt=TestDType} {dims=[3, 1]} "gr.hw" (Const 1.0)
  hb <- param {ex=TestExecutor} {dt=TestDType} {dims=[3]}    "gr.hb" (Const 0.0)
  pure (MkGru iw ib hw hb Nothing)

inp1 : IO (Tensor [1] TestExecutor TestDType WithGrad)
inp1 = retypeGrad <$> tensor {ex=TestExecutor} {dt=TestDType} {dims=[1]} (Const 1.0)

stepCarriesState : IO Bool
stepCarriesState = do
  g0 <- mkGru1
  x  <- inp1
  (o1, o2) <- Control.Linear.LIO.run (do
     (MkBang a # g1) <- recurStep g0 x
     (MkBang b # g2) <- recurStep g1 x
     discard g2
     pure (a, b))
  let v1 = primItem1d {ex=TestExecutor} o1.tensorPtr 0
  let v2 = primItem1d {ex=TestExecutor} o2.tensorPtr 0
  check ("GRU hidden bounded + state carried (got " ++ show v1 ++ ", " ++ show v2 ++ ")")
        (abs v1 < 1.0 && abs v2 < 1.0 && abs (v2 - v1) > 1.0e-3)

resetRestores : IO Bool
resetRestores = do
  gA <- mkGru1
  gB <- mkGru1
  x  <- inp1
  (oA, oR) <- Control.Linear.LIO.run (do
     (MkBang a # ga1) <- recurStep gA x
     discard ga1
     (MkBang _ # gb1) <- recurStep gB x
     (MkBang r # gb2) <- recurStep (recurReset gb1) x
     discard gb2
     pure (a, r))
  let vA = primItem1d {ex=TestExecutor} oA.tensorPtr 0
  let vR = primItem1d {ex=TestExecutor} oR.tensorPtr 0
  check ("GRU recurReset restores step-1 output (got " ++ show vA ++ " vs " ++ show vR ++ ")")
        (abs (vA - vR) < 1.0e-9)

paramsExposed : IO Bool
paramsExposed = do
  g0 <- mkGru1
  check ("Params (Gru) lists 4 learnable tensors (got " ++ show (mapMaybe paramName (params g0)) ++ ")")
        (mapMaybe paramName (params g0) == ["gr.iw", "gr.ib", "gr.hw", "gr.hb"])

smartCtorNames : IO Bool
smartCtorNames = do
  _ <- runInit $ scoped "enc" (gru {ex=TestExecutor} {dt=TestDType} {g=WithGrad} {i=3} {o=4})
  cnt <- getParamCount {ex=TestExecutor}
  names <- traverse (\i => getParamName {ex=TestExecutor} i) [0 .. cnt - 1]
  check "gru registers enc.gru_0.weight_ih + .weight_hh"
        (("enc.gru_0.weight_ih" `elem` names) && ("enc.gru_0.weight_hh" `elem` names))

-- Init contract: every recurrent weight matrix is Xavier-*uniform*,
-- `U(±√(6/(fan_in+fan_out)))`, matching the reference's
-- `nn.init.xavier_uniform_`. The Idris side used a normal of the same
-- variance until 2026-07-31, which is why the check is on the tail rather
-- than the spread: at equal variance only a uniform keeps every draw inside
-- the bound. Over 1536 elements a normal clears it with probability
-- 0.9167^1536, i.e. never.
gruInitIsXavierUniform : IO Bool
gruInitIsXavierUniform = do
  m <- runInit $ scoped "gi" (gru {ex=TestExecutor} {dt=TestDType} {g=WithGrad} {i=32} {o=16})
  let bound = sqrt (6.0 / cast {to=Double} (32 + 48))
      ws    = case params m of
                (p :: _) => [ primItem1d {ex=TestExecutor} p.paramPtr k
                            | k <- map (cast {to=Int}) [the Nat 0 .. 1535] ]
                []       => []
  check ("gru weight_ih ~ U(±√(6/(fan_in+fan_out))) (bound " ++ show bound
         ++ ", max " ++ show (foldl (\a, w => max a (abs w)) 0.0 ws) ++ ")")
        (all (\w => abs w <= bound) ws && any (\w => w /= 0.0) ws)

export
tests : List (IO Bool)
tests = [stepCarriesState, resetRestores, paramsExposed, smartCtorNames, gruInitIsXavierUniform]
