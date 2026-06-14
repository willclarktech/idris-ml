module Test.Nn.Gru

import Data.List
import Data.Vect

import Executor
import Nn.Gru
import Nn.Init
import Nn.Module
import Nn.Recurrent
import Tensor
import Test.Config
import Test.Harness

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
  (g1, o1) <- recurStep g0 x
  (_,  o2) <- recurStep g1 x
  let v1 = primItem1d {ex=TestExecutor} o1.tensorPtr 0
  let v2 = primItem1d {ex=TestExecutor} o2.tensorPtr 0
  check ("GRU hidden bounded + state carried (got " ++ show v1 ++ ", " ++ show v2 ++ ")")
        (abs v1 < 1.0 && abs v2 < 1.0 && abs (v2 - v1) > 1.0e-3)

resetRestores : IO Bool
resetRestores = do
  g0 <- mkGru1
  x  <- inp1
  (_, oA) <- recurStep g0 x
  (g1, _) <- recurStep g0 x
  (_, oR) <- recurStep (recurReset g1) x
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
  _ <- runInit $ scoped "enc" (gru {ex=TestExecutor} {dt=TestDType} {i=3} {o=4})
  cnt <- getParamCount {ex=TestExecutor}
  names <- traverse (\i => getParamName {ex=TestExecutor} i) [0 .. cnt - 1]
  check "gru registers enc.gru_0.weight_ih + .weight_hh"
        (("enc.gru_0.weight_ih" `elem` names) && ("enc.gru_0.weight_hh" `elem` names))

export
tests : List (IO Bool)
tests = [stepCarriesState, resetRestores, paramsExposed, smartCtorNames]
