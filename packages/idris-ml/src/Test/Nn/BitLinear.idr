module Test.Nn.BitLinear

import Data.List
import Data.Vect

import Ml.Executor
import Ml.Nn.BitLinear
import Ml.Nn.Module
import Ml.Tensor
import Test.Config
import Test.Harness

-- Same PyTorch-verified fixture as Test.BitLinear: ternary weight from
-- packed bytes 0x71/0x17/0x4C ([3,4]), scale [0.5,0.25,0.75], bias
-- [0.1,-0.2,0.3], input [1,2,-0.5,0.25] → y = [0.975, -0.075, -1.0125].
fixtureBytes : IO (AnyPtr, Int)
fixtureBytes = do
  let b0 = prim__allocBytes 3
      b1 = prim__setByte b0 0 0x71
      b2 = prim__setByte b1 1 0x17
      b3 = prim__setByte b2 2 0x4C
  pure (b3, 3)

vecG : {n : Nat} -> Vect n Double -> IO (Tensor [n] TestExecutor TestDType WithGrad)
vecG xs = retypeGrad <$> tensor {ex=TestExecutor} {dt=TestDType} {dims=[n]} (FromVect xs)

vecN : {n : Nat} -> Vect n Double -> IO (Tensor [n] TestExecutor TestDType NoGrad)
vecN xs = tensor {ex=TestExecutor} {dt=TestDType} {dims=[n]} (FromVect xs)

mkFixture : IO (BitLinear 4 3 TestExecutor TestDType WithGrad)
mkFixture = do
  (bytes, cnt) <- fixtureBytes
  w <- tCreateTernaryPacked2d {ex=TestExecutor} {o=3} {i=4} bytes cnt
  s <- vecN (the (Vect 3 Double) [0.5, 0.25, 0.75])
  b <- vecG (the (Vect 3 Double) [0.1, -0.2, 0.3])
  pure (bitLinear w s b)

forwardMatchesOracle : IO Bool
forwardMatchesOracle = do
  bl <- mkFixture
  x  <- vecG (the (Vect 4 Double) [1.0, 2.0, -0.5, 0.25])
  y  <- bitLinearForward bl x
  let y0 = primItem1d {ex=TestExecutor} y.tensorPtr 0
  let y1 = primItem1d {ex=TestExecutor} y.tensorPtr 1
  let y2 = primItem1d {ex=TestExecutor} y.tensorPtr 2
  check ("BitLinear forward matches PyTorch oracle (got [" ++ show y0 ++ ", " ++ show y1 ++ ", " ++ show y2 ++ "])")
        (abs (y0 - 0.975) < 1.0e-6 && abs (y1 + 0.075) < 1.0e-6 && abs (y2 + 1.0125) < 1.0e-6)

paramsExposesAllThree : IO Bool
paramsExposesAllThree = do
  bl <- mkFixture
  check ("Params (BitLinear) lists weight+scale+bias (got " ++ show (length (params bl)) ++ ")")
        (length (params bl) == 3)

export
tests : List (IO Bool)
tests = [forwardMatchesOracle, paramsExposesAllThree]
