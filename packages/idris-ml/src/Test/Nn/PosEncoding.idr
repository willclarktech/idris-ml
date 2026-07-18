module Test.Nn.PosEncoding

import Control.Linear.LIO
import Data.Vect

import Ml.Executor
import Ml.Nn.PosEncoding
import Ml.Tensor
import Test.Harness

import Test.Config

-- Value pins for the sinusoidal positional-encoding table (Vaswani et al
-- 2017): PE[pos,2i] = sin(pos / 10000^(2i/dModel)),
--        PE[pos,2i+1] = cos(pos / 10000^(2i/dModel)).
-- Pure host-side F64 both sides; 1e-12 leaves headroom for sin/cos ulps.
tol : Double
tol = 1.0e-12

near : String -> Double -> Double -> IO Bool
near label got want =
  if abs (got - want) < tol
    then check label True
    else do
      putStrLn ("  FAIL: " ++ label ++ " got " ++ show got ++ " expected " ++ show want)
      pure False

----------------------------------------------------------------------
-- Bucket 1: posEncVal pure host math
----------------------------------------------------------------------

-- pos=0 → angle 0 for every dim: sin 0 = 0 (even), cos 0 = 1 (odd).
testPosZeroEven : IO Bool
testPosZeroEven = near "posEncVal _ 0 0 = sin 0 = 0" (posEncVal 4 0 0) 0.0

testPosZeroOdd : IO Bool
testPosZeroOdd = near "posEncVal _ 0 1 = cos 0 = 1" (posEncVal 4 0 1) 1.0

-- pos=1, dim=0: i=0, angle = 1/10000^0 = 1 → sin 1.
testPosOneDim0 : IO Bool
testPosOneDim0 = near "posEncVal 4 1 0 = sin 1" (posEncVal 4 1 0) (sin 1.0)

-- pos=1, dim=2: even → sin; i = 2/2 = 1; angle = 1/10000^(2/4) = 0.01.
testPosOneDim2 : IO Bool
testPosOneDim2 = near "posEncVal 4 1 2 = sin 0.01" (posEncVal 4 1 2) (sin 0.01)

-- pos=1, dim=3: odd → cos; same angle 0.01.
testPosOneDim3 : IO Bool
testPosOneDim3 = near "posEncVal 4 1 3 = cos 0.01" (posEncVal 4 1 3) (cos 0.01)

----------------------------------------------------------------------
-- Bucket 2: materialized [seqLen, dModel] tensor matches posEncVal
----------------------------------------------------------------------

testTableMatches : IO Bool
testTableMatches = do
  pe <- the (IO (Tensor [3, 4] TestExecutor TestDType WithGrad))
            (sinusoidalPE {ex=TestExecutor} {dt=TestDType} {seqLen=3} {dModel=4})
  let at : Int -> Int -> Double
      at i j = primItem2d {ex=TestExecutor} pe.tensorPtr i j
      pairs  = the (List (Double, Double))
                  [ (at 0 0, 0.0), (at 0 1, 1.0)
                  , (at 1 0, sin 1.0), (at 1 2, sin 0.01), (at 1 3, cos 0.01) ]
      worst = foldl (\m, (g, w) => max m (abs (g - w))) 0.0 pairs
  if worst < tol
    then check "sinusoidalPE [3,4] table matches posEncVal" True
    else do
      putStrLn ("  FAIL: table drifted from posEncVal by " ++ show worst)
      pure False

----------------------------------------------------------------------
-- Bucket 3: the `L IO` twin composes + matches the IO table
----------------------------------------------------------------------

-- `sinusoidalPEL` built inside a `Control.Linear.LIO.run` block (no
-- `liftIO1` at the call site) yields the same [3,4] table as `sinusoidalPE`.
testTwinComposes : IO Bool
testTwinComposes = do
  pe <- Control.Linear.LIO.run
          (sinusoidalPEL {ex=TestExecutor} {dt=TestDType} {seqLen=3} {dModel=4}
            {g=WithGrad})
  let at : Int -> Int -> Double
      at i j = primItem2d {ex=TestExecutor} pe.tensorPtr i j
      pairs  = the (List (Double, Double))
                  [ (at 0 0, 0.0), (at 0 1, 1.0)
                  , (at 1 0, sin 1.0), (at 1 2, sin 0.01), (at 1 3, cos 0.01) ]
      worst = foldl (\m, (g, w) => max m (abs (g - w))) 0.0 pairs
  if worst < tol
    then check "sinusoidalPEL composes in L IO + matches posEncVal" True
    else do
      putStrLn ("  FAIL: L twin table drifted by " ++ show worst)
      pure False

export
tests : List (IO Bool)
tests =
  [ testPosZeroEven
  , testPosZeroOdd
  , testPosOneDim0
  , testPosOneDim2
  , testPosOneDim3
  , testTableMatches
  , testTwinComposes
  ]
