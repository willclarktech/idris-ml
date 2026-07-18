module Test.Backend

import Data.Vect

import Ml.Executor
import Ml.Tensor
import Test.Config
import Test.Harness

-- The positive probe for the `Backend ex dt` bundle: a helper generic
-- in (ex, dt) and constrained ONLY by the bundle. Its body needs, via
-- superclass projection: UserExecutorTraining + RuntimeDType + Linked +
-- Compatible (dtCreateScalar), UserExecutorStreamed (deviceStreamTag,
-- through the Training aggregate), and UserExecutorCore (tadd /
-- tmulScalar / tensorItem, through the Training -> Conv -> NN ->
-- Linear -> Core chain). The call site below resolves the bundle from
-- the concrete TestExecutor/TestDType leaf instances via the blanket
-- implementation — both directions of the bundle exercised in one test.

mkScalar : {0 ex : Executor} -> {0 dt : DType} -> Backend ex dt =>
           Double -> IO (Tensor [] ex dt NoGrad)
mkScalar v = ioRerun (\_ =>
  MkTensor (dtCreateScalar {ex} {t=dt} v 0 (deviceStreamTag {ex})) Nothing)

fiveTimesSum : {0 ex : Executor} -> {0 dt : DType} -> Backend ex dt =>
               Double -> Double -> IO Double
fiveTimesSum x y = do
  a <- mkScalar {ex} {dt} x
  b <- mkScalar {ex} {dt} y
  s <- tadd a b
  m <- tmulScalar s 5.0
  pure (tensorItem m)

bundleArithmetic : IO Bool
bundleArithmetic = do
  r <- fiveTimesSum {ex=TestExecutor} {dt=TestDType} 2.0 3.0
  check "Backend bundle: (2 + 3) * 5 via bundle-only helper" (r == 25.0)

export
tests : List (IO Bool)
tests = [bundleArithmetic]
