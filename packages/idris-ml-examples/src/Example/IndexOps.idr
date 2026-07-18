||| Type-safe integral index API demo (torch-only).
|||
||| Exercises the typed `targsort` / `tgather` / `tscatterAdd` surface end to
||| end: argsort returns an `I64` index tensor, which then drives a gather and
||| a scatter-add. Readouts use an order-sensitive positional dot (multiply by
||| `[1,2,3,4]`, then sum) so the check actually pins the *ordering* and the
||| scatter placement, not just the resulting multiset.
|||
||| These ops are `Compatible` only where an integer dtype exists (`I64`),
||| i.e. `TorchExecutor TCpu` / `TCuda` — Metal has no F64/int and tape/mlx store
||| F64 only. So this builds under `BACKEND=torch TORCH_DEVICE=cpu` only (see
||| the `example-index-ops` Makefile target). It is deliberately NOT listed in
||| `idris-ml-examples.ipkg`: that package builds on every backend, and this
||| module calls `targsort` (needs `Compatible ExampleExecutor I64`), which has
||| no instance on tape/mlx. The Makefile target compiles it standalone.
module Example.IndexOps

import Data.Vect
import System

import BuildConfig
import Ml.Array
import Ml.DType.Core
import Ml.Executor
import Ml.Tensor

-- src to sort/index. Distinct values so the permutation is unambiguous.
srcVals : Vector 4 Double
srcVals = VArray [SArray 3.0, SArray 1.0, SArray 4.0, SArray 1.5]

-- Positional weights for the order-sensitive readout dot.
weightVals : Vector 4 Double
weightVals = VArray [SArray 1.0, SArray 2.0, SArray 3.0, SArray 4.0]

||| Build a [4] NoGrad tensor at the example dtype from a 4-vector.
mkVec : Vector 4 Double -> TVec 4 ExampleExecutor ExampleDType NoGrad
mkVec vals = MkTensor (bulkToTensor {ex=ExampleExecutor} {dt=ExampleDType} vals) Nothing

||| Reduce a [4] vector to its scalar sum and read it out.
scalarSum : TVec 4 ExampleExecutor ExampleDType NoGrad -> Double
scalarSum v =
  tensorItem {ex=ExampleExecutor}
    (the (Tensor [] ExampleExecutor ExampleDType NoGrad)
         (MkTensor (primSum {ex=ExampleExecutor} v.tensorPtr) Nothing))

||| Order-sensitive scalar readout: sum(v * weights).
dotWeights : TVec 4 ExampleExecutor ExampleDType NoGrad -> IO Double
dotWeights v = do
  prod <- tmul v (mkVec weightVals)
  pure (scalarSum prod)

approxEq : Double -> Double -> Bool
approxEq a b = abs (a - b) < 1.0e-9

main : IO ()
main = do
  requireMachine {m = ChosenMachine}
  putStrLn $ "=== index ops [" ++ backendName {ex=ExampleExecutor} ++ "] ==="
  let src = mkVec srcVals

  -- argsort ascending: [3,1,4,1.5] -> indices [1,3,0,2]
  idx <- targsort {ex=ExampleExecutor} 0 False src

  -- gather: sorted ascending [1.0, 1.5, 3.0, 4.0]; dot [1,2,3,4] = 29.0
  sorted    <- tgather src idx
  sortedDot <- dotWeights sorted

  -- scatter-add src at the argsort indices into a fresh [4] zero vector:
  -- out[1]+=3, out[3]+=1, out[0]+=4, out[2]+=1.5 -> [4, 3, 1.5, 1];
  -- dot [1,2,3,4] = 18.5
  scattered    <- tscatterAdd 4 idx src
  scatteredDot <- dotWeights scattered

  putStrLn $ "gather sorted-dot    = " ++ show sortedDot ++ " (expect 29.0)"
  putStrLn $ "scatter placement-dot = " ++ show scatteredDot ++ " (expect 18.5)"

  if approxEq sortedDot 29.0 && approxEq scatteredDot 18.5
    then putStrLn "PASS: typed argsort/gather/scatterAdd round-trip"
    else do putStrLn "FAIL: index-op readout mismatch"
            exitFailure
