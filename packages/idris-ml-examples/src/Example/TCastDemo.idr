||| Runtime smoke test for `tcast` and `tcastUnsafe`.
|||
||| Verifies the per-dtype FFI plumbing (`tensor_cast_dtype_<dt>`
||| primitives across all backends) by round-tripping a tensor
||| through the safe and unsafe cast surfaces at the build's
||| ExampleDType. Same-dtype casts are the only case all three
||| backends share today (tape has F64 only; mlx-gpu has F32 only);
||| the actual cross-dtype path on mlx/torch is exercised
||| separately when the build runs in mixed-dtype configurations.
|||
||| Expected output: original and cast tensors match to fp ULP.
module Example.TCastDemo

import Data.Vect
import System

import Array
import Executor
import Tensor
import BuildConfig


-- A fixed test vector. Same shape across builds.
testValues : Vector 4 Double
testValues = VArray [SArray 1.5, SArray (-2.7), SArray 3.14159, SArray 0.0]


-- Read all 4 elements out of a 1D tensor into a Vect.
readBack : TVec 4 ExampleExecutor ExampleDType WithGrad -> Vect 4 Double
readBack v = [ primItem1d {ex=ExampleExecutor} v.tensorPtr 0
             , primItem1d {ex=ExampleExecutor} v.tensorPtr 1
             , primItem1d {ex=ExampleExecutor} v.tensorPtr 2
             , primItem1d {ex=ExampleExecutor} v.tensorPtr 3
             ]


showVec : Vect 4 Double -> String
showVec [a, b, c, d] = "[" ++ show a ++ ", " ++ show b ++ ", "
                         ++ show c ++ ", " ++ show d ++ "]"


main : IO ()
main = do
  requireMachine {m = ChosenMachine}
  putStrLn $ "=== tcast smoke test [" ++ backendName {ex=ExampleExecutor} ++ "] ==="
  let srcPtr = bulkToTensor {ex=ExampleExecutor} {dt=ExampleDType} testValues
      src    = the (TVec 4 ExampleExecutor ExampleDType WithGrad) (MkTensor srcPtr Nothing)
  let origVals = readBack src
  putStrLn $ "original     : " ++ showVec origVals

  -- Safe cast (lossless): `UpcastableTo from to` required. The
  -- same-dtype case is satisfied via the `upcastLteRefl` %hint in
  -- `DType/Core.idr`, which bypasses Idris-2's auto-search depth
  -- cap for `LTE n n` at large n (the F64 case `LTE 64 64` would
  -- otherwise fall off the default ~50 search ceiling).
  safe <- tcast src
  let safeVals = readBack safe
  putStrLn $ "tcast        : " ++ showVec safeVals

  -- Unsafe cast: explicit target dtype, no `UpcastableTo` constraint.
  unsafe <- tcastUnsafe ExampleDType src
  let unsafeVals = readBack unsafe
  putStrLn $ "tcastUnsafe  : " ++ showVec unsafeVals

  let allMatch = origVals == safeVals && origVals == unsafeVals
  if allMatch
    then putStrLn "PASS"
    else do putStrLn "FAIL: values diverged"
            exitFailure
