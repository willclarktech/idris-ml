||| Cross-language dtype serialization demo (torch-only).
|||
||| Constructs tensors at the inference dtypes (bf16, f16, i32), registers
||| them under names via `registerParam`, and writes a `.safetensors` file.
||| The companion verifier `scripts/verify_dtypes.py` loads the file with
||| `safetensors.torch.load_file()` and asserts the on-disk dtypes + values
||| — independently confirming our writer's byte layout matches the
||| SafeTensors spec / HuggingFace reader.
|||
||| bf16/f16/int are `Compatible` only on torch, so this builds under
||| `BACKEND=torch` only (see the `example-dtype-serialize` Makefile target).
||| It is deliberately NOT listed in `idris-ml-examples.ipkg`: that package
||| is built on every backend, and this module constructs `BF16`/`F16`/`I32`
||| in `main` (no `Compatible TapeExecutor (BFloat 16)` etc.). The Makefile target
||| compiles it standalone instead.
module Example.DTypeSerialize

import Data.Vect
import System

import Array
import BuildConfig
import Checkpoint
import DType.Core
import Executor
import Tensor

-- Values chosen to be exactly representable in bf16/f16 (binary fractions
-- and powers of two) and in i32, so the cross-language check is exact.
floatVals : Vector 4 Double
floatVals = VArray [SArray 1.5, SArray (-2.0), SArray 256.0, SArray (-0.5)]

intVals : Vector 4 Double
intVals = VArray [SArray 1.0, SArray (-2.0), SArray 1000.0, SArray (-42.0)]

||| Build a NoGrad tensor at dtype `dt` from a 4-vector and register it.
saveOne : RuntimeDType dt => Compatible ExampleExecutor dt =>
          String -> Vector 4 Double -> IO ()
saveOne name vals = do
  _ <- registerParam {ex=ExampleExecutor} name
         (the (TVec 4 ExampleExecutor dt NoGrad)
              (MkTensor (bulkToTensor {ex=ExampleExecutor} {dt} vals) Nothing))
  pure ()

main : IO ()
main = do
  requireMachine {m = ChosenMachine}
  args <- getArgs
  let path = case args of
               (_ :: p :: _) => p
               _             => "/tmp/idrisml-dtypes.safetensors"
  putStrLn $ "=== dtype serialize [" ++ backendName {ex=ExampleExecutor} ++ "] -> " ++ path ++ " ==="

  saveOne {dt=BF16} "w_bf16" floatVals
  saveOne {dt=F16}  "w_f16"  floatVals
  saveOne {dt=I32}  "w_i32"  intVals

  ok <- (== Right ()) <$> saveAll {ex=ExampleExecutor} path
  if ok
    then putStrLn "PASS: wrote bf16/f16/i32 tensors"
    else do putStrLn "FAIL: saveAll returned an error"
            exitFailure
