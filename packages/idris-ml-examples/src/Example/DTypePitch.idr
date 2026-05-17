||| Demo: type-system pitch for the (device, dtype) Compatible check.
|||
||| The MLX Metal GPU dropped float64 support in mlx 0.31. PyTorch
||| catches this with a runtime `RuntimeError: Float64 not supported
||| on Metal`. Our static `Compatible` capability interface lifts
||| that check to compile time: `Compatible (MlxDev MCpu) F64`
||| exists, but `Compatible (MlxDev MGpu) F64` does not — so
||| spelling `Tensor [..] MlxGpu F64 WithGrad` at a constructor
||| call site fails to typecheck.
|||
||| This file demonstrates the type-level check using a stub creator.
||| The "demoBadCase" definition at the bottom is COMMENTED OUT
||| because uncommenting it would block compilation; that is exactly
||| the demo. See `docs/develop/dtype-parameter.md` for the design.
module Example.DTypePitch

import Data.Vect

import Device
import Device.Mlx
import Tensor


-- A stub tensor creator that requires `Compatible d dt`. The actual
-- body doesn't matter for the typecheck demo (`believe_me` was
-- explicitly rejected by the codebase's no-unsafe-coercion policy,
-- so we leave the body as a hole that won't execute).
demoCreate : {0 d : Device} -> {0 dt : DType} -> Compatible d dt =>
             IO (Tensor [4] d dt WithGrad)
demoCreate = ?demoCreateImpl  -- not executable; type-checks if Compatible holds


-- POSITIVE CASES — these compile because the relevant Compatible
-- instances exist:

okCpuF64 : IO (Tensor [4] CPU F64 WithGrad)
okCpuF64 = demoCreate

okMlxCpuF64 : IO (Tensor [4] MlxCpu F64 WithGrad)
okMlxCpuF64 = demoCreate

okMlxCpuF32 : IO (Tensor [4] MlxCpu F32 WithGrad)
okMlxCpuF32 = demoCreate

okMlxGpuF32 : IO (Tensor [4] (MlxDev MGpu) F32 WithGrad)
okMlxGpuF32 = demoCreate


-- NEGATIVE CASE — uncomment to see the compile-time rejection:
--
--   When checking type of Example.DTypePitch.failMlxGpuF64:
--   Can't find an implementation for Compatible (MlxDev MGpu) F64
--
-- failMlxGpuF64 : IO (Tensor [4] (MlxDev MGpu) F64 WithGrad)
-- failMlxGpuF64 = demoCreate


-- Same demo on the lossless-upcast partial order. `F32 → F64` is
-- a lossless conversion (UpcastableTo F32 F64 exists); `F64 → F32`
-- is not (no instance, would be a narrowing cast).
demoUpcast : UpcastableTo from to => IO ()
demoUpcast = pure ()

okF32ToF64 : IO ()
okF32ToF64 = demoUpcast {from = F32} {to = F64}

-- Uncomment to see "Can't find an implementation for LTE 64 32":
-- failF64ToF32 : IO ()
-- failF64ToF32 = demoUpcast {from = F64} {to = F32}


main : IO ()
main = do
  putStrLn "=== DType pitch ==="
  putStrLn "If you see this, the type-system check passed for:"
  putStrLn "  * Tensor [4] CPU F64 WithGrad"
  putStrLn "  * Tensor [4] MlxCpu F64 WithGrad"
  putStrLn "  * Tensor [4] MlxCpu F32 WithGrad"
  putStrLn "  * Tensor [4] MlxGpu F32 WithGrad"
  putStrLn ""
  putStrLn "Try uncommenting `failMlxGpuF64` for the type rejection demo."
