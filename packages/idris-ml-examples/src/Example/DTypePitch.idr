||| Demo: the (device, dtype) `Compatible` capability as a compile-time gate.
|||
||| Some hardware genuinely can't represent some dtypes: the MLX Metal
||| GPU dropped float64 in mlx 0.31, and libtorch rejects float64 at
||| MPS tensor *construction*. PyTorch surfaces this as a runtime
||| `RuntimeError`/abort. idris-ml lifts it to compile time: the
||| `Compatible (0 d : Device) (0 t : DType)` capability has instances
||| only for representable pairs, and every tensor constructor carries
||| a `Compatible d dt =>` constraint. Spelling an unrepresentable pair
||| at a constructor call site fails to typecheck.
|||
||| The `ok*` definitions below call the REAL constructor
||| (`tconstScalar`) — they are compile-time witnesses that the gate
||| admits each representable cell. They type-check; only the
||| build-selected cell (`ExampleDevice`/`ExampleDType`) is actually
||| executed in `main`, so the demo runs on whatever backend is linked.
|||
||| The `bad*` definitions are commented out: uncommenting either is a
||| compile error, because the matching `Compatible` instance
||| deliberately does not exist. That rejection IS the demo.
|||
||| See `docs/develop/dtype-parameter.md` for the design.
module Example.DTypePitch

import Data.Vect

import Device
import Tensor
import BuildConfig


-- ALLOWED — each typechecks because the matching Compatible instance
-- exists. Real constructor, not a stub.

okTapeF64 : IO (Tensor [] TapeDev F64 WithGrad)
okTapeF64 = tconstScalar {d = TapeDev} {dt = F64} 0.0

okTorchCpuF64 : IO (Tensor [] (TorchDev TCpu) F64 WithGrad)
okTorchCpuF64 = tconstScalar {d = TorchDev TCpu} {dt = F64} 0.0

okTorchCpuF32 : IO (Tensor [] (TorchDev TCpu) F32 WithGrad)
okTorchCpuF32 = tconstScalar {d = TorchDev TCpu} {dt = F32} 0.0

okTorchMpsF32 : IO (Tensor [] (TorchDev TMps) F32 WithGrad)
okTorchMpsF32 = tconstScalar {d = TorchDev TMps} {dt = F32} 0.0

okMlxCpuF64 : IO (Tensor [] (MlxDev MCpu) F64 WithGrad)
okMlxCpuF64 = tconstScalar {d = MlxDev MCpu} {dt = F64} 0.0

okMlxCpuF32 : IO (Tensor [] (MlxDev MCpu) F32 WithGrad)
okMlxCpuF32 = tconstScalar {d = MlxDev MCpu} {dt = F32} 0.0

okMlxGpuF32 : IO (Tensor [] (MlxDev MGpu) F32 WithGrad)
okMlxGpuF32 = tconstScalar {d = MlxDev MGpu} {dt = F32} 0.0


-- DISALLOWED — uncomment either line and the build fails with exactly
-- the quoted error. No `Compatible` instance exists for these cells.
--
--   Can't find an implementation for Compatible (MlxDev MGpu) (Float 64).
-- badMlxGpuF64 : IO (Tensor [] (MlxDev MGpu) F64 WithGrad)
-- badMlxGpuF64 = tconstScalar {d = MlxDev MGpu} {dt = F64} 0.0
--
--   Can't find an implementation for Compatible (TorchDev TMps) (Float 64).
-- badTorchMpsF64 : IO (Tensor [] (TorchDev TMps) F64 WithGrad)
-- badTorchMpsF64 = tconstScalar {d = TorchDev TMps} {dt = F64} 0.0


-- Same idea on the lossless-upcast partial order: `F32 → F64` has an
-- `UpcastableTo` instance (lossless), `F64 → F32` does not (narrowing).
demoUpcast : UpcastableTo from to => IO ()
demoUpcast = pure ()

okF32ToF64 : IO ()
okF32ToF64 = demoUpcast {from = F32} {to = F64}

--   Can't find an implementation for LTE 64 32.
-- failF64ToF32 : IO ()
-- failF64ToF32 = demoUpcast {from = F64} {to = F32}


main : IO ()
main = do
  putStrLn "=== Compatible (device, dtype) gate ==="
  putStrLn "These cells typecheck against the real constructor:"
  putStrLn "  TapeDev F64 | TorchDev {TCpu F64, TCpu F32, TMps F32}"
  putStrLn "  MlxDev {MCpu F64, MCpu F32, MGpu F32}"
  putStrLn "Deliberately rejected (no instance):"
  putStrLn "  MlxDev MGpu F64 | TorchDev TMps F64"
  putStrLn ""
  -- Actually construct on the build-selected cell to prove the allowed
  -- path runs end-to-end on the linked backend.
  v <- tconstScalar {d = ExampleDevice} {dt = ExampleDType} 42.0
  let x = tensorItem v
  putStrLn $ "Constructed Tensor [] on " ++ deviceName {d = ExampleDevice}
               ++ " " ++ dtypeName {t = ExampleDType}
               ++ " holding " ++ show x
  putStrLn $ "RESULT\tgate=ok\tdevice=" ++ deviceName {d = ExampleDevice}
               ++ "\tdtype=" ++ dtypeName {t = ExampleDType}
