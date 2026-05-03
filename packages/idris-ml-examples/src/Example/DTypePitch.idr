||| Demo: the two compile-time device capability gates.
|||
||| idris-ml has two orthogonal compile-time capabilities, each an empty
||| marker interface with curated instances:
|||
|||   * `Compatible (0 d : Device) (0 t : DType)` — dtype admissibility.
|||     Some hardware can't represent some dtypes: the MLX Metal GPU
|||     dropped float64 in mlx 0.31; libtorch rejects float64 at MPS
|||     construction. There is no `Compatible (MlxDev MGpu) F64` /
|||     `Compatible (TorchDev TMps) F64` instance. Build-independent —
|||     every backend's Compatible instances are always in scope.
|||
|||   * `Linked (0 d : Device)` — backend linkage. Only backends compiled
|||     into this `libidrisml` (the `BACKEND` list) get a `Linked`
|||     instance, emitted by the generated `HwConfig`. A tape-only build
|||     has no `Linked (MlxDev _)`, so naming an mlx device at a
|||     constructor fails to typecheck. Build-dependent.
|||
||| Real tensor constructors carry BOTH `Compatible d dt =>` and
||| `Linked d =>`, so this file isolates each axis with a witness that
||| requires only that one capability (no construction, so the dtype
||| axis stays build-independent). `main` then does a real construction
||| on the build-selected cell, which satisfies both.
|||
||| The `bad*` / non-linked lines are commented out: uncommenting one is
||| a compile error, and that rejection is the demo.
|||
||| See `docs/develop/dtype-parameter.md` and
||| `docs/develop/device-availability-gating.md`.
module Example.DTypePitch

import Data.Vect

import Device
import Tensor
import BuildConfig


-- Axis 1: Compatible (dtype admissibility). Build-independent — these
-- need only the Compatible instance, which exists for every backend.

compatOK : Compatible d dt => ()
compatOK = ()

okTapeF64    : ()
okTapeF64    = compatOK {d = TapeDev}        {dt = F64}
okTorchCpuF64 : ()
okTorchCpuF64 = compatOK {d = TorchDev TCpu}  {dt = F64}
okTorchCpuF32 : ()
okTorchCpuF32 = compatOK {d = TorchDev TCpu}  {dt = F32}
okTorchMpsF32 : ()
okTorchMpsF32 = compatOK {d = TorchDev TMps}  {dt = F32}
okMlxCpuF64  : ()
okMlxCpuF64  = compatOK {d = MlxDev MCpu}     {dt = F64}
okMlxCpuF32  : ()
okMlxCpuF32  = compatOK {d = MlxDev MCpu}     {dt = F32}
okMlxGpuF32  : ()
okMlxGpuF32  = compatOK {d = MlxDev MGpu}     {dt = F32}

-- Uncomment either: "Can't find an implementation for Compatible ... F64".
-- badMlxGpuF64   : () ; badMlxGpuF64   = compatOK {d = MlxDev MGpu}    {dt = F64}
-- badTorchMpsF64 : () ; badTorchMpsF64 = compatOK {d = TorchDev TMps}  {dt = F64}


-- Axis 2: Linked (backend linkage). Build-DEPENDENT — only the
-- compiled-in backends have a Linked instance. ExampleDevice is always
-- linked (it's this build's device).

linkedOK : Linked d => ()
linkedOK = ()

linkedExample : ()
linkedExample = linkedOK {d = ExampleDevice}

-- Uncomment on a build whose BACKEND omits that backend and it fails
-- with "Can't find an implementation for Linked ...":
-- linkedMlxGpu  : () ; linkedMlxGpu  = linkedOK {d = MlxDev MGpu}
-- linkedTorchCpu : () ; linkedTorchCpu = linkedOK {d = TorchDev TCpu}


-- Lossless-upcast partial order: `F32 → F64` is lossless, `F64 → F32`
-- is not.
demoUpcast : UpcastableTo from to => ()
demoUpcast = ()

okF32ToF64 : ()
okF32ToF64 = demoUpcast {from = F32} {to = F64}
-- failF64ToF32 : () ; failF64ToF32 = demoUpcast {from = F64} {to = F32}  -- LTE 64 32


main : IO ()
main = do
  putStrLn "=== device capability gates ==="
  putStrLn "Compatible (dtype) admits, on every build:"
  putStrLn "  TapeDev F64 | TorchDev {TCpu F64/F32, TMps F32} | MlxDev {MCpu F64/F32, MGpu F32}"
  putStrLn "  rejected: MlxDev MGpu F64 | TorchDev TMps F64"
  putStrLn "Linked (linkage) admits only this build's BACKEND."
  putStrLn ""
  -- Real construction satisfies BOTH Compatible and Linked.
  v <- tconstScalar {d = ExampleDevice} {dt = ExampleDType} 42.0
  putStrLn $ "Constructed Tensor [] on " ++ deviceName {d = ExampleDevice}
               ++ " " ++ dtypeName {t = ExampleDType}
               ++ " holding " ++ show (tensorItem v)
  putStrLn $ "RESULT\tgate=ok\tdevice=" ++ deviceName {d = ExampleDevice}
               ++ "\tdtype=" ++ dtypeName {t = ExampleDType}
