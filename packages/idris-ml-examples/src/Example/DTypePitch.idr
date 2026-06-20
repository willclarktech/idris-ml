||| Demo: the three compile-time dtype/device capability gates.
|||
||| idris-ml has three orthogonal compile-time capabilities, each an empty
||| marker interface with curated instances:
|||
|||   * `Compatible (0 ex : Executor) (0 t : DType)` — dtype admissibility.
|||     Some hardware can't represent some dtypes: the MLX Metal GPU
|||     dropped float64 in mlx 0.31; libtorch rejects float64 at MPS
|||     construction. There is no `Compatible (MlxExecutor MGpu) F64` /
|||     `Compatible (TorchExecutor TMps) F64` instance. Build-independent —
|||     every backend's Compatible instances are always in scope.
|||
|||   * `Linked (0 ex : Executor)` — backend linkage. Only backends compiled
|||     into this `libidrisml` (the `BACKEND` list) get a `Linked`
|||     instance, emitted by the generated `HwConfig`. A tape-only build
|||     has no `Linked (MlxExecutor _)`, so naming an mlx device at a
|||     constructor fails to typecheck. Build-dependent.
|||
|||   * `IsFloating (0 t : DType)` / `IsIntegral (0 t : DType)` — op-level
|||     dtype-kind gates. Restriction at the *operation*, not just the
|||     backend: a loss / gradient / softmax is real-valued, so the loss
|||     fns and `runBackward`/`trainStep` carry `IsFloating dt =>`.
|||     `Bool` is neither floating nor integral. Build-independent.
|||
||| Real tensor constructors carry BOTH `Compatible ex dt =>` and
||| `Linked ex =>`, so this file isolates each axis with a witness that
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

import BuildConfig
import Executor
import Tensor

-- Axis 1: Compatible (dtype admissibility). Build-independent — these
-- need only the Compatible instance, which exists for every backend.

compatOK : Compatible ex dt => ()
compatOK = ()

okTapeF64    : ()
okTapeF64    = compatOK {ex=TapeExecutor}        {dt = F64}
okTorchCpuF64 : ()
okTorchCpuF64 = compatOK {ex=TorchExecutor TCpu}  {dt = F64}
okTorchCpuF32 : ()
okTorchCpuF32 = compatOK {ex=TorchExecutor TCpu}  {dt = F32}
okTorchMpsF32 : ()
okTorchMpsF32 = compatOK {ex=TorchExecutor TMps}  {dt = F32}
okMlxCpuF64  : ()
okMlxCpuF64  = compatOK {ex=MlxExecutor MCpu}     {dt = F64}
okMlxCpuF32  : ()
okMlxCpuF32  = compatOK {ex=MlxExecutor MCpu}     {dt = F32}
okMlxGpuF32  : ()
okMlxGpuF32  = compatOK {ex=MlxExecutor MGpu}     {dt = F32}

-- Uncomment either: "Can't find an implementation for Compatible ... F64".
-- badMlxGpuF64   : () ; badMlxGpuF64   = compatOK {ex=MlxExecutor MGpu}    {dt = F64}
-- badTorchMpsF64 : () ; badTorchMpsF64 = compatOK {ex=TorchExecutor TMps}  {dt = F64}

-- Axis 2: Linked (backend linkage). Build-DEPENDENT — only the
-- compiled-in backends have a Linked instance. ExampleExecutor is always
-- linked (it's this build's device).

linkedOK : Linked ex => ()
linkedOK = ()

linkedExample : ()
linkedExample = linkedOK {ex=ExampleExecutor}

-- Uncomment on a build whose BACKEND omits that backend and it fails
-- with "Can't find an implementation for Linked ...":
-- linkedMlxGpu  : () ; linkedMlxGpu  = linkedOK {ex=MlxExecutor MGpu}
-- linkedTorchCpu : () ; linkedTorchCpu = linkedOK {ex=TorchExecutor TCpu}

-- Lossless-upcast partial order: `F32 → F64` is lossless, `F64 → F32`
-- is not.
demoUpcast : UpcastableTo from to => ()
demoUpcast = ()

okF32ToF64 : ()
okF32ToF64 = demoUpcast {from = F32} {to = F64}
-- failF64ToF32 : () ; failF64ToF32 = demoUpcast {from = F64} {to = F32}  -- LTE 64 32

-- Axis 3: op-level dtype-kind gates. Beyond *which dtypes a backend
-- admits* (Compatible), some *operations* only make sense for a dtype
-- kind: a gradient / loss / softmax is real-valued (`IsFloating`), an
-- index is integral (`IsIntegral`). `Bool` is neither. These are
-- build-independent (no device, no construction) — they constrain the
-- dtype tag itself.

floatingOK : IsFloating dt => ()
floatingOK = ()

integralOK : IsIntegral dt => ()
integralOK = ()

okFloatF32  : () ; okFloatF32 = floatingOK {dt = F32}
okFloatF64  : () ; okFloatF64 = floatingOK {dt = F64}
okFloatBF16 : () ; okFloatBF16 = floatingOK {dt = BF16}
okIntI32    : () ; okIntI32 = integralOK {dt = I32}
okIntU8     : () ; okIntU8  = integralOK {dt = U8}

-- Uncomment any: "Can't find an implementation for IsFloating/IsIntegral …".
-- This is the same rejection the gated ops enforce — the loss fns
-- (`tnllLoss`/`tbceLoss`/`tmseLoss`) and the gradient surface
-- (`runBackward`/`trainStep`) carry `IsFloating dt =>`, so a loss
-- on, or backprop through, a `Bool`/`Int` tensor is a compile error.
-- badFloatI32  : () ; badFloatI32  = floatingOK {dt = I32}   -- no IsFloating (IntN 32)
-- badFloatBool : () ; badFloatBool = floatingOK {dt = Bool}  -- no IsFloating Bool
-- badIntF32    : () ; badIntF32    = integralOK {dt = F32}   -- no IsIntegral (Float 32)

main : IO ()
main = do
  requireMachine {m = ChosenMachine}
  putStrLn "=== device capability gates ==="
  putStrLn "Compatible (dtype) admits, on every build:"
  putStrLn "  TapeExecutor F64 | TorchExecutor {TCpu F64/F32, TMps F32} | MlxExecutor {MCpu F64/F32, MGpu F32}"
  putStrLn "  rejected: MlxExecutor MGpu F64 | TorchExecutor TMps F64"
  putStrLn "Linked (linkage) admits only this build's BACKEND."
  putStrLn "IsFloating/IsIntegral (op kind) gate ops on the dtype tag:"
  putStrLn "  floating: F32/F64/BF16/F16 | integral: I8/I16/I32/I64/U8 | Bool: neither"
  putStrLn "  rejected: a loss / backprop on a Bool or Int tensor"
  putStrLn ""
  -- Real construction satisfies BOTH Compatible and Linked.
  v <- tconstScalar {ex=ExampleExecutor} {dt = ExampleDType} 42.0
  putStrLn $ "Constructed Tensor [] on " ++ deviceName {ex=ExampleExecutor}
               ++ " " ++ dtypeName {t = ExampleDType}
               ++ " holding " ++ show (tensorItem v)
  putStrLn $ "RESULT\tgate=ok\tdevice=" ++ deviceName {ex=ExampleExecutor}
               ++ "\tdtype=" ++ dtypeName {t = ExampleDType}
