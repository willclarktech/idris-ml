||| Preset — per-(Backend, Hardware) default Executor + dtype.
|||
||| `Preset b h` declares: for primary Backend `b` on Hardware `h`, the
||| Executor type and DType that example code should target. Phase B's
||| Makefile sed populates `BuildConfig.idr` with `PrimaryBackend`,
||| `ChosenHardware`, and `ChosenMachine`; this typeclass then resolves
||| `ExampleExecutor` / `ExampleDType` at example-compile time.
|||
||| Open by extension: users override built-in presets (e.g. BF16 on
||| AppleGpu for mlx instead of F32) by writing their own `Preset`
||| instance in a module imported ahead of the built-ins in
||| `BuildConfig.idr`.
|||
||| Compile-time errors when no instance is found map to clear failures:
||| picking `tape × AppleGpu` errors with "no instance Preset
||| TapeBackend AppleGpu" since the tape backend has no Apple GPU
||| support.
module Preset

import BackendLib
import Hardware

public export
interface Preset (0 b : BackendLib) (0 h : Hardware) where
  presetExecutor : Type
  presetDType    : Type
