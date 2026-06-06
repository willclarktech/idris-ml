||| Hardware — open kind of compute hardware classes.
|||
||| Parallel to `Executor = Type`: an open kind whose inhabitants are
||| singleton types representing the physical hardware an Executor runs
||| on. A Machine (separate kind in `Machine.idr`) provides one or more
||| Hardware classes; each (Executor × Hardware) pair lives in the
||| `RunsOn` membership typeclass.
|||
||| Distinct from `HardwareClass` in `Executor.Core` — that one is a
||| *runtime* classification value used by `someExecutor` discovery for
||| grouping/reporting. This kind is the *type-level* dispatch surface
||| used by the `Preset` lookup and any future
||| polymorphic-over-Metal-or-CUDA code. The two coexist by accident of
||| naming (both have an "AppleGpu" inhabitant) — Idris disambiguates by
||| context since one is a Type and one is a HardwareClass constructor.
|||
||| Open by extension: a custom backend declares its own `Hardware` tag
||| (e.g. `data Rocm : Type where MkRocm : Rocm`) plus `RunsOn` instances
||| in its own module.
module Hardware

import Executor.Core

public export 0 Hardware : Type
Hardware = Type

||| Host CPU — any ISA. The default for tape and the CPU streams of
||| torch / mlx.
public export data Cpu : Type where MkCpu : Cpu

||| Apple Silicon GPU. Reachable via Metal API surface; covers both
||| `TorchExecutor TMps` and `MlxExecutor MGpu`.
public export data AppleGpu : Type where MkAppleGpu : AppleGpu

||| NVIDIA CUDA device indexed by ordinal. `Cuda 0` is the first GPU,
||| `Cuda 1` the second, etc. Multi-GPU enumeration is the
||| High-Priority CUDA TODO row's job.
public export data Cuda : Nat -> Type where MkCuda : (idx : Nat) -> Cuda idx

||| An Executor runs on a Hardware class. Membership claim only — no
||| methods. Each backend declares which of its hardware variants map
||| to which Hardware kind.
public export
interface RunsOn (0 d : Executor) (0 h : Hardware) where
