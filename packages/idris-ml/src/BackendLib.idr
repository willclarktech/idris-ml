||| BackendLib — open kind of backend-library names.
|||
||| Parallel to `Executor = Type` and `Hardware = Type`: an open kind
||| whose inhabitants are singleton types naming a backend library
||| (tape, libtorch, mlx, …). Used by the `Preset` typeclass (`Preset.idr`)
||| to dispatch per-(primary-backend, hardware) example defaults.
|||
||| Distinct from the value-level `backendTag : Int` method on
||| `UserExecutorTransfer` — that one is the runtime intra-vs-cross-
||| backend discriminator used by `toExecutor` for fast handle migration.
||| This kind is the type-level dispatch surface used by build-time
||| presets and any future polymorphic-over-backend code. The two
||| coexist (different layers, different uses).
|||
||| Open by extension: a custom backend declares its own `BackendLib` tag
||| (`data Rocm : Type where MkRocm : Rocm`) plus a `RunsVia` instance
||| linking its Executor type(s) to that tag in its own module.
module BackendLib

import Executor.Core

public export 0 BackendLib : Type
BackendLib = Type

||| The tape backend — pure-C autograd-on-arena, CPU only.
public export data TapeBackend : Type where MkTapeBackend : TapeBackend

||| The libtorch backend — wraps PyTorch's C++ library.
public export data TorchBackend : Type where MkTorchBackend : TorchBackend

||| The mlx backend — Apple's array library (CPU + Metal streams).
public export data MlxBackend : Type where MkMlxBackend : MlxBackend

||| An Executor is provided by a Backend. Membership claim only — no
||| methods. Each backend's `Executor/<Name>.idr` declares the instance
||| linking its `Executor` type(s) to its `BackendLib` tag.
public export
interface RunsVia (0 ex : Executor) (0 b : BackendLib) where
