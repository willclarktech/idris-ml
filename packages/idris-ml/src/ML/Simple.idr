||| `ML.Simple` — `ML` plus the build's default `(executor, dtype)` cell
||| pinned as `Ex` / `F`. Tutorial and example code writes `Tensor dims Ex F
||| g` (and constructs via the smart constructors with the result type
||| annotated to `Ex`/`F`), so Idris infers `{ex=Ex}{dt=F}` everywhere and no
||| `{ex=}` is ever spelled. The pin comes from the generated `ML.Config`
||| (one cell per build, same mechanism as the examples' `BuildConfig`).
module ML.Simple

import public ML
import public ML.Config

||| The build's default executor (e.g. `TapeExecutor`, `TorchExecutor TMps`).
public export
Ex : Type
Ex = DefaultExecutor

||| The build's default dtype (e.g. `F64` on tape/CPU, `F32` on Metal).
public export
F : Type
F = DefaultDType
