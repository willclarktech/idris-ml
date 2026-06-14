||| Backend — the user-facing constraint bundle.
|||
||| One constraint instead of four. The canonical exported signature
||| used to lead with
|||
|||   {0 ex : Executor} -> UserExecutorTraining ex => RuntimeDType dt =>
|||   Linked ex => Compatible ex dt => ...
|||
||| (~92 stacks across the library; api-critique.md §S9). `Backend ex dt`
||| bundles those four leaves; signatures read `Backend ex dt => ...` and
||| bodies still resolve every leaf method through the superclass chain
||| (`UserExecutorCore` arrives via `UserExecutorTraining`'s aggregate).
|||
||| The blanket implementation below means any `(ex, dt)` pair whose four
||| leaf instances exist gets `Backend ex dt` for free — including
||| bring-your-own backends (declare the same leaf instances as today, per
||| `Example/BringYourOwn.idr`; the bundle costs you nothing). `Linked`
||| stays a leaf so the per-build availability gating (HwConfig.idr) keeps
||| working: a tape-only build cannot resolve `Backend (MlxExecutor s) dt`
||| because `Linked (MlxExecutor s)` is not generated.
|||
||| `IsFloating dt` is deliberately NOT in the bundle — it gates the
||| loss/training axis (a handful of sites), not backend admissibility,
||| and bundling it would wrongly exclude integer dtypes everywhere.
|||
||| Tier names `BackendCore` / `BackendInference` / `BackendStreamed` are
||| reserved for when thinner aggregates earn a population (today
||| Core-only sites carry no dtype constraints, so an `(ex, dt)` bundle
||| has nothing to offer them).
module Backend

import Executor.Core
import DType.Core

public export
interface (UserExecutorTraining ex,
           RuntimeDType dt,
           Linked ex,
           Compatible ex dt) =>
          Backend (0 ex : Executor) (0 dt : DType) where

||| Blanket implementation: the four leaves imply the bundle.
||| (Precedent: `LosslessTo from to => UpcastableTo from to` in
||| DType.Core.)
public export
(UserExecutorTraining ex, RuntimeDType dt, Linked ex, Compatible ex dt) => Backend ex dt where

||| Assemble a Backend dictionary from explicitly-chosen dtype-side
||| leaves. For bridge code (e.g. the AsMixed LayerLike adapter) whose
||| scope carries TWO candidate (RuntimeDType, Compatible) dict pairs:
||| plain auto-search finds multiple solutions there and idris2
||| rejects the ambiguity; this pins one side. Inside this body the
||| explicit args are the only candidates, so the blanket resolves
||| uniquely.
export
backendFrom : UserExecutorTraining ex => Linked ex =>
              (rdt : RuntimeDType dt) -> (cmp : Compatible ex dt) ->
              Backend ex dt
backendFrom rdt cmp = %search
