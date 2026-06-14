-- | GradMode: type-level grad-tracking phantom for Tensor / Network.
-- |
-- | The runtime `withNoGrad` (Tensor.idr) gates tape construction via
-- | a C-side depth counter. `GradMode` is the static cousin: it lifts
-- | "this tensor is not being tracked for backward" into the type, so
-- | `runBackward` / `nativeTrainStep` can statically reject NoGrad
-- | inputs instead of silently no-opping.
-- |
-- | See `docs/grad-mode-and-device-typing.md` for the full design.

module GradMode

import Decidable.Equality

----------------------------------------------------------------------
-- GradMode Type
----------------------------------------------------------------------

public export
data GradMode = WithGrad | NoGrad

----------------------------------------------------------------------
-- Instances
----------------------------------------------------------------------

public export
Show GradMode where
  show WithGrad = "WithGrad"
  show NoGrad   = "NoGrad"

public export
Eq GradMode where
  WithGrad == WithGrad = True
  NoGrad == NoGrad     = True
  _ == _               = False

public export
DecEq GradMode where
  decEq WithGrad WithGrad = Yes Refl
  decEq NoGrad NoGrad     = Yes Refl
  decEq WithGrad NoGrad   = No (\case Refl impossible)
  decEq NoGrad WithGrad   = No (\case Refl impossible)

----------------------------------------------------------------------
-- Runtime witness for grad-mode-polymorphic construction
----------------------------------------------------------------------

||| Singleton linking a runtime `GradMode` value to its type-level index.
||| `g` is erased, so a grad-mode-polymorphic builder needs a runtime
||| witness to decide whether to `weakenGrad` its params to `NoGrad` at
||| construction (build the inference form directly, no post-construction
||| `eval` flip). Matching `SWithGrad`/`SNoGrad` also *refines* `g` in each
||| branch, so both builds type-check against `… g`.
public export
data SGrad : GradMode -> Type where
  SWithGrad : SGrad WithGrad
  SNoGrad   : SGrad NoGrad

||| `KnownGrad g` supplies the `SGrad g` witness as an auto-implicit, so a
||| caller writes `hfFooModel {g = NoGrad} …` and resolution finds it — the
||| inference-vs-training choice becomes a single type-application at the
||| construction site.
public export
interface KnownGrad (0 g : GradMode) where
  sgrad : SGrad g

public export
KnownGrad WithGrad where
  sgrad = SWithGrad

public export
KnownGrad NoGrad where
  sgrad = SNoGrad
