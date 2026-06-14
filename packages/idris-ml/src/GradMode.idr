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
  show NoGrad = "NoGrad"

public export
Eq GradMode where
  WithGrad == WithGrad = True
  NoGrad == NoGrad = True
  _ == _ = False

public export
DecEq GradMode where
  decEq WithGrad WithGrad = Yes Refl
  decEq NoGrad NoGrad = Yes Refl
  decEq WithGrad NoGrad = No (\case Refl impossible)
  decEq NoGrad WithGrad = No (\case Refl impossible)
