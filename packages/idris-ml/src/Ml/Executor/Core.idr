||| Pluggable-Executor interface, split into slice modules. See
||| `docs/develop/design-decisions.md` "Pluggable Executor via sliced
||| `UserExecutor` interfaces" for the design.
|||
||| Users implementing their own backend declare an empty type and the
||| relevant `UserExecutor*` instances; the built-in `TapeExecutor` /
||| `TorchExecutor` / `MlxExecutor` (in `Executor.Tape`, `Executor.Torch`,
||| `Executor.Mlx`) forward to the per-backend C symbols.
|||
||| This is the umbrella re-export; the declarations live in the
||| `Executor.Core.*` sub-modules. `import Executor.Core` is unchanged.
module Ml.Executor.Core

import public Ml.Executor.Core.Aggregate
import public Ml.Executor.Core.Compute
import public Ml.Executor.Core.Kind
import public Ml.Executor.Core.Training
import public Ml.Executor.Core.Transfer
