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
module Executor.Core

import public Executor.Core.Aggregate
import public Executor.Core.Compute
import public Executor.Core.Kind
import public Executor.Core.Training
import public Executor.Core.Transfer
