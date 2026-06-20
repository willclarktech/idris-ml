||| `MlxExecutor` backend, split into per-slice instance
||| modules (Core / Linear / Nn / Training / Transfer). Umbrella
||| re-export — `import Executor.Mlx` is unchanged.
module Executor.Mlx

import Executor.Core
import public Executor.Mlx.Core
import public Executor.Mlx.Linear
import public Executor.Mlx.Nn
import public Executor.Mlx.Training
import public Executor.Mlx.Transfer

public export
{s : MlxStream} -> UserExecutorTraining (MlxExecutor s) where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  -- <<< END GENERATED <<<
