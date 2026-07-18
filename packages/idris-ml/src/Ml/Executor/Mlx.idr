||| `MlxExecutor` backend, split into per-slice instance
||| modules (Core / Linear / Nn / Training / Transfer). Umbrella
||| re-export — `import Executor.Mlx` is unchanged.
module Ml.Executor.Mlx

import Ml.Executor.Core
import public Ml.Executor.Mlx.Core
import public Ml.Executor.Mlx.Linear
import public Ml.Executor.Mlx.Nn
import public Ml.Executor.Mlx.Training
import public Ml.Executor.Mlx.Transfer

public export
{s : MlxStream} -> UserExecutorTraining (MlxExecutor s) where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  -- <<< END GENERATED <<<
