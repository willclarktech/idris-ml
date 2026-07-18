||| `TorchExecutor` backend, split into per-slice instance
||| modules (Core / Linear / Nn / Training / Transfer). Umbrella
||| re-export — `import Executor.Torch` is unchanged.
module Ml.Executor.Torch

import Ml.Executor.Core
import public Ml.Executor.Torch.Core
import public Ml.Executor.Torch.Linear
import public Ml.Executor.Torch.Nn
import public Ml.Executor.Torch.Training
import public Ml.Executor.Torch.Transfer

public export
{d : TorchHwDev} -> UserExecutorTraining (TorchExecutor d) where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  -- <<< END GENERATED <<<
