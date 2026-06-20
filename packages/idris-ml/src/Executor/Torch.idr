||| `TorchExecutor` backend, split into per-slice instance
||| modules (Core / Linear / Nn / Training / Transfer). Umbrella
||| re-export — `import Executor.Torch` is unchanged.
module Executor.Torch

import Executor.Core
import public Executor.Torch.Core
import public Executor.Torch.Linear
import public Executor.Torch.Nn
import public Executor.Torch.Training
import public Executor.Torch.Transfer

public export
{d : TorchHwDev} -> UserExecutorTraining (TorchExecutor d) where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  -- <<< END GENERATED <<<
