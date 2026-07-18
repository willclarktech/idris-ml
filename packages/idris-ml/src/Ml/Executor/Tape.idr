||| `TapeExecutor` backend, split into per-slice instance
||| modules (Core / Linear / Nn / Training / Transfer). Umbrella
||| re-export — `import Executor.Tape` is unchanged.
module Ml.Executor.Tape

import Ml.Executor.Core
import public Ml.Executor.Tape.Core
import public Ml.Executor.Tape.Linear
import public Ml.Executor.Tape.Nn
import public Ml.Executor.Tape.Training
import public Ml.Executor.Tape.Transfer

public export
UserExecutorTraining TapeExecutor where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  -- <<< END GENERATED <<<
