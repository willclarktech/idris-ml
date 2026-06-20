||| `TapeExecutor` backend, split into per-slice instance
||| modules (Core / Linear / Nn / Training / Transfer). Umbrella
||| re-export — `import Executor.Tape` is unchanged.
module Executor.Tape

import Executor.Core
import public Executor.Tape.Core
import public Executor.Tape.Linear
import public Executor.Tape.Nn
import public Executor.Tape.Training
import public Executor.Tape.Transfer

public export
UserExecutorTraining TapeExecutor where
  -- >>> GENERATED FROM ffi_manifest.py — gen-executor-instances.py >>>
  -- <<< END GENERATED <<<
