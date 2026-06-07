||| Executor — barrel re-export module.
|||
||| `import Executor` brings the device taxonomy into scope: the
||| `UserExecutorCore` / `UserExecutorLinear` / `UserExecutorNN` /
||| `UserExecutorConv` / `UserExecutorTraining` / `UserExecutorTransfer`
||| interfaces from `Executor.Core`, plus the three built-in backend
||| device tags from `Executor.{Tape,Torch,Mlx}`, plus the DType
||| taxonomy from `DType.Core`.
|||
||| Every Tensor in the codebase carries one of:
|||
|||   * `TapeExecutor`              — the tape backend (host CPU only)
|||   * `TorchExecutor TCpu`        — libtorch on host CPU
|||   * `TorchExecutor TMps`        — libtorch on Apple Metal
|||   * `TorchExecutor (TCuda n)`   — libtorch on NVIDIA CUDA device n
|||   * `MlxExecutor MCpu`          — mlx CPU stream
|||   * `MlxExecutor MGpu`          — mlx Metal stream
|||
||| as its `(0 d : Type)` phantom. Each carries an associated
||| `backendTag` (via `UserExecutorTransfer`) used by `toExecutor` for
||| cross-backend transfer dispatch. Users adding their own backend
||| declare a new type with `UserExecutorCore` (+ optional
||| `UserExecutorTransfer`) instances — see
||| `packages/idris-ml-examples/src/Example/BringYourOwn.idr` for
||| the recipe.
module Executor

import public Backend
import public Executor.Core
import public Executor.Tape
import public Executor.Torch
import public Executor.Mlx
import public DType.Core
import public HwConfig
