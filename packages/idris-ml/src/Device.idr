||| Device — barrel re-export module.
|||
||| `import Device` brings the device taxonomy into scope: the
||| `UserDeviceCore` / `UserDeviceLinear` / `UserDeviceNN` /
||| `UserDeviceConv` / `UserDeviceTape` / `UserDeviceTransfer`
||| interfaces from `Device.Core`, plus the three built-in backend
||| device tags from `Device.{Tape,Torch,Mlx}`, plus the DType
||| taxonomy from `DType.Core`.
|||
||| Every Tensor in the codebase carries one of:
|||
|||   * `TapeDev`              — the tape backend (host CPU only)
|||   * `TorchDev TCpu`        — libtorch on host CPU
|||   * `TorchDev TMps`        — libtorch on Apple Metal
|||   * `TorchDev (TCuda n)`   — libtorch on NVIDIA CUDA device n
|||   * `MlxDev MCpu`          — mlx CPU stream
|||   * `MlxDev MGpu`          — mlx Metal stream
|||
||| as its `(0 d : Type)` phantom. Each carries an associated
||| `backendTag` (via `UserDeviceTransfer`) used by `toDevice` for
||| cross-backend transfer dispatch. Users adding their own backend
||| declare a new type with `UserDeviceCore` (+ optional
||| `UserDeviceTransfer`) instances — see
||| `packages/idris-ml-examples/src/Example/BringYourOwn.idr` for
||| the recipe.
module Device

import public Device.Core
import public Device.Tape
import public Device.Torch
import public Device.Mlx
import public DType.Core
import public HwConfig
