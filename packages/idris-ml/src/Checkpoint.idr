||| SafeTensors model serialization.
||| Save/load registered parameters to/from .safetensors files.
module Checkpoint

import Device
import Tensor

-- SafeTensors I/O dispatches per-backend through `UserDeviceTape d`:
-- each backend's param/optimizer registry is TU-local, so `{d}`
-- selects which one is serialized.

||| Save all registered parameters to a .safetensors file.
||| Returns True on success.
export
saveModel : UserDeviceTape d => String -> IO Bool
saveModel path = do
  rc <- primIO (primParamSave {d} path)
  pure (rc == 0)

||| Load parameters from a .safetensors file into the existing registry.
||| Strict-dtype mode: any param whose on-disk dtype differs from the
||| in-memory destination is an error (the load reports the offending
||| param name to stderr and returns `False`). After loading, use
||| `emap refreshValue` on the network to update cached values.
|||
||| For cross-dtype loads (e.g. an F32-saved checkpoint into an F64
||| model), use `loadModelAllowCast` to opt in to silent precision
||| conversion at load time.
export
loadModel : UserDeviceTape d => String -> IO Bool
loadModel path = do
  rc <- primIO (primParamLoad {d} path)
  pure (rc == 0)

||| Same as `loadModel` but routes through `param_load_with_policy`
||| with `allow_cast=1`. On dtype mismatch, the on-disk bytes are
||| read in their source width (F32 -> 4 bytes/elem, F64 -> 8) and
||| widened to doubles before being loaded into the destination param
||| (which the backend then narrows back to its actual storage dtype
||| as needed). F32 -> F64 is lossless; F64 -> F32 incurs precision
||| loss but is well-defined.
export
loadModelAllowCast : UserDeviceTape d => String -> IO Bool
loadModelAllowCast path = do
  rc <- primIO (primParamLoadWithPolicy {d} path 1)
  pure (rc == 0)

||| Save optimizer state (momentum/velocity buffers) to a .safetensors file.
||| Returns True on success.
export
saveOptimizer : UserDeviceTape d => String -> NativeOptimizer d -> IO Bool
saveOptimizer path opt = do
  rc <- primIO (primOptimizerSave {d} opt.handle path)
  pure (rc == 0)

||| Load optimizer state from a .safetensors file.
||| Returns True on success.
export
loadOptimizer : UserDeviceTape d => String -> NativeOptimizer d -> IO Bool
loadOptimizer path opt = do
  rc <- primIO (primOptimizerLoad {d} opt.handle path)
  pure (rc == 0)
