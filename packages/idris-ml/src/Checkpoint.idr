||| SafeTensors model serialization.
||| Save/load registered parameters to/from .safetensors files.
module Checkpoint

import Tensor

%foreign "C:param_save,libidrisml"
prim__paramSave : String -> PrimIO Int

%foreign "C:param_load,libidrisml"
prim__paramLoad : String -> PrimIO Int

%foreign "C:param_load_with_policy,libidrisml"
prim__paramLoadWithPolicy : String -> Int -> PrimIO Int

%foreign "C:optimizer_save,libidrisml"
prim__optimizerSave : AnyPtr -> String -> PrimIO Int

%foreign "C:optimizer_load,libidrisml"
prim__optimizerLoad : AnyPtr -> String -> PrimIO Int

||| Save all registered parameters to a .safetensors file.
||| Returns True on success.
export
saveModel : String -> IO Bool
saveModel path = do
  rc <- primIO (prim__paramSave path)
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
loadModel : String -> IO Bool
loadModel path = do
  rc <- primIO (prim__paramLoad path)
  pure (rc == 0)

||| Same as `loadModel` but routes through `param_load_with_policy`
||| with `allow_cast=1`. On dtype mismatch, the on-disk bytes are
||| read in their source width (F32 -> 4 bytes/elem, F64 -> 8) and
||| widened to doubles before being loaded into the destination param
||| (which the backend then narrows back to its actual storage dtype
||| as needed). F32 -> F64 is lossless; F64 -> F32 incurs precision
||| loss but is well-defined.
export
loadModelAllowCast : String -> IO Bool
loadModelAllowCast path = do
  rc <- primIO (prim__paramLoadWithPolicy path 1)
  pure (rc == 0)

||| Save optimizer state (momentum/velocity buffers) to a .safetensors file.
||| Returns True on success.
export
saveOptimizer : String -> NativeOptimizer -> IO Bool
saveOptimizer path opt = do
  rc <- primIO (prim__optimizerSave opt.handle path)
  pure (rc == 0)

||| Load optimizer state from a .safetensors file.
||| Returns True on success.
export
loadOptimizer : String -> NativeOptimizer -> IO Bool
loadOptimizer path opt = do
  rc <- primIO (prim__optimizerLoad opt.handle path)
  pure (rc == 0)
