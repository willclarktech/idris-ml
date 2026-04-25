||| SafeTensors model serialization.
||| Save/load registered parameters to/from .safetensors files.
module Checkpoint

import Variable

%foreign "C:param_save,libidrisml"
prim__paramSave : String -> PrimIO Int

%foreign "C:param_load,libidrisml"
prim__paramLoad : String -> PrimIO Int

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
||| After loading, use `emap refreshValue` on the network to update cached values.
||| Returns True on success.
export
loadModel : String -> IO Bool
loadModel path = do
  rc <- primIO (prim__paramLoad path)
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
