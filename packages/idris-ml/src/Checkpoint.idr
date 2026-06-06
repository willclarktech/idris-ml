||| SafeTensors model serialization.
||| Save/load registered parameters to/from .safetensors files.
module Checkpoint

import Data.Either
import Data.List
import Data.String
import System.Clock
import System.Directory
import System.File

import Executor
import Tensor

-- SafeTensors I/O dispatches per-backend through `UserExecutorTraining d`:
-- each backend's param/optimizer registry is TU-local, so `{d}`
-- selects which one is serialized.

||| Save all registered parameters to a .safetensors file.
||| Returns True on success.
export
saveModel : UserExecutorTraining d => String -> IO Bool
saveModel path = do
  rc <- primIO (primParamSave {d} path)
  pure (rc == 0)

||| Load parameters from a .safetensors file into the existing registry.
||| Strict-dtype mode: any param whose on-disk dtype differs from the
||| in-memory destination is an error (the load reports the offending
||| param name to stderr and returns `False`). The load mutates the
||| C-side parameter buffers in place, so subsequent forward passes see
||| the loaded weights with no further refresh needed.
|||
||| For cross-dtype loads (e.g. an F32-saved checkpoint into an F64
||| model), use `loadModelAllowCast` to opt in to silent precision
||| conversion at load time.
export
loadModel : UserExecutorTraining d => String -> IO Bool
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
loadModelAllowCast : UserExecutorTraining d => String -> IO Bool
loadModelAllowCast path = do
  rc <- primIO (primParamLoadWithPolicy {d} path 1)
  pure (rc == 0)

||| Save optimizer state (momentum/velocity buffers) to a .safetensors file.
||| Returns True on success.
export
saveOptimizer : UserExecutorTraining d => String -> NativeOptimizer d -> IO Bool
saveOptimizer path opt = do
  rc <- primIO (primOptimizerSave {d} opt.handle path)
  pure (rc == 0)

||| Load optimizer state from a .safetensors file.
||| Returns True on success.
export
loadOptimizer : UserExecutorTraining d => String -> NativeOptimizer d -> IO Bool
loadOptimizer path opt = do
  rc <- primIO (primOptimizerLoad {d} opt.handle path)
  pure (rc == 0)


----------------------------------------------------------------------
-- Checkpoint policy (training-loop integration)
----------------------------------------------------------------------

-- Resume metadata rides in an HF-Trainer-style `trainer_state.json`
-- sidecar written here in pure Idris (no C change). The safetensors
-- files carry the heavy state (params + optimizer m/v buffers); the
-- sidecar carries only the scalar resume state (epoch + best metric).

||| Find the substring immediately after the first occurrence of
||| `needle` in `hay`. Used to locate a JSON key.
afterNeedle : List Char -> List Char -> Maybe (List Char)
afterNeedle needle hay =
  if isPrefixOf needle hay
    then Just (drop (length needle) hay)
    else case hay of
           []        => Nothing
           (_ :: cs) => afterNeedle needle cs

||| Extract the bare value token following `"<key>":` in a flat JSON
||| object — everything up to the next `,`, `}`, or newline, with
||| whitespace and quotes stripped. We own both writer and reader, so
||| this lenient scan is sufficient (no nested objects).
extractField : String -> String -> Maybe String
extractField key src = do
  rest <- afterNeedle (unpack ("\"" ++ key ++ "\"")) (unpack src)
  let afterColon = drop 1 (dropWhile (/= ':') rest)
      tok        = takeWhile (\c => c /= ',' && c /= '}' && c /= '\n') afterColon
      cleaned    = filter (\c => not (isSpace c) && c /= '"') tok
  pure (pack cleaned)

||| Write the resume sidecar. Returns True on success.
writeTrainerState : String -> Nat -> Double -> IO Bool
writeTrainerState path ep best = do
  now <- clockTime UTC
  let json = "{\n  \"epoch\": " ++ show ep
          ++ ",\n  \"best\": " ++ show best
          ++ ",\n  \"timestamp\": " ++ show (seconds now)
          ++ "\n}\n"
  res <- writeFile path json
  pure (isRight res)

||| Read the resume sidecar. Returns `Nothing` when the file is absent
||| or unparseable (treated as a fresh start).
readTrainerState : String -> IO (Maybe (Nat, Double))
readTrainerState path = do
  res <- readFile path
  case res of
    Left _ => pure Nothing
    Right contents =>
      case (extractField "epoch" contents, extractField "best" contents) of
        (Just eStr, Just bStr) =>
          pure (Just (cast (the Integer (cast eStr)), cast bStr))
        _ => pure Nothing

||| Policy describing how the training loop persists and restores state.
||| Built by `fileCheckpoint`. `runTrainingIO` consults the policy after
||| each epoch (periodic + keep-best) and once before the loop (resume).
|||
||| `monitor` selects the scalar to keep-best on (lower is better);
||| `Nothing` tracks the per-epoch training loss the loop already has.
||| It's an `IO Double` (not `model -> IO Double`) so the policy stays
||| free of the model type — an override closes over its own eval state,
||| the same idiom the `metrics` callback uses.
||| `saveState prefix epoch best` writes the model + optimizer + sidecar
||| under `<prefix>.*`; `loadState prefix` restores them and returns the
||| `(resumeEpoch, bestMetric)` from the sidecar, or `Nothing` for a
||| fresh start.
public export
record CheckpointPolicy where
  constructor MkCheckpointPolicy
  dir       : String
  everyN    : Nat
  keepBest  : Bool
  monitor   : Maybe (IO Double)
  saveState : String -> Nat -> Double -> IO Bool
  loadState : String -> IO (Maybe (Nat, Double))

saveCheckpointFiles : UserExecutorTraining d => NativeOptimizer d -> String -> String -> Nat -> Double -> IO Bool
saveCheckpointFiles opt dir pfx ep best = do
  _   <- createDir dir
  ok1 <- saveModel {d} (pfx ++ ".model.safetensors")
  ok2 <- saveOptimizer {d} (pfx ++ ".opt.safetensors") opt
  ok3 <- writeTrainerState (pfx ++ ".trainer_state.json") ep best
  pure (ok1 && ok2 && ok3)

loadCheckpointFiles : UserExecutorTraining d =>
  NativeOptimizer d -> String -> IO (Maybe (Nat, Double))
loadCheckpointFiles opt pfx = do
  st <- readTrainerState (pfx ++ ".trainer_state.json")
  case st of
    Nothing     => pure Nothing
    Just epBest => do
      _ <- loadModel {d} (pfx ++ ".model.safetensors")
      _ <- loadOptimizer {d} (pfx ++ ".opt.safetensors") opt
      pure (Just epBest)

||| Build a file-backed checkpoint policy. The param registry is global,
||| so saving needs only the optimizer handle (not the model value).
||| Periodic saves use the `<dir>/last` prefix; keep-best uses
||| `<dir>/best` — so a periodic save never clobbers the best one.
||| Override `monitor` via record update to keep-best on a val metric.
export
fileCheckpoint : UserExecutorTraining d =>
  (dir : String) -> (everyN : Nat) -> (keepBest : Bool) ->
  NativeOptimizer d -> CheckpointPolicy
fileCheckpoint dir everyN keepBest opt =
  MkCheckpointPolicy dir everyN keepBest Nothing
    (saveCheckpointFiles opt dir)
    (loadCheckpointFiles opt)
