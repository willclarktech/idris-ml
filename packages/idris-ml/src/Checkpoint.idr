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

-- SafeTensors I/O dispatches per-backend through `UserExecutorTraining ex`:
-- each backend's param/optimizer registry is TU-local, so `{ex}`
-- selects which one is serialized.

||| Save all registered parameters to a .safetensors file.
||| Returns True on success.
export
saveModel : UserExecutorTraining ex => String -> IO Bool
saveModel path = do
  rc <- primIO (primParamSave {ex} path)
  pure (rc == 0)

||| Save everything registered, one file — the registry-wide escape
||| hatch. Model-scoped saves arrive with the models-as-records work.
export
saveAll : UserExecutorTraining ex => String -> IO Bool
saveAll path = saveModel {ex} path

----------------------------------------------------------------------
-- Typed load surface (v1)
----------------------------------------------------------------------

||| Why a load failed. Mirrors `param_load`'s C return codes
||| (see backend.h); `LoadFailed n` is the total escape hatch for
||| codes this vocabulary doesn't know yet.
|||
||| A file key missing from the registry is a SKIP, not an error —
||| the warm-start path (backbone via `only`, fresh head) depends on
||| it, so there is deliberately no MissingParam constructor.
public export
data LoadError
  = FileNotFound
  | MalformedFile
  | DtypeMismatch
  | ShapeMismatch
  | UnsupportedDtype
  | ReadFailed
  | LoadFailed Int
  | ConfigError String  -- config.json missing/malformed/missing-field (Idris-side only)

export
Eq LoadError where
  FileNotFound     == FileNotFound     = True
  MalformedFile    == MalformedFile    = True
  DtypeMismatch    == DtypeMismatch    = True
  ShapeMismatch    == ShapeMismatch    = True
  UnsupportedDtype == UnsupportedDtype = True
  ReadFailed       == ReadFailed       = True
  LoadFailed n     == LoadFailed m     = n == m
  ConfigError a    == ConfigError b    = a == b
  _                == _                = False

export
Show LoadError where
  show FileNotFound     = "FileNotFound"
  show MalformedFile    = "MalformedFile"
  show DtypeMismatch    = "DtypeMismatch"
  show ShapeMismatch    = "ShapeMismatch"
  show UnsupportedDtype = "UnsupportedDtype"
  show ReadFailed       = "ReadFailed"
  show (LoadFailed n)   = "LoadFailed " ++ show n
  show (ConfigError m)  = "ConfigError: " ++ m

decodeLoadError : Int -> LoadError
decodeLoadError (-1) = FileNotFound
decodeLoadError (-2) = MalformedFile
decodeLoadError (-3) = DtypeMismatch
decodeLoadError (-4) = ShapeMismatch
decodeLoadError (-5) = UnsupportedDtype
decodeLoadError (-6) = ReadFailed
decodeLoadError n    = LoadFailed n

||| Load options. Build from `defaultLoadOpts` with record updates.
|||
||| `allowCast = False` (the safe default) errors on any on-disk vs
||| destination dtype difference; `True` reads source bytes in their
||| on-disk width, widens to doubles, and narrows into the
||| destination's storage dtype (F32 -> F64 lossless, F64 -> F32
||| precision loss, both well-defined).
|||
||| `only = Just pfx` loads only the safetensors keys whose name
||| starts with `pfx`, leaving every other in-memory param untouched
||| (warm-start: backbone under "bert.", fresh head elsewhere).
|||
||| `remap = Just f` translates registry names to on-disk keys at load
||| time: each registered param `nm` reads from the file key `f nm`
||| (`f nm = Nothing` skips that param). The symmetric inverse of
||| `saveModelMatchingRenamed`'s transform — use it to read a
||| foreign-named checkpoint (e.g. a peft adapter whose JSON keys are
||| `base_model.model.[...].lora_A.default.weight`) into registry params
||| under idris-ml names, without registering params under the foreign
||| names first. When set it takes precedence over `only` (the
||| transform is the filter). Params whose `f nm` key is absent from
||| the file are skipped, not errored.
public export
record LoadOpts where
  constructor MkLoadOpts
  allowCast : Bool
  only      : Maybe String
  remap     : Maybe (String -> Maybe String)

public export
defaultLoadOpts : LoadOpts
defaultLoadOpts = MkLoadOpts False Nothing Nothing

-- Forward declaration — `collectRenamedNames` is defined in the
-- filtered-save section below; `load`'s remap path reuses the same
-- registry walker, so its type needs to be in scope here.
collectRenamedNames : UserExecutorTraining ex =>
  (transform : String -> Maybe String) -> IO (String, String, Int)

||| Load parameters from a .safetensors file into the existing
||| registry, by name. The load mutates the C-side parameter buffers
||| in place, so subsequent forward passes see the loaded weights with
||| no further refresh needed. File keys missing from the registry are
||| skipped (not an error); per-entry failures don't abort the rest of
||| the load, and the FIRST failure becomes the returned `LoadError`.
export
load : UserExecutorTraining ex =>
       (path : String) -> LoadOpts -> IO (Either LoadError ())
load path opts = do
  let castFlag : Int
      castFlag = if opts.allowCast then 1 else 0
  rc <- case opts.remap of
          Just f  => do
            -- Walk the registry, building (registryName, onDiskKey) pairs
            -- via the same collector the renamed-save side uses. Zero
            -- matches = nothing to load (not an error, unlike save).
            (regNames, diskNames, count) <- collectRenamedNames {ex} f
            if count == 0
              then pure 0
              else primIO (primParamLoadRenamed {ex} path castFlag regNames diskNames count)
          Nothing => case opts.only of
            Nothing  => primIO (primParamLoadWithPolicy {ex} path castFlag)
            Just pfx => primIO (primParamLoadWithPrefix {ex} path castFlag pfx)
  pure (if rc == 0 then Right () else Left (decodeLoadError rc))

||| Strict-dtype load as a bare Bool (wrapper over `load`).
export
loadModel : UserExecutorTraining ex => String -> IO Bool
loadModel path = isRight <$> load {ex} path defaultLoadOpts

||| `loadModel` with silent dtype conversion (wrapper over `load`
||| with `allowCast`).
export
loadModelAllowCast : UserExecutorTraining ex => String -> IO Bool
loadModelAllowCast path =
  isRight <$> load {ex} path ({ allowCast := True } defaultLoadOpts)

||| Load only keys under `pfx` (wrapper over `load` with `only`).
export
loadModelPrefix : UserExecutorTraining ex => (path : String) -> (pfx : String) -> IO Bool
loadModelPrefix path pfx =
  isRight <$> load {ex} path ({ only := Just pfx } defaultLoadOpts)

||| Prefix-filtered load with silent dtype conversion (wrapper over
||| `load`).
export
loadModelPrefixAllowCast : UserExecutorTraining ex =>
                           (path : String) -> (pfx : String) -> IO Bool
loadModelPrefixAllowCast path pfx =
  isRight <$> load {ex} path ({ allowCast := True, only := Just pfx } defaultLoadOpts)

----------------------------------------------------------------------
-- Filtered save (adapter-only checkpoints)
----------------------------------------------------------------------

-- Join a list of strings with newlines (no trailing newline).
joinNewlines : List String -> String
joinNewlines []        = ""
joinNewlines [x]       = x
joinNewlines (x :: xs) = x ++ "\n" ++ joinNewlines xs

-- Walk the registry and collect names matching the predicate. Order
-- matches registry order (which the C-side preserves on disk via
-- `param_save_by_name`). Returns the matched names as a single
-- newline-joined String + the count — the shape `primParamSaveByName`
-- accepts on the C side.
collectMatchingNames : UserExecutorTraining ex =>
  (String -> Bool) -> IO (String, Int)
collectMatchingNames pred = do
  n <- getParamCount {ex}
  go n 0 []
  where
    -- `acc` accumulates kept names; we build the joined string at
    -- the end so we don't repeatedly concat-with-newline mid-walk.
    go : Int -> Int -> List String -> IO (String, Int)
    go end i acc =
      if i >= end
        then pure (joinNewlines (reverse acc), cast (length acc))
        else do
          nm <- getParamName {ex} i
          let acc' = if pred nm then nm :: acc else acc
          go end (i + 1) acc'

||| Save only those registered params whose paramId satisfies
||| `predicate`. The C-side `param_save_by_name` builds a fresh
||| safetensors file containing only the matching tensors; on-disk
||| names are the registry names verbatim (no rename hook here —
||| if you need name remapping, register your params under the
||| target on-disk names directly).
|||
||| The order on disk matches registry order (not the order names
||| would appear if iterated in any other way). Returns False if
||| no params match (empty list is treated as an error by the C
||| layer, matching `saveModel`'s "no params registered" guard).
|||
||| Use case: LoRA adapter-only checkpoints — call
|||
|||     saveModelMatching path (\nm => isSuffixOf "lora_A" nm
|||                                 || isSuffixOf "lora_B" nm)
|||
||| to write a small `.safetensors` (~200KB for bert-tiny + r=8)
||| containing just the trainable A / B matrices while the multi-MB
||| backbone stays untouched on disk.
export
saveModelMatching : UserExecutorTraining ex =>
  (path : String) -> (predicate : String -> Bool) -> IO Bool
saveModelMatching path pred = do
  (names, count) <- collectMatchingNames {ex} pred
  if count == 0
    then pure False
    else do
      rc <- primIO (primParamSaveByName {ex} path names count)
      pure (rc == 0)

||| Convenience wrapper: save only params whose paramId ends with
||| any of the given suffixes. Common LoRA call: `saveModelSuffixes
||| path ["lora_A", "lora_B"]`.
export
saveModelSuffixes : UserExecutorTraining ex =>
  (path : String) -> (suffixes : List String) -> IO Bool
saveModelSuffixes path sfxs =
  saveModelMatching {ex} path (\nm => any (\s => isSuffixOf s nm) sfxs)

-- Walk the registry collecting matching name pairs (registryName, ondiskName).
-- The transform takes a registry name and returns `Just on_disk_name` to
-- include (with rename) or `Nothing` to skip. (Type forward-declared above
-- the `load` definition so the remap path can reuse it.)
collectRenamedNames transform = do
  n <- getParamCount {ex}
  go n 0 [] []
  where
    go : Int -> Int -> List String -> List String -> IO (String, String, Int)
    go end i lookups ondisks =
      if i >= end
        then pure ( joinNewlines (reverse lookups)
                  , joinNewlines (reverse ondisks)
                  , cast (length lookups))
        else do
          nm <- getParamName {ex} i
          case transform nm of
            Nothing      => go end (i + 1) lookups ondisks
            Just renamed => go end (i + 1) (nm :: lookups) (renamed :: ondisks)

||| Like `saveModelMatching`, but the per-param transform produces
||| the on-disk name (or `Nothing` to skip). The C-side writer
||| receives both lists in lockstep: registry names for tensor
||| lookup, override names for the JSON header.
|||
||| Use case: peft-compatible LoRA adapter export. The transform
|||
|||     \nm => if endsWithLoraAorB nm
|||              then Just ("base_model.model." ++ nm ++ ".default.weight")
|||              else Nothing
|||
||| picks out the adapters and wraps them in peft's on-disk
||| decorations so the resulting file loads cleanly via
||| `peft.PeftModel.from_pretrained` in Python.
export
saveModelMatchingRenamed : UserExecutorTraining ex =>
  (path : String) -> (transform : String -> Maybe String) -> IO Bool
saveModelMatchingRenamed path transform = do
  (lookups, ondisks, count) <- collectRenamedNames {ex} transform
  if count == 0
    then pure False
    else do
      rc <- primIO (primParamSaveByNameRenamed {ex} path lookups ondisks count)
      pure (rc == 0)

||| Save optimizer state (momentum/velocity buffers) to a .safetensors file.
||| Returns True on success.
export
saveOptimizer : UserExecutorTraining ex => String -> NativeOptimizer ex -> IO Bool
saveOptimizer path opt = do
  rc <- primIO (primOptimizerSave {ex} opt.handle path)
  pure (rc == 0)

||| Load optimizer state from a .safetensors file.
||| Returns True on success.
export
loadOptimizer : UserExecutorTraining ex => String -> NativeOptimizer ex -> IO Bool
loadOptimizer path opt = do
  rc <- primIO (primOptimizerLoad {ex} opt.handle path)
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
      tok     = takeWhile (\c => c /= ',' && c /= '}' && c /= '\n') afterColon
      cleaned = filter (\c => not (isSpace c) && c /= '"') tok
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
    Left _         => pure Nothing
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

saveCheckpointFiles : UserExecutorTraining ex => NativeOptimizer ex -> String -> String -> Nat -> Double -> IO Bool
saveCheckpointFiles opt dir pfx ep best = do
  _   <- createDir dir
  ok1 <- saveModel {ex} (pfx ++ ".model.safetensors")
  ok2 <- saveOptimizer {ex} (pfx ++ ".opt.safetensors") opt
  ok3 <- writeTrainerState (pfx ++ ".trainer_state.json") ep best
  pure (ok1 && ok2 && ok3)

loadCheckpointFiles : UserExecutorTraining ex =>
  NativeOptimizer ex -> String -> IO (Maybe (Nat, Double))
loadCheckpointFiles opt pfx = do
  st <- readTrainerState (pfx ++ ".trainer_state.json")
  case st of
    Nothing     => pure Nothing
    Just epBest => do
      _ <- loadModel {ex} (pfx ++ ".model.safetensors")
      _ <- loadOptimizer {ex} (pfx ++ ".opt.safetensors") opt
      pure (Just epBest)

||| Build a file-backed checkpoint policy. The param registry is global,
||| so saving needs only the optimizer handle (not the model value).
||| Periodic saves use the `<dir>/last` prefix; keep-best uses
||| `<dir>/best` — so a periodic save never clobbers the best one.
||| Override `monitor` via record update to keep-best on a val metric.
export
fileCheckpoint : UserExecutorTraining ex =>
  (dir : String) -> (everyN : Nat) -> (keepBest : Bool) ->
  NativeOptimizer ex -> CheckpointPolicy
fileCheckpoint dir everyN keepBest opt =
  MkCheckpointPolicy dir everyN keepBest Nothing
    (saveCheckpointFiles opt dir)
    (loadCheckpointFiles opt)
