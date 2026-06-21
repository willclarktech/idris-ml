||| peft-compatible adapter file I/O for HF-loaded BERT LoRA models.
|||
||| `saveLoraAdapter outputDir config` writes a directory matching
||| HuggingFace `peft`'s on-disk layout:
|||
|||     outputDir/
|||       adapter_config.json    -- LoRA hyperparams (rank, alpha, targets, etc.)
|||       adapter_model.safetensors -- A / B tensors under peft's wrapped names
|||
||| The cross-tool guarantee: an idris-ml-saved adapter directory is
||| loadable in Python via
|||
|||     from peft import PeftModel
|||     PeftModel.from_pretrained(base_model, "outputDir")
|||
||| The trick is the on-disk name wrapping. idris-ml's in-memory
||| paramId for an adapter weight is
|||
|||     bert.encoder.layer.0.attention.self.query.lora_A
|||
||| (HF-aligned for the load path). peft wraps both sides with two
||| decorations:
|||
|||     base_model.model.bert.encoder.layer.0.attention.self.query.lora_A.default.weight
|||                ----- peft -----                                ----- peft -----
|||
||| `idrisToPeftName` adds those at save time; `peftToIdrisName`
||| strips them at load time. The actual tensor data is unchanged.
module Transformers.LoraIO

import Data.Either
import Data.List
import Data.String
import System.Directory
import System.File

import Checkpoint
import Executor
import Tensor

----------------------------------------------------------------------
-- Adapter config (mirror of peft.LoraConfig fields used here)
----------------------------------------------------------------------

public export
record LoraAdapterConfig where
  constructor MkLoraAdapterConfig
  rank          : Nat         -- peft `r`
  alpha         : Double      -- peft `lora_alpha`
  targetModules : List String -- peft `target_modules`, e.g. ["query","value"]
  taskType      : String      -- e.g. "SEQ_CLS"

----------------------------------------------------------------------
-- Name remap helpers
----------------------------------------------------------------------

peftPrefix : String
peftPrefix = "base_model.model."

peftSuffix : String
peftSuffix = ".default.weight"

||| Strip peft's `base_model.model.` prefix + `.default.weight`
||| suffix from a name. Returns `Nothing` if either decoration is
||| missing (means the name didn't come from a peft-saved adapter).
export
peftToIdrisName : String -> Maybe String
peftToIdrisName name =
  if isPrefixOf peftPrefix name && isSuffixOf peftSuffix name
    then
      let preLen = length peftPrefix
          sufLen  = length peftSuffix
          dropped = substr (cast preLen) (length name `minus` (preLen + sufLen)) name
      in Just dropped
    else Nothing

||| Wrap an idris-ml paramId with peft's on-disk decorations.
||| Inverse of `peftToIdrisName`.
export
idrisToPeftName : String -> String
idrisToPeftName name = peftPrefix ++ name ++ peftSuffix

-- Predicate: does this paramId look like a LoRA adapter weight?
-- Matches the L1/L3 naming convention: `<...>.lora_A` or `<...>.lora_B`.
isAdapterName : String -> Bool
isAdapterName nm = isSuffixOf ".lora_A" nm || isSuffixOf ".lora_B" nm

----------------------------------------------------------------------
-- adapter_config.json — hand-rolled JSON emitter
----------------------------------------------------------------------

quoteString : String -> String
quoteString s = "\"" ++ s ++ "\""

joinComma : List String -> String
joinComma []        = ""
joinComma [x]       = x
joinComma (x :: xs) = x ++ ", " ++ joinComma xs

renderTargetModules : List String -> String
renderTargetModules ms =
  "[" ++ joinComma (map quoteString ms) ++ "]"

-- Mirror peft's adapter_config.json schema with just the fields we need
-- + the fields peft validates on load. peft is strict about required
-- keys (peft_type, task_type, r, lora_alpha, target_modules, bias,
-- lora_dropout, fan_in_fan_out, inference_mode, modules_to_save).
renderAdapterConfig : LoraAdapterConfig -> String
renderAdapterConfig cfg =
  "{\n"
  ++ "  \"peft_type\": \"LORA\",\n"
  ++ "  \"task_type\": \"" ++ cfg.taskType ++ "\",\n"
  ++ "  \"r\": " ++ show cfg.rank ++ ",\n"
  ++ "  \"lora_alpha\": " ++ show cfg.alpha ++ ",\n"
  ++ "  \"target_modules\": " ++ renderTargetModules cfg.targetModules ++ ",\n"
  ++ "  \"lora_dropout\": 0.0,\n"
  ++ "  \"bias\": \"none\",\n"
  ++ "  \"fan_in_fan_out\": false,\n"
  ++ "  \"modules_to_save\": null,\n"
  ++ "  \"inference_mode\": false\n"
  ++ "}\n"

----------------------------------------------------------------------
-- adapter_config.json — flat lenient parser
----------------------------------------------------------------------
--
-- We own both writer and reader, so a flat scan is sufficient.
-- Mirrors the `extractField` helper in Checkpoint.idr (used by
-- trainer_state.json).

afterNeedle : List Char -> List Char -> Maybe (List Char)
afterNeedle needle hay =
  if isPrefixOf needle hay
    then Just (drop (length needle) hay)
    else case hay of
           []        => Nothing
           (_ :: cs) => afterNeedle needle cs

extractScalar : String -> String -> Maybe String
extractScalar key src = do
  rest <- afterNeedle (unpack ("\"" ++ key ++ "\"")) (unpack src)
  let afterColon = drop 1 (dropWhile (/= ':') rest)
      tok     = takeWhile (\c => c /= ',' && c /= '}' && c /= '\n') afterColon
      cleaned = filter (\c => not (isSpace c) && c /= '"') tok
  pure (pack cleaned)

-- Split a List Char by separator into List (List Char).
splitChars : Char -> List Char -> List (List Char)
splitChars sep cs = go cs []
  where
    go : List Char -> List Char -> List (List Char)
    go []        acc = if isNil acc then [] else [reverse acc]
    go (c :: cs) acc =
      if c == sep
        then reverse acc :: go cs []
        else go cs (c :: acc)

extractArray : String -> String -> Maybe (List String)
extractArray key src = do
  rest <- afterNeedle (unpack ("\"" ++ key ++ "\"")) (unpack src)
  let afterColon = drop 1 (dropWhile (/= '[') rest)
      tok      = takeWhile (/= ']') afterColon
      pieces   = splitChars ',' tok
      stripped = map (pack . filter (\c => c /= '"' && not (isSpace c))) pieces
      nonempty = filter (\s => s /= "") stripped
  pure nonempty

----------------------------------------------------------------------
-- Save / Load
----------------------------------------------------------------------

||| Save a peft-compatible LoRA adapter directory. Writes
|||  outputDir/adapter_config.json  + adapter_model.safetensors.
||| The adapter weights' on-disk names are wrapped in peft's
||| `base_model.model.[...].default.weight` decorations so the
||| resulting directory loads cleanly via `PeftModel.from_pretrained`
||| in Python.
|||
||| Skips creating outputDir if it already exists (mirror of
||| Checkpoint.saveCheckpointFiles).
export
saveLoraAdapter : UserExecutorTraining ex
              => (outputDir : String)
              -> LoraAdapterConfig
              -> IO Bool
saveLoraAdapter outputDir cfg = do
  _ <- createDir outputDir
  cfgOk <- writeFile (outputDir ++ "/adapter_config.json")
                     (renderAdapterConfig cfg)
  case cfgOk of
    Left _   => pure False
    Right () => do
      let safePath = outputDir ++ "/adapter_model.safetensors"
      saveModelMatchingRenamed {ex} safePath
        (\nm => if isAdapterName nm
                  then Just (idrisToPeftName nm)
                  else Nothing)

||| Load a peft-saved LoRA adapter directory back into the in-memory
||| registry. The reverse of `saveLoraAdapter`: reads
||| adapter_config.json into a `LoraAdapterConfig`, then loads the
||| safetensors by stripping peft's wrapping at the key level (the
||| existing `load … {only := Just pfx}` machinery handles the actual tensor
||| read — we just need to ensure the on-disk keys match the
||| in-memory paramIds after the strip).
|||
||| Caveat: the C-side load path looks up tensors by EXACT registry
||| name from the file's on-disk name. To load a peft-saved file we'd
||| need the symmetric C-side rename hook. Adding it tracks the load
||| follow-up TODO row; the canonical idris-ml LoRA workflow today
||| is "train + save in idris-ml; consume via peft in Python", which
||| only needs the save side. This function reads the config + emits
||| a stderr note + returns `Right cfg` so callers can still inspect
||| hyperparams even before the symmetric load lands.
export
loadLoraAdapter : (outputDir : String)
              -> IO (Either String LoraAdapterConfig)
loadLoraAdapter outputDir = do
  res <- readFile (outputDir ++ "/adapter_config.json")
  case res of
    Left _         => pure (Left ("could not read adapter_config.json at " ++ outputDir))
    Right contents =>
      case ( extractScalar "r" contents
           , extractScalar "lora_alpha" contents
           , extractArray  "target_modules" contents
           , extractScalar "task_type" contents
           ) of
        (Just rStr, Just aStr, Just tms, Just task) =>
          let r = cast {to=Nat} (the Integer (cast rStr))
              a = the Double (cast aStr)
          in pure (Right (MkLoraAdapterConfig r a tms task))
        _ => pure (Left "adapter_config.json missing required fields")
