||| Arch-agnostic `config.json` reading for the HF-aligned adapters.
|||
||| Every `fromPretrained` reads a HuggingFace `config.json` into the
||| adapter's own config record, then packs the model into a dependent
||| pair `(cfg ** Model cfg ex dt g)`. The JSON plumbing — read the
||| file, parse it, pull integer fields by their HF key — is identical
||| across architectures; only the *set* of fields is per-arch (and
||| lives in each `Transformers.*` module, beside its param-name
||| catalogue). This module owns the shared plumbing.
|||
||| Errors surface as `Checkpoint.LoadError`:
|||   - missing file            → `FileNotFound`
|||   - unparseable JSON        → `ConfigError "..."`
|||   - missing / non-numeric   → `ConfigError "..."` (names the field)
||| so `fromPretrained : String -> IO (Either LoadError ...)` reports
||| config failures in the same channel as weight-load failures.
module Transformers.Config

import Data.List
import public Language.JSON
import System.File

import Checkpoint

----------------------------------------------------------------------
-- Read + parse
----------------------------------------------------------------------

||| Read a `config.json` off disk and parse it to a `JSON` value.
||| A missing file is `FileNotFound`; an unparseable one is a
||| `ConfigError` naming the path.
export
readConfigFile : String -> IO (Either LoadError JSON)
readConfigFile path = do
  Right contents <- readFile path
    | Left _ => pure (Left FileNotFound)
  case parse contents of
    Nothing => pure (Left (ConfigError ("could not parse JSON: " ++ path)))
    Just j  => pure (Right j)

----------------------------------------------------------------------
-- Field extraction
----------------------------------------------------------------------

||| Pull a required non-negative integer field out of a JSON object by
||| its HF key (e.g. `"hidden_size"`). HF stores these as JSON numbers,
||| so we truncate the parsed `Double`; a missing or non-numeric field
||| is a `ConfigError` that names the key.
export
natField : JSON -> (key : String) -> Either LoadError Nat
natField (JObject fields) key =
  case lookup key fields of
    Just (JNumber d) => Right (integerToNat (cast {to=Integer} (the Int (cast d))))
    Just _           => Left (ConfigError ("field is not a number: " ++ key))
    Nothing          => Left (ConfigError ("missing field: " ++ key))
natField _ _ = Left (ConfigError "config root is not a JSON object")

||| Like `natField`, but returns `def` when the key is absent. Use for
||| HF fields that default when omitted (e.g. `type_vocab_size`). A
||| present-but-non-numeric value is still a `ConfigError`.
export
natFieldOr : JSON -> (key : String) -> (def : Nat) -> Either LoadError Nat
natFieldOr (JObject fields) key def =
  case lookup key fields of
    Nothing => Right def
    Just _  => natField (JObject fields) key
natFieldOr _ _ def = Right def
