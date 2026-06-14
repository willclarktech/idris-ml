||| Typed Idris wrapper over the HuggingFace Python tokenizer.
|||
||| `Tokenizer vocab` carries the on-disk vocab size as a type-level Nat.
||| `mkTokenizer` constructs one by probing the actual tokenizer's vocab
||| via `scripts/hf_tokenize.py` and validating it matches the claimed
||| size — feeding the wrong `vocab` to a model is a *type error*, not a
||| silent runtime drift.
|||
||| `tokenize` returns an existential pair `(n ** Vect n (Fin vocab))`:
||| the length `n` is runtime-determined but type-tracked, and every
||| token ID is statically bounded by `Fin vocab` (the FFI boundary
||| lifts each ID via `natToFin` and surfaces `TokIdOutOfRange` for
||| anything past the limit — that means the Python tokenizer's vocab
||| disagrees with what the model claims, the kind of misalignment a
||| typed boundary is supposed to catch).
|||
||| The backing is a Python subprocess (`AutoTokenizer.from_pretrained`)
||| rather than an in-process FFI to the Rust `huggingface/tokenizers`
||| crate. Python is already a dev dep (oracle scripts use it); ~1s
||| startup per call is fine for the example pattern of "tokenize the
||| prompt once, generate N tokens in-process, detokenize once". A
||| future row in TODO.md tracks the Rust-FFI upgrade if perf or a
||| non-HF use case forces the issue.
|||
||| Three subprocess interactions:
|||   - `vocab`   →  `mkTokenizer` (one call at construction)
|||   - `encode`  →  `tokenize`     (one call per input string)
|||   - `decode`  →  `detokenize`   (one call per ID Vect)
|||
||| Each call writes the input to `/tmp/idris-hf-tokenize-in.txt`,
||| invokes `python3 hf_tokenize.py …` with stdout redirected to
||| `/tmp/idris-hf-tokenize-out.txt`, then reads the result. NOT
||| concurrency-safe — the temp paths are deterministic. Examples run
||| single-threaded; concurrent use is an explicit non-goal for v1.
module Transformers.Tokenizer

import Data.Fin
import Data.List
import Data.String
import Data.Vect
import System
import System.File

----------------------------------------------------------------------
-- Errors
----------------------------------------------------------------------

public export
data TokError : Type where
  ||| The probed on-disk vocab disagrees with the type-level `vocab` Nat.
  ||| Caller probably passed the wrong size for the model's embedding table.
  TokVocabMismatch  : (claimed : Nat) -> (onDisk : Nat) -> TokError
  ||| Subprocess returned a non-zero exit code.
  TokSubprocessFail : (cmd : String) -> (rc : Int) -> TokError
  ||| Could not parse a number we expected as an int.
  TokParseFail      : (raw : String) -> TokError
  ||| A token ID landed outside `Fin vocab`. Tokenizer's vocab is bigger
  ||| than the model claims.
  TokIdOutOfRange   : (id : Nat) -> (vocab : Nat) -> TokError
  ||| Reading the captured subprocess stdout failed.
  TokReadFail       : (path : String) -> TokError

export
Show TokError where
  show (TokVocabMismatch claimed onDisk) =
    "Tokenizer vocab mismatch: claimed " ++ show claimed ++
    " but on-disk reports " ++ show onDisk
  show (TokSubprocessFail cmd rc) =
    "Tokenizer subprocess failed (rc=" ++ show rc ++ "): " ++ cmd
  show (TokParseFail raw) =
    "Tokenizer parse failure on: " ++ show raw
  show (TokIdOutOfRange i v) =
    "Token ID " ++ show i ++ " >= vocab " ++ show v
  show (TokReadFail path) =
    "Failed to read tokenizer subprocess output: " ++ path

----------------------------------------------------------------------
-- Tokenizer handle
----------------------------------------------------------------------

||| A tokenizer constructed against the HF repo `repo`, with vocab size
||| pinned at the type level. The only way to build one is `mkTokenizer`,
||| which validates the type-level `vocab` against the on-disk vocab.
public export
data Tokenizer : (vocab : Nat) -> Type where
  MkTokenizer : (repo : String) -> Tokenizer vocab

----------------------------------------------------------------------
-- Subprocess plumbing
----------------------------------------------------------------------

-- Deterministic temp paths. Examples run single-threaded — concurrent
-- use is an explicit non-goal for v1. If/when it matters, switch to
-- `mkstemp`-style unique paths via a getPID + counter combination.
tmpIn : String
tmpIn = "/tmp/idris-hf-tokenize-in.txt"

tmpOut : String
tmpOut = "/tmp/idris-hf-tokenize-out.txt"

tmpErr : String
tmpErr = "/tmp/idris-hf-tokenize-err.txt"

-- The script lives under packages/idris-transformers/scripts/.
-- Examples run from the repo root so this relative path resolves.
scriptPath : String
scriptPath = "packages/idris-transformers/scripts/hf_tokenize.py"

-- Wrap the script invocation in `cd packages/pytorch && uv run python …`
-- so the HF transformers dep (managed by the pytorch package's uv venv)
-- is on PYTHONPATH. Mirrors how save_oracle.py is invoked from the
-- existing Makefile targets.
buildCmd : (repo : String) -> (mode : String) -> (extraArgs : String) -> String
buildCmd repo mode extraArgs =
  "cd packages/pytorch && uv run python ../idris-transformers/scripts/hf_tokenize.py " ++
  repo ++ " " ++ mode ++ " " ++ extraArgs ++
  " > " ++ tmpOut ++ " 2> " ++ tmpErr

-- Run a command via System.system, return Either TokError String holding
-- whatever landed on stdout. On failure, the captured stderr is folded
-- into the TokSubprocessFail message — previously discarded via
-- 2>/dev/null, which ate Python's actual exception text and made tokenizer
-- failures opaque ("rc=1" with no clue why).
runCapture : (cmd : String) -> IO (Either TokError String)
runCapture cmd = do
  rc <- system cmd
  if rc /= 0
    then do
      errText <- readFile tmpErr
      let stderrSnippet = case errText of
            Right s => "\n  stderr:\n" ++ s
            Left _  => ""
      pure (Left (TokSubprocessFail (cmd ++ stderrSnippet) rc))
    else do
      r <- readFile tmpOut
      case r of
        Left _   => pure (Left (TokReadFail tmpOut))
        Right ok => pure (Right ok)

----------------------------------------------------------------------
-- Vocab probe + mkTokenizer
----------------------------------------------------------------------

-- Probe the tokenizer's on-disk vocab size by running the `vocab`
-- subcommand and parsing the integer that comes back.
probeVocab : (repo : String) -> IO (Either TokError Nat)
probeVocab repo = do
  let cmd = buildCmd repo "vocab" ""
  r <- runCapture cmd
  case r of
    Left err  => pure (Left err)
    Right raw => case parsePositive (trim raw) of
      Nothing => pure (Left (TokParseFail raw))
      Just n  => pure (Right n)

||| Construct a tokenizer + validate its vocab against the type-level
||| Nat. Surfaces vocab mismatch as `Left TokVocabMismatch`; surfaces
||| subprocess / parse failures as the corresponding `TokError`.
|||
||| Example:
|||   Right bertTok <- mkTokenizer "google/bert_uncased_L-2_H-128_A-2" 30522
|||     | Left err => panic ("bert tokenizer setup: " ++ show err)
public export
mkTokenizer : (repo : String) -> (vocab : Nat) -> IO (Either TokError (Tokenizer vocab))
mkTokenizer repo vocab = do
  r <- probeVocab repo
  case r of
    Left err     => pure (Left err)
    Right onDisk =>
      if onDisk == vocab
        then pure (Right (MkTokenizer repo))
        else pure (Left (TokVocabMismatch vocab onDisk))

----------------------------------------------------------------------
-- encode / decode
----------------------------------------------------------------------

-- Parse a space-separated list of integers, lifting each to `Fin vocab`.
-- Fails on first parse / out-of-range.
parseIds : (vocab : Nat) -> String -> Either TokError (List (Fin vocab))
parseIds vocab raw =
  let pieces = filter (\s => not (s == "")) (words (trim raw))
  in traverse parseOne pieces
  where
    parseOne : String -> Either TokError (Fin vocab)
    parseOne s = case parsePositive s of
      Nothing => Left (TokParseFail s)
      Just n  => case natToFin n vocab of
        Nothing => Left (TokIdOutOfRange n vocab)
        Just f  => Right f

-- Build an existential `(n ** Vect n a)` from a List a — the dependent
-- pair binds `n` to the list's length and uses Vect.fromList.
toExistentialVect : (xs : List a) -> (n ** Vect n a)
toExistentialVect xs = (length xs ** fromList xs)

||| Encode a string to a Vect of statically-bounded IDs. The length `n`
||| is runtime-determined but type-tracked; each ID is `Fin vocab`.
|||
||| Returns `Left TokError` if the subprocess fails or any ID is out of
||| range. The latter only fires if the Python tokenizer's vocab grew
||| past the model's claimed size (i.e. somebody bumped the tokenizer
||| but not the embedding table). With `mkTokenizer` already validating
||| vocab on construction, this should be impossible — but the typed
||| boundary catches it if it happens.
public export
tokenize : {vocab : Nat}
        -> Tokenizer vocab
        -> (text : String)
        -> IO (Either TokError (n ** Vect n (Fin vocab)))
tokenize (MkTokenizer repo) text = do
  Right () <- writeFile tmpIn text
    | Left _ => pure (Left (TokReadFail tmpIn))
  let cmd = buildCmd repo "encode" ("--input-file " ++ tmpIn)
  r <- runCapture cmd
  case r of
    Left err  => pure (Left err)
    Right raw => case parseIds vocab raw of
      Left err  => pure (Left err)
      Right ids => pure (Right (toExistentialVect ids))

||| Decode a Vect of token IDs to a string. Round-trips through the
||| HF tokenizer's `decode` (which un-merges BPE / WordPiece pieces and
||| handles the `Ġ` / `▁` whitespace conventions correctly).
public export
detokenize : Tokenizer vocab -> Vect n (Fin vocab) -> IO (Either TokError String)
detokenize (MkTokenizer repo) ids = do
  let idStrs = map (show . finToNat) (toList ids)
      payload = unwords idStrs
  Right () <- writeFile tmpIn payload
    | Left _ => pure (Left (TokReadFail tmpIn))
  let cmd = buildCmd repo "decode" ("--input-file " ++ tmpIn)
  runCapture cmd
