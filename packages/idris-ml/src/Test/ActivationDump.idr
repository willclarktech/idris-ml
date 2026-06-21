module Test.ActivationDump

import Data.List
import Data.String
import Data.Vect
import System
import System.Directory
import System.File

import Executor
import Tensor
import Test.Config
import Test.Harness
import Util.Log

-- Tests for the activation-dump machinery — the C-side
-- `param_save_by_name` flush the TRACE log level uses to dump
-- per-layer activations. Tests the
-- load-bearing registry round-trip directly so coverage is
-- independent of `IDRISML_LOG_LEVEL` (which can only be raised to
-- TRACE in a build with `IDRISML_LOG=trace` set at make time, and
-- the test process inherits the default INFO ceiling).
--
-- End-to-end "TRACE actually flips behavior" coverage lives in
-- the perf-run gate that runs `IDRISML_LOG_LEVEL=trace make
-- example-mnist` and inspects the produced files; see Commit 3.

-- Use the host's `stat` via System.system: the simplest portable
-- "file exists with non-trivial size" check that doesn't depend on
-- fGetChars's binary-safety quirks. `stat -f %z` (BSD/macOS) returns
-- size; non-existent file yields non-zero exit.
fileExistsAndNonEmpty : String -> IO Bool
fileExistsAndNonEmpty path = do
  Right f <- openFile path Read | Left _ => pure False
  closeFile f
  -- File opened — exists. Now check size > 8 via stat.
  rc <- system ("test $(wc -c < " ++ path ++ ") -gt 8")
  pure (rc == 0)

mechanicsRoundTrip : IO Bool
mechanicsRoundTrip = do
  let dir  = "/tmp/idrisml-act-test"
      path = dir ++ "/mech.safetensors"
  _ <- createDir dir   -- ignore "already exists"
  -- Register two wrap-on-return scalar handles under synthetic
  -- activation names. The Scheme-side FFI template wraps + retains
  -- on return, so the AnyPtr is registry-safe.
  let v1 = primCreateScalar {ex=TestExecutor} 1.5 0
      v2 = primCreateScalar {ex=TestExecutor} 2.5 0
  _ <- ioRerun (\_ => primParamRegister {ex=TestExecutor} "__act/mech/0" v1)
  _ <- ioRerun (\_ => primParamRegister {ex=TestExecutor} "__act/mech/1" v2)
  -- Flush to disk via the same `param_save_by_name` primitive the dump uses.
  let names = "__act/mech/0\n__act/mech/1\n"
  rc <- primIO (primParamSaveByName {ex=TestExecutor} path names 2)
  -- Erase the synthetic entries (the TRACE-branch cleanup step).
  primIO (primParamEraseByPrefix {ex=TestExecutor} "__act/mech/")
  exists <- fileExistsAndNonEmpty path
  check ("mechanics: register-save-erase produced "
          ++ path
          ++ " (rc=" ++ show rc ++ ", non-empty=" ++ show exists ++ ")")
        (rc == 0 && exists)

export
tests : List (IO Bool)
tests =
  [ mechanicsRoundTrip
  ]
