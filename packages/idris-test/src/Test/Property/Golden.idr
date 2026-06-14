-- Test.Property.Golden — fixture-based output assertions.
--
-- Pairs an `IO String` (the test's actual output) with a path to a
-- fixture file containing the expected output. Returns True iff
-- the two strings match byte-for-byte.
--
-- Re-baseline mode: set GOLDEN_UPDATE=1 in the env and the test
-- rewrites the fixture instead of asserting. Useful when the
-- expected output legitimately changed (rendered docs format
-- change, schema dump format change, …) and you want to bulk-
-- update fixtures rather than hand-edit each.
--
-- This is the in-Idris primitive. For shell-test-style goldens
-- (where a `run.sh` produces the output and an `expected` file
-- pairs with it), contrib's Test.Golden module (depend on `test`)
-- is the canonical mechanism — both coexist; pick the one whose
-- shape fits the test.
--
-- Why not extend Test.Harness's `check`: this primitive needs file
-- I/O and an env-var check, both of which are heavier than the
-- pure `String -> Bool -> IO Bool` of `check`. Keeping them
-- separate keeps Test.Harness lightweight.
module Test.Property.Golden

import Data.String
import System
import System.File

import Test.Harness

-- Not marked total: readFile transitively uses Data.Fuel.forever
-- (file I/O is the canonical "unbounded" computation). The function
-- terminates in practice for any real fixture; the totality marker
-- would be a lie.

||| Assert that running `action` produces exactly the contents of
||| the file at `fixturePath`. With `GOLDEN_UPDATE=1` in the env,
||| rewrites the fixture instead of asserting (the test always
||| passes in update mode).
|||
||| Returns True iff (a) we're in update mode and the file was
||| written, or (b) the file's contents match the action's output.
export
checkGolden : String -> String -> IO String -> IO Bool
checkGolden name fixturePath action = do
  actual <- action
  update <- getEnv "GOLDEN_UPDATE"
  case update of
    Just "1" => do
      Right () <- writeFile fixturePath actual
        | Left err => do
            putStrLn ("  FAIL: " ++ name ++ " — couldn't write fixture: " ++ show err)
            pure False
      putStrLn ("  UPDATE: " ++ name ++ " (rewrote " ++ fixturePath ++ ")")
      pure True
    _ => do
      contents <- readFile fixturePath
      case contents of
        Left err => do
          putStrLn ("  FAIL: " ++ name ++ " — couldn't read fixture " ++ fixturePath
                  ++ ": " ++ show err)
          pure False
        Right expected =>
          if expected == actual
            then check name True
            else do
              putStrLn ("  FAIL: " ++ name ++ " — output differs from fixture")
              putStrLn ("    fixture: " ++ fixturePath)
              putStrLn ("    actual length: " ++ show (length actual))
              putStrLn ("    expected length: " ++ show (length expected))
              putStrLn ("    Re-baseline: GOLDEN_UPDATE=1 make ...")
              pure False
