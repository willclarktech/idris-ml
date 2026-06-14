||| idris-fmt: a compiler-native Idris 2 source formatter.
|||
||| It parses with the *compiler's own* parser (`Idris.Parser`) and gates
||| every reformat behind the round-trip oracle in `Format.Roundtrip`, so it
||| can never emit code that differs in meaning from its input.
module Main

import Data.List
import Data.String
import System
import System.File

import Format.Render
import Format.Roundtrip

usage : String
usage = unlines
  [ "idris-fmt - Idris 2 source formatter"
  , ""
  , "usage:"
  , "  idris-fmt FILE...             format to stdout"
  , "  idris-fmt -w|--write FILE...  format files in place"
  , "  idris-fmt -c|--check FILE...  exit 1 if any file is not formatted"
  , "  idris-fmt --parse-check FILE...  parse every file (no formatting)"
  ]

||| Read a file; on error print a message and yield Nothing.
slurp : String -> IO (Maybe String)
slurp fn = do
  Right s <- readFile fn
    | Left e => do putStrLn (fn ++ ": read error: " ++ show e); pure Nothing
  pure (Just s)

||| --check: True iff already formatted.
checkOne : String -> IO Bool
checkOne fn = do
  Just s <- slurp fn | Nothing => pure False
  if isFormatted s
    then pure True
    else do putStrLn ("would reformat: " ++ fn); pure False

||| --write: format in place; True on success.
writeOne : String -> IO Bool
writeOne fn = do
  Just s <- slurp fn | Nothing => pure False
  let out = format s
  if out == s
    then pure True
    else do
      Right () <- writeFile fn out
        | Left e => do putStrLn (fn ++ ": write error: " ++ show e); pure False
      putStrLn ("formatted: " ++ fn)
      pure True

||| default: format to stdout.
stdoutOne : String -> IO Bool
stdoutOne fn = do
  Just s <- slurp fn | Nothing => pure False
  putStr (format s)
  pure True

||| --parse-check: parse only (coverage probe). True iff it parses.
parseOne : String -> IO Bool
parseOne fn = do
  Just s <- slurp fn | Nothing => pure False
  if parses s then pure True else do putStrLn (fn ++ ": PARSE FAIL"); pure False

runMode : (String -> IO Bool) -> List String -> IO ()
runMode _ [] = putStrLn usage
runMode f fs = do
  oks <- traverse f fs
  if all id oks then exitSuccess else exitFailure

main : IO ()
main = do
  args <- drop 1 <$> getArgs
  case args of
    [] => putStrLn usage
    ("-w" :: fs) => runMode writeOne fs
    ("--write" :: fs) => runMode writeOne fs
    ("-c" :: fs) => runMode checkOne fs
    ("--check" :: fs) => runMode checkOne fs
    ("--parse-check" :: fs) => runMode parseOne fs
    fs => runMode stdoutOne fs
