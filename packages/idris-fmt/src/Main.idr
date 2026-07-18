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

import Args

import Format.Ipkg
import Format.Render
import Format.Roundtrip

||| What to do with each input file. (Prefixed to dodge name clashes
||| with the idris2 compiler API, which is in scope via Format.*.)
data FmtMode = FStdout | FWrite | FCheck | FParse

||| CLI flags, parsed by the repo's own idris-args package. The config is
||| just the mode; long-form switches select it (last wins) and
||| positionals are the files. `--help`/`-h` come from idris-args. The
||| import grouping's local tier is derived per file from the owning ipkg
||| (`Format.Ipkg`), so there is nothing to configure.
flags : List (Flag FmtMode)
flags =
  [ switch "write" "format files in place" (const FWrite)
  , switch "check" "exit 1 if any file is not formatted" (const FCheck)
  , switch "parse-check" "parse every file, no formatting" (const FParse)
  ]

noFilesHint : String
noFilesHint = "idris-fmt: no input files (try --help)"

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
  locals <- localModulesFor fn
  if isFormattedWith locals s
    then pure True
    else do putStrLn ("would reformat: " ++ fn); pure False

||| --write: format in place; True on success.
writeOne : String -> IO Bool
writeOne fn = do
  Just s <- slurp fn | Nothing => pure False
  locals <- localModulesFor fn
  let out = formatWith locals s
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
  locals <- localModulesFor fn
  putStr (formatWith locals s)
  pure True

||| --parse-check: parse only (coverage probe). True iff it parses.
parseOne : String -> IO Bool
parseOne fn = do
  Just s <- slurp fn | Nothing => pure False
  if parses s then pure True else do putStrLn (fn ++ ": PARSE FAIL"); pure False

handler : FmtMode -> (String -> IO Bool)
handler FStdout = stdoutOne
handler FWrite  = writeOne
handler FCheck  = checkOne
handler FParse  = parseOne

runMode : (String -> IO Bool) -> List String -> IO ()
runMode _ [] = putStrLn noFilesHint
runMode f fs = do
  oks <- traverse f fs
  if all id oks then exitSuccess else exitFailure

main : IO ()
main = do
  args <- drop 1 <$> getArgs
  case parseArgs "idris-fmt" flags FStdout args of
    ShowHelp txt      => putStr txt
    ParseError e      => do putStrLn e; exitFailure
    Parsed mode files => runMode (handler mode) files
