||| idris-fmt: a compiler-native Idris 2 source formatter.
|||
||| Current entry point is the parse-coverage probe: parse every .idr path
||| passed on argv with the *compiler's own* parser (`Idris.Parser.prog`) and
||| report which files fail. This proves the parser covers 100% of a target
||| codebase's surface syntax (multi-kB FFI strings, long dependent type
||| sigs, `:=` record updates, `do`/`!` sugar, ...) before any reformatting
||| is attempted — and doubles as a cheap CI sanity gate.
|||
||| The formatting pipeline (declaration-layout printer + comment
||| re-association + style) is built on top of this same parse step.
module Main

import Core.Core
import Core.FC
import Parser.Source
import Idris.Parser
import Idris.Syntax

import Data.List
import Data.String
import System
import System.File

||| Parse one file. Returns Nothing on success, Just <message> on failure.
parseOne : String -> IO (Maybe String)
parseOne fname = do
  Right str <- readFile fname
    | Left err => pure (Just (fname ++ ": read error: " ++ show err))
  let origin = Virtual Interactive
  case runParser origin Nothing str (prog origin) of
    Left err => pure (Just (fname ++ ": PARSE FAIL: " ++ show err))
    Right (_, _, _) => pure Nothing

report : (numFiles : Nat) -> (failures : List String) -> IO ()
report numFiles failures = do
  traverse_ putStrLn failures
  let nfail = length failures
  putStrLn "----------------------------------------------------------------------"
  putStrLn $ "parsed " ++ show (numFiles `minus` nfail) ++ "/" ++ show numFiles
              ++ " files; " ++ show nfail ++ " failure(s)"
  if nfail == 0 then exitSuccess else exitFailure

run : List String -> IO ()
run files = do
  results <- traverse parseOne files
  report (length files) (mapMaybe id results)

main : IO ()
main = do
  files <- drop 1 <$> getArgs
  if null files
    then putStrLn "usage: idris-fmt <file.idr> [<file.idr> ...]  (parse-coverage probe)"
    else run files
