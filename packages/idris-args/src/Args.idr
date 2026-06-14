||| Typed CLI flag parsing with zero dependencies beyond base.
|||
||| General-purpose: no ML, no backend, no contrib. Flags are
||| long-form only (`--epochs 500` or `--epochs=500`); `--help`/`-h`
||| and the `--` positional terminator are built in. Unknown flags
||| and unparseable values are errors, not skips — a typo'd flag must
||| never be silently ignored.
module Args

import Data.List
import Data.Maybe
import Data.String
import System
import System.File

----------------------------------------------------------------------
-- Readers
----------------------------------------------------------------------

||| Parse one raw flag value. Left carries a human-readable reason.
public export
0 Reader : Type -> Type
Reader a = String -> Either String a

export
natArg : Reader Nat
natArg s = case parsePositive s of
  Just n  => Right n
  Nothing => Left ("expected a natural number, got \"" ++ s ++ "\"")

export
integerArg : Reader Integer
integerArg s = case parseInteger s of
  Just n  => Right n
  Nothing => Left ("expected an integer, got \"" ++ s ++ "\"")

export
bits64Arg : Reader Bits64
bits64Arg s = case parseInteger {a = Integer} s of
  Just n  => if n < 0
               then Left ("expected a non-negative integer, got \"" ++ s ++ "\"")
               else Right (cast n)
  Nothing => Left ("expected a non-negative integer, got \"" ++ s ++ "\"")

export
doubleArg : Reader Double
doubleArg s = case parseDouble s of
  Just x  => Right x
  Nothing => Left ("expected a number, got \"" ++ s ++ "\"")

export
stringArg : Reader String
stringArg = Right

||| Reader from a fixed table of admissible spellings.
export
enumArg : List (String, a) -> Reader a
enumArg table s = case lookup s table of
  Just v  => Right v
  Nothing => Left ("expected one of "
                ++ concat (intersperse "|" (map fst table))
                ++ ", got \"" ++ s ++ "\"")

----------------------------------------------------------------------
-- Flag specifications
----------------------------------------------------------------------

public export
data Flag : Type -> Type where
  Option : (long : String) -> (metavar : String) -> (help : String)
        -> (parse : String -> Either String (cfg -> cfg)) -> Flag cfg
  Switch : (long : String) -> (help : String) -> (set : cfg -> cfg) -> Flag cfg

||| Value-taking flag from a typed reader and a setter.
export
option : (long : String) -> (metavar : String) -> (help : String)
      -> Reader b -> (b -> cfg -> cfg) -> Flag cfg
option long metavar help rd set = Option long metavar help (\s => map set (rd s))

||| Boolean flag: presence applies the update.
export
switch : (long : String) -> (help : String) -> (cfg -> cfg) -> Flag cfg
switch = Switch

longOf : Flag cfg -> String
longOf (Option long _ _ _) = long
longOf (Switch long _ _)   = long

findFlag : String -> List (Flag cfg) -> Maybe (Flag cfg)
findFlag name = find (\f => longOf f == name)

----------------------------------------------------------------------
-- Help text
----------------------------------------------------------------------

export
usage : (prog : String) -> List (Flag cfg) -> String
usage prog flags =
  let rows = map row flags ++ [("--help, -h", "show this help and exit")]
      w    = foldl (\acc, r => max acc (length (fst r))) 0 rows
      line = \r => "  " ++ padRight w ' ' (fst r) ++ "  " ++ snd r
  in "Usage: " ++ prog ++ " [options]\n\nOptions:\n"
  ++ unlines (map line rows)
  where
    row : Flag cfg -> (String, String)
    row (Option long metavar help _) = ("--" ++ long ++ " " ++ metavar, help)
    row (Switch long help _)         = ("--" ++ long, help)

----------------------------------------------------------------------
-- Parsing
----------------------------------------------------------------------

public export
data ParseResult cfg
  = Parsed cfg (List String)
  | ShowHelp String
  | ParseError String

||| Split a `--name[=value]` token into name and optional value.
splitFlag : String -> (String, Maybe String)
splitFlag t =
  let body = unpack (substr 2 (length t) t)
      (n, v) = break (== '=') body
  in case v of
       ('=' :: vs) => (pack n, Just (pack vs))
       _           => (pack n, Nothing)

||| Parse argv (without the program name) against the flag specs.
||| Non-flag tokens are collected as positionals in order; `--`
||| terminates flag parsing.
export
parseArgs : (prog : String) -> List (Flag cfg) -> cfg -> List String -> ParseResult cfg
parseArgs prog flags def args = go def [] args
  where
    bad : String -> ParseResult cfg
    bad msg = ParseError (prog ++ ": " ++ msg ++ " (try --help)")

    go : cfg -> List String -> List String -> ParseResult cfg
    go c pos []             = Parsed c (reverse pos)
    go c pos ("--" :: rest) = Parsed c (reverse pos ++ rest)
    go c pos (t :: rest)    =
      if t == "--help" || t == "-h" then ShowHelp (usage prog flags)
      else if isPrefixOf "--" t then
        let (name, eq) = splitFlag t in
        case findFlag name flags of
          Nothing               => bad ("unknown flag --" ++ name)
          Just (Switch _ _ set) => case eq of
            Just _  => bad ("flag --" ++ name ++ " does not take a value")
            Nothing => go (set c) pos rest
          Just (Option _ metavar _ parse) => case eq of
            Just v  => case parse v of
              Left err => bad ("invalid value for --" ++ name ++ ": " ++ err)
              Right f  => go (f c) pos rest
            Nothing => case rest of
              []           => bad ("flag --" ++ name ++ " requires a value " ++ metavar)
              (v :: rest') => case parse v of
                Left err => bad ("invalid value for --" ++ name ++ ": " ++ err)
                Right f  => go (f c) pos rest'
      else if t /= "-" && isPrefixOf "-" t then bad ("unknown flag " ++ t)
      else go c (t :: pos) rest

||| Parse the program's own argv. On `--help`: print usage, exit 0.
||| On error: print to stderr, exit 1. Returns (config, positionals).
export
getOpts : List (Flag cfg) -> cfg -> IO (cfg, List String)
getOpts flags def = do
  args <- getArgs
  let prog = fromMaybe "program" (head' args)
  let rest = drop 1 args
  case parseArgs prog flags def rest of
    Parsed c pos => pure (c, pos)
    ShowHelp txt => do putStr txt; exitSuccess
    ParseError e => do ignore (fPutStrLn stderr e); exitWith (ExitFailure 1)
