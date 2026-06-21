# idris-args

Typed command-line flag parsing for Idris 2, with zero dependencies beyond `base`. Long-form
flags (`--name value` or `--name=value`), typed readers that drive the setter's type, a built-in
`--help`/`-h`, and a `--` positional terminator. Unknown or unparseable flags are hard errors,
never silently skipped.

## Usage

Describe your config as a record, list the flags as updates to it, and call `getOpts`:

```idris
import Args

record Config where
  constructor MkConfig
  epochs : Nat
  lr     : Double
  quiet  : Bool

flags : List (Flag Config)
flags =
  [ option "epochs" "N"   "training epochs" natArg    (\v, c => { epochs := v } c)
  , option "lr"     "F"   "learning rate"   doubleArg (\v, c => { lr     := v } c)
  , switch "quiet"        "suppress logging"           ({ quiet := True })
  ]

main : IO ()
main = do
  (cfg, positionals) <- getOpts flags (MkConfig 100 0.01 False)
  ...
```

`getOpts` reads `argv`, applies the parsed flags to the defaults, prints usage and exits 0 on
`--help`, or prints the error to stderr and exits 1 on a bad flag.

## API

- **Readers** (`Reader a = String -> Either String a`): `natArg`, `integerArg`, `bits64Arg`,
  `doubleArg`, `stringArg`, and `enumArg [("name", value), …]` for closed sets. A reader's result
  type fixes the setter's type, so there's no boilerplate cast at the call site.
- **Flag specs**: `option long metavar help reader setter` (takes a value) and
  `switch long help setter` (boolean).
- **Entry points**: `getOpts flags defaults : IO (cfg, List String)` (the high-level driver),
  `parseArgs prog flags defaults args : ParseResult cfg` (pure), and `usage prog flags : String`.

Used throughout the repo for example CLIs (`--epochs` / `--lr` / `--seed`) and by
[idris-fmt](../idris-fmt/)'s command line.
