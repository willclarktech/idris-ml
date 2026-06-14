module Test.Args

import Data.String
import Test.Harness
import Args

record Cfg where
  constructor MkCfg
  epochs : Nat
  lr     : Double
  seed   : Bits64
  resume : Bool
  device : String

defaults : Cfg
defaults = MkCfg 100 0.01 42 False "cpu"

flags : List (Flag Cfg)
flags =
  [ option "epochs" "N" "number of training epochs" natArg (\n => { epochs := n })
  , option "lr" "X" "learning rate" doubleArg (\x => { lr := x })
  , option "seed" "S" "rng seed" bits64Arg (\s => { seed := s })
  , option "device" "DEV" "execution device"
      (enumArg [("cpu", "cpu"), ("gpu", "gpu")]) (\d => { device := d })
  , switch "resume" "resume from last checkpoint" ({ resume := True })
  ]

parsed : List String -> Maybe (Cfg, List String)
parsed args = case parseArgs "prog" flags defaults args of
  Parsed c pos => Just (c, pos)
  _            => Nothing

errOf : List String -> Maybe String
errOf args = case parseArgs "prog" flags defaults args of
  ParseError e => Just e
  _            => Nothing

helpOf : List String -> Maybe String
helpOf args = case parseArgs "prog" flags defaults args of
  ShowHelp t => Just t
  _          => Nothing

errMentions : List String -> List String -> Bool
errMentions args needles = case errOf args of
  Just e  => all (\n => isInfixOf n e) needles
  Nothing => False

export
tests : List (IO Bool)
tests =
  [ check "empty args keep defaults, no positionals" $
      case parsed [] of
        Just (c, []) => c.epochs == 100 && c.lr == 0.01 && not c.resume
        _            => False
  , check "space-form option (--epochs 500)" $
      case parsed ["--epochs", "500"] of
        Just (c, []) => c.epochs == 500
        _            => False
  , check "equals-form option (--lr=0.5)" $
      case parsed ["--lr=0.5"] of
        Just (c, []) => c.lr == 0.5
        _            => False
  , check "switch (--resume)" $
      case parsed ["--resume"] of
        Just (c, []) => c.resume
        _            => False
  , check "later flag overrides earlier" $
      case parsed ["--epochs", "1", "--epochs", "2"] of
        Just (c, []) => c.epochs == 2
        _            => False
  , check "bits64 value (--seed 7)" $
      case parsed ["--seed", "7"] of
        Just (c, []) => c.seed == 7
        _            => False
  , check "enum accepts listed value (--device gpu)" $
      case parsed ["--device", "gpu"] of
        Just (c, []) => c.device == "gpu"
        _            => False
  , check "positionals collected in order" $
      case parsed ["alpha", "--epochs", "5", "beta"] of
        Just (c, pos) => c.epochs == 5 && pos == ["alpha", "beta"]
        _             => False
  , check "-- terminates flag parsing" $
      case parsed ["--", "--epochs"] of
        Just (c, pos) => c.epochs == 100 && pos == ["--epochs"]
        _             => False
  , check "bare - is a positional" $
      case parsed ["-"] of
        Just (c, pos) => pos == ["-"]
        _             => False
  , check "unknown long flag is an error naming the flag" $
      errMentions ["--bogus"] ["--bogus"]
  , check "unknown short flag is an error" $
      errMentions ["-x"] ["-x"]
  , check "bad value names flag and value" $
      errMentions ["--epochs", "abc"] ["--epochs", "abc"]
  , check "negative rejected by natArg" $
      errMentions ["--epochs", "-5"] ["--epochs", "-5"]
  , check "missing value is an error" $
      errMentions ["--epochs"] ["--epochs"]
  , check "switch with =value is an error" $
      errMentions ["--resume=yes"] ["--resume"]
  , check "enum rejects unlisted value, lists choices" $
      errMentions ["--device", "tpu"] ["--device", "tpu", "cpu", "gpu"]
  , check "--help renders flags, metavars, prog name" $
      case helpOf ["--help"] of
        Just t  => isInfixOf "--epochs" t && isInfixOf "N" t
                && isInfixOf "--resume" t && isInfixOf "prog" t
        Nothing => False
  , check "-h is --help" $
      case helpOf ["--seed", "9", "-h"] of
        Just t  => isInfixOf "--help" t
        Nothing => False
  ]
