module Test.Roundtrip

import Test.Harness
import Format.Roundtrip

-- A small but real module used as a parse/round-trip fixture.
clean : String
clean = "module M\n\nfoo : Nat\nfoo = 1\n"

export
tests : List (IO Bool)
tests =
  [ check "codeSig ignores whitespace" $
      codeSig "foo  =   1" == codeSig "foo = 1"
  , check "codeSig distinguishes tokens" $
      codeSig "foo = 1" /= codeSig "foo = 2"
  , check "codeSig is Just for lexable input" $
      case codeSig clean of
        Just _ => True
        Nothing => False
  , check "safeReformat accepts a pure-whitespace reflow" $
      safeReformat clean "module M\nfoo : Nat\nfoo = 1\n"
  , check "safeReformat rejects a changed token" $
      not (safeReformat clean "module M\n\nfoo : Nat\nfoo = 2\n")
  , check "safeReformat rejects output that does not parse" $
      not (safeReformat clean "where\n")
  , check "parses accepts a real module" $
      parses clean
  , check "parses rejects garbage" $
      not (parses "where where where\n")
  ]
