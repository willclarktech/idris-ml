module Test.Render

import Test.Harness
import Format.Render
import Format.Roundtrip

clean : String
clean = "module M\n\nfoo : Nat\nfoo = 1\n"

export
tests : List (IO Bool)
tests =
  [ check "strips trailing whitespace" $
      format "module M\n\nfoo : Nat   \nfoo = 1\n" == clean
  , check "collapses runs of blank lines" $
      format "module M\n\n\n\nfoo : Nat\nfoo = 1\n" == clean
  , check "adds a missing trailing newline" $
      format "module M\n\nfoo : Nat\nfoo = 1" == clean
  , check "trims leading and trailing blank lines" $
      format "\n\nmodule M\n\nfoo : Nat\nfoo = 1\n\n\n" == clean
  , check "already-formatted source is a fixed point" $
      isFormatted clean
  , check "format is idempotent" $
      format (format "module M\n\n\nfoo : Nat  \nfoo = 1") ==
        format "module M\n\n\nfoo : Nat  \nfoo = 1"
  , check "format preserves code (round-trip safe)" $
      safeReformat clean (format clean)
  , check "messy-but-valid source still round-trips" $
      let messy = "module M\n\n\nfoo : Nat    \n\n\nfoo = 1\n\n"
      in safeReformat messy (format messy)
  ]
