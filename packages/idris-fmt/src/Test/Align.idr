module Test.Align

import Test.Harness
import Format.Align
import Format.Render
import Format.Roundtrip

recIn : String
recIn = "module M\n\nrecord R where\n  constructor MkR\n  a : Int\n  bb : Int\n"

recOut : String
recOut = "module M\n\nrecord R where\n  constructor MkR\n  a  : Int\n  bb : Int\n"

dataIn : String
dataIn = "module M\n\ndata D : Type where\n  A : D\n  Bee : Int -> D\n"

dataOut : String
dataOut = "module M\n\ndata D : Type where\n  A   : D\n  Bee : Int -> D\n"

export
tests : List (IO Bool)
tests =
  [ check "aligns record-field colons (excludes constructor line)" $
      format recIn == recOut
  , check "aligns data-constructor colons" $
      format dataIn == dataOut
  , check "alignment is a fixed point" $
      format recOut == recOut
  , check "single annotation line is left untouched" $
      alignColons "foo : Nat\nfoo = 1\n" == "foo : Nat\nfoo = 1\n"
  , check "does not touch `:` inside an expression / ::" $
      alignColons "xs : List Nat\nys = 1 :: 2 :: []\n" ==
        "xs : List Nat\nys = 1 :: 2 :: []\n"
  , check "alignment output passes the round-trip oracle" $
      safeReformat recIn (format recIn)
  ]
