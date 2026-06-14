module Test.Align

import Format.Align
import Format.Render
import Format.Roundtrip
import Test.Harness

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
  -- equals alignment: indented binding group (let/where)
  , check "aligns `=` in an indented binding group" $
      alignEquals "  a = 1\n  bb = 2\n" == "  a  = 1\n  bb = 2\n"
  -- equals alignment: top-level multi-clause (shared LHS head)
  , check "aligns `=` across top-level multi-clause def (shared head)" $
      alignEquals "f Zero = 0\nf (S n) = n\n" == "f Zero  = 0\nf (S n) = n\n"
  -- equals alignment: unrelated top-level defs are left alone (no churn)
  , check "leaves unrelated top-level def `=` untouched" $
      alignEquals "foo = 1\nbarbaz = 2\n" == "foo = 1\nbarbaz = 2\n"
  -- equals alignment: never matches `==` / `=>` / `:=`
  , check "does not align `==` as `=`" $
      alignEquals "  x == y\n  zzz == w\n" == "  x == y\n  zzz == w\n"
  -- arrows alignment: case / with arms
  , check "aligns `=>` across case arms" $
      alignArrows "  Zero => z\n  S n => s\n" == "  Zero => z\n  S n  => s\n"
  ]
