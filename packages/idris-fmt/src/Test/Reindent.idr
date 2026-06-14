module Test.Reindent

import Format.Reindent
import Format.Roundtrip
import Test.Harness

-- Canonical layout: own-line `where` at parent+2, where-items at parent+4.
wb : String
wb = "module M\n\nf : Nat\nf = g\n  where\n    g : Nat\n    g = 1\n"

export
tests : List (IO Bool)
tests =
  [ check "reindents an over-indented where-block to 4 spaces" $
      reindent "module M\n\nf : Nat\nf = g\n  where\n      g : Nat\n      g = 1\n" == wb
  , check "reindents over-indented data constructors to 2 spaces" $
      reindent "module M\n\ndata D : Type where\n    A : D\n    B : D\n"
        == "module M\n\ndata D : Type where\n  A : D\n  B : D\n"
  , check "already-canonical source is a fixed point" $
      reindent wb == wb
  , check "leaves a parse failure untouched" $
      reindent "module M\n\nwhere where\n" == "module M\n\nwhere where\n"
  , check "reindent output passes the safeReindent oracle" $
      let messy = "module M\n\nf : Nat\nf = g\n  where\n      g : Nat\n      g = 1\n"
      in safeReindent messy (reindent messy)
  ]
