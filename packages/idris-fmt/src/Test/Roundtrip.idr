module Test.Roundtrip

import Format.Roundtrip
import Test.Harness

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
        Just _  => True
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
  -- AST-equivalence oracle (astSig: FC-insensitive decl signature)
  , check "astSig ignores whitespace + blank lines" $
      astSig "module M\n\nf : Nat\nf = 1\n" ==
        astSig "module M\n\n\nf  :  Nat\nf   =   1\n"
  , check "astSig is structure-sensitive (changed RHS)" $
      astSig "module M\n\nf : Nat\nf = 1\n" /=
        astSig "module M\n\nf : Nat\nf = 2\n"
  , check "astSig unchanged when only imports reorder" $
      astSig "module M\n\nimport Data.List\nimport Data.Vect\n" ==
        astSig "module M\n\nimport Data.Vect\nimport Data.List\n"
  -- import-sort oracle (safeImportSort: decls fixed, imports same set)
  , check "safeImportSort accepts a reordering" $
      safeImportSort "module M\n\nimport Data.Vect\nimport Data.List\n"
                     "module M\n\nimport Data.List\nimport Data.Vect\n"
  , check "safeImportSort accepts dedup of an exact duplicate" $
      safeImportSort "module M\n\nimport Data.List\nimport Data.List\n"
                     "module M\n\nimport Data.List\n"
  , check "safeImportSort rejects a dropped import" $
      not (safeImportSort "module M\n\nimport Data.List\nimport Data.Vect\n"
                          "module M\n\nimport Data.List\n")
  , check "safeImportSort rejects a changed declaration" $
      not (safeImportSort "module M\n\nimport Data.List\n\nf : Nat\nf = 1\n"
                          "module M\n\nimport Data.List\n\nf : Nat\nf = 2\n")
  , check "old codeSig oracle rejects a reordering (why astSig is needed)" $
      not (safeReformat "module M\n\nimport Data.Vect\nimport Data.List\n"
                        "module M\n\nimport Data.List\nimport Data.Vect\n")
  -- deepSig: descends where-blocks / nested decls that astSig (Show) collapses
  , check "deepSig sees a where-block RHS change that astSig is blind to" $
      let a = "module M\n\nf : Nat\nf = g\n  where\n    g : Nat\n    g = 1\n"
          b = "module M\n\nf : Nat\nf = g\n  where\n    g : Nat\n    g = 2\n"
      in astSig a == astSig b && deepSig a /= deepSig b
  , check "deepSig ignores the indentation of a where-block" $
      let a = "module M\n\nf : Nat\nf = g\n  where\n    g : Nat\n    g = 1\n"
          b = "module M\n\nf : Nat\nf = g\n   where\n       g : Nat\n       g = 1\n"
      in deepSig a == deepSig b
  , check "deepSig sees a nested-decl-count change inside a where-block" $
      let a = "module M\n\nf : Nat\nf = g\n  where\n    g : Nat\n    g = 1\n"
          b = "module M\n\nf : Nat\nf = g\n  where\n    g : Nat\n    g = 1\n    h : Nat\n    h = 2\n"
      in deepSig a /= deepSig b
  , check "safeReindent accepts a pure reindent of a where-block" $
      safeReindent "module M\n\nf : Nat\nf = g\n  where\n    g : Nat\n    g = 1\n"
                   "module M\n\nf : Nat\nf = g\n      where\n          g : Nat\n          g = 1\n"
  ]
