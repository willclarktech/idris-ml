module Test.Imports

import Format.Imports
import Format.Render
import Format.Roundtrip
import Test.Harness

unsorted : String
unsorted = "module M\n\nimport Tensor\nimport Data.Vect\nimport Layer\nimport Data.List\nimport Data.List\n\nfoo : Nat\nfoo = 1\n"

sorted : String
sorted = "module M\n\nimport Data.List\nimport Data.Vect\n\nimport Layer\nimport Tensor\n\nfoo : Nat\nfoo = 1\n"

withComment : String
withComment = "module M\n\nimport Tensor\n-- a note\nimport Data.List\n\nfoo : Nat\nfoo = 1\n"

-- Three-tier grouping: stdlib, then cross-package libraries, then
-- package-local modules, blank-line separated. "Local" is exact
-- membership in the owning ipkg's module list (derived by Format.Ipkg;
-- the tests pass it directly).
localSet : List String
localSet = ["BuildConfig", "Generate"]

threeTierUnsorted : String
threeTierUnsorted = "module M\n\nimport BuildConfig\nimport Ml.Tensor\nimport Data.Vect\nimport Generate\nimport Gym.Space\nimport Transformers.Bert\n\nfoo : Nat\nfoo = 1\n"

threeTierSorted : String
threeTierSorted = "module M\n\nimport Data.Vect\n\nimport Gym.Space\nimport Ml.Tensor\nimport Transformers.Bert\n\nimport BuildConfig\nimport Generate\n\nfoo : Nat\nfoo = 1\n"

-- Postfix projections (`r.tensorPtr`) leak FCs through `Show PTerm`, so a
-- regroup that changes the import block's line count shifts every following
-- decl's positions; the oracle must ignore that (it is documented as
-- FC-insensitive) or projection-using files can never be regrouped.
projUnsorted : String
projUnsorted = "module M\n\nimport Ml.Tensor\nimport Data.Vect\nimport BuildConfig\n\ngetPtr : a -> b\ngetPtr r = r.tensorPtr\n"

projSorted : String
projSorted = "module M\n\nimport Data.Vect\n\nimport Ml.Tensor\n\nimport BuildConfig\n\ngetPtr : a -> b\ngetPtr r = r.tensorPtr\n"

export
tests : List (IO Bool)
tests =
  [ check "sorts + groups + dedups imports" $
      format unsorted == sorted
  , check "sorted output is a fixed point" $
      format sorted == sorted
  , check "leaves comment-interleaved imports untouched" $
      sortImports [] withComment == Nothing
  , check "format preserves a comment in the import block" $
      format withComment == withComment
  , check "import-sort output passes the safeImportSort oracle" $
      safeImportSort unsorted sorted
  , check "no-import module is unchanged by sort" $
      sortImports [] "module M\n\nfoo : Nat\nfoo = 1\n" == Nothing
  , check "groups stdlib / library / local imports in three tiers" $
      formatWith localSet threeTierUnsorted == threeTierSorted
  , check "three-tier output is a fixed point" $
      formatWith localSet threeTierSorted == threeTierSorted
  , check "no known locals degrades to two tiers" $
      formatWith [] threeTierUnsorted ==
        "module M\n\nimport Data.Vect\n\nimport BuildConfig\nimport Generate\nimport Gym.Space\nimport Ml.Tensor\nimport Transformers.Bert\n\nfoo : Nat\nfoo = 1\n"
  , check "three-tier output passes the safeImportSort oracle" $
      safeImportSort threeTierUnsorted threeTierSorted
  , check "regroups a file whose body uses postfix projections" $
      formatWith ["BuildConfig"] projUnsorted == projSorted
  , check "projection FCs don't fail the safeImportSort oracle" $
      safeImportSort projUnsorted projSorted
  ]
