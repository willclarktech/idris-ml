module Test.Imports

import Test.Harness
import Format.Render
import Format.Imports
import Format.Roundtrip

unsorted : String
unsorted = "module M\n\nimport Tensor\nimport Data.Vect\nimport Layer\nimport Data.List\nimport Data.List\n\nfoo : Nat\nfoo = 1\n"

sorted : String
sorted = "module M\n\nimport Data.List\nimport Data.Vect\n\nimport Layer\nimport Tensor\n\nfoo : Nat\nfoo = 1\n"

withComment : String
withComment = "module M\n\nimport Tensor\n-- a note\nimport Data.List\n\nfoo : Nat\nfoo = 1\n"

export
tests : List (IO Bool)
tests =
  [ check "sorts + groups + dedups imports" $
      format unsorted == sorted
  , check "sorted output is a fixed point" $
      format sorted == sorted
  , check "leaves comment-interleaved imports untouched" $
      sortImports withComment == Nothing
  , check "format preserves a comment in the import block" $
      format withComment == withComment
  , check "import-sort output passes the safeImportSort oracle" $
      safeImportSort unsorted sorted
  , check "no-import module is unchanged by sort" $
      sortImports "module M\n\nfoo : Nat\nfoo = 1\n" == Nothing
  ]
