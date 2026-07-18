module Test.Ipkg

import Format.Ipkg
import Test.Harness

-- Mirrors the repo's ipkg shape: leading `package`, a multi-line
-- `depends` block (must NOT leak into modules), a multi-line `modules`
-- block, `main`, `executable`.
sample : String
sample = "package foo\n\nsourcedir = \"src\"\n\ndepends = base\n        , idris-ml\n\nmodules = Compat.Random\n        , Array\n        , Nn.Linear\n\nmain = Main\n\nexecutable = foo\n"

export
tests : List (IO Bool)
tests =
  [ check "parseIpkg reads sourcedir" $
      (parseIpkg sample).sourcedir == "src"
  , check "parseIpkg reads modules block + main, not depends" $
      (parseIpkg sample).modules == ["Compat.Random", "Array", "Nn.Linear", "Main"]
  , check "sourcedir defaults to . when absent" $
      (parseIpkg "package foo\n\nmodules = A\n").sourcedir == "."
  , check "moduleNameFor maps a sourcedir-relative path" $
      moduleNameFor "packages/x" "src" "packages/x/src/Nn/Linear.idr" == Just "Nn.Linear"
  , check "moduleNameFor rejects a file outside the sourcedir" $
      moduleNameFor "packages/x" "src" "packages/y/src/Nn/Linear.idr" == Nothing
  , check "parentDir strips one segment" $
      (parentDir "a/b/c.idr" == "a/b") && (parentDir "a" == "")
  ]
