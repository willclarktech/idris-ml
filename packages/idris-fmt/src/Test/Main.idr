module Test.Main

import Test.Harness
import Test.Roundtrip
import Test.Render
import Test.Imports

main : IO ()
main = runAll
  [ ("Roundtrip", Test.Roundtrip.tests)
  , ("Render", Test.Render.tests)
  , ("Imports", Test.Imports.tests)
  ]
