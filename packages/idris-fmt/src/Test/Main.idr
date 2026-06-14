module Test.Main

import Test.Harness
import Test.Imports
import Test.Render
import Test.Roundtrip

main : IO ()
main = runAll
  [ ("Roundtrip", Test.Roundtrip.tests)
  , ("Render", Test.Render.tests)
  , ("Imports", Test.Imports.tests)
  ]
