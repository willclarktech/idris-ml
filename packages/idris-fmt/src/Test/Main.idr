module Test.Main

import Test.Align
import Test.Harness
import Test.Imports
import Test.Reindent
import Test.Render
import Test.Roundtrip

main : IO ()
main = runAll
  [ ("Roundtrip", Test.Roundtrip.tests)
  , ("Render", Test.Render.tests)
  , ("Imports", Test.Imports.tests)
  , ("Align", Test.Align.tests)
  , ("Reindent", Test.Reindent.tests)
  ]
