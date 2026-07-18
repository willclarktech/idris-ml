module Test.Main

import Test.Harness

import Test.Align
import Test.Imports
import Test.Ipkg
import Test.Reindent
import Test.Render
import Test.Roundtrip

main : IO ()
main = runAll
  [ ("Roundtrip", Test.Roundtrip.tests)
  , ("Render", Test.Render.tests)
  , ("Imports", Test.Imports.tests)
  , ("Ipkg", Test.Ipkg.tests)
  , ("Align", Test.Align.tests)
  , ("Reindent", Test.Reindent.tests)
  ]
