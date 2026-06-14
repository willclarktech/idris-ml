module Test.Main

import Test.Harness
import Test.Roundtrip
import Test.Render

main : IO ()
main = runAll
  [ ("Roundtrip", Test.Roundtrip.tests)
  , ("Render", Test.Render.tests)
  ]
