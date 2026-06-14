module Test.Main

import Test.Generate
import Test.Harness
import Test.Reinforce

main : IO ()
main = runAll
  [ ("Generate", Test.Generate.tests)
  , ("Reinforce", Test.Reinforce.tests)
  ]
