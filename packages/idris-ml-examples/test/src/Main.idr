module Main

import Harness
import Test.Generate
import Test.Reinforce

main : IO ()
main = runAll
  [ ("Generate", Test.Generate.tests)
  , ("Reinforce", Test.Reinforce.tests)
  ]
