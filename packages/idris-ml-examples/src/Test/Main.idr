module Test.Main

import Test.Harness
import Test.Reinforce

main : IO ()
main = runAll
  [ ("Reinforce", Test.Reinforce.tests)
  ]
