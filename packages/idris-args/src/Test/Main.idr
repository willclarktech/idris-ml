module Test.Main

import Test.Args
import Test.Harness

main : IO ()
main = runAll
  [ ("Args", Test.Args.tests)
  ]
