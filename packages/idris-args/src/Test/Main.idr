module Test.Main

import Test.Harness
import Test.Args


main : IO ()
main = runAll
  [ ("Args", Test.Args.tests)
  ]
