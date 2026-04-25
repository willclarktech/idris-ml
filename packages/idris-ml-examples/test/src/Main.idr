module Main

import Harness
import Test.Generate

main : IO ()
main = runAll
  [ ("Generate", Test.Generate.tests)
  ]
