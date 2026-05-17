module Main

import Harness
import Test.HfBert
import Test.HfGpt2


main : IO ()
main = runAll (
  [ ("package wiring",
     [ check "ipkg parses and harness links" True
     ])
  ] ++ Test.HfBert.suite ++ Test.HfGpt2.suite)
