module Main

import Harness
import Test.HfBert


main : IO ()
main = runAll (
  [ ("package wiring",
     [ check "ipkg parses and harness links" True
     ])
  ] ++ Test.HfBert.suite)
