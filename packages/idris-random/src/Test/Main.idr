module Test.Main

import Test.Harness

import Test.Dist
import Test.Source
import Test.SplitMix
import Test.Xoshiro

main : IO ()
main = runAll
  [ ("SplitMix", Test.SplitMix.tests)
  , ("Xoshiro",  Test.Xoshiro.tests)
  , ("Source",   Test.Source.tests)
  , ("Dist",     Test.Dist.tests)
  ]
