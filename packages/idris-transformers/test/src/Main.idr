module Main

import Harness
import Test.HfBert
import Test.HfGpt2
import Test.HfLlama
import Test.Tokenizer


main : IO ()
main = runAll (
  [ ("package wiring",
     [ check "ipkg parses and harness links" True
     ])
  ] ++ Test.HfBert.suite ++ Test.HfGpt2.suite ++ Test.HfLlama.suite ++ Test.Tokenizer.suite)
