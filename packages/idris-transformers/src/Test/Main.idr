module Test.Main

import Test.Harness
import Test.HfBert
import Test.HfBertAttentionMask
import Test.HfBertForClassification
import Test.HfBitNet
import Test.HfDataset
import Test.HfGpt2
import Test.HfLlama
import Test.KVCache
import Test.Tokenizer


main : IO ()
main = runAll (
  [ ("package wiring",
     [ check "ipkg parses and harness links" True
     ])
  ] ++ Test.HfBert.suite ++ Test.HfBertAttentionMask.suite
    ++ Test.HfBertForClassification.suite
    ++ Test.HfDataset.suite
    ++ Test.HfGpt2.suite ++ Test.HfLlama.suite
    ++ Test.HfBitNet.suite ++ Test.KVCache.suite ++ Test.Tokenizer.suite)
