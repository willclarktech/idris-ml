module Test.Main

import Test.Harness

import Test.Bert
import Test.BertAttentionMask
import Test.BertForClassification
import Test.BertLoraInject
import Test.BitNet
import Test.Dataset
import Test.Gpt2
import Test.KVCache
import Test.Llama
import Test.LoraIO
import Test.Tokenizer

main : IO ()
main = runAll (
  [ ("package wiring",
     [ check "ipkg parses and harness links" True
     ])
  ] ++ Test.Bert.suite ++ Test.BertAttentionMask.suite
    ++ Test.BertForClassification.suite
    ++ Test.BertLoraInject.suite
    ++ Test.LoraIO.suite
    ++ Test.Dataset.suite
    ++ Test.Gpt2.suite ++ Test.Llama.suite
    ++ Test.BitNet.suite ++ Test.KVCache.suite ++ Test.Tokenizer.suite)
