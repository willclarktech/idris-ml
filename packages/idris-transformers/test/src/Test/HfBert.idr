||| Unit tests for `HfBert` — pure-Idris piece (the param-name
||| catalogue). Model construction + forward-pass tests arrive in
||| follow-up commits.
module Test.HfBert

import Data.List
import Data.String

import HfBert
import Harness


-- isInfixOf in Idris-2's Data.List works on `List a`, not `String`.
-- Cast both sides through `unpack` for the convention check below.
strContains : String -> String -> Bool
strContains needle hay = isInfixOf (unpack needle) (unpack hay)


-- The canonical list of param names that `google/bert_uncased_L-2_H-128_A-2`
-- exposes via its `model.safetensors` header, in HfBert's registration
-- order (embeddings, encoder layer 0, encoder layer 1, pooler).
--
-- Confirmed against the live model header on 2026-05-26:
--   python3 -c "..." against the downloaded fixture printed exactly
--   these 39 strings (modulo the dict-sorted display order; HfBert
--   uses state_dict() insertion order which groups embeddings first
--   then each encoder layer's substructure).
expectedBertTinyParamNames : List String
expectedBertTinyParamNames =
  [ "bert.embeddings.word_embeddings.weight"
  , "bert.embeddings.position_embeddings.weight"
  , "bert.embeddings.token_type_embeddings.weight"
  , "bert.embeddings.LayerNorm.weight"
  , "bert.embeddings.LayerNorm.bias"
  , "bert.encoder.layer.0.attention.self.query.weight"
  , "bert.encoder.layer.0.attention.self.query.bias"
  , "bert.encoder.layer.0.attention.self.key.weight"
  , "bert.encoder.layer.0.attention.self.key.bias"
  , "bert.encoder.layer.0.attention.self.value.weight"
  , "bert.encoder.layer.0.attention.self.value.bias"
  , "bert.encoder.layer.0.attention.output.dense.weight"
  , "bert.encoder.layer.0.attention.output.dense.bias"
  , "bert.encoder.layer.0.attention.output.LayerNorm.weight"
  , "bert.encoder.layer.0.attention.output.LayerNorm.bias"
  , "bert.encoder.layer.0.intermediate.dense.weight"
  , "bert.encoder.layer.0.intermediate.dense.bias"
  , "bert.encoder.layer.0.output.dense.weight"
  , "bert.encoder.layer.0.output.dense.bias"
  , "bert.encoder.layer.0.output.LayerNorm.weight"
  , "bert.encoder.layer.0.output.LayerNorm.bias"
  , "bert.encoder.layer.1.attention.self.query.weight"
  , "bert.encoder.layer.1.attention.self.query.bias"
  , "bert.encoder.layer.1.attention.self.key.weight"
  , "bert.encoder.layer.1.attention.self.key.bias"
  , "bert.encoder.layer.1.attention.self.value.weight"
  , "bert.encoder.layer.1.attention.self.value.bias"
  , "bert.encoder.layer.1.attention.output.dense.weight"
  , "bert.encoder.layer.1.attention.output.dense.bias"
  , "bert.encoder.layer.1.attention.output.LayerNorm.weight"
  , "bert.encoder.layer.1.attention.output.LayerNorm.bias"
  , "bert.encoder.layer.1.intermediate.dense.weight"
  , "bert.encoder.layer.1.intermediate.dense.bias"
  , "bert.encoder.layer.1.output.dense.weight"
  , "bert.encoder.layer.1.output.dense.bias"
  , "bert.encoder.layer.1.output.LayerNorm.weight"
  , "bert.encoder.layer.1.output.LayerNorm.bias"
  , "bert.pooler.dense.weight"
  , "bert.pooler.dense.bias"
  ]


-- Compare two lists element-by-element and report the first mismatch,
-- so a wrong name doesn't get lost in a sea of equal entries.
firstMismatch : List String -> List String -> Maybe (Nat, String, String)
firstMismatch xs ys = go Z xs ys
  where
    go : Nat -> List String -> List String -> Maybe (Nat, String, String)
    go _ []        []        = Nothing
    go n []        (y :: _)  = Just (n, "<missing>", y)
    go n (x :: _)  []        = Just (n, x, "<missing>")
    go n (x :: xs) (y :: ys) =
      if x == y then go (S n) xs ys else Just (n, x, y)


testParamCount : IO Bool
testParamCount =
  let got = length (bertParamNames bertTinyConfig "bert")
      expected = 39 -- 5 embeddings + 16*2 encoder layers + 2 pooler
  in check ("bertParamNames length = 39 (got " ++ show got ++ ")")
           (got == expected)

testParamNamesMatchHfReference : IO Bool
testParamNamesMatchHfReference =
  let got = bertParamNames bertTinyConfig "bert"
  in case firstMismatch got expectedBertTinyParamNames of
       Nothing => check "all 39 param names match HF reference exactly" True
       Just (i, g, e) => do
         putStrLn ("  FAIL: param[" ++ show i ++ "] mismatch:")
         putStrLn ("    got:      " ++ g)
         putStrLn ("    expected: " ++ e)
         pure False

-- Sanity: HF names must use `.` as separator (not `_`), `.weight`
-- (not `_weights`), and avoid idris-ml's plural conventions. A drift
-- like `_weights` would skim through `loadModel` as silent skip.
testNamingConvention : IO Bool
testNamingConvention =
  let names = bertParamNames bertTinyConfig "bert"
      hasPlural = any (\n => strContains "_weights" n || strContains "_biases" n) names
      missingDots = any (\n => not (strContains "." n)) names
  in check "no `_weights`/`_biases` plural; every name uses `.` separator"
           (not hasPlural && not missingDots)


export
suite : List (String, List (IO Bool))
suite =
  [ ("HfBert param-name catalogue",
     [ testParamCount
     , testParamNamesMatchHfReference
     , testNamingConvention
     ])
  ]
