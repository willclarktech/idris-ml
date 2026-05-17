||| Unit tests for `HfBert`.
|||
||| The pure-Idris piece (the param-name catalogue) is the first
||| suite. The FFI piece — constructing a tiny model and asserting
||| the C-side param registry holds exactly the catalogue's names in
||| the catalogue's order — is the second.
|||
||| The FFI bucket pins `{d=TapeDev}` directly rather than going
||| through a generated `TestConfig.idr` like idris-ml's test suite
||| does. That means `make BACKEND=torch test-transformers` or
||| `BACKEND=mlx test-transformers` would currently fail at FFI
||| resolution (tape-suffixed C symbols not linked). The CI runs
||| `make test-transformers` with whatever backend `make install`
||| produced (tape by default), so the gate works. Extending to a
||| `TestConfig.idr.in` generator is a follow-up.
module Test.HfBert

import Data.List
import Data.String

import HfBert
import Harness

import Device
import Device.Core
import Device.Tape
import Tensor


----------------------------------------------------------------------
-- Reference catalogue (mirrors the live model's safetensors header)
----------------------------------------------------------------------

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


-- isInfixOf from Data.List works on List a, not String. Cast through
-- unpack for the convention check below.
strContains : String -> String -> Bool
strContains needle hay = isInfixOf (unpack needle) (unpack hay)

-- Walk two lists, report first index where they diverge.
firstMismatch : List String -> List String -> Maybe (Nat, String, String)
firstMismatch xs ys = go Z xs ys
  where
    go : Nat -> List String -> List String -> Maybe (Nat, String, String)
    go _ []        []        = Nothing
    go n []        (y :: _)  = Just (n, "<missing>", y)
    go n (x :: _)  []        = Just (n, x, "<missing>")
    go n (x :: xs) (y :: ys) =
      if x == y then go (S n) xs ys else Just (n, x, y)


----------------------------------------------------------------------
-- Bucket 1 — pure Idris (catalogue correctness)
----------------------------------------------------------------------

testParamCount : IO Bool
testParamCount =
  let got      = length (bertParamNames bertTinyConfig "bert")
      expected = 39
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

testNamingConvention : IO Bool
testNamingConvention =
  let names = bertParamNames bertTinyConfig "bert"
      hasPlural   = any (\n => strContains "_weights" n || strContains "_biases" n) names
      missingDots = any (\n => not (strContains "." n)) names
  in check "no `_weights`/`_biases` plural; every name uses `.` separator"
           (not hasPlural && not missingDots)


----------------------------------------------------------------------
-- Bucket 2 — FFI (constructor actually registers under those names)
----------------------------------------------------------------------

-- Read all registered param names off the C registry. The registry
-- accumulates in registration order; the tape primary's `param_name(i)`
-- returns the name registered at slot i.
readAllParamNames : IO (List String)
readAllParamNames = do
  count <- primIO (primParamCount {d=TapeDev})
  go count 0
  where
    go : Int -> Int -> IO (List String)
    go end i = if i >= end
                 then pure []
                 else do
                   name <- primIO (primParamName {d=TapeDev} i)
                   rest <- go end (i + 1)
                   pure (name :: rest)

testConstructorRegistersHfNames : IO Bool
testConstructorRegistersHfNames = do
  -- Use bertTinyConfig dims as implicits to the polymorphic constructor.
  let cfg = bertTinyConfig
  -- Build the model. Discard the returned state — we only care that
  -- the C-side param registry now holds all 39 HF-native names.
  _ <- hfBertModel {d=TapeDev} {dt=F64}
                   {vocab = cfg.vocabSize}
                   {hidden = cfg.hidden}
                   {numLayers = cfg.numLayers}
                   {numHeads = cfg.numHeads}
                   {intermediate = cfg.intermediate}
                   {maxPos = cfg.maxPosition}
                   {typeVocab = cfg.typeVocabSize}
                   "bert"
  registered <- readAllParamNames
  let expected = bertParamNames cfg "bert"
  case firstMismatch registered expected of
    Nothing => check "C-side param registry matches catalogue exactly" True
    Just (i, g, e) => do
      putStrLn ("  FAIL: registry[" ++ show i ++ "] mismatch:")
      putStrLn ("    got:      " ++ g)
      putStrLn ("    expected: " ++ e)
      putStrLn ("    (total registered: " ++ show (length registered) ++
                ", expected: " ++ show (length expected) ++ ")")
      pure False


----------------------------------------------------------------------
-- Test suite
----------------------------------------------------------------------

export
suite : List (String, List (IO Bool))
suite =
  [ ("HfBert param-name catalogue",
     [ testParamCount
     , testParamNamesMatchHfReference
     , testNamingConvention
     ])
  , ("HfBert constructor registers HF-native names",
     [ testConstructorRegistersHfNames
     ])
  ]
