||| Unit tests for `HfBert`.
|||
||| The pure-Idris piece (the param-name catalogue) is the first
||| suite. The FFI piece — constructing a tiny model and asserting
||| the C-side param registry holds exactly the catalogue's names in
||| the catalogue's order — is the second.
|||
||| The FFI bucket resolves `{ex=TestExecutor}` / `{dt=TestDType}` from
||| the Makefile-generated `Test.Config`, mirroring idris-ml's test
||| suite — so the same suite elaborates and runs against any
||| F64-admissible primary (tape, torch-cpu, torch-cuda, mlx-cpu).
module Test.Bert

import Data.List
import Data.String
import Data.Vect
import System.File

import Transformers.Bert
import Test.Harness
import Test.Common

import Array
import Checkpoint
import Executor
import Executor.Core
import Tensor
import Test.Config

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
  assertHfModelParamCount "bertParamNames"
                          (bertParamNames bertTinyConfig "bert")
                          39

testParamNamesMatchHfReference : IO Bool
testParamNamesMatchHfReference =
  let got = bertParamNames bertTinyConfig "bert"
  in case firstMismatch got expectedBertTinyParamNames of
       Nothing        => check "all 39 param names match HF reference exactly" True
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
-- accumulates in registration order; the active primary's `param_name(i)`
-- returns the name registered at slot i.
readAllParamNames : IO (List String)
readAllParamNames = do
  count <- primIO (primParamCount {ex=TestExecutor})
  go count 0
  where
    go : Int -> Int -> IO (List String)
    go end i = if i >= end
                 then pure []
                 else do
                   name <- primIO (primParamName {ex=TestExecutor} i)
                   rest <- go end (i + 1)
                   pure (name :: rest)

testConstructorRegistersHfNames : IO Bool
testConstructorRegistersHfNames = do
  -- Use bertTinyConfig dims as implicits to the polymorphic constructor.
  let cfg = bertTinyConfig
  -- Build the model. Discard the returned state — we only care that
  -- the C-side param registry now holds all 39 HF-native names.
  _ <- hfBertModel {ex=TestExecutor} {dt=TestDType} {g=WithGrad}
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
    Nothing        => check "C-side param registry matches catalogue exactly" True
    Just (i, g, e) => do
      putStrLn ("  FAIL: registry[" ++ show i ++ "] mismatch:")
      putStrLn ("    got:      " ++ g)
      putStrLn ("    expected: " ++ e)
      putStrLn ("    (total registered: " ++ show (length registered) ++
                ", expected: " ++ show (length expected) ++ ")")
      pure False

----------------------------------------------------------------------
-- Bucket 3 — forward pass shape + finite smoke
----------------------------------------------------------------------

-- Build a Tensor [n] from a Vect of doubles. Wraps bulkToTensor (which
-- copies into a fresh C buffer) + tinput1d (which records the handle
-- as a non-parameter input). The values represent token IDs encoded
-- as doubles — same convention as Layer.Embedding's input contract.
mkIdsTensor : {n : Nat} -> Vect n Double -> Tensor [n] TestExecutor TestDType WithGrad
mkIdsTensor xs =
  let raw = bulkToTensor {ex=TestExecutor} {dt=TestDType}
                         (VArray (map SArray xs))
  in tinput1d {n} raw

-- Read out an [N]-shape Tensor's values via primItem1d, one at a time.
-- Threads the raw pointer through the where-helper so it doesn't have
-- to capture the outer Tensor's implicit `n`.
readOut : {n : Nat} -> AnyPtr -> IO (List Double)
readOut {n} p = loop (cast {to=Int} n) 0 []
  where
    loop : Int -> Int -> List Double -> IO (List Double)
    loop end i acc =
      if i >= end
        then pure (reverse acc)
        else let v = primItem1d {ex=TestExecutor} p i
             in loop end (i + 1) (v :: acc)

-- Finite ≡ neither NaN nor ±Inf. NaN self-inequality + magnitude
-- check rules both out without depending on a stdlib helper.
isFinite : Double -> Bool
isFinite x = x == x && abs x < 1.0e100

testForwardShapeAndFinite : IO Bool
testForwardShapeAndFinite = do
  -- Tiny config: hidden=8, layers=1, heads=2, headDim=4, intermediate=16.
  -- Distinct paramPrefix from bertTinyConfig's "bert" so this test
  -- doesn't collide with the bucket-2 registry.
  model <- hfBertModel {ex=TestExecutor} {dt=TestDType} {g=WithGrad}
                       {vocab        = 4}
                       {hidden       = 8}
                       {numLayers    = 1}
                       {numHeads     = 2}
                       {intermediate = 16}
                       {maxPos       = 4}
                       {typeVocab    = 2}
                       "fwdtest"
  -- seqLen=3 input. IDs all < their respective vocab caps. Use the
  -- explicit Vect 3 type annotation so the literal isn't ambiguous
  -- between List and Vect at the bare-bracket level.
  let inputIds = mkIdsTensor (the (Vect 3 Double) [1.0, 2.0, 3.0])
      posIds  = mkIdsTensor (the (Vect 3 Double) [0.0, 1.0, 2.0])
      typeIds = mkIdsTensor (the (Vect 3 Double) [0.0, 0.0, 0.0])
  out <- hfBertForward {ex=TestExecutor} {dt=TestDType}
                       {seqLen       = 3}
                       {vocab        = 4}
                       {hidden       = 8}
                       {numLayers    = 1}
                       {numHeads     = 2}
                       {headDim      = 4}
                       {intermediate = 16}
                       {maxPos       = 4}
                       {typeVocab    = 2}
                       model inputIds posIds typeIds Nothing
  -- primItem1d takes an AnyPtr directly; no need to coerce the
  -- Tensor's gradmode.
  vals <- readOut {n=8} out.tensorPtr
  if length vals /= 8
    then do
      putStrLn ("  FAIL: expected 8 output values, got " ++ show (length vals))
      pure False
    else if not (all isFinite vals)
      then do
        putStrLn "  FAIL: forward output contains non-finite values"
        putStrLn ("    values: " ++ show vals)
        pure False
      else check ("forward produced 8 finite values "
                    ++ "(sample: " ++ show (take 3 vals) ++ "...)") True

----------------------------------------------------------------------
-- Bucket 4 — MLM-head catalogue (cls.predictions.* naming gate)
----------------------------------------------------------------------

-- The MLM head's 5 params in HF's exact spelling. Mirrors what
-- save_oracle.py would emit under `cls.predictions.*` if it loaded
-- BertForMaskedLM instead of BertModel.
expectedMlmHeadParamNames : List String
expectedMlmHeadParamNames =
  [ "cls.predictions.transform.dense.weight"
  , "cls.predictions.transform.dense.bias"
  , "cls.predictions.transform.LayerNorm.weight"
  , "cls.predictions.transform.LayerNorm.bias"
  , "cls.predictions.bias"
  ]

testMlmParamCount : IO Bool
testMlmParamCount =
  let got = length (mlmHeadParamNames "cls")
  in check ("mlmHeadParamNames length = 5 (got " ++ show got ++ ")")
           (got == 5)

testMlmParamNamesMatchHfReference : IO Bool
testMlmParamNamesMatchHfReference =
  let got = mlmHeadParamNames "cls"
  in case firstMismatch got expectedMlmHeadParamNames of
       Nothing        => check "all 5 MLM-head names match HF reference exactly" True
       Just (i, g, e) => do
         putStrLn ("  FAIL: mlm[" ++ show i ++ "] mismatch:")
         putStrLn ("    got:      " ++ g)
         putStrLn ("    expected: " ++ e)
         pure False

testMaskedLmCombinedCatalogue : IO Bool
testMaskedLmCombinedCatalogue =
  let got = bertForMaskedLmParamNames bertTinyConfig "bert" "cls"
      expected = expectedBertTinyParamNames ++ expectedMlmHeadParamNames
  in case firstMismatch got expected of
       Nothing        => check "bertForMaskedLmParamNames concatenates correctly (44 = 39 + 5)" True
       Just (i, g, e) => do
         putStrLn ("  FAIL: combined[" ++ show i ++ "] mismatch:")
         putStrLn ("    got:      " ++ g)
         putStrLn ("    expected: " ++ e)
         pure False

----------------------------------------------------------------------
-- Bucket 4 — readBertConfig (config.json → BertConfig)
----------------------------------------------------------------------

-- Distinct per-field values (not bert-tiny's) so a swapped key mapping
-- is caught: e.g. reading hidden_size into numLayers would make
-- `numLayers cfg == 3` fail.
testReadBertConfig : IO Bool
testReadBertConfig = do
  let path = "/tmp/idris_bert_config_full.json"
  Right () <- writeFile path "{\"vocab_size\": 99, \"hidden_size\": 8, \"num_hidden_layers\": 3, \"num_attention_heads\": 2, \"intermediate_size\": 16, \"max_position_embeddings\": 32, \"type_vocab_size\": 2}"
    | Left e => do putStrLn ("  FAIL: writeFile: " ++ show e); pure False
  Right cfg <- readBertConfig path
    | Left e => do putStrLn ("  FAIL: readBertConfig: " ++ show e); pure False
  check "readBertConfig maps all 7 HF keys to the right fields"
        (vocabSize cfg == 99 && hidden cfg == 8 && numLayers cfg == 3 &&
         numHeads cfg == 2 && intermediate cfg == 16 && maxPosition cfg == 32 &&
         typeVocabSize cfg == 2)

testReadBertConfigDefault : IO Bool
testReadBertConfigDefault = do
  let path = "/tmp/idris_bert_config_default.json"
  Right () <- writeFile path "{\"vocab_size\": 10, \"hidden_size\": 4, \"num_hidden_layers\": 1, \"num_attention_heads\": 2, \"intermediate_size\": 8, \"max_position_embeddings\": 16}"
    | Left e => do putStrLn ("  FAIL: writeFile: " ++ show e); pure False
  Right cfg <- readBertConfig path
    | Left e => do putStrLn ("  FAIL: readBertConfig: " ++ show e); pure False
  check "type_vocab_size defaults to 2 when the key is omitted" (typeVocabSize cfg == 2)

testReadBertConfigMissing : IO Bool
testReadBertConfigMissing = do
  let path = "/tmp/idris_bert_config_missing.json"
  Right () <- writeFile path "{\"vocab_size\": 10}"
    | Left e => do putStrLn ("  FAIL: writeFile: " ++ show e); pure False
  res <- readBertConfig path
  case res of
    Left (ConfigError _) => check "a missing required field surfaces as ConfigError" True
    Left e               => do putStrLn ("  FAIL: wrong error variant: " ++ show e); pure False
    Right _              => do putStrLn "  FAIL: expected ConfigError, got Right"; pure False

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
  , ("HfBert forward pass — shape + finite",
     [ testForwardShapeAndFinite
     ])
  , ("HfBert MLM-head catalogue",
     [ testMlmParamCount
     , testMlmParamNamesMatchHfReference
     , testMaskedLmCombinedCatalogue
     ])
  , ("readBertConfig — config.json parsing",
     [ testReadBertConfig
     , testReadBertConfigDefault
     , testReadBertConfigMissing
     ])
  ]
