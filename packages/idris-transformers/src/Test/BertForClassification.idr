||| Unit tests for `Transformers.BertForClassification`. Mirrors the 4-bucket
||| layout of `Test.Bert`:
|||
|||   1. classifier-head catalogue (pure Idris)
|||   2. combined backbone + head catalogue (pure Idris)
|||   3. FFI registry — constructor registers HF-native names in order
|||   4. forward pass — shape + finite values
|||
||| The FFI bucket resolves `{ex=TestExecutor}` / `{dt=TestDType}`
||| from `Test.Config`; the suite runs against every F64-admissible
||| primary.
module Test.BertForClassification

import Data.List
import Data.String
import Data.Vect

import Transformers.Bert
import Transformers.BertForClassification
import Test.Harness

import Executor
import Executor.Core
import Nn.Derive
import Nn.Module
import Test.Config
import Tensor
import Array

----------------------------------------------------------------------
-- Reference catalogue
----------------------------------------------------------------------

expectedClassifierHeadParamNames : List String
expectedClassifierHeadParamNames =
  [ "classifier.weight"
  , "classifier.bias"
  ]

-- The full 41-name reference for `BertForSequenceClassification`
-- at bertTinyConfig + numClasses=3 (3 labels chosen by the worked
-- example in FT3; the head count is the same regardless of value).
expectedBertSeqClassifyParamNames : List String
expectedBertSeqClassifyParamNames =
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
  , "classifier.weight"
  , "classifier.bias"
  ]

-- (Shared with Test.Bert; lifted here so this test file is
-- self-contained against module-test churn.)
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
-- Bucket 1 — pure Idris (classifier-head catalogue)
----------------------------------------------------------------------

testClassifierHeadParamCount : IO Bool
testClassifierHeadParamCount =
  let got = length (classifierHeadParamNames "classifier")
  in check ("classifierHeadParamNames length = 2 (got " ++ show got ++ ")")
           (got == 2)

testClassifierHeadParamNames : IO Bool
testClassifierHeadParamNames =
  let got = classifierHeadParamNames "classifier"
  in case firstMismatch got expectedClassifierHeadParamNames of
       Nothing        => check "classifier head names match HF reference" True
       Just (i, g, e) => do
         putStrLn ("  FAIL: classifier[" ++ show i ++ "] mismatch:")
         putStrLn ("    got:      " ++ g)
         putStrLn ("    expected: " ++ e)
         pure False

----------------------------------------------------------------------
-- Bucket 2 — pure Idris (combined catalogue is base + head)
----------------------------------------------------------------------

testSeqClassifyCombinedCatalogue : IO Bool
testSeqClassifyCombinedCatalogue =
  let got = bertForSequenceClassificationParamNames bertTinyConfig "bert" "classifier"
  in case firstMismatch got expectedBertSeqClassifyParamNames of
       Nothing        => check "bertForSequenceClassificationParamNames concatenates correctly (41 = 39 + 2)" True
       Just (i, g, e) => do
         putStrLn ("  FAIL: combined[" ++ show i ++ "] mismatch:")
         putStrLn ("    got:      " ++ g)
         putStrLn ("    expected: " ++ e)
         pure False

----------------------------------------------------------------------
-- Bucket 3 — FFI (constructor registers HF-native names in order)
----------------------------------------------------------------------

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

-- Filter the registry to names matching the prefixes this test cares
-- about. The shared registry accumulates entries from prior tests in
-- the same process, so we look for our distinct-prefixed pair.
filterByPrefixes : List String -> List String -> List String
filterByPrefixes prefixes = filter (\n => any (\p => isPrefixOf p n) prefixes)

testConstructorRegistersClassifierHead : IO Bool
testConstructorRegistersClassifierHead = do
  let cfg = bertTinyConfig
  -- Use distinct prefixes to dodge any registry pollution from
  -- earlier Test.Bert buckets ("bert" / "fwdtest" / "clstest").
  _ <- hfBertForSequenceClassification {ex=TestExecutor} {dt=TestDType}
                                       {vocab        = cfg.vocabSize}
                                       {hidden       = cfg.hidden}
                                       {numLayers    = cfg.numLayers}
                                       {numHeads     = cfg.numHeads}
                                       {intermediate = cfg.intermediate}
                                       {maxPos       = cfg.maxPosition}
                                       {typeVocab    = cfg.typeVocabSize}
                                       {numClasses   = 3}
                                       "ftbert" "ftclassifier"
  registered <- readAllParamNames
  let ours    = filterByPrefixes ["ftbert.", "ftclassifier."] registered
      -- Build the expected list under the same custom prefixes.
      expected = bertForSequenceClassificationParamNames cfg "ftbert" "ftclassifier"
  case firstMismatch ours expected of
    Nothing => check ("classifier model registers 41 HF-shaped names "
                       ++ "(got " ++ show (length ours) ++ ")")
                     (length ours == 41)
    Just (i, g, e) => do
      putStrLn ("  FAIL: registry[" ++ show i ++ "] mismatch (filtered):")
      putStrLn ("    got:      " ++ g)
      putStrLn ("    expected: " ++ e)
      pure False

----------------------------------------------------------------------
-- Bucket 4 — forward pass shape + finite smoke
----------------------------------------------------------------------

mkIdsTensor : {n : Nat} -> Vect n Double -> Tensor [n] TestExecutor TestDType WithGrad
mkIdsTensor xs =
  let raw = bulkToTensor {ex=TestExecutor} {dt=TestDType}
                         (VArray (map SArray xs))
  in tinput1d {n} raw

readOut : {n : Nat} -> AnyPtr -> IO (List Double)
readOut {n} p = loop (cast {to=Int} n) 0 []
  where
    loop : Int -> Int -> List Double -> IO (List Double)
    loop end i acc =
      if i >= end
        then pure (reverse acc)
        else let v = primItem1d {ex=TestExecutor} p i
             in loop end (i + 1) (v :: acc)

isFinite : Double -> Bool
isFinite x = x == x && abs x < 1.0e100

testForwardShapeAndFinite : IO Bool
testForwardShapeAndFinite = do
  -- Tiny config: hidden=8, layers=1, heads=2, headDim=4, intermediate=16, numClasses=3.
  -- Distinct prefixes so registry entries don't collide with earlier buckets.
  model <- hfBertForSequenceClassification {ex=TestExecutor} {dt=TestDType}
                                           {vocab        = 4}
                                           {hidden       = 8}
                                           {numLayers    = 1}
                                           {numHeads     = 2}
                                           {intermediate = 16}
                                           {maxPos       = 4}
                                           {typeVocab    = 2}
                                           {numClasses   = 3}
                                           "ftfwdb" "ftfwdc"
  let inputIds = mkIdsTensor (the (Vect 3 Double) [1.0, 2.0, 3.0])
      posIds  = mkIdsTensor (the (Vect 3 Double) [0.0, 1.0, 2.0])
      typeIds = mkIdsTensor (the (Vect 3 Double) [0.0, 0.0, 0.0])
  out <- hfBertSeqClassifyForward {ex=TestExecutor} {dt=TestDType}
                                  {seqLen       = 3}
                                  {vocab        = 4}
                                  {hidden       = 8}
                                  {numLayers    = 1}
                                  {numHeads     = 2}
                                  {headDim      = 4}
                                  {intermediate = 16}
                                  {maxPos       = 4}
                                  {typeVocab    = 2}
                                  {numClasses   = 3}
                                  model inputIds posIds typeIds Nothing
  vals <- readOut {n=3} out.tensorPtr
  if length vals /= 3
    then do
      putStrLn ("  FAIL: expected 3 logits, got " ++ show (length vals))
      pure False
    else if not (all isFinite vals)
      then do
        putStrLn "  FAIL: forward output contains non-finite values"
        putStrLn ("    values: " ++ show vals)
        pure False
      else check ("forward produced 3 finite logits "
                    ++ "(sample: " ++ show vals ++ ")") True

-- The derived `gparams` (backbone reuses BertModelState's derived GCast;
-- classifier head derived here) must visit exactly the combined HF
-- catalogue.
testDerivedGparamsMatchesCatalogue : IO Bool
testDerivedGparamsMatchesCatalogue = do
  let cfg = bertTinyConfig
  m <- hfBertForSequenceClassification {ex=TestExecutor} {dt=TestDType}
                                       {vocab        = cfg.vocabSize}
                                       {hidden       = cfg.hidden}
                                       {numLayers    = cfg.numLayers}
                                       {numHeads     = cfg.numHeads}
                                       {intermediate = cfg.intermediate}
                                       {maxPos       = cfg.maxPosition}
                                       {typeVocab    = cfg.typeVocabSize}
                                       {numClasses   = 3}
                                       "ftbert" "ftclassifier"
  let got      = sort (mapMaybe paramName (gparams m))
      expected = sort (bertForSequenceClassificationParamNames cfg "ftbert" "ftclassifier")
  case firstMismatch got expected of
    Nothing        => check "derived gparams visits exactly the BertForSeqClassify catalogue" True
    Just (i, g, e) => do
      putStrLn ("  FAIL: param[" ++ show i ++ "] mismatch:")
      putStrLn ("    got:      " ++ g)
      putStrLn ("    expected: " ++ e)
      pure False

----------------------------------------------------------------------
-- Test suite
----------------------------------------------------------------------

export
suite : List (String, List (IO Bool))
suite =
  [ ("Transformers.BertForClassification head catalogue",
     [ testClassifierHeadParamCount
     , testClassifierHeadParamNames
     ])
  , ("Transformers.BertForClassification combined catalogue",
     [ testSeqClassifyCombinedCatalogue
     ])
  , ("Transformers.BertForClassification constructor registers HF-native names",
     [ testConstructorRegistersClassifierHead
     ])
  , ("Transformers.BertForClassification derived GCast traversal",
     [ testDerivedGparamsMatchesCatalogue
     ])
  , ("Transformers.BertForClassification forward — shape + finite",
     [ testForwardShapeAndFinite
     ])
  ]
