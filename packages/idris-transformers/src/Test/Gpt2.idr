||| Unit tests for `HfGpt2`.
|||
||| Same three-bucket structure as `Test.HfBert`:
|||   1. Pure-Idris param-name catalogue tests (exact HF-naming match).
|||   2. FFI bucket — the smart constructor actually registers under
|||      those names in C-side param registry order.
|||   3. Forward-pass shape + finite smoke on the tinyGpt2Config.
|||
||| Resolves `{ex=TestExecutor}` / `{dt=TestDType}` from `Test.Config`;
||| runs on every F64-admissible primary.
module Test.Gpt2

import Data.List
import Data.String
import Data.Vect
import System.File

import Array
import Checkpoint
import Executor
import Executor.Core
import Nn.Derive
import Nn.Module
import Tensor
import Test.Common
import Test.Config
import Test.Harness
import Transformers.Gpt2

----------------------------------------------------------------------
-- Reference catalogue (mirrors `sshleifer/tiny-gpt2`'s safetensors header)
----------------------------------------------------------------------

-- Build the 12 per-layer names so we don't have to repeat them 5 times.
oneLayer : Nat -> List String
oneLayer i =
  let p = "transformer.h." ++ show i in
  [ p ++ ".ln_1.weight",     p ++ ".ln_1.bias"
  , p ++ ".attn.c_attn.weight", p ++ ".attn.c_attn.bias"
  , p ++ ".attn.c_proj.weight", p ++ ".attn.c_proj.bias"
  , p ++ ".ln_2.weight",     p ++ ".ln_2.bias"
  , p ++ ".mlp.c_fc.weight", p ++ ".mlp.c_fc.bias"
  , p ++ ".mlp.c_proj.weight", p ++ ".mlp.c_proj.bias"
  ]

||| 76 names: 2 embeddings + 6 layers * 12 params/layer + 2 final-norm.
||| Matches the distilgpt2 on-disk safetensors header exactly.
expectedDistilGpt2ParamNames : List String
expectedDistilGpt2ParamNames =
  [ "transformer.wte.weight"
  , "transformer.wpe.weight"
  ]
  ++ concatMap oneLayer (the (List Nat) [0, 1, 2, 3, 4, 5])
  ++
  [ "transformer.ln_f.weight"
  , "transformer.ln_f.bias"
  ]

strContains : String -> String -> Bool
strContains needle hay = isInfixOf (unpack needle) (unpack hay)

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
-- Bucket 1 — pure Idris catalogue correctness
----------------------------------------------------------------------

testParamCount : IO Bool
testParamCount =
  assertHfModelParamCount "hfGpt2ParamNames"
                          (hfGpt2ParamNames distilGpt2Config "")
                          76

testParamNamesMatchHfReference : IO Bool
testParamNamesMatchHfReference =
  let got = hfGpt2ParamNames distilGpt2Config ""
  in case firstMismatch got expectedDistilGpt2ParamNames of
       Nothing        => check "all 76 param names match HF reference exactly" True
       Just (i, g, e) => do
         putStrLn ("  FAIL: param[" ++ show i ++ "] mismatch:")
         putStrLn ("    got:      " ++ g)
         putStrLn ("    expected: " ++ e)
         pure False

testNamingConvention : IO Bool
testNamingConvention =
  let names = hfGpt2ParamNames distilGpt2Config ""
      hasPlural   = any (\n => strContains "_weights" n || strContains "_biases" n) names
      missingDots = any (\n => not (strContains "." n)) names
  in check "no `_weights`/`_biases` plural; every name uses `.` separator"
           (not hasPlural && not missingDots)

----------------------------------------------------------------------
-- Bucket 2 — FFI: smart constructor registers exactly those names
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

testConstructorRegistersHfNames : IO Bool
testConstructorRegistersHfNames = do
  -- The param registry accumulates across all prior tests in this
  -- process; take the count BEFORE we construct and slice off only
  -- what GPT-2 added.
  --
  -- Small dims, BUT n_layer=6 so the param-name math at the layer
  -- count matches distilgpt2 (76 names). The actual distilgpt2 model
  -- has ~82M params (300MB+ embedding tensor alone); constructing
  -- that in a unit test would burn ~600MB of host RAM via the
  -- normalSample-per-element init loop. The small dims here exercise
  -- the same code path against the same catalogue without that cost.
  --
  -- Literal Nats so `hidden = numHeads * headDim` reduces to Refl
  -- (`4 = 2 * 2`). Going through `cfg.hidden` keeps the proof
  -- existentially-quantified and the auto-implicit can't resolve.
  preCount <- primIO (primParamCount {ex=TestExecutor})
  _ <- hfGpt2Model {ex=TestExecutor} {dt=TestDType} {g=WithGrad}
                   {vocab        = 8}
                   {hidden       = 4}
                   {numLayers    = 6}
                   {numHeads     = 2}
                   {headDim      = 2}
                   {intermediate = 8}
                   {maxPos       = 8}
                   ""
  allNames <- readAllParamNames
  let registered = drop (cast {to=Nat} preCount) allNames
      expected   = hfGpt2ParamNames distilGpt2Config ""
  case firstMismatch registered expected of
    Nothing        => check "C-side param registry matches catalogue exactly" True
    Just (i, g, e) => do
      putStrLn ("  FAIL: registry[" ++ show i ++ "] mismatch:")
      putStrLn ("    got:      " ++ g)
      putStrLn ("    expected: " ++ e)
      putStrLn ("    (total registered: " ++ show (length registered) ++
                ", expected: " ++ show (length expected) ++ ")")
      pure False

-- The derived `gparams` (from `%runElab derive` on the GPT-2 records)
-- must visit exactly the leaf params the HF catalogue lists — proving
-- the generated cascade is correct. `gparams` reads the model's own
-- handles, so it's robust to the shared C registry.
testDerivedGparamsMatchesCatalogue : IO Bool
testDerivedGparamsMatchesCatalogue = do
  m <- hfGpt2Model {ex=TestExecutor} {dt=TestDType} {g=WithGrad}
                   {vocab        = 8}
                   {hidden       = 4}
                   {numLayers    = 6}
                   {numHeads     = 2}
                   {headDim      = 2}
                   {intermediate = 8}
                   {maxPos       = 8}
                   ""
  let got      = sort (mapMaybe paramName (gparams m))
      expected = sort (hfGpt2ParamNames distilGpt2Config "")
  case firstMismatch got expected of
    Nothing        => check "derived gparams visits exactly the GPT-2 catalogue" True
    Just (i, g, e) => do
      putStrLn ("  FAIL: param[" ++ show i ++ "] mismatch:")
      putStrLn ("    got:      " ++ g)
      putStrLn ("    expected: " ++ e)
      pure False

----------------------------------------------------------------------
-- readGpt2Config (config.json → Gpt2Config)
----------------------------------------------------------------------

-- Distinct per-field values so a swapped key mapping is caught. headDim
-- is derived (n_embd / n_head = 8 / 2 = 4); n_inner is present here.
testReadGpt2Config : IO Bool
testReadGpt2Config = do
  let path = "/tmp/idris_gpt2_config_full.json"
  Right () <- writeFile path "{\"vocab_size\": 99, \"n_embd\": 8, \"n_layer\": 3, \"n_head\": 2, \"n_inner\": 16, \"n_positions\": 32}"
    | Left e => do putStrLn ("  FAIL: writeFile: " ++ show e); pure False
  Right cfg <- readGpt2Config path
    | Left e => do putStrLn ("  FAIL: readGpt2Config: " ++ show e); pure False
  check "readGpt2Config maps GPT-2 keys + derives head_dim = n_embd / n_head"
        (vocabSize cfg == 99 && hidden cfg == 8 && numLayers cfg == 3 &&
         numHeads cfg == 2 && headDim cfg == 4 && intermediate cfg == 16 &&
         maxPosition cfg == 32)

testReadGpt2ConfigInnerDefault : IO Bool
testReadGpt2ConfigInnerDefault = do
  let path = "/tmp/idris_gpt2_config_noinner.json"
  Right () <- writeFile path "{\"vocab_size\": 50257, \"n_embd\": 768, \"n_layer\": 6, \"n_head\": 12, \"n_positions\": 1024}"
    | Left e => do putStrLn ("  FAIL: writeFile: " ++ show e); pure False
  Right cfg <- readGpt2Config path
    | Left e => do putStrLn ("  FAIL: readGpt2Config: " ++ show e); pure False
  check "n_inner defaults to 4 * n_embd when omitted (distilgpt2: 3072)"
        (intermediate cfg == 3072 && headDim cfg == 64)

----------------------------------------------------------------------
-- Suite export
----------------------------------------------------------------------

public export
suite : List (String, List (IO Bool))
suite =
  [ ("HfGpt2 — param name catalogue",
     [ testParamCount
     , testParamNamesMatchHfReference
     , testNamingConvention
     ])
  , ("HfGpt2 — FFI constructor registry",
     [ testConstructorRegistersHfNames
     ])
  , ("HfGpt2 — derived GCast traversal",
     [ testDerivedGparamsMatchesCatalogue
     ])
  , ("readGpt2Config — config.json parsing",
     [ testReadGpt2Config
     , testReadGpt2ConfigInnerDefault
     ])
  ]
