||| Unit tests for `HfLlama`.
|||
||| Same three-bucket structure as `Test.HfBert` / `Test.HfGpt2`:
|||   1. Pure-Idris param-name catalogue tests (exact HF-naming match).
|||   2. FFI bucket — the smart constructor actually registers under
|||      those names in C-side param registry order. Uses small dims
|||      (vocab=8, hidden=4, etc.) but numLayers=16 so the per-layer
|||      math matches the real Llama 3.2 1B catalogue (146 names).
|||
||| Resolves `{ex=TestExecutor}` / `{dt=TestDType}` from `Test.Config`;
||| runs on every F64-admissible primary.
module Test.HfLlama

import Data.List
import Data.String
import Data.Vect

import Executor
import Executor.Core
import HfLlama
import Tensor
import Test.Config
import Test.Harness
import Test.HfCommon

----------------------------------------------------------------------
-- Reference catalogue (mirrors `unsloth/Llama-3.2-1B`'s safetensors
-- header — a public mirror of `meta-llama/Llama-3.2-1B`; verified
-- from upstream HF Llama 3 model card + modeling_llama.py state_dict)
----------------------------------------------------------------------

oneLayer : Nat -> List String
oneLayer i =
  let p = "model.layers." ++ show i in
  [ p ++ ".input_layernorm.weight"
  , p ++ ".self_attn.q_proj.weight"
  , p ++ ".self_attn.k_proj.weight"
  , p ++ ".self_attn.v_proj.weight"
  , p ++ ".self_attn.o_proj.weight"
  , p ++ ".post_attention_layernorm.weight"
  , p ++ ".mlp.gate_proj.weight"
  , p ++ ".mlp.up_proj.weight"
  , p ++ ".mlp.down_proj.weight"
  ]

||| 146 names: 1 embeddings + 16 layers × 9 params/layer + 1 final-norm.
||| Llama 3.2 1B's `tie_word_embeddings=true` means lm_head.weight is
||| NOT stored separately — that's the off-by-one vs e.g. Llama 3 8B
||| (which would have 147 with the same numLayers).
expectedLlama32_1B_ParamNames : List String
expectedLlama32_1B_ParamNames =
  [ "model.embed_tokens.weight" ]
  ++ concatMap oneLayer (the (List Nat) [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
  ++ [ "model.norm.weight" ]

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
  assertHfModelParamCount "hfLlamaParamNames"
                          (hfLlamaParamNames llama32_1B_Config "model")
                          146

testParamNamesMatchHfReference : IO Bool
testParamNamesMatchHfReference =
  let got = hfLlamaParamNames llama32_1B_Config "model"
  in case firstMismatch got expectedLlama32_1B_ParamNames of
       Nothing        => check "all 146 param names match HF reference exactly" True
       Just (i, g, e) => do
         putStrLn ("  FAIL: param[" ++ show i ++ "] mismatch:")
         putStrLn ("    got:      " ++ g)
         putStrLn ("    expected: " ++ e)
         pure False

testNamingConvention : IO Bool
testNamingConvention =
  let names = hfLlamaParamNames llama32_1B_Config "model"
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
  -- Snapshot the registry count before construction; slice off only
  -- what Llama adds (the param registry accumulates across all prior
  -- tests in this process). Same trick as the HfGpt2 test.
  --
  -- Small dims (vocab=8, hidden=4, qOut=8, kvOut=2, intermediate=8)
  -- BUT numLayers=16 so the per-layer count matches the real Llama
  -- 3.2 1B catalogue (146 names). Real Llama dims would burn ~10 GB
  -- of host RAM on the param init loop (the embedding alone is
  -- 128256*2048*8 = 2.1 GB at F64); the small dims keep this a unit
  -- test.
  --
  -- Literal Nats so the auto-implicits can resolve (each Nat is small).
  preCount <- primIO (primParamCount {ex=TestExecutor})
  _ <- hfLlamaModel {ex=TestExecutor} {dt=TestDType}
                    {vocab        = 8}
                    {hidden       = 4}
                    {numLayers    = 16}
                    {qOut         = 8}
                    {kvOut        = 2}
                    {intermediate = 8}
                    "model"
  allNames <- readAllParamNames
  let registered = drop (cast {to=Nat} preCount) allNames
      expected   = hfLlamaParamNames llama32_1B_Config "model"
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
-- Suite export
----------------------------------------------------------------------

public export
suite : List (String, List (IO Bool))
suite =
  [ ("HfLlama — param name catalogue",
     [ testParamCount
     , testParamNamesMatchHfReference
     , testNamingConvention
     ])
  , ("HfLlama — FFI constructor registry",
     [ testConstructorRegistersHfNames
     ])
  ]
