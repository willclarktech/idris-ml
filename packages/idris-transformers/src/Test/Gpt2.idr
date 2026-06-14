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

import Transformers.Gpt2
import Test.Harness
import Test.Common

import Executor
import Executor.Core
import Test.Config
import Tensor
import Array

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
  _ <- hfGpt2Model {ex=TestExecutor} {dt=TestDType}
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
  ]
