||| Unit tests for `HfBitNet`.
|||
||| Same three-bucket structure as `Test.HfLlama`:
|||   1. Pure-Idris param-name catalogue tests (exact HF-naming match).
|||   2. FFI bucket — the smart constructor registers the FLOAT params
|||      (norms, embeddings, weight_scales, lm_head) under their HF
|||      names. The TERNARY weights themselves are NOT registered in
|||      this commit — they go through a separate load helper (filed
|||      under the HfBitNetLoader follow-up) because the param registry
|||      today assumes float-dtype storage. The test checks the
|||      registered subset matches `hfBitnetRegisteredParamNames`.
|||
||| Pins `{d=TapeDev}`.
module Test.HfBitNet

import Data.List
import Data.String
import Data.Vect

import HfBitNet
import Harness

import Device
import Device.Core
import Device.Tape
import Tensor


----------------------------------------------------------------------
-- Reference catalogue
----------------------------------------------------------------------

oneLayerFull : Nat -> List String
oneLayerFull i =
  let p = "model.layers." ++ show i in
  [ p ++ ".input_layernorm.weight"
  , p ++ ".self_attn.q_proj.weight"
  , p ++ ".self_attn.q_proj.weight_scale"
  , p ++ ".self_attn.k_proj.weight"
  , p ++ ".self_attn.k_proj.weight_scale"
  , p ++ ".self_attn.v_proj.weight"
  , p ++ ".self_attn.v_proj.weight_scale"
  , p ++ ".self_attn.attn_sub_norm.weight"
  , p ++ ".self_attn.o_proj.weight"
  , p ++ ".self_attn.o_proj.weight_scale"
  , p ++ ".post_attention_layernorm.weight"
  , p ++ ".mlp.gate_proj.weight"
  , p ++ ".mlp.gate_proj.weight_scale"
  , p ++ ".mlp.up_proj.weight"
  , p ++ ".mlp.up_proj.weight_scale"
  , p ++ ".mlp.ffn_sub_norm.weight"
  , p ++ ".mlp.down_proj.weight"
  , p ++ ".mlp.down_proj.weight_scale"
  ]

-- Just the names actually registered by `hfBitnetModel` in this
-- commit. Excludes the seven ternary `…weight` names per layer (the
-- ternary weights are NOT registered; they're loaded by the custom
-- path landing in the follow-up commit).
oneLayerRegistered : Nat -> List String
oneLayerRegistered i =
  let p = "model.layers." ++ show i in
  [ p ++ ".input_layernorm.weight"
  , p ++ ".self_attn.q_proj.weight_scale"
  , p ++ ".self_attn.k_proj.weight_scale"
  , p ++ ".self_attn.v_proj.weight_scale"
  , p ++ ".self_attn.attn_sub_norm.weight"
  , p ++ ".self_attn.o_proj.weight_scale"
  , p ++ ".post_attention_layernorm.weight"
  , p ++ ".mlp.gate_proj.weight_scale"
  , p ++ ".mlp.up_proj.weight_scale"
  , p ++ ".mlp.ffn_sub_norm.weight"
  , p ++ ".mlp.down_proj.weight_scale"
  ]

range30 : List Nat
range30 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]

||| 543 names: 1 embeddings + 30 layers × 18 params/layer + 1 final-
||| norm + 1 lm_head. BitNet 2B-4T's `tie_word_embeddings=false` means
||| `lm_head.weight` IS stored separately (the off-by-one vs Llama
||| 3.2 1B, which ties it).
expectedBitnet2B4T_ParamNames : List String
expectedBitnet2B4T_ParamNames =
  [ "model.embed_tokens.weight" ]
  ++ concatMap oneLayerFull range30
  ++ [ "model.norm.weight" ]
  ++ [ "lm_head.weight" ]

||| 333 names: 1 embeddings + 30 layers × 11 params/layer (4 norms +
||| 7 weight_scales) + 1 final-norm + 1 lm_head. This is the subset
||| the C-side param registry actually has after `hfBitnetModel`.
expectedBitnet2B4T_RegisteredParamNames : List String
expectedBitnet2B4T_RegisteredParamNames =
  [ "model.embed_tokens.weight" ]
  ++ concatMap oneLayerRegistered range30
  ++ [ "model.norm.weight" ]
  ++ [ "lm_head.weight" ]


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
  let got      = length (hfBitnetParamNames bitnet2B4T_Config "model")
      expected = 543
  in check ("hfBitnetParamNames length = 543 (got " ++ show got ++ ")")
           (got == expected)

testParamNamesMatchHfReference : IO Bool
testParamNamesMatchHfReference =
  let got = hfBitnetParamNames bitnet2B4T_Config "model"
  in case firstMismatch got expectedBitnet2B4T_ParamNames of
       Nothing => check "all 543 param names match HF reference exactly" True
       Just (i, g, e) => do
         putStrLn ("  FAIL: param[" ++ show i ++ "] mismatch:")
         putStrLn ("    got:      " ++ g)
         putStrLn ("    expected: " ++ e)
         pure False

testNamingConvention : IO Bool
testNamingConvention =
  let names = hfBitnetParamNames bitnet2B4T_Config "model"
      hasPlural   = any (\n => strContains "_weights" n || strContains "_biases" n) names
      missingDots = any (\n => not (strContains "." n)) names
  in check "no `_weights`/`_biases` plural; every name uses `.` separator"
           (not hasPlural && not missingDots)


----------------------------------------------------------------------
-- Bucket 2 — FFI: smart constructor registers the float subset
----------------------------------------------------------------------

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

missingFrom : List String -> List String -> List String
missingFrom names expected =
  filter (\e => not (elem e names)) expected

testConstructorRegistersHfNames : IO Bool
testConstructorRegistersHfNames = do
  -- Small dims (vocab=8, hidden=4, qOut=8, kvOut=2, intermediate=8)
  -- BUT numLayers=30 so the per-layer count matches BitNet 2B-4T's
  -- catalogue (333 registered names = 1 + 30×11 + 1 + 1). Real BitNet
  -- dims would burn ~5 GB of host RAM on the param init loop; the
  -- small dims keep this a unit test.
  --
  -- The C-side `param_register` is name-keyed and idempotent: re-
  -- registering an existing name REPLACES the tensor at that slot
  -- without incrementing the count. Earlier `Test.HfLlama` already
  -- populated `model.embed_tokens.weight`, `model.layers.0..15.*`, and
  -- `model.norm.weight` with `HfLlama`'s tensors — so 34 of the 333
  -- BitNet names overwrite in-place rather than appending new slots.
  -- Validate by SET MEMBERSHIP (every expected name present in the
  -- registry post-construction), not by counting "newly added" entries.
  _ <- hfBitnetModel {d=TapeDev} {dt=F64}
                     {vocab        = 8}
                     {hidden       = 4}
                     {numLayers    = 30}
                     {qOut         = 8}
                     {kvOut        = 2}
                     {intermediate = 8}
                     "model"
  allNames <- readAllParamNames
  let expected = expectedBitnet2B4T_RegisteredParamNames
      missing  = missingFrom allNames expected
  case missing of
    [] => check "every float-subset param name present in C-side registry" True
    _  => do
      putStrLn ("  FAIL: " ++ show (length missing) ++ " of " ++
                show (length expected) ++ " expected names missing:")
      for_ (take 10 missing) $ \name => putStrLn ("    - " ++ name)
      pure False


----------------------------------------------------------------------
-- Suite export
----------------------------------------------------------------------

public export
suite : List (String, List (IO Bool))
suite =
  [ ("HfBitNet — param name catalogue",
     [ testParamCount
     , testParamNamesMatchHfReference
     , testNamingConvention
     ])
  , ("HfBitNet — FFI constructor registry (float subset)",
     [ testConstructorRegistersHfNames
     ])
  ]
