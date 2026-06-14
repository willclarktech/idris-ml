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
|||   3. Forward-pass smoke — construct a tiny model end-to-end and
|||      verify the LM forward emits a finite `[seq, vocab]` tensor.
|||      Correctness against an HF reference is gated by the roundtrip
|||      target landing in the follow-up commit.
|||
||| Resolves `{ex=TestExecutor}` / `{dt=TestDType}` from `Test.Config`.
module Test.HfBitNet

import Data.List
import Data.String
import Data.Vect

import Executor
import Executor.Core
import HfBitNet
import Nn.RoPE
import Tensor
import Test.Config
import Test.Harness
import Test.HfCommon

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

||| 542 names: 1 embeddings + 30 layers × 18 params/layer + 1 final-
||| norm. BitNet 2B-4T's `tie_word_embeddings=true` means there's NO
||| separate `lm_head.weight` on disk — the embedding tensor serves
||| as both, just like Llama 3.2 1B.
expectedBitnet2B4T_ParamNames : List String
expectedBitnet2B4T_ParamNames =
  [ "model.embed_tokens.weight" ]
  ++ concatMap oneLayerFull range30
  ++ [ "model.norm.weight" ]

||| 332 names: 1 embeddings + 30 layers × 11 params/layer (4 norms +
||| 7 weight_scales) + 1 final-norm. This is the subset the C-side
||| param registry actually has after `hfBitnetModel` (no lm_head —
||| tied to the embedding).
expectedBitnet2B4T_RegisteredParamNames : List String
expectedBitnet2B4T_RegisteredParamNames =
  [ "model.embed_tokens.weight" ]
  ++ concatMap oneLayerRegistered range30
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
  assertHfModelParamCount "hfBitnetParamNames"
                          (hfBitnetParamNames bitnet2B4T_Config "model")
                          542

testParamNamesMatchHfReference : IO Bool
testParamNamesMatchHfReference =
  let got = hfBitnetParamNames bitnet2B4T_Config "model"
  in case firstMismatch got expectedBitnet2B4T_ParamNames of
       Nothing        => check "all 542 param names match HF reference exactly" True
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
  _ <- hfBitnetModel {ex=TestExecutor} {dt=TestDType}
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

----------------------------------------------------------------------
-- Bucket 3 — Forward-pass smoke
----------------------------------------------------------------------

-- Walk a List of Doubles, return True if any element is NaN or ±Inf.
-- NaN detected via `x /= x` (IEEE 754: NaN compares unequal to itself);
-- ±Inf via comparison against `1.0/0.0` / `-1.0/0.0`. Same idiom as
-- `Train.showFix`'s fallback rendering.
anyNonFinite : List Double -> Bool
anyNonFinite = any nonFinite
  where
    nonFinite : Double -> Bool
    nonFinite x = x /= x || x == 1.0/0.0 || x == -1.0/0.0

testForwardLmSmoke : IO Bool
testForwardLmSmoke = do
  -- Tiny config: vocab=8, hidden=4 (= numHeads*headDim = 2*2),
  -- intermediate=8, numKvHeads=1 (GQA 2:1), numLayers=2, maxPos=16.
  -- seq=2 (a two-token "prompt").
  model <- hfBitnetModel {ex=TestExecutor} {dt=TestDType}
                         {vocab        = 8}
                         {hidden       = 4}
                         {numLayers    = 2}
                         {qOut         = 4}
                         {kvOut        = 2}
                         {intermediate = 8}
                         "bitnet_smoke"
  tables <- buildLlamaRoPETables {ex=TestExecutor} {dt=TestDType}
                                 {maxPos=16} {headDim=2}
                                 500000.0 bitnetRopeScaling
  -- Two-token input: token IDs 1 and 3 (both within vocab=8).
  let tokBuf  = prim__allocDoubles 2
      tokBuf' = prim__setDouble tokBuf 0 1.0
      tokBuf2 = prim__setDouble tokBuf' 1 3.0
      tokPtr  = dtCreateState1d {ex=TestExecutor} {t=TestDType} 2 tokBuf2 (deviceStreamTag {ex=TestExecutor})
      tokens  : Tensor [2] TestExecutor TestDType WithGrad
      tokens  = MkTensor tokPtr Nothing
  logits <- hfBitnetForwardLm {ex=TestExecutor} {dt=TestDType}
                              {seq=2} {vocab=8} {hidden=4} {numLayers=2}
                              {numHeads=2} {numKvHeads=1} {headDim=2}
                              {intermediate=8} {maxPos=16}
                              1.0e-5 model tables tokens
  -- Extract all 2*8=16 logits as doubles to assert finiteness.
  let logitVals : List Double
      logitVals = collect logits
  if anyNonFinite logitVals
    then do
      putStrLn ("  FAIL: non-finite logit in forward output:")
      putStrLn ("    sample: " ++ show (take 6 logitVals))
      pure False
    else check ("forward emitted " ++ show (length logitVals)
                ++ " finite logits (sample: "
                ++ show (take 3 logitVals) ++ "...)")
               (length logitVals == 16)
  where
    -- Read every element of the [seq=2, vocab=8] logits tensor by
    -- per-position narrow+reshape+item. seq*vocab=16 elements total.
    collect : Tensor [2, 8] TestExecutor TestDType WithGrad -> List Double
    collect t = go 0 0 []
      where
        go : Int -> Int -> List Double -> List Double
        go 2 _ acc = reverse acc
        go r 8 acc = go (r + 1) 0 acc
        go r c acc =
          let row1d  = primReshape1d {ex=TestExecutor}
                         (primNarrow {ex=TestExecutor} t.tensorPtr 0 r 1) 8
              scalar = primNarrow {ex=TestExecutor} row1d 0 c 1
              v      = primItem {ex=TestExecutor} scalar
          in go r (c + 1) (v :: acc)

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
  , ("HfBitNet — forward-pass smoke",
     [ testForwardLmSmoke
     ])
  ]
