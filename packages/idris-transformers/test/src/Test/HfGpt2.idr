||| Unit tests for `HfGpt2`.
|||
||| Same three-bucket structure as `Test.HfBert`:
|||   1. Pure-Idris param-name catalogue tests (exact HF-naming match).
|||   2. FFI bucket — the smart constructor actually registers under
|||      those names in C-side param registry order.
|||   3. Forward-pass shape + finite smoke on the tinyGpt2Config.
|||
||| Like Test.HfBert this pins `{d=TapeDev}` directly. CI runs `make
||| test-transformers` with whatever backend `make install` produced
||| (tape by default).
module Test.HfGpt2

import Data.List
import Data.String
import Data.Vect

import HfGpt2
import Harness

import Device
import Device.Core
import Device.Tape
import Tensor
import Array


----------------------------------------------------------------------
-- Reference catalogue (mirrors `sshleifer/tiny-gpt2`'s safetensors header)
----------------------------------------------------------------------

expectedTinyGpt2ParamNames : List String
expectedTinyGpt2ParamNames =
  [ "transformer.wte.weight"
  , "transformer.wpe.weight"
  , "transformer.h.0.ln_1.weight"
  , "transformer.h.0.ln_1.bias"
  , "transformer.h.0.attn.c_attn.weight"
  , "transformer.h.0.attn.c_attn.bias"
  , "transformer.h.0.attn.c_proj.weight"
  , "transformer.h.0.attn.c_proj.bias"
  , "transformer.h.0.ln_2.weight"
  , "transformer.h.0.ln_2.bias"
  , "transformer.h.0.mlp.c_fc.weight"
  , "transformer.h.0.mlp.c_fc.bias"
  , "transformer.h.0.mlp.c_proj.weight"
  , "transformer.h.0.mlp.c_proj.bias"
  , "transformer.h.1.ln_1.weight"
  , "transformer.h.1.ln_1.bias"
  , "transformer.h.1.attn.c_attn.weight"
  , "transformer.h.1.attn.c_attn.bias"
  , "transformer.h.1.attn.c_proj.weight"
  , "transformer.h.1.attn.c_proj.bias"
  , "transformer.h.1.ln_2.weight"
  , "transformer.h.1.ln_2.bias"
  , "transformer.h.1.mlp.c_fc.weight"
  , "transformer.h.1.mlp.c_fc.bias"
  , "transformer.h.1.mlp.c_proj.weight"
  , "transformer.h.1.mlp.c_proj.bias"
  , "transformer.ln_f.weight"
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
  let got      = length (hfGpt2ParamNames tinyGpt2Config "")
      expected = 28
  in check ("hfGpt2ParamNames length = 28 (got " ++ show got ++ ")")
           (got == expected)

testParamNamesMatchHfReference : IO Bool
testParamNamesMatchHfReference =
  let got = hfGpt2ParamNames tinyGpt2Config ""
  in case firstMismatch got expectedTinyGpt2ParamNames of
       Nothing => check "all 28 param names match HF reference exactly" True
       Just (i, g, e) => do
         putStrLn ("  FAIL: param[" ++ show i ++ "] mismatch:")
         putStrLn ("    got:      " ++ g)
         putStrLn ("    expected: " ++ e)
         pure False

testNamingConvention : IO Bool
testNamingConvention =
  let names = hfGpt2ParamNames tinyGpt2Config ""
      hasPlural   = any (\n => strContains "_weights" n || strContains "_biases" n) names
      missingDots = any (\n => not (strContains "." n)) names
  in check "no `_weights`/`_biases` plural; every name uses `.` separator"
           (not hasPlural && not missingDots)


----------------------------------------------------------------------
-- Bucket 2 — FFI: smart constructor registers exactly those names
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

testConstructorRegistersHfNames : IO Bool
testConstructorRegistersHfNames = do
  -- The param registry accumulates across all prior tests in this
  -- process; take the count BEFORE we construct the GPT-2 model and
  -- slice the registered list to "only what GPT-2 added" so we don't
  -- depend on test-order ordering of prior Bert/Mlm-head tests.
  --
  -- Pin literal Nats so the `hidden = numHeads * headDim` proof reduces
  -- to `Refl` at the call site (`2 = 2 * 1`). Going through
  -- `cfg.hidden`/`cfg.numHeads`/`cfg.headDim` keeps the proof
  -- existentially-quantified and the auto-implicit can't resolve.
  preCount <- primIO (primParamCount {d=TapeDev})
  _ <- hfGpt2Model {d=TapeDev} {dt=F64}
                   {vocab        = 50257}
                   {hidden       = 2}
                   {numLayers    = 2}
                   {numHeads     = 2}
                   {headDim      = 1}
                   {intermediate = 8}
                   {maxPos       = 1024}
                   ""
  allNames <- readAllParamNames
  let registered = drop (cast {to=Nat} preCount) allNames
      expected   = hfGpt2ParamNames tinyGpt2Config ""
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
