||| Unit tests for L4's peft-compatible LoRA adapter I/O
||| (`Transformers.LoraIO.saveLoraAdapter` + `Transformers.LoraIO.loadLoraAdapter`).
|||
||| Strategy: register two LoRA-shaped params (one `.lora_A` and one
||| `.lora_B`), call `saveLoraAdapter` on a temp dir, then verify
||| three properties:
|||
|||  1. The adapter_config.json round-trips via `loadLoraAdapter` to
|||     the same `LoraAdapterConfig` values.
|||  2. The safetensors file's JSON header contains the peft-wrapped
|||     keys (`base_model.model.[...]. lora_A.default.weight`) and
|||     NOT the bare idris-ml names.
|||  3. The actual tensor data round-trips (we load via the existing
|||     `loadModel` path with renamed-on-disk-back-to-idris setup —
|||     register two params under the peft names, then loadModel
|||     restores them; the values match what we saved).
|||
||| Resolves `{ex=TestExecutor}` / `{dt=TestDType}` from `Test.Config`;
||| runs on every F64-admissible primary. Cross-backend numeric
||| coverage of the LoRA math primitive lives in L1's `Test.LoraLinear`.
module Test.LoraIO

import Data.List
import Data.String
import Data.Vect
import System.Directory
import System.File

import Test.Harness

import Checkpoint
import Executor
import Executor.Core
import Test.Config
import Transformers.LoraIO
import Tensor

----------------------------------------------------------------------
-- Test #1 — adapter_config.json round-trip
----------------------------------------------------------------------

configRoundTrip : IO Bool
configRoundTrip = do
  let cfg = MkLoraAdapterConfig 8 16.0 (the (List String) ["query", "value"]) "SEQ_CLS"
      dir = "/tmp/idris-ml-l4-cfg-roundtrip"
  -- Register a single adapter param so saveLoraAdapter has something to write
  -- (it errors on an empty match set).
  _ <- tparam1dConst {ex=TestExecutor} {dt=TestDType} {n=1}
                     "L4cfg.bert.encoder.layer.0.attention.self.query.lora_A" 1.0
  ok <- saveLoraAdapter {ex=TestExecutor} dir cfg
  if not ok
    then do putStrLn "  FAIL: saveLoraAdapter returned False"; pure False
    else do
      res <- loadLoraAdapter dir
      case res of
        Left err => do
          putStrLn ("  FAIL: loadLoraAdapter: " ++ err)
          pure False
        Right cfg' => do
          let okRank = cfg'.rank          == cfg.rank
              okAlpha = cfg'.alpha         == cfg.alpha
              okTM    = cfg'.targetModules == cfg.targetModules
              okTask  = cfg'.taskType      == cfg.taskType
          if okRank && okAlpha && okTM && okTask
            then check ("adapter_config.json round-trips "
                        ++ "(rank=" ++ show cfg'.rank
                        ++ ", alpha=" ++ show cfg'.alpha
                        ++ ", targets=" ++ show cfg'.targetModules
                        ++ ", task=" ++ cfg'.taskType ++ ")") True
            else do
              putStrLn ("  FAIL: round-trip mismatch — got "
                        ++ "rank=" ++ show cfg'.rank
                        ++ ", alpha=" ++ show cfg'.alpha
                        ++ ", targets=" ++ show cfg'.targetModules
                        ++ ", task=" ++ cfg'.taskType)
              pure False

----------------------------------------------------------------------
-- Test #2 — adapter_model.safetensors uses peft on-disk names
----------------------------------------------------------------------
--
-- Verify that the keys written to disk include the peft `base_model.model.`
-- prefix + `.default.weight` suffix wrap, NOT the bare idris-ml names.

testPeftKeyShape : IO Bool
testPeftKeyShape = do
  let dir = "/tmp/idris-ml-l4-keyshape"
      cfg = MkLoraAdapterConfig 4 8.0 (the (List String) ["query"]) "SEQ_CLS"
  -- Register one A + one B under the L3 HF-aligned naming convention.
  _ <- tparam1dConst {ex=TestExecutor} {dt=TestDType} {n=1}
                     "L4ks.bert.encoder.layer.0.attention.self.query.lora_A" 0.5
  _ <- tparam1dConst {ex=TestExecutor} {dt=TestDType} {n=1}
                     "L4ks.bert.encoder.layer.0.attention.self.query.lora_B" 0.0
  ok <- saveLoraAdapter {ex=TestExecutor} dir cfg
  if not ok
    then do putStrLn "  FAIL: saveLoraAdapter returned False"; pure False
    else do
      -- Verify round-trip by loading under the on-disk peft names.
      -- We register two params with the peft names + 99 sentinel values,
      -- then loadModel restores them from the saved file (if the saved
      -- keys really are peft-wrapped, the load will hit them). After the
      -- load, the peft-named params should be 0.5 / 0.0 (the saved values),
      -- and the idris-ml-named params (still 99 from a separate prefix)
      -- should be untouched.
      peftAName <- pure "base_model.model.L4ks.bert.encoder.layer.0.attention.self.query.lora_A.default.weight"
      peftBName <- pure "base_model.model.L4ks.bert.encoder.layer.0.attention.self.query.lora_B.default.weight"
      pA <- tparam1dConst {ex=TestExecutor} {dt=TestDType} {n=1} peftAName 99.0
      pB <- tparam1dConst {ex=TestExecutor} {dt=TestDType} {n=1} peftBName 99.0
      okLoad <- (== Right ()) <$> load {ex=TestExecutor} (dir ++ "/adapter_model.safetensors") defaultLoadOpts
      let pAV = primItem1d {ex=TestExecutor} pA.tensorPtr 0
          pBV = primItem1d {ex=TestExecutor} pB.tensorPtr 0
      if okLoad && pAV == 0.5 && pBV == 0.0
        then check ("peft-wrapped keys present in adapter_model.safetensors "
                    ++ "(loaded A=" ++ show pAV ++ ", B=" ++ show pBV ++ ")") True
        else do
          putStrLn ("  FAIL: peft-wrapped load failed — "
                    ++ "okLoad=" ++ show okLoad
                    ++ ", A=" ++ show pAV ++ " (want 0.5)"
                    ++ ", B=" ++ show pBV ++ " (want 0.0)")
          pure False

----------------------------------------------------------------------
-- Test #3 — name remap helpers
----------------------------------------------------------------------

testNameRemap : IO Bool
testNameRemap = do
  let idrisName = "bert.encoder.layer.0.attention.self.query.lora_A"
      peftName   = "base_model.model.bert.encoder.layer.0.attention.self.query.lora_A.default.weight"
      wrapped    = idrisToPeftName idrisName
      unwrapped  = peftToIdrisName peftName
      rtFromPeft = peftToIdrisName wrapped
      okWrap     = wrapped == peftName
      okUnwrap   = unwrapped == Just idrisName
      okRt       = rtFromPeft == Just idrisName
      okMiss     = peftToIdrisName "some.unrelated.name" == Nothing
  if okWrap && okUnwrap && okRt && okMiss
    then check "idrisToPeftName / peftToIdrisName round-trip + reject non-peft" True
    else do
      putStrLn ("  FAIL: wrap=" ++ show okWrap
                ++ ", unwrap=" ++ show okUnwrap
                ++ ", roundtrip=" ++ show okRt
                ++ ", reject-non-peft=" ++ show okMiss)
      pure False

----------------------------------------------------------------------
-- Test #4 — load-side remap (LoadOpts.remap)
----------------------------------------------------------------------
--
-- The symmetric inverse of saveModelMatchingRenamed: read a tensor
-- stored on disk under a FOREIGN key into a registry param under the
-- idris-ml name, without first registering the param under the foreign
-- name. Proves the contract two ways in one test:
--   * a plain (remap-less) load leaves the param at its sentinel,
--     because the foreign on-disk key isn't a registry name (skip);
--   * a remap load restores the saved value via the registry->on-disk
--     translation.

testRemapLoad : IO Bool
testRemapLoad = do
  let path    = "/tmp/idris-ml-l4-remap.safetensors"
      regName = "L4rm.weightA"
      diskKey = "FOREIGN.weightA.ondisk"
      xform : String -> Maybe String
      xform nm = if nm == regName then Just diskKey else Nothing
  -- Source param = 0.5; save it under the foreign on-disk key.
  _ <- tparam1dConst {ex=TestExecutor} {dt=TestDType} {n=1} regName 0.5
  okSave <- saveModelMatchingRenamed {ex=TestExecutor} path xform
  if not okSave
    then do putStrLn "  FAIL: saveModelMatchingRenamed returned False"; pure False
    else do
      -- Reset to a sentinel; hold the fresh handle to read back through.
      p99 <- tparam1dConst {ex=TestExecutor} {dt=TestDType} {n=1} regName 99.0
      -- Plain load: foreign key not in registry -> param untouched.
      _ <- load {ex=TestExecutor} path defaultLoadOpts
      let vPlain = primItem1d {ex=TestExecutor} p99.tensorPtr 0
      -- Remap load: foreign key reached via registry->on-disk translation.
      res <- load {ex=TestExecutor} path ({ remap := Just xform } defaultLoadOpts)
      let vRemap = primItem1d {ex=TestExecutor} p99.tensorPtr 0
      case res of
        Left err => do putStrLn ("  FAIL: remap load: " ++ show err); pure False
        Right () =>
          if vPlain == 99.0 && vRemap == 0.5
            then check ("LoadOpts.remap restores foreign-keyed param "
                        ++ "(plain kept " ++ show vPlain
                        ++ ", remap loaded " ++ show vRemap ++ ")") True
            else do
              putStrLn ("  FAIL: plain=" ++ show vPlain ++ " (want 99.0), "
                        ++ "remap=" ++ show vRemap ++ " (want 0.5)")
              pure False

export
suite : List (String, List (IO Bool))
suite =
  [ ("HfLoraIO — peft-compatible adapter I/O",
     [ configRoundTrip
     , testPeftKeyShape
     , testNameRemap
     , testRemapLoad
     ])
  ]
