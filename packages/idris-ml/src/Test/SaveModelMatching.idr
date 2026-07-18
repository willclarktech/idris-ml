||| Tests for saveModelMatching / saveModelSuffixes + freezeBySuffix /
||| unfreezeBySuffix.
|||
||| Strategy mirrors Test.CheckpointSubset (the prefix-filtered load
||| tests for FT1): save a known-value subset, replace EVERY registry
||| entry with 99.0, then `loadModel` from the saved file and assert
||| which params got restored (= were in the file) vs left at 99.0
||| (= weren't in the file). This sidesteps the unreliable
||| `readFile`-on-binary-safetensors path.
module Test.SaveModelMatching

import Data.String
import Data.Vect

import Ml.Checkpoint
import Ml.Executor
import Ml.Optimizer
import Ml.Tensor
import Ml.Train.Freeze
import Test.Harness

import Test.Config

-- Read the single element out of a registered 1-D param.
readScalar1d : Tensor (the (Vect 1 Nat) [1]) TestExecutor TestDType WithGrad -> Double
readScalar1d t = primItem1d {ex=TestExecutor} t.tensorPtr 0

-- ---------------------------------------------------------------
-- Test #1 — saveModelMatching writes only the predicate-matching
-- params. Load the file back into a "noise" registry: only the
-- saved name should be restored; the unsaved one stays at 99.0.
-- ---------------------------------------------------------------

saveMatchingExactNameTest : IO Bool
saveMatchingExactNameTest = do
  let path = "/tmp/idris-ml-l2-savematch-exact.safetensors"
  -- Register two params with distinct names and known values.
  _ <- tparam1dConst {ex=TestExecutor} {dt=TestDType} {n=1} "L2.exact.alpha" 1.5
  _ <- tparam1dConst {ex=TestExecutor} {dt=TestDType} {n=1} "L2.exact.beta"  2.5
  -- Save only "L2.exact.alpha" via predicate.
  okSave <- saveModelMatching {ex=TestExecutor} path
                              (\nm => nm == "L2.exact.alpha")
  -- Re-register both with 99.0 — values now nonsense in registry.
  a <- tparam1dConst {ex=TestExecutor} {dt=TestDType} {n=1} "L2.exact.alpha" 99.0
  b <- tparam1dConst {ex=TestExecutor} {dt=TestDType} {n=1} "L2.exact.beta"  99.0
  -- Load whatever's in the file.
  okLoad <- (== Right ()) <$> load {ex=TestExecutor} path defaultLoadOpts
  let aV = readScalar1d a
      bV = readScalar1d b
  r0 <- check "save+load returned ok" (okSave && okLoad)
  r1 <- check ("alpha restored to 1.5 (got " ++ show aV ++ ")") (aV == 1.5)
  r2 <- check ("beta untouched at 99.0 (got " ++ show bV ++ ")") (bV == 99.0)
  pure (r0 && r1 && r2)

-- ---------------------------------------------------------------
-- Test #2 — saveModelSuffixes writes every param whose name ends
-- with one of the given suffixes. Three params under a backbone
-- prefix; saving by ["lora_A", "lora_B"] should checkpoint BOTH
-- adapter params and exclude the unrelated backbone weight.
-- ---------------------------------------------------------------

saveBySuffixesPicksAdaptersTest : IO Bool
saveBySuffixesPicksAdaptersTest = do
  let path = "/tmp/idris-ml-l2-savematch-suffixes.safetensors"
  -- A backbone weight + two LoRA-style adapter params (matches the
  -- L3 HF-aligned naming convention).
  let wName = "L2.sfx.bert.encoder.layer.0.self.query.weight"
      aName = "L2.sfx.bert.encoder.layer.0.self.query.lora_A"
      bName = "L2.sfx.bert.encoder.layer.0.self.query.lora_B"
  _ <- tparam1dConst {ex=TestExecutor} {dt=TestDType} {n=1} wName 1.0
  _ <- tparam1dConst {ex=TestExecutor} {dt=TestDType} {n=1} aName 2.0
  _ <- tparam1dConst {ex=TestExecutor} {dt=TestDType} {n=1} bName 3.0
  okSave <- saveModelSuffixes {ex=TestExecutor} path ["lora_A", "lora_B"]
  -- Re-register all three with nonsense.
  w <- tparam1dConst {ex=TestExecutor} {dt=TestDType} {n=1} wName 99.0
  a <- tparam1dConst {ex=TestExecutor} {dt=TestDType} {n=1} aName 99.0
  b <- tparam1dConst {ex=TestExecutor} {dt=TestDType} {n=1} bName 99.0
  okLoad <- (== Right ()) <$> load {ex=TestExecutor} path defaultLoadOpts
  let wV = readScalar1d w
      aV = readScalar1d a
      bV = readScalar1d b
  r0 <- check "save+load returned ok" (okSave && okLoad)
  r1 <- check ("backbone weight UNTOUCHED at 99.0 (got " ++ show wV ++ ")")
              (wV == 99.0)
  r2 <- check ("lora_A restored to 2.0 (got " ++ show aV ++ ")") (aV == 2.0)
  r3 <- check ("lora_B restored to 3.0 (got " ++ show bV ++ ")") (bV == 3.0)
  pure (r0 && r1 && r2 && r3)

-- ---------------------------------------------------------------
-- Test #3 — empty predicate match returns False (no degenerate
-- empty safetensors written). Mirrors saveModel's "no params"
-- guard so the LoRA adapter-save path can early-error cleanly.
-- ---------------------------------------------------------------

emptyMatchFailsTest : IO Bool
emptyMatchFailsTest = do
  let path = "/tmp/idris-ml-l2-savematch-empty.safetensors"
  ok <- saveModelMatching {ex=TestExecutor} path (\_ => False)
  check "empty predicate returns False" (not ok)

-- ---------------------------------------------------------------
-- Test #4 — freezeBySuffix smoke. Registers two params with the
-- target suffix; freezeBySuffix walks the registry without
-- erroring. Real freeze semantics (LR=0 prevents updates) are
-- covered by the existing native-optimizer tests + the L5
-- worked example.
-- ---------------------------------------------------------------

freezeGroupBySuffixSmokeTest : IO Bool
freezeGroupBySuffixSmokeTest = do
  opt <- sgd {ex=TestExecutor} 0.1 defaultOpts
  _ <- tparam1dConst {ex=TestExecutor} {dt=TestDType} {n=1} "L2.freeze.x.lora_A" 0.0
  _ <- tparam1dConst {ex=TestExecutor} {dt=TestDType} {n=1} "L2.freeze.x.lora_B" 0.0
  _ <- tparam1dConst {ex=TestExecutor} {dt=TestDType} {n=1} "L2.freeze.x.weight" 0.0
  freezeGroup   {ex=TestExecutor} opt !(namesMatching {ex=TestExecutor} (isSuffixOf "lora_A"))
  freezeGroup   {ex=TestExecutor} opt !(namesMatching {ex=TestExecutor} (isSuffixOf "lora_B"))
  unfreezeGroup {ex=TestExecutor} opt !(namesMatching {ex=TestExecutor} (isSuffixOf "lora_A"))
  unfreezeGroup {ex=TestExecutor} opt !(namesMatching {ex=TestExecutor} (isSuffixOf "lora_B"))
  check "freezeGroup / unfreezeGroup by suffix run without error" True

export
tests : List (IO Bool)
tests =
  [ saveMatchingExactNameTest
  , saveBySuffixesPicksAdaptersTest
  , emptyMatchFailsTest
  , freezeGroupBySuffixSmokeTest
  ]
