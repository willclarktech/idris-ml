||| Tests for prefix-filtered subset load + freeze-by-prefix.
|||
||| Subset load: register four 1-element params under two prefixes
||| (`ft1.aa.*` and `ft1.bb.*`), save, replace every value with 99.0
||| via re-registration, then a prefix-filtered `load` (`only = Just
||| "ft1.aa."`). Assert
||| `aa.*` restored to their original values + `bb.*` untouched at
||| 99.0 — exactly the "warm-start backbone, leave head fresh" guarantee
||| the API exists to provide.
|||
||| Freeze: smoke-test that `freezeByPrefix` walks the registry and
||| applies `setParamLR ... 0.0` to matching names without erroring. The
||| convergence-side guarantee that LR=0 actually freezes the
||| parameter is already covered by the existing native-optimizer
||| tests + the FT3 worked example.
module Test.CheckpointSubset

import Data.String
import Data.Vect

import Test.Harness
import Test.Config

import Executor
import Optimizer
import Tensor
import Checkpoint
import Train.Freeze

-- Read the single element out of a registered 1-D param.
readScalar1d : Tensor (the (Vect 1 Nat) [1]) TestExecutor TestDType WithGrad -> Double
readScalar1d t = primItem1d {ex=TestExecutor} t.tensorPtr 0

subsetLoadTest : IO Bool
subsetLoadTest = do
  let path = "/tmp/idris-ml-ft1-subset.safetensors"
  -- Create 4 params (1-element each) with known values.
  _   <- tparam1dConst {ex=TestExecutor} {dt=TestDType} {n=1} "ft1.aa.0" 1.0
  _   <- tparam1dConst {ex=TestExecutor} {dt=TestDType} {n=1} "ft1.aa.1" 2.0
  _   <- tparam1dConst {ex=TestExecutor} {dt=TestDType} {n=1} "ft1.bb.0" 3.0
  _   <- tparam1dConst {ex=TestExecutor} {dt=TestDType} {n=1} "ft1.bb.1" 4.0
  -- Save the initial values.
  _   <- saveAll {ex=TestExecutor} path
  -- Re-register with 99.0 (replaces in registry; the returned handles
  -- point at the new C-side tensors that the prefix-filtered `load`
  -- will mutate in place for matching names).
  aa0 <- tparam1dConst {ex=TestExecutor} {dt=TestDType} {n=1} "ft1.aa.0" 99.0
  aa1 <- tparam1dConst {ex=TestExecutor} {dt=TestDType} {n=1} "ft1.aa.1" 99.0
  bb0 <- tparam1dConst {ex=TestExecutor} {dt=TestDType} {n=1} "ft1.bb.0" 99.0
  bb1 <- tparam1dConst {ex=TestExecutor} {dt=TestDType} {n=1} "ft1.bb.1" 99.0
  -- Subset-load only the `ft1.aa.` prefix.
  ok  <- (== Right ()) <$> load {ex=TestExecutor} path ({ only := Just "ft1.aa." } defaultLoadOpts)
  let aa0v = readScalar1d aa0
      aa1v = readScalar1d aa1
      bb0v = readScalar1d bb0
      bb1v = readScalar1d bb1
  r0 <- check ("subset-load returned ok") ok
  r1 <- check ("subset-load: aa.0 restored to 1.0 (got " ++ show aa0v ++ ")") (aa0v == 1.0)
  r2 <- check ("subset-load: aa.1 restored to 2.0 (got " ++ show aa1v ++ ")") (aa1v == 2.0)
  r3 <- check ("subset-load: bb.0 untouched at 99.0 (got " ++ show bb0v ++ ")") (bb0v == 99.0)
  r4 <- check ("subset-load: bb.1 untouched at 99.0 (got " ++ show bb1v ++ ")") (bb1v == 99.0)
  pure (r0 && r1 && r2 && r3 && r4)

freezeGroupSmokeTest : IO Bool
freezeGroupSmokeTest = do
  opt <- sgd {ex=TestExecutor} 0.1 defaultOpts
  -- These params get registered by subsetLoadTest; if the test is run
  -- standalone we register them here too. Re-registration is a no-op
  -- on existing names.
  _ <- tparam1dConst {ex=TestExecutor} {dt=TestDType} {n=1} "ft1.aa.0" 1.0
  _ <- tparam1dConst {ex=TestExecutor} {dt=TestDType} {n=1} "ft1.bb.0" 3.0
  freezeGroup {ex=TestExecutor} opt !(namesMatching {ex=TestExecutor} (isPrefixOf "ft1.aa."))
  check "freezeGroup: completed without error" True

export
tests : List (IO Bool)
tests =
  [ subsetLoadTest
  , freezeGroupSmokeTest
  ]
