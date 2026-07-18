||| Unit tests for L3's `loraInjectBert` + `hfBertForwardWithLora`.
|||
||| Resolves `{ex=TestExecutor}` / `{dt=TestDType}` from the
||| Makefile-generated `Test.Config`; runs on every F64-admissible
||| primary. Cross-backend numeric coverage of the LoRA math primitive
||| itself lives in L1's `Test.LoraLinear`; this module's job is the
||| PLUMBING gate: confirm the adapter struct is correctly threaded
||| into the encoder forward.
module Test.BertLoraInject

import Data.List
import Data.Vect

import Transformers.Bert
import Transformers.BertLora
import Test.Harness

import Ml.Executor
import Ml.Executor.Core
import Test.Config
import Ml.Tensor
import Ml.Array

----------------------------------------------------------------------
-- Helpers (mirror Test.HfBertAttentionMask)
----------------------------------------------------------------------

mkIdsTensor : {n : Nat} -> Vect n Double -> IO (Tensor [n] TestExecutor TestDType WithGrad)
mkIdsTensor xs = do
  raw <- ioRerun (\_ => bulkToTensor {ex=TestExecutor} {dt=TestDType}
                                     (VArray (map SArray xs)))
  pure (tinput1d {n} raw)

readOut : {n : Nat} -> AnyPtr -> IO (List Double)
readOut {n} p = loop (cast {to=Int} n) 0 []
  where
    loop : Int -> Int -> List Double -> IO (List Double)
    loop end i acc =
      if i >= end
        then pure (reverse acc)
        else let v = primItem1d {ex=TestExecutor} p i
             in loop end (i + 1) (v :: acc)

maxAbsDiff : List Double -> List Double -> Double
maxAbsDiff actual expected = go actual expected 0.0
  where
    go : List Double -> List Double -> Double -> Double
    go []        _         m = m
    go _         []        m = m
    go (a :: as) (b :: bs) m =
      let d = abs (a - b)
      in go as bs (if d > m then d else m)

-- ---------------------------------------------------------------
-- Shared tiny BERT (hidden=8, layers=1, heads=2)
-- ---------------------------------------------------------------

buildModel : (pfx : String)
          -> IO (Transformers.Bert.BertModelState 4 8 1 16 4 2 TestExecutor TestDType WithGrad)
buildModel pfx =
  hfBertModel {ex=TestExecutor} {dt=TestDType}
              {vocab        = 4}
              {hidden       = 8}
              {numLayers    = 1}
              {numHeads     = 2}
              {intermediate = 16}
              {maxPos       = 4}
              {typeVocab    = 2}
              pfx

-- Run hfBertForwardWithLora with given adapters + return the
-- [Hidden=8]-shape pooled output as a List Double.
runForwardWith :
     Transformers.Bert.BertModelState 4 8 1 16 4 2 TestExecutor TestDType WithGrad
  -> Maybe (BertLoraAdapters 1 8 4 TestExecutor TestDType WithGrad)
  -> IO (List Double)
runForwardWith model lora = do
  inputIds <- mkIdsTensor (the (Vect 3 Double) [1.0, 2.0, 3.0])
  posIds   <- mkIdsTensor (the (Vect 3 Double) [0.0, 1.0, 2.0])
  typeIds  <- mkIdsTensor (the (Vect 3 Double) [0.0, 0.0, 0.0])
  out <- hfBertForwardWithLora {ex=TestExecutor} {dt=TestDType}
                               {seqLen       = 3}
                               {vocab        = 4}
                               {hidden       = 8}
                               {numLayers    = 1}
                               {numHeads     = 2}
                               {headDim      = 4}
                               {intermediate = 16}
                               {maxPos       = 4}
                               {typeVocab    = 2}
                               {r            = 4}
                               model lora inputIds posIds typeIds Nothing
  readOut {n=8} out.tensorPtr

----------------------------------------------------------------------
-- Assertion 1: t=0 LoRA (B=0 init) bit-matches the no-adapter path.
----------------------------------------------------------------------
--
-- loraInjectBert init's B to zero, so at construction `(α/r) · B · A · x`
-- is identically zero — the LoRA delta added to Q and V is zero in
-- every layer. The pooled output of `hfBertForwardWithLora model (Just
-- lora) ...` must be bit-identical to `hfBertForwardWithLora model
-- Nothing ...`. Catches: missed-adapter-threading, wrong base-vs-delta
-- composition order, sign errors in the scale factor.

testInjectBitMatchAtInit : IO Bool
testInjectBitMatchAtInit = do
  model <- buildModel "lora_inject_init.bert"
  lora  <- loraInjectBert {ex=TestExecutor} {dt=TestDType} {hidden=8}
                          "lora_inject_init.bert" 1 4 16.0
  outNothing <- runForwardWith model Nothing
  outWith    <- runForwardWith model (Just lora)
  let d = maxAbsDiff outWith outNothing
  if d == 0.0
    then check ("t=0 LoRA forward bit-matches no-adapter path "
                ++ "(max-abs-diff " ++ show d ++ ")") True
    else do
      putStrLn ("  FAIL: with-adapter forward differs from Nothing by " ++ show d)
      putStrLn ("    nothing: " ++ show (take 3 outNothing) ++ "...")
      putStrLn ("    with:    " ++ show (take 3 outWith)    ++ "...")
      pure False

----------------------------------------------------------------------
-- Assertion 2: nonzero-B LoRA measurably changes the forward.
----------------------------------------------------------------------
--
-- Re-register the per-layer query lora_B with a small constant value.
-- Re-registration REPLACES the C-side entry by name (per
-- `feedback_param_registry_dedup`), but the BertLoraAdapters value's
-- Idris-side Tensor handle still points at the OLD backing — so we
-- need to construct a fresh adapters value with the new tensors.
--
-- Simplest path: manually build the adapter pair with a nonzero-B
-- constant + the same Gaussian-init A as `loraInjectBert` produces,
-- then construct `BertLoraAdapters` directly. The forward must differ
-- from the no-adapter baseline by at least ~1e-10 (well above tape's
-- F64 numerical noise floor), confirming the LoRA pipeline ACTUALLY
-- applies. Exact-value verification of the delta lives in L1's
-- `Test.LoraLinear.testNonZeroDelta`; this is the integration gate.

mkAdaptersWithNonzeroB : IO (BertLoraAdapters 1 8 4 TestExecutor TestDType WithGrad)
mkAdaptersWithNonzeroB = do
  let qPfx = "lora_inject_nz.bert.encoder.layer.0.attention.self.query"
      vPfx = "lora_inject_nz.bert.encoder.layer.0.attention.self.value"
  aQ <- tparam2dNormal {ex=TestExecutor} {dt=TestDType} {o=4} {i=8}
                       (qPfx ++ ".lora_A") 0.0 0.5
  bQ <- tparam2dConst  {ex=TestExecutor} {dt=TestDType} {o=8} {i=4}
                       (qPfx ++ ".lora_B") 0.25  -- NONZERO
  aV <- tparam2dNormal {ex=TestExecutor} {dt=TestDType} {o=4} {i=8}
                       (vPfx ++ ".lora_A") 0.0 0.5
  bV <- tparam2dConst  {ex=TestExecutor} {dt=TestDType} {o=8} {i=4}
                       (vPfx ++ ".lora_B") 0.25  -- NONZERO
  pure (MkBertLoraAdapters 4 16.0 [MkLoraAdapter aQ bQ] [MkLoraAdapter aV bV])

testInjectNonzeroDelta : IO Bool
testInjectNonzeroDelta = do
  model <- buildModel "lora_inject_nz.bert"
  lora  <- mkAdaptersWithNonzeroB
  outNothing <- runForwardWith model Nothing
  outWith    <- runForwardWith model (Just lora)
  let d = maxAbsDiff outWith outNothing
  if d > 1.0e-10
    then check ("nonzero-B LoRA forward measurably differs from baseline "
                ++ "(max-abs-diff " ++ show d ++ ")") True
    else do
      putStrLn ("  FAIL: nonzero-B forward did NOT differ measurably "
                ++ "from baseline — max-abs-diff = " ++ show d)
      putStrLn ("    nothing: " ++ show (take 3 outNothing) ++ "...")
      putStrLn ("    with:    " ++ show (take 3 outWith)    ++ "...")
      pure False

export
suite : List (String, List (IO Bool))
suite =
  [ ("Transformers.BertLora — adapter inject + forward plumbing",
     [ testInjectBitMatchAtInit
     , testInjectNonzeroDelta
     ])
  ]
