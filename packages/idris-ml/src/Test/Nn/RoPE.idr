module Test.Nn.RoPE

import Data.Vect

import Nn.RoPE
import Test.Harness

-- Value-pin tests for the relocated `Nn.RoPE` free functions against the
-- same Python oracle as `Test.RoPE` (which stays on `Layer.RoPE` until the
-- migration sweep). Oracle generator:
--   `packages/idris-transformers/scripts/save_rope_oracle.py`
-- which ports HF transformers' `_compute_llama3_parameters` to F64.
--
-- Pure host-side F64 math both sides; expected exact agreement modulo
-- math-library ulps for cos/sin (1e-12 leaves headroom).
tol : Double
tol = 1.0e-12

-- Llama 3.2 1B's RoPE config: headDim=64, base=500000, NTK factor=32.
headDim : Nat
headDim = 64

ropeBase : Double
ropeBase = 500000.0

-- Vect index helper (Vect.head needs a non-empty proof that `div 64 2`
-- doesn't reduce automatically; this stays partial-free and total).
listIdx : Nat -> Vect n a -> Maybe a
listIdx _ []            = Nothing
listIdx Z (x :: _)      = Just x
listIdx (S k) (_ :: xs) = listIdx k xs

----------------------------------------------------------------------
-- Bucket 1: base inverse frequencies (pre-NTK-scaling)
----------------------------------------------------------------------

testBaseInvFreqZero : IO Bool
testBaseInvFreqZero =
  let invFreqs = baseInvFreq headDim ropeBase
  in case listIdx 0 invFreqs of
       Just f0 =>
         if abs (f0 - 1.0) < tol
           then check "baseInvFreq[0] = 1.0 (=  1 / base^0)" True
           else do
             putStrLn ("  FAIL: got " ++ show f0 ++ " expected 1.0")
             pure False
       Nothing => do
         putStrLn "  FAIL: baseInvFreq returned empty Vect"
         pure False

----------------------------------------------------------------------
-- Bucket 2: Llama-3 NTK-scaled inverse frequencies
----------------------------------------------------------------------
--
-- Oracle slice from save_rope_oracle.py:
--   inv_freq[0]   = 1.00000000000000000e+00   (high-freq band → no scale)
--   inv_freq[16]  = 4.29556796559368154e-04   (mid-freq band  → smooth interp)
--   inv_freq[31]  = 9.41830672543490868e-08   (low-freq band  → / 32)

checkInvFreqAt : Nat -> Double -> IO Bool
checkInvFreqAt k expected = do
  let scaled = llamaInvFreq headDim ropeBase llama3Scaling
  case listIdx k scaled of
    Nothing => do
      putStrLn ("  FAIL: llamaInvFreq has no index " ++ show k)
      pure False
    Just got =>
      if abs (got - expected) < tol
        then check ("llamaInvFreq[" ++ show k ++ "] = " ++ show expected) True
        else do
          putStrLn ("  FAIL: llamaInvFreq[" ++ show k ++ "] got " ++
                    show got ++ " expected " ++ show expected)
          pure False

testInvFreqHighFreqBand : IO Bool
testInvFreqHighFreqBand = checkInvFreqAt 0 1.0

testInvFreqMidFreqBand : IO Bool
testInvFreqMidFreqBand =
  checkInvFreqAt 16 4.29556796559368154e-04

testInvFreqLowFreqBand : IO Bool
testInvFreqLowFreqBand =
  checkInvFreqAt 31 9.41830672543490868e-08

----------------------------------------------------------------------
-- Bucket 3: noScaling is the factor=1 identity short-circuit
----------------------------------------------------------------------
--
-- `noScaling` (factor=1.0) must reproduce `baseInvFreq` exactly — the
-- factor==1 short-circuit in `applyLlamaFreqScaling`. This is the
-- relocated-module invariant the migrated `Transformers.BitNet` relies
-- on (`bitnetRopeScaling` = `MkRopeScaling 1.0 1.0 1.0 0`).

testNoScalingIsIdentity : IO Bool
testNoScalingIsIdentity =
  let base    = baseInvFreq headDim ropeBase
      noScale = llamaInvFreq headDim ropeBase noScaling
      diffs   = zipWith (\a, b => abs (a - b)) base noScale
      maxDiff = foldl max 0.0 diffs
  in if maxDiff < tol
       then check "noScaling reproduces baseInvFreq (factor=1 short-circuit)" True
       else do
         putStrLn ("  FAIL: noScaling drifted from baseInvFreq by " ++ show maxDiff)
         pure False

----------------------------------------------------------------------
-- Suite
----------------------------------------------------------------------

export
tests : List (IO Bool)
tests =
  [ testBaseInvFreqZero
  , testInvFreqHighFreqBand
  , testInvFreqMidFreqBand
  , testInvFreqLowFreqBand
  , testNoScalingIsIdentity
  ]
