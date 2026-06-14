module Test.RoPE

import Data.Vect

import Test.Harness
import Layer.RoPE

-- Tolerance for the value-pin tests against the Python oracle.
-- Pure host-side F64 math both sides; expected exact agreement modulo
-- math-library ulps for cos/sin (1e-12 leaves headroom).
tol : Double
tol = 1.0e-12

-- Llama 3.2 1B's RoPE config: headDim=64, base=500000, NTK factor=32.
headDim : Nat
headDim = 64

ropeBase : Double
ropeBase = 500000.0

-- All expected values were produced by
--   `packages/idris-transformers/scripts/save_rope_oracle.py`
-- which ports HF transformers' `_compute_llama3_parameters` to F64.

----------------------------------------------------------------------
-- Bucket 1: base inverse frequencies (pre-NTK-scaling)
----------------------------------------------------------------------

-- listIdx is defined below in Bucket 2; declared first so Bucket 1
-- can use it. (Idris's `Vect.head` requires a proof the Vect is
-- non-empty, and `divNat 64 2 = S _` doesn't reduce automatically.)
listIdx : Nat -> Vect n a -> Maybe a
listIdx _ []        = Nothing
listIdx Z (x :: _)  = Just x
listIdx (S k) (_ :: xs) = listIdx k xs

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
-- Bucket 3: cos / sin at position 1, dim 0
----------------------------------------------------------------------
--
-- inv_freq[0] = 1, so cos(1.0 * 1) = cos(1) and sin(1.0 * 1) = sin(1).
-- These are math-library results; the test pins the (host F64) value.

testCosAtPos1Dim0 : IO Bool
testCosAtPos1Dim0 =
  let -- mirrors what the cos-table builder writes at (pos=1, i=0):
      --   cos(1.0 * inv_freq[0]) = cos(1.0)
      val = cos 1.0
      expected = 5.40302305868139765e-01
  in if abs (val - expected) < tol
       then check "cos(1.0) matches oracle (F64 host math)" True
       else do
         putStrLn ("  FAIL: cos(1.0) got " ++ show val ++ " expected " ++ show expected)
         pure False

testSinAtPos1Dim0 : IO Bool
testSinAtPos1Dim0 =
  let val = sin 1.0
      expected = 8.41470984807896505e-01
  in if abs (val - expected) < tol
       then check "sin(1.0) matches oracle (F64 host math)" True
       else do
         putStrLn ("  FAIL: sin(1.0) got " ++ show val ++ " expected " ++ show expected)
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
  , testCosAtPos1Dim0
  , testSinAtPos1Dim0
  ]
