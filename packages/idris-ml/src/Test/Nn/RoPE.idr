module Test.Nn.RoPE

import Control.Linear.LIO
import Data.Vect

import Ml.Executor
import Ml.Nn.RoPE
import Ml.Tensor
import Test.Harness

import Test.Config

-- Value-pin tests for the `Nn.RoPE` free functions against the Python
-- oracle. Oracle generator:
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
-- Bucket 4: the `L IO` twins compose + match the IO rotations
----------------------------------------------------------------------
--
-- `applyRopeL`/`applyRopeAllHeadsL` are thin `liftIO1` lifts of the IO
-- rotations, so values must agree exactly; the point of the test is that
-- they compose inside a `Control.Linear.LIO.run` block with no `liftIO1`
-- at the call site (the model-forward use case).

ropeInput8 : Vect 8 Double
ropeInput8 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

testApplyRopeLComposes : IO Bool
testApplyRopeLComposes = do
  tables <- buildLlamaRoPETables {ex=TestExecutor} {dt=TestDType} {maxPos=4} {headDim=4}
              ropeBase noScaling
  x0 <- tensor {ex=TestExecutor} {dt=TestDType} {dims=[2, 4]} (FromVect ropeInput8)
  let x = the (Tensor [2, 4] TestExecutor TestDType WithGrad) (retypeGrad x0)
  ioOut <- applyRope  {seq=2} {headDim=4} {maxPos=4} tables 0 x
  lOut  <- Control.Linear.LIO.run (applyRopeL {seq=2} {headDim=4} {maxPos=4} tables 0 x)
  let rd : AnyPtr -> Int -> Int -> Double
      rd p i j = primItem2d {ex=TestExecutor} p i j
      idxs     = the (List (Int, Int)) [(0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2),(1,3)]
      worst    = foldl (\m, (i, j) => max m (abs (rd ioOut.tensorPtr i j - rd lOut.tensorPtr i j))) 0.0 idxs
  if worst < tol
    then check "applyRopeL composes in L IO + equals applyRope" True
    else do
      putStrLn ("  FAIL: applyRopeL drifted from applyRope by " ++ show worst)
      pure False

testApplyRopeAllHeadsLComposes : IO Bool
testApplyRopeAllHeadsLComposes = do
  tables <- buildLlamaRoPETables {ex=TestExecutor} {dt=TestDType} {maxPos=4} {headDim=4}
              ropeBase noScaling
  x0 <- tensor {ex=TestExecutor} {dt=TestDType} {dims=[2, 1, 4]} (FromVect ropeInput8)
  let x = the (Tensor [2, 1, 4] TestExecutor TestDType WithGrad) (retypeGrad x0)
  ioOut <- applyRopeAllHeads  {seq=2} {numHeads=1} {headDim=4} {maxPos=4} tables 0 x
  lOut  <- Control.Linear.LIO.run
             (applyRopeAllHeadsL {seq=2} {numHeads=1} {headDim=4} {maxPos=4} tables 0 x)
  -- No primItem3d; reshape the [2,1,4] outputs to [2,4] to read elementwise.
  let ioR  = primReshape2d {ex=TestExecutor} ioOut.tensorPtr 2 4
      lR   = primReshape2d {ex=TestExecutor} lOut.tensorPtr 2 4
      idxs = the (List (Int, Int)) [(0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2),(1,3)]
      worst = foldl (\m, (i, j) => max m (abs (primItem2d {ex=TestExecutor} ioR i j
                                              - primItem2d {ex=TestExecutor} lR i j))) 0.0 idxs
  if worst < tol
    then check "applyRopeAllHeadsL composes in L IO + equals applyRopeAllHeads" True
    else do
      putStrLn ("  FAIL: applyRopeAllHeadsL drifted by " ++ show worst)
      pure False

----------------------------------------------------------------------
-- Bucket 5: RoPE tables are non-learnable state constants
----------------------------------------------------------------------
--
-- RoPE cos/sin come from `dtCreateState2d` — precomputed non-learnable
-- STATE (no paramId, requires_grad==0). `RoPETables` carries NO GradMode
-- index (the rotation bodies touch the tables only as raw `.tensorPtr`, so
-- a phantom `g` there would be vestigial); the grad-mode that matters lives
-- on the `applyRope*` activation, independent of the tables.
testTablesAreStateConstants : IO Bool
testTablesAreStateConstants = do
  MkRoPETables cosT sinT <- buildLlamaRoPETables {ex=TestExecutor} {dt=TestDType}
                              {maxPos=4} {headDim=4} ropeBase noScaling
  let rgC = primRequiresGrad {ex=TestExecutor} cosT.tensorPtr
      rgS = primRequiresGrad {ex=TestExecutor} sinT.tensorPtr
  check ("RoPE tables are non-learnable state constants (requires_grad cos=" ++ show rgC
         ++ " sin=" ++ show rgS ++ ")")
        (rgC == 0 && rgS == 0)

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
  , testApplyRopeLComposes
  , testApplyRopeAllHeadsLComposes
  , testTablesAreStateConstants
  ]
