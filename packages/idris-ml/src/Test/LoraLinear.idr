module Test.LoraLinear

import Data.Vect

import Test.Harness
import Executor
import Tensor
import Array
import Layer.Core
import Layer.Linear
import Layer.LoraLinear
import Test.Config

----------------------------------------------------------------------
-- Helpers (mirror Test.SwiGLU / Test.MixedLayerLike conventions)
----------------------------------------------------------------------

readVec : (n : Nat) -> AnyPtr -> IO (List Double)
readVec n p = go (cast {to=Int} n) 0 []
  where
    go : Int -> Int -> List Double -> IO (List Double)
    go end i acc =
      if i >= end
        then pure (reverse acc)
        else let v = primItem1d {ex=TestExecutor} p i
             in go end (i + 1) (v :: acc)

maxAbsDiff : List Double -> List Double -> Double
maxAbsDiff actual expected = go actual expected 0.0
  where
    go : List Double -> List Double -> Double -> Double
    go []        _         m = m
    go _         []        m = m
    go (a :: as) (b :: bs) m =
      let d = abs (a - b)
      in go as bs (if d > m then d else m)

-- Element-wise subtract: `actual[i] - base[i]`. Shorter input wins.
diffList : List Double -> List Double -> List Double
diffList []        _         = []
diffList _         []        = []
diffList (a :: as) (b :: bs) = (a - b) :: diffList as bs

-- `ioRerun` around `bulkToTensor` per feedback_pure_typed_ffi_reorders:
-- a `let raw = bulkToTensor ...` form silently gets reordered past
-- sibling do-block IO actions (this is the MixedLayerLike crash the
-- BUG-A fix in 2026-06-07 chased down).
mkInput : {n : Nat} -> Vect n Double ->
          IO (Tensor [n] TestExecutor TestDType WithGrad)
mkInput xs = do
  raw <- ioRerun (\_ => bulkToTensor {ex=TestExecutor} {dt=TestDType}
                                     (VArray (map SArray xs)))
  pure (tinput1d {n} raw)

-- Zero a flat Double buffer over [off, off + k).
zeroBufN : Int -> Int -> AnyPtr -> AnyPtr
zeroBufN _   0 b = b
zeroBufN off k b = zeroBufN (off + 1) (k - 1) (prim__setDouble b off 0.0)

-- Write `val` at flat offset `off`.
setAt : Int -> Double -> AnyPtr -> AnyPtr
setAt off val b = prim__setDouble b off val

----------------------------------------------------------------------
-- Test #1 — Zero-B initialisation makes LoRA forward bit-identical
-- to the bare base Linear. This is the load-bearing init property:
-- LoRA papers (and peft's `LoraConfig`) ALL initialise B to zero
-- precisely so the t=0 delta is identically zero. Any plumbing bug
-- (wrong scale factor applied at t=0, wrong sign on B, wrong
-- chain of tmv/tmulScalar) would show up here.
----------------------------------------------------------------------

testInitZeroEquivalence : IO Bool
testInitZeroEquivalence = do
  base <- linearLayer {ex=TestExecutor} {dt=TestDType} {i=4} {o=3}
                       "lora_init_eq.base"
  lora <- mkLoraLinear {ex=TestExecutor} {dt=TestDType} {i=4} {o=3}
                       "lora_init_eq" 2 4.0 base
  input <- mkInput (the (Vect 4 Double) [0.5, -1.0, 0.0, 1.0])
  (_, baseOut) <- applyVar base input
  loraOut       <- applyLoraLinear lora input
  baseVals <- readVec 3 baseOut.tensorPtr
  loraVals <- readVec 3 loraOut.tensorPtr
  let mdiff = maxAbsDiff baseVals loraVals
  if mdiff == 0.0
    then check ("LoRA t=0 forward bit-matches base Linear "
                ++ "(max-abs-diff " ++ show mdiff ++ ")") True
    else do
      putStrLn ("  FAIL: max-abs-diff " ++ show mdiff)
      putStrLn ("    base: " ++ show baseVals)
      putStrLn ("    lora: " ++ show loraVals)
      pure False

----------------------------------------------------------------------
-- Test #2 — Manually-set B produces the analytically-expected
-- scaled delta. Catches off-by-one bugs in the `(alpha / r)` scale
-- factor (a common error: dividing by `rank` only at construction
-- vs at the per-step scale).
--
-- Configuration: i=2, o=3, rank=2, alpha=4.0
--   A = [[1.0, 2.0],     (row-major flat: 1.0, 2.0, 3.0, 4.0)
--        [3.0, 4.0]]
--   B = [[0.0, 0.0],     (row-major flat: 0,0, 1.0,0, 0,0)
--        [1.0, 0.0],
--        [0.0, 0.0]]
--   x = [1.0, 0.0]
--   A·x   = [1.0, 3.0]
--   B·A·x = [0, 1.0, 0]
--   scale = alpha / rank = 4.0 / 2.0 = 2.0
--   delta = scale · B·A·x = [0.0, 2.0, 0.0]
----------------------------------------------------------------------

testNonZeroDelta : IO Bool
testNonZeroDelta = do
  base <- linearLayer {ex=TestExecutor} {dt=TestDType} {i=2} {o=3}
                       "lora_nz_delta.base"

  -- A : Tensor [2, 2] row-major [1.0, 2.0, 3.0, 4.0]
  let aBuf = prim__allocDoubles 4
      aBuf' = setAt 3 4.0 (setAt 2 3.0 (setAt 1 2.0 (setAt 0 1.0 aBuf)))
  a <- tparam2d {ex=TestExecutor} {dt=TestDType} {o=2} {i=2}
                "lora_nz_delta.lora_A" aBuf'

  -- B : Tensor [3, 2] row-major [0,0, 1.0,0, 0,0]
  --   = B[0][0..1] = (0,0), B[1][0..1] = (1.0,0), B[2][0..1] = (0,0)
  let bBuf  = prim__allocDoubles 6
      bBuf' = zeroBufN 0 6 bBuf
      bBuf2 = setAt 2 1.0 bBuf'       -- row 1, col 0
  b <- tparam2d {ex=TestExecutor} {dt=TestDType} {o=3} {i=2}
                "lora_nz_delta.lora_B" bBuf2

  let lora : LoraLinearState 2 3 TestExecutor TestDType WithGrad
      lora = MkLoraLinear {rank=2} base a b 4.0

  input <- mkInput (the (Vect 2 Double) [1.0, 0.0])
  (_, baseOut) <- applyVar base input
  loraOut       <- applyLoraLinear lora input
  baseVals <- readVec 3 baseOut.tensorPtr
  loraVals <- readVec 3 loraOut.tensorPtr
  let expectedDelta : List Double
      expectedDelta = [0.0, 2.0, 0.0]
      actualDelta : List Double
      actualDelta = diffList loraVals baseVals
      mdiff = maxAbsDiff actualDelta expectedDelta
  if mdiff < 1.0e-12
    then check ("LoRA non-zero-B delta matches (alpha/r)·B·A·x exactly "
                ++ "(max-abs-diff " ++ show mdiff ++ ")") True
    else do
      putStrLn ("  FAIL: max-abs-diff " ++ show mdiff)
      putStrLn ("    base:           " ++ show baseVals)
      putStrLn ("    lora:           " ++ show loraVals)
      putStrLn ("    actual delta:   " ++ show actualDelta)
      putStrLn ("    expected delta: " ++ show expectedDelta)
      pure False

export
tests : List (IO Bool)
tests =
  [ testInitZeroEquivalence
  , testNonZeroDelta
  ]
