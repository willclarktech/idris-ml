||| Unit tests for `KVCache` — the per-layer K/V cache module.
|||
||| The non-trivial thing this exercises is the rank-2 axis-0 concat
||| primitive (`tconcat2dAxis0`) under the cache's append semantics:
||| starting from `Empty`, append 2 rows, then 1 more row, verify the
||| cache length is 3 and the row-major data is exactly what
||| concat-along-axis-0 should produce.
|||
||| Pins `{d=TapeDev}` directly (same shape as `Test.HfLlama` — it's a
||| FFI-level test and tape's C arena is the predictable lane).
module Test.KVCache

import Data.Vect

import Array
import Device
import Device.Core
import Device.Tape
import Test.Harness
import KVCache
import Tensor


----------------------------------------------------------------------
-- Helpers
----------------------------------------------------------------------

mkRow2 : {n : Nat} -> Vect n Double -> IO (Tensor [1, n] TapeDev F64 NoGrad)
mkRow2 xs = do
  raw <- ioRerun (\_ => bulkToTensor2d {d=TapeDev} {dt=F64} {b=1} {i=n}
                                       [VArray (map SArray xs)])
  weakenGrad {d=TapeDev} (tinput2d {m=1} {n} raw)

mkRows2 : {m, n : Nat} -> Vect m (Vect n Double) ->
          IO (Tensor [m, n] TapeDev F64 NoGrad)
mkRows2 xss = do
  raw <- ioRerun (\_ => bulkToTensor2d {d=TapeDev} {dt=F64} {b=m} {i=n}
                                       (map (\row => VArray (map SArray row)) xss))
  weakenGrad {d=TapeDev} (tinput2d {m} {n} raw)


-- Read a [m, n] tensor's raw buffer into row-major List Double via
-- the same per-element accessor the other tensor-value tests use.
readMat : (m, n : Nat) -> AnyPtr -> IO (List Double)
readMat m n p = go (cast {to=Int} (m * n)) 0 []
  where
    go : Int -> Int -> List Double -> IO (List Double)
    go end i acc =
      if i >= end
        then pure (reverse acc)
        else let v = primItem1d {d=TapeDev} p i
             in go end (i + 1) (v :: acc)


----------------------------------------------------------------------
-- Bucket 1 — Empty.cacheLen = 0
----------------------------------------------------------------------

testEmptyLen : IO Bool
testEmptyLen =
  let c : KVCache 4 TapeDev F64
      c = emptyKVCache
  in check "emptyKVCache.cacheLen = 0" (cacheLen c == 0)


----------------------------------------------------------------------
-- Bucket 2 — Append into Empty gives Filled s newK newV
----------------------------------------------------------------------

testAppendIntoEmpty : IO Bool
testAppendIntoEmpty = do
  -- Append a [2, 3] row pair into an Empty cache.
  -- K data: [[1,2,3], [4,5,6]]; V data: [[7,8,9], [10,11,12]]
  k0 <- mkRows2 {m=2} {n=3}
          (the (Vect 2 (Vect 3 Double))
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  v0 <- mkRows2 {m=2} {n=3}
          (the (Vect 2 (Vect 3 Double))
               [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
  c1 <- appendKV {s=2} {kvOut=3} (the (KVCache 3 TapeDev F64) emptyKVCache) k0 v0
  case c1 of
    Empty => do
      putStrLn "  FAIL: appendKV onto Empty returned Empty"
      pure False
    Filled len k v => do
      if len /= 2
        then do
          putStrLn ("  FAIL: post-append len = " ++ show len ++ " (expected 2)")
          pure False
        else do
          kVals <- readMat 2 3 k.tensorPtr
          vVals <- readMat 2 3 v.tensorPtr
          let kExpect = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
              vExpect = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
          if kVals == kExpect && vVals == vExpect
            then check "appendKV Empty [2,3]: len=2, K + V preserved" True
            else do
              putStrLn ("  FAIL: K = " ++ show kVals ++ ", expected " ++ show kExpect)
              putStrLn ("        V = " ++ show vVals ++ ", expected " ++ show vExpect)
              pure False


----------------------------------------------------------------------
-- Bucket 3 — Append into Filled concatenates along axis 0
----------------------------------------------------------------------

testAppendIntoFilled : IO Bool
testAppendIntoFilled = do
  -- Seed with [2, 3]; append [1, 3]; expect Filled 3 with the rows
  -- in order.
  k0 <- mkRows2 {m=2} {n=3}
          (the (Vect 2 (Vect 3 Double))
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  v0 <- mkRows2 {m=2} {n=3}
          (the (Vect 2 (Vect 3 Double))
               [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
  c1 <- appendKV {s=2} {kvOut=3} (the (KVCache 3 TapeDev F64) emptyKVCache) k0 v0
  k1 <- mkRow2 {n=3} (the (Vect 3 Double) [7.0, 8.0, 9.0])
  v1 <- mkRow2 {n=3} (the (Vect 3 Double) [70.0, 80.0, 90.0])
  c2 <- appendKV {s=1} {kvOut=3} c1 k1 v1
  case c2 of
    Empty => do
      putStrLn "  FAIL: appendKV onto Filled returned Empty"
      pure False
    Filled len k v => do
      if len /= 3
        then do
          putStrLn ("  FAIL: post-append len = " ++ show len ++ " (expected 3)")
          pure False
        else do
          kVals <- readMat 3 3 k.tensorPtr
          vVals <- readMat 3 3 v.tensorPtr
          -- Concat-along-axis-0 of [[1,2,3],[4,5,6]] and [[7,8,9]] →
          -- [[1,2,3],[4,5,6],[7,8,9]] = [1,2,3,4,5,6,7,8,9] row-major.
          let kExpect = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
              vExpect = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
          if kVals == kExpect && vVals == vExpect
            then check "appendKV Filled[2,3] + [1,3]: len=3, axis-0 concat preserves row order" True
            else do
              putStrLn ("  FAIL: K = " ++ show kVals ++ ", expected " ++ show kExpect)
              putStrLn ("        V = " ++ show vVals ++ ", expected " ++ show vExpect)
              pure False


----------------------------------------------------------------------
-- Bucket 4 — cacheLen accumulates correctly across multiple appends
----------------------------------------------------------------------

testCacheLenAccumulates : IO Bool
testCacheLenAccumulates = do
  k1 <- mkRow2 {n=3} (the (Vect 3 Double) [1.0, 2.0, 3.0])
  v1 <- mkRow2 {n=3} (the (Vect 3 Double) [4.0, 5.0, 6.0])
  k2 <- mkRow2 {n=3} (the (Vect 3 Double) [1.0, 2.0, 3.0])
  v2 <- mkRow2 {n=3} (the (Vect 3 Double) [4.0, 5.0, 6.0])
  k3 <- mkRow2 {n=3} (the (Vect 3 Double) [1.0, 2.0, 3.0])
  v3 <- mkRow2 {n=3} (the (Vect 3 Double) [4.0, 5.0, 6.0])
  k4 <- mkRow2 {n=3} (the (Vect 3 Double) [1.0, 2.0, 3.0])
  v4 <- mkRow2 {n=3} (the (Vect 3 Double) [4.0, 5.0, 6.0])
  c0 <- pure (the (KVCache 3 TapeDev F64) emptyKVCache)
  c1 <- appendKV {s=1} {kvOut=3} c0 k1 v1
  c2 <- appendKV {s=1} {kvOut=3} c1 k2 v2
  c3 <- appendKV {s=1} {kvOut=3} c2 k3 v3
  c4 <- appendKV {s=1} {kvOut=3} c3 k4 v4
  if cacheLen c4 == 4
    then check "cacheLen accumulates across 4 single-row appends" True
    else do
      putStrLn ("  FAIL: cacheLen c4 = " ++ show (cacheLen c4) ++ " (expected 4)")
      pure False


----------------------------------------------------------------------
-- Suite export
----------------------------------------------------------------------

public export
suite : List (String, List (IO Bool))
suite =
  [ ("KVCache — empty + append + cacheLen",
     [ testEmptyLen
     , testAppendIntoEmpty
     , testAppendIntoFilled
     , testCacheLenAccumulates
     ])
  ]
