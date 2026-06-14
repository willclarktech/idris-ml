module Test.SwiGLU

import Data.Vect

import Array
import Executor
import Layer.SwiGLU
import Tensor
import Test.Config
import Test.Harness

-- Tolerance for the value-pinning test against the all-ones weight
-- reference. The forward composes ~7 ops. On tape + torch the F64
-- arithmetic is bit-stable so `1e-9` leaves headroom against the
-- PyTorch F64 oracle. On mlx the silu/multiply paths run through
-- Accelerate kernels that introduce ~1e-7 of round-off even at F64
-- storage (observed: 1.14e-7 max-abs-diff vs PyTorch F64 on
-- mlx-cpu), so the oracle tolerance loosens to `1e-6` there — still
-- well below the F32 cross-language gate's 4.96e-05 floor.
tol : Double
tol = if TestPrimaryBackend == "mlx" then 1.0e-6 else 1.0e-9

-- Read a [n] tensor's raw pointer into a List Double.
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

mkInput : {n : Nat} -> Vect n Double -> Tensor [n] TestExecutor TestDType WithGrad
mkInput xs =
  let raw = bulkToTensor {ex=TestExecutor} {dt=TestDType}
                         (VArray (map SArray xs))
  in tinput1d {n} raw

-- Fill a buffer with 1.0 at every position. Top-level so the
-- elaborator doesn't choke on it as a nested let-binding inside a do.
fillOnes : Int -> Int -> AnyPtr -> AnyPtr
fillOnes _   0 b = b
fillOnes off k b = fillOnes (off + 1) (k - 1) (prim__setDouble b off 1.0)

mkOnesWeight : {o, i : Nat} -> (name : String) ->
               IO (Tensor [o, i] TestExecutor TestDType WithGrad)
mkOnesWeight {o} {i} name =
  let nElts = cast {to=Int} (o * i)
      buf  = prim__allocDoubles nElts
      buf' = fillOnes 0 nElts buf
  in tparam2d {o} {i} name buf'

-- Construct a SwiGLUState whose three weights are all-ones. Bypasses
-- swigluLayer's Xavier-uniform init so the forward has an analytically
-- computable reference.
mkAllOnesSwiGLU : {hidden, intermediate : Nat} -> (pfx : String) ->
                   IO (SwiGLUState hidden intermediate TestExecutor TestDType WithGrad)
mkAllOnesSwiGLU pfx = do
  gateW <- mkOnesWeight {o=intermediate} {i=hidden}       (pfx ++ "_gate_weight")
  upW   <- mkOnesWeight {o=intermediate} {i=hidden}       (pfx ++ "_up_weight")
  downW <- mkOnesWeight {o=hidden}       {i=intermediate} (pfx ++ "_down_weight")
  pure (MkSwiGLU gateW upW downW)

-- For weights all-ones, hidden=2, intermediate=3, input=[1,2]:
--   gate = upW = [s, s, s] where s = 1 + 2 = 3
--   silu(3) = 3 * sigmoid(3) = 3 / (1 + exp(-3)) ≈ 2.857722...
--   mid = silu(gate) * up = [silu(3)*3, ...] ≈ 8.5731...
--   out = downW @ mid = [sum_3 mid, sum_3 mid] = [3 * silu(3) * 3, …]
-- Computed exactly by PyTorch F64 (see test description in commit message).
testForwardAllOnesAt12 : IO Bool
testForwardAllOnesAt12 = do
  sw <- mkAllOnesSwiGLU {hidden=2} {intermediate=3} "sw_ones"
  let input = mkInput (the (Vect 2 Double) [1.0, 2.0])
  (_, out) <- applySwiGLU sw input
  vals <- readVec 2 out.tensorPtr
  -- PyTorch F64 reference (matches Idris F64 bit-for-byte):
  --   gateW = upW = ones(3, 2); downW = ones(2, 3); x = [1, 2]
  --   gate = up = [3, 3, 3]
  --   silu(3) ≈ 2.8577…; mid = silu(gate) * up
  --   out = downW @ mid = [25.7195014…, 25.7195014…]
  let expected : List Double
      expected = [25.7195014242057, 25.7195014242057]
      mdiff    = maxAbsDiff vals expected
  if mdiff < tol
    then check ("SwiGLU all-ones forward at [1,2] matches PyTorch F64 " ++
                "(max-abs-diff " ++ show mdiff ++ ")") True
    else do
      putStrLn ("  FAIL: max-abs-diff " ++ show mdiff ++ " > tol " ++ show tol)
      putStrLn ("    got:      " ++ show vals)
      putStrLn ("    expected: " ++ show expected)
      pure False

testShapeAndFinite : IO Bool
testShapeAndFinite = do
  sw <- swigluLayer {ex=TestExecutor} {dt=TestDType}
                    {hidden=4} {intermediate=11}  -- intermediate ≠ 4*hidden to catch hard-coded ratios
                    "sw_shape"
  let input = mkInput (the (Vect 4 Double) [0.5, -1.5, 2.0, -0.25])
  (_, out) <- applySwiGLU sw input
  vals <- readVec 4 out.tensorPtr
  let allFinite = all (\x => x == x && abs x < 1.0e100) vals
  check "SwiGLU h=4 i=11 with mixed-sign input produces 4 finite values" allFinite

export
tests : List (IO Bool)
tests =
  [ testForwardAllOnesAt12
  , testShapeAndFinite
  ]
