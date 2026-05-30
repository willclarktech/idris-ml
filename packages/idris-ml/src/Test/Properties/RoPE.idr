-- Test.Properties.RoPE — round-trip invariant for Llama-style RoPE.
--
-- Tests: applyRopeInverse · applyRope ≈ id (within F64 ULPs).
--
-- The 2D rotation underlying RoPE has an algebraic inverse — rotate
-- by -θ, which sign-flips the sin term — so round-tripping should
-- return the input bit-close to exact. This property exercises the
-- composition of `applyRope` + `applyRopeInverse` over random inputs
-- of shape [seq, headDim], catching:
--   * a sign flip in either direction (drift ≈ 2x the rotation)
--   * a swap of first/second halves (drift = full magnitude)
--   * a cos/sin table indexing offset (drift varies by position)
--
-- Uses `checkPropertyIO` since the body calls IO-typed smart
-- constructors (`buildLlamaRoPETables`, `applyRope`,
-- `applyRopeInverse`). See `packages/idris-test/src/Test/Property.idr`
-- for the trade-off (no shrinking).
module Test.Properties.RoPE

import Data.Vect

import Test.Property
import Test.Config
import Test.Harness as Harness

import Device
import Tensor
import Array
import Layer.RoPE

%default partial

-- Tiny dims keep the counterexample readable (Show output) and the
-- per-case wall short. The invariant is shape-invariant — n=4
-- exercises the rotation, narrow, concat path same as n=4096.
SEQ : Nat
SEQ = 4

HEAD_DIM : Nat
HEAD_DIM = 8  -- halfDim = 4

MAX_POS : Nat
MAX_POS = 16

-- Standard RoPE base. Llama-3 uses 500000; either works as long as
-- forward + inverse share it, which they do here (same tables).
ROPE_BASE : Double
ROPE_BASE = 10000.0

mkInput2d : {seq, headDim : Nat} ->
            Vect seq (Vect headDim Double) ->
            Tensor [seq, headDim] TestDevice TestDType WithGrad
mkInput2d xs =
  let rows = map (\r => VArray (map SArray r)) xs
      raw  = bulkToTensor2d {d=TestDevice} {dt=TestDType} {b=seq} {i=headDim} rows
  in tinput2d {m=seq} {n=headDim} raw

-- Element-wise readback into a row-major flat List Double. Simpler
-- than reconstructing the dependent Vect rows.
readMatFlat : (rows, cols : Nat) -> AnyPtr -> IO (List Double)
readMatFlat rows cols p = go (cast {to=Int} rows) (cast {to=Int} cols) 0 0 []
  where
    go : Int -> Int -> Int -> Int -> List Double -> IO (List Double)
    go rEnd cEnd r c acc =
      if r >= rEnd
        then pure (reverse acc)
        else if c >= cEnd
          then go rEnd cEnd (r + 1) 0 acc
          else
            let v = primItem2d {d=TestDevice} p r c
            in go rEnd cEnd r (c + 1) (v :: acc)

flattenRows : Vect r (Vect c Double) -> List Double
flattenRows xs = concat (map toList (toList xs))

maxAbsDiff : List Double -> List Double -> Double
maxAbsDiff []        _         = 0.0
maxAbsDiff _         []        = 0.0
maxAbsDiff (a :: as) (b :: bs) = max (abs (a - b)) (maxAbsDiff as bs)

prop_rope_inverse_commutativity_body : Vect SEQ (Vect HEAD_DIM Double) -> IO Bool
prop_rope_inverse_commutativity_body xs = do
  tables <- buildLlamaRoPETables {d=TestDevice} {dt=TestDType}
                                 {maxPos=MAX_POS} {headDim=HEAD_DIM}
                                 ROPE_BASE noScaling
  let inputT = mkInput2d xs
  rotated  <- applyRope        {d=TestDevice} {seq=SEQ} {headDim=HEAD_DIM}
                               {maxPos=MAX_POS} tables 0 inputT
  restored <- applyRopeInverse {d=TestDevice} {seq=SEQ} {headDim=HEAD_DIM}
                               {maxPos=MAX_POS} tables 0 rotated
  outVals <- readMatFlat SEQ HEAD_DIM restored.tensorPtr
  let inVals  = flattenRows xs
      mdiff   = maxAbsDiff outVals inVals
  if mdiff < 1.0e-10
    then pure True
    else do
      putStrLn $ "    max-abs-diff = " ++ show mdiff ++ "  (tol 1e-10)"
      putStrLn $ "    input  = " ++ show inVals
      putStrLn $ "    output = " ++ show outVals
      pure False

prop_rope_inverse_commutativity : IO Bool
prop_rope_inverse_commutativity = checkPropertyIOn
  "rope_inverse_commutativity"
  25
  (vect SEQ (vect HEAD_DIM (double (linearFracFrom 0.0 (-1.0) 1.0))))
  prop_rope_inverse_commutativity_body

export
tests : List (IO Bool)
tests = [ prop_rope_inverse_commutativity ]
