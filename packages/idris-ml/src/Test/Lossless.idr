module Test.Lossless

import Data.Nat

import Ml.DType.Core
import Test.Harness

-- ---------------------------------------------------------------
-- A0.5: cross-family LosslessTo float-cast witnesses
-- ---------------------------------------------------------------
--
-- These tests assert that `LosslessTo from to` resolves at the type
-- level for known-safe float upcasts (mantissa-bits non-decreasing
-- AND exponent-bits non-decreasing from source to target). If any
-- `%search` here fails, this module won't compile — that's the
-- behavioural surface this test gates.
--
-- The complementary refutation — "lossy edges do NOT resolve" — lives
-- in a `neg/` compile-fail file (see `LossyDirectionRejected.idr`).

-- BFloat 16 → Float 32: mantissa 7→23 (grows), exponent 8→8 (same).
proofBF16ToF32 : LosslessTo (BFloat 16) (Float 32)
proofBF16ToF32 = %search

-- Float 16 → Float 32: mantissa 10→23, exponent 5→8.
proofF16ToF32 : LosslessTo (Float 16) (Float 32)
proofF16ToF32 = %search

-- Float 32 → Float 64: mantissa 23→52, exponent 8→11.
proofF32ToF64 : LosslessTo (Float 32) (Float 64)
proofF32ToF64 = %search

-- BFloat 16 → Float 64: mantissa 7→52, exponent 8→11.
proofBF16ToF64 : LosslessTo (BFloat 16) (Float 64)
proofBF16ToF64 = %search

-- Float 16 → Float 64: mantissa 10→52, exponent 5→11.
proofF16ToF64 : LosslessTo (Float 16) (Float 64)
proofF16ToF64 = %search

-- ---------------------------------------------------------------
-- F1 (#412): int/uint/bool → float lossless witnesses
-- ---------------------------------------------------------------
--
-- The float-only LosslessTo from A0.5 generalised to a typeclass
-- with per-family-pair instances. New witnesses:
--
-- IntN n → Float m: n ≤ mantissaBits + 2.
--   F64 mantissa = 52, so I32 (n=32 ≤ 54) ✓, I16 ✓, I8 ✓; I64 ✗.
proofI32ToF64 : LosslessTo (IntN 32) (Float 64)
proofI32ToF64 = %search

proofI16ToF32 : LosslessTo (IntN 16) (Float 32)
proofI16ToF32 = %search

proofI8ToF16 : LosslessTo (IntN 8) (Float 16)
proofI8ToF16 = %search

-- UInt n → Float m: n ≤ mantissaBits + 1.
--   F32 mantissa = 23, so U8 ✓, U16 ✓, U24 boundary ✓; U25 ✗.
proofU8ToF32 : LosslessTo (UInt 8) (Float 32)
proofU8ToF32 = %search

proofU16ToF64 : LosslessTo (UInt 16) (Float 64)
proofU16ToF64 = %search

-- Bool → any float / BFloat (trivially: 0 and 1 always representable).
proofBoolToF32 : LosslessTo Bool (Float 32)
proofBoolToF32 = %search

proofBoolToBF16 : LosslessTo Bool (BFloat 16)
proofBoolToBF16 = %search

-- Bool → IntN m (m ≥ 2) / UInt m (m ≥ 1).
proofBoolToI8 : LosslessTo Bool (IntN 8)
proofBoolToI8 = %search

proofBoolToU8 : LosslessTo Bool (UInt 8)
proofBoolToU8 = %search

-- ---------------------------------------------------------------
-- B1 (#411): Ternary / Binary lossless witnesses
-- ---------------------------------------------------------------
--
-- {-1, 0, +1} and {-1, +1} fit exactly into every IEEE float (the
-- mantissa needs only to hold integers up to 1, which it does in
-- F16). The LosslessTo instances are FloatPrecision-gated to inherit
-- the float-family scope from the existing typeclass setup; no
-- additional Nat bound is needed.

proofTernaryToF32 : LosslessTo Ternary (Float 32)
proofTernaryToF32 = %search

proofTernaryToBF16 : LosslessTo Ternary (BFloat 16)
proofTernaryToBF16 = %search

proofTernaryToF16 : LosslessTo Ternary (Float 16)
proofTernaryToF16 = %search

proofTernaryToI8 : LosslessTo Ternary (IntN 8)
proofTernaryToI8 = %search

proofBinaryToF32 : LosslessTo Binary (Float 32)
proofBinaryToF32 = %search

proofBinaryToI8 : LosslessTo Binary (IntN 8)
proofBinaryToI8 = %search

-- ---------------------------------------------------------------
-- F1 (#412): UpcastableTo bridge — LosslessTo edges thread into
-- the existing tcast-resolution surface
-- ---------------------------------------------------------------
--
-- These probe that the `LosslessTo from to => UpcastableTo from to`
-- bridge wires every cross-family lossless edge into `tcast`'s
-- typeclass constraint. Failure of any of these to compile means
-- the bridge isn't firing.

0 upcastableBF16ToF32 : UpcastableTo (BFloat 16) (Float 32)
upcastableBF16ToF32   = %search

0 upcastableI32ToF64 : UpcastableTo (IntN 32) (Float 64)
upcastableI32ToF64   = %search

0 upcastableU8ToF32 : UpcastableTo (UInt 8) (Float 32)
upcastableU8ToF32   = %search

0 upcastableBoolToF32 : UpcastableTo Bool (Float 32)
upcastableBoolToF32   = %search

-- Smoke: if all proofs above compile, this module loads and the
-- assertion is trivially true. The actual test is the compile-time
-- check; this runtime check just gives the test harness a row to
-- report.
losslessResolvesForKnownEdges : IO Bool
losslessResolvesForKnownEdges = do
  let _ = proofBF16ToF32
      _ = proofF16ToF32
      _ = proofF32ToF64
      _ = proofBF16ToF64
      _ = proofF16ToF64
  check "LosslessTo resolves for known-safe cross-family float upcasts" True

crossFamilyIntToFloatResolves : IO Bool
crossFamilyIntToFloatResolves = do
  let _ = proofI32ToF64
      _ = proofI16ToF32
      _ = proofI8ToF16
      _ = proofU8ToF32
      _ = proofU16ToF64
  check "LosslessTo resolves IntN/UInt → Float at the right widths" True

boolToAnyResolves : IO Bool
boolToAnyResolves = do
  let _ = proofBoolToF32
      _ = proofBoolToBF16
      _ = proofBoolToI8
      _ = proofBoolToU8
  check "LosslessTo resolves Bool → Float / BFloat / IntN / UInt" True

ternaryAndBinaryResolve : IO Bool
ternaryAndBinaryResolve = do
  let _ = proofTernaryToF32
      _ = proofTernaryToBF16
      _ = proofTernaryToF16
      _ = proofTernaryToI8
      _ = proofBinaryToF32
      _ = proofBinaryToI8
  check "LosslessTo resolves Ternary / Binary → Float / BFloat / IntN" True

upcastableBridgeWiresThrough : IO Bool
upcastableBridgeWiresThrough = do
  -- These references are at the type level — if the bridge instance
  -- isn't firing, this module doesn't compile and the test runner
  -- never reaches us.
  check "LosslessTo → UpcastableTo bridge resolves cross-family" True

export
tests : List (IO Bool)
tests =
  [ losslessResolvesForKnownEdges
  , crossFamilyIntToFloatResolves
  , boolToAnyResolves
  , ternaryAndBinaryResolve
  , upcastableBridgeWiresThrough
  ]
