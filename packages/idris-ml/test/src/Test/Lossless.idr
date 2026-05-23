module Test.Lossless

import Data.Nat

import Harness
import DType.Core


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


export
tests : List (IO Bool)
tests =
  [ losslessResolvesForKnownEdges
  ]
