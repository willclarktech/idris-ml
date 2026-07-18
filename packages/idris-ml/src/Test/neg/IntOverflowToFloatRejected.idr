||| Negative compile test: F1 (#412) — confirms `LosslessTo` REFUSES
||| to resolve when an integer source's range overflows the target
||| float's exact-integer range. This file MUST NOT type-check. If
||| it ever starts to compile, the int → float lossless gate has
||| regressed: silent mantissa-overflowing casts (`IntN 64` values
||| > 2^24 cast to F32, with the upper bits silently rounded away)
||| would be permitted under the typeclass, defeating the whole
||| point of the structural witness.
|||
||| Direction tested: IntN 64 → Float 32. Max IntN 64 value is
||| 2^63 ≈ 9.2 × 10^18; F32 mantissa is 23 bits so the exact-integer
||| range is `[-2^24, 2^24]` ≈ ±16M. The required `LTE 64 25`
||| (`64 ≤ mantissaBits(F32) + 2 = 25`) is unsolvable.

module IntOverflowToFloatRejected

import Ml.DType.Core

-- This proof should fail to typecheck:
--   `LosslessTo (IntN 64) (Float 32)` requires `LTE 64 25`, which
--   has no inhabitant. Auto-search refuses, compile error.
proofI64ToF32Lossy : LosslessTo (IntN 64) (Float 32)
proofI64ToF32Lossy = %search
