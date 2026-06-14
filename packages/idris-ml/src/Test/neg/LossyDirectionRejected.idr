||| Negative compile test: confirms that `LosslessTo from to` REFUSES
||| to resolve for a lossy direction. This file MUST NOT type-check.
||| If it ever starts to compile, the cross-family lossless-cast gate
||| has regressed — silent F32 → BF16 mid-graph casts would be
||| permitted under the typeclass, which is exactly what idris-ml is
||| set up to refuse.
|||
||| The lossy direction here: Float 32 → BFloat 16. Mantissa shrinks
||| 23 → 7 (the cast loses ~16 bits of precision). Exponent stays at
||| 8 (BF16 inherited F32's exponent layout by design). The mantissa
||| dimension's `LTE 23 7` is unsolvable; auto-search refuses the
||| `LosslessTo` definition.

module LossyDirectionRejected

import DType.Core

-- This proof should fail to typecheck:
--   `LosslessTo (Float 32) (BFloat 16)` requires `LTE 23 7`, which
--   is `LTESucc (LTESucc (LTESucc ... LTEZero))` — but `23 ≤ 7` has
--   no value. Auto-search refuses, compile error.
proofF32ToBF16Lossy : LosslessTo (Float 32) (BFloat 16)
proofF32ToBF16Lossy = %search
