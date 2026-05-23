||| Compile-time smoke test for `DType.Core`. Pure type-level — every
||| definition is `0`-quantity so the file produces no runtime code.
||| Purpose: prove that the `UpcastableTo` derivation actually
||| resolves via Idris's auto-search, not just that the module
||| declares the interface.
|||
||| Remove or move into a dedicated test package once the precision-
||| parameter work has landed and the demo examples carry the
||| equivalent assertions.
module DType.Test

import DType.Core
-- Note: `LTE` is re-exported by DType.Core via `import public Data.Nat`,
-- so users of UpcastableTo don't need to import Data.Nat separately.


-- 1. IsDType resolution: dtypeName / dtypeBytes are reachable for
--    each family alias.

0 f32Name : String
f32Name = dtypeName {t = F32}

0 i16Name : String
i16Name = dtypeName {t = I16}

0 bf16Name : String
bf16Name = dtypeName {t = BF16}

0 u8Name : String
u8Name = dtypeName {t = U8}


-- 2. Precision resolution: precisionRank exists for parameterized
--    families, not just floats.

0 f32Rank : Nat
f32Rank = precisionRank {t = F32}

0 i16Rank : Nat
i16Rank = precisionRank {t = I16}

0 bf16Rank : Nat
bf16Rank = precisionRank {t = BF16}


-- 3. UpcastableTo: derived upcasts auto-resolve.
--    Float / BFloat pairs resolve via the LosslessTo → UpcastableTo
--    bridge (F1 of #410); integer within-family pairs resolve via
--    the per-family ladders in DType.Core. Cross-family lossless
--    edges (BF16→F32, I32→F64, Bool→F32) also auto-resolve via
--    LosslessTo.

0 reflF32     : UpcastableTo F32 F32
reflF32       = %search

0 upcastF32F64 : UpcastableTo F32 F64
upcastF32F64   = %search

0 reflI16     : UpcastableTo I16 I16
reflI16       = %search

0 upcastI8I64  : UpcastableTo (IntN 8) (IntN 64)
upcastI8I64    = %search

0 upcastU8U16  : UpcastableTo (UInt 8) (UInt 16)
upcastU8U16    = %search

0 upcastBF16F32 : UpcastableTo BF16 F32
upcastBF16F32   = %search

0 upcastF16F32 : UpcastableTo (Float 16) F32
upcastF16F32   = %search

0 upcastI32F64 : UpcastableTo (IntN 32) F64
upcastI32F64   = %search

0 upcastU8F32 : UpcastableTo (UInt 8) F32
upcastU8F32   = %search

0 upcastBoolF32 : UpcastableTo Bool F32
upcastBoolF32   = %search

-- Direct LTE probe: are these provable via auto-search?
0 lte8_64 : LTE 8 64
lte8_64 = %search

0 lte16_32 : LTE 16 32
lte16_32 = %search


-- 4. Negative cases — these need the corresponding type signatures
--    to fail to typecheck. They live in the design memo's "what
--    fails" table rather than here; %search would fail at build time
--    and prevent compilation. Smoke verification is the positive
--    cases above.
