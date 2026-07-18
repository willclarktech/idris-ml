||| DType — type-safe tensor element type, with parameterized dtype
||| families and a derived precision partial-order on float upcasts.
|||
||| Dtype families are open `Nat`-parameterized type constructors:
|||
||| * `Float n` — IEEE 754 float with `n` bits (`F32 = Float 32`).
||| * `BFloat n` — brain-float family. Distinct from `Float`: BF16 and
|||   F16 are both 16 bits but have incomparable mantissa/exponent
|||   layouts, so cross-family conversion is *always* explicit.
||| * `IntN n` / `UInt n` — signed/unsigned integers (Idris reserves
|||   the bare name `Int`, hence the `N` suffix).
||| * `Bool` — boolean storage.
|||
||| Three typeclasses split the work:
|||
||| * `IsDType t` — t is a valid tensor element type. Carries
|||   `dtypeName`/`dtypeBytes` metadata. One polymorphic instance per
|||   family (`IsDType (Float n)`, `IsDType (IntN n)`, ...).
||| * `Precision t` — t has a bit-width rank that drives within-family
|||   lossless upcasts. `precisionRank = n` for `Float n`, `BFloat n`,
|||   `IntN n`, `UInt n`. `Bool` has no `Precision` instance (no
|||   bit-width concept).
||| * `UpcastableTo from to` — lossless conversion witness. Derived
|||   per-family from `LTE m n` on the bit-widths: `Float m → Float n`
|||   iff `LTE m n`, same for `BFloat` / `IntN` / `UInt`.
|||   **No cross-family derivation** — converting a `UInt 8` to a
|||   `Float 16` or a `BFloat 16` to a `Float 32` always requires an
|||   explicit `tcast` op, since the compiler can't safely auto-detect
|||   lossless cross-boundary conversions in general (a `UInt 8` value
|||   fits in F16's mantissa, but the result type is different — only
|||   the user knows whether the cast is semantically correct).
|||
||| `Compatible (0 ex : Executor) (0 t : DType)` is the empty capability
||| marker — device d can store dtype t. Instance head IS the proof.
|||
||| Design memo: `docs/develop/precision-parameter.md`.
module Ml.DType.Core

import public Data.Nat

import Ml.Executor.Core

----------------------------------------------------------------------
-- `DType` kind alias
--
-- Kind-level slot for Tensor's `t` parameter. 0-quantity, no runtime
-- enforcement — exists for documentation at every kind-binder site.
-- Same trick as `Executor.Core`'s `Executor = Type`.
----------------------------------------------------------------------

public export
0 DType : Type
DType = Type

----------------------------------------------------------------------
-- Dtype families
--
-- Each family is a `Nat`-parameterized type constructor. Users
-- write `Float 32` directly, or use the aliases (`F32`) below.
-- Constructors are present to mirror the Executor tag convention
-- (`TapeExecutor` / `TorchExecutor d` / `MlxExecutor s`); user code never
-- constructs values of these types — they exist only as
-- type-level tags.
----------------------------------------------------------------------

||| IEEE 754 binary floating point of n bits. `F32`, `F64`, `F16`
||| are aliases.
public export
data Float : Nat -> Type where MkFloat : Float n

||| Brain-float family. Distinct from `Float`: BF16 and F16 share a
||| bit width but have incomparable representations
||| (BF16 trades mantissa for exponent range, F16 the opposite), so
||| there is no `UpcastableTo` between the two — only to a wider
||| target like `Float 32`. The `BFloat` ladder gets its own
||| `UpcastableTo` instance below.
public export
data BFloat : Nat -> Type where MkBFloat : BFloat n

||| Signed two's-complement integer of n bits. (Idris reserves the
||| bare name `Int`; `IntN n` is this codebase's parameterized
||| variant. `I32` alias below.)
public export
data IntN : Nat -> Type where MkIntN : IntN n

||| Unsigned integer of n bits.
public export
data UInt : Nat -> Type where MkUInt : UInt n

----------------------------------------------------------------------
-- Common aliases
--
-- Exported for ergonomics. `Tensor [4] TapeExecutor F64 WithGrad` reads
-- better than `Tensor [4] TapeExecutor (Float 64) WithGrad`.
----------------------------------------------------------------------

||| Ternary value-set {-1, 0, +1}, packed 4 elements per byte using
||| a 2-bit two's-complement encoding (00 -> 0, 01 -> +1, 11 -> -1,
||| 10 -> reserved/invalid). Distinct from `IntN 2` (which has full
||| -2..1 range) — Ternary's value set is fixed to BitNet b1.58's
||| three-value alphabet so backend kernels can specialise for it.
||| Pack/unpack helpers in `packages/backends/shared_utils.c`.
public export
data Ternary : Type where MkTernary : Ternary

||| Binary value-set {-1, +1}, packed 8 elements per byte (1 bit each,
||| 0 -> +1, 1 -> -1). Slot reserved for future BitNet 1-bit variants
||| (e.g. BitNet b1, original 2023 paper). No kernels in B1; the
||| typeclass instances exist so callers can spell `tcast` targets
||| without a backend round-trip.
public export
data Binary : Type where MkBinary : Binary

public export
F16 : Type
F16 = Float 16

public export
F32 : Type
F32 = Float 32

public export
F64 : Type
F64 = Float 64

public export
BF16 : Type
BF16 = BFloat 16

public export
I8 : Type
I8 = IntN 8

public export
I16 : Type
I16 = IntN 16

public export
I32 : Type
I32 = IntN 32

public export
I64 : Type
I64 = IntN 64

public export
U8 : Type
U8 = UInt 8

----------------------------------------------------------------------
-- IsDType — valid-element-type capability
--
-- One polymorphic instance per family. Carries diagnostic metadata
-- (human-readable name, bytes-per-element). Runtime support — the
-- per-dtype FFI primitives — lives on the separate `RuntimeDType`
-- interface below, so that polymorphic dtypes (e.g. `Float n` for
-- arbitrary n) can claim `IsDType` for type-system purposes without
-- needing a C backend implementation.
----------------------------------------------------------------------

public export
interface IsDType (0 t : Type) where
  ||| Human-readable tag ("f32", "f64", "bf16", "i32", ...). Used for
  ||| diagnostic printing; runtime dispatch goes through `RuntimeDType`'s
  ||| per-dtype FFI symbols, not via this name.
  dtypeName  : String

  ||| Storage size of one element in bytes.
  dtypeBytes : Int

public export
{n : Nat} -> IsDType (Float n) where
  dtypeName  = "f" ++ show n
  dtypeBytes = cast n `div` 8

public export
{n : Nat} -> IsDType (BFloat n) where
  dtypeName  = "bf" ++ show n
  dtypeBytes = cast n `div` 8

public export
{n : Nat} -> IsDType (IntN n) where
  dtypeName  = "i" ++ show n
  dtypeBytes = cast n `div` 8

public export
{n : Nat} -> IsDType (UInt n) where
  dtypeName  = "u" ++ show n
  dtypeBytes = cast n `div` 8

public export
IsDType Bool where
  dtypeName  = "bool"
  dtypeBytes = 1

-- Sub-byte dtypes: dtypeBytes is 0 as a "size-is-not-1-byte-per-element"
-- sentinel — callers that compute buffer sizes must consult the dtype's
-- pack rate (4 ternary or 8 binary slots per byte) rather than
-- multiplying numel by dtypeBytes. The sentinel is intentional: a code
-- path that asks dtypeBytes for a packed dtype is using the wrong
-- arithmetic and crashes loudly (numel * 0 = 0 buffer) rather than
-- silently undersizing.
public export
IsDType Ternary where
  dtypeName  = "ternary"
  dtypeBytes = 0

public export
IsDType Binary where
  dtypeName  = "binary"
  dtypeBytes = 0

----------------------------------------------------------------------
-- RuntimeDType — per-dtype FFI primitive capability
--
-- Carries the concrete `prim__create*` family for each dtype that has
-- a C-side implementation. Each instance binds the methods to its own
-- per-dtype FFI symbols (e.g. `tensor_create_scalar_f32` vs `_f64`),
-- so dispatch is static through typeclass resolution — no global
-- enum, no runtime tag passed across the FFI.
--
-- Unlike `IsDType` (polymorphic across `Float n` etc.), `RuntimeDType`
-- instances are concrete-per-dtype: only dtypes with a working C
-- runtime declare them. Backend asymmetry (e.g. mlx Metal has no F64,
-- no half-precision, no integer storage) is expressed by which
-- `Compatible ex dt` pairs hold — the unified dtag-dispatch create
-- symbols are present on every backend, but the per-backend body
-- aborts on unsupported dtags as defence-in-depth.
--
-- Method signatures mirror the existing top-level `prim__create*`
-- bindings in `Tensor.idr`; instances are defined where those
-- primitives are accessible.
----------------------------------------------------------------------

public export
interface RuntimeDType (0 t : Type) where
  ||| Runtime selector for this dtype, kind-major / precision-minor
  ||| (see `Tensor.idr` for the full layout — `0` is reserved as
  ||| invalid; defined slots are `1=Bool, 4=U8, 8/9/10/11=I8/I16/I32/I64,
  ||| 13/14/15=F16/F32/F64, 17=BF16`; sub-byte families 24-31 are
  ||| reserved for future quantization dtypes). The `dtCreate*` free
  ||| functions (in `Tensor`) pass it to the device's
  ||| `primCreate*Streamed` method, which switches on it to pick the
  ||| concrete C-side dtype. This is how the type-level `(d, dt)`
  ||| pair drives both backend dispatch (via `d`) and dtype dispatch
  ||| (via this tag) without a 2-D typeclass.
  dtypeTag : Int

----------------------------------------------------------------------
-- Precision — bit-width rank for within-family upcasts
--
-- Applies to every `Nat`-parameterized family: `Float n`, `BFloat n`,
-- `IntN n`, `UInt n`. Each carries a `precisionRank = n` (the bit
-- width) used by `UpcastableTo`'s per-family derivation.
--
-- `Bool` deliberately has no `Precision` instance (no bit-width
-- precision concept). Cross-family upcasts have no derivation —
-- the per-family `UpcastableTo` instances below scope each ladder
-- to its own family, so `Float 32` and `IntN 32` sharing
-- `precisionRank = 32` doesn't accidentally make them mutually
-- upcastable.
----------------------------------------------------------------------

public export
interface IsDType t => Precision (0 t : Type) where
  ||| Bit-width of the storage. `LTE m n` on two `Precision` instances
  ||| of the same family is sufficient (and necessary) for an
  ||| `UpcastableTo` instance to exist.
  precisionRank : Nat

public export
{n : Nat} -> Precision (Float n) where
  precisionRank = n

public export
{n : Nat} -> Precision (BFloat n) where
  precisionRank = n

public export
{n : Nat} -> Precision (IntN n) where
  precisionRank = n

public export
{n : Nat} -> Precision (UInt n) where
  precisionRank = n

----------------------------------------------------------------------
-- UpcastableTo — lossless conversion within a single dtype family
--
-- One instance per family ladder. Each requires an `LTE m n` proof
-- on the bit widths and Idris's auto-search synthesises it from
-- `LTEZero`/`LTESucc`/`lteRefl`. Cross-family combinations (e.g.
-- `UpcastableTo (BFloat 16) (Float 32)`, `UpcastableTo (UInt 8) F16`)
-- have no matching instance and therefore fail to typecheck — they
-- require an explicit `tcastUnsafe` op.
--
-- Used as `=>` constraint on `toDeviceAs` and `tcast`'s lossless
-- variant. `toExecutor` (same-dtype across devices) doesn't need it.
----------------------------------------------------------------------

||| Reflexive `LTE n n` exposed as a `%hint` so auto-search resolves
||| same-bit-width upcasts in O(1) search steps instead of recursing
||| `LTESucc` N times. Without this hint, Idris-2's default auto-implicit
||| search depth (~50) caps below the reflexive F64 case `LTE 64 64`,
||| so `tcast` on a same-dtype tensor (e.g. `F64 → F64`) fails to find
||| `UpcastableTo (Float 64) (Float 64)`. Marked `%hint` so the prelude
||| `lteRefl`-style structural recursion is bypassed by direct
||| reflexivity; the body still reduces to a normal `LTE` value when
||| elaborated, but as a single function call rather than a 64-deep
||| search tree.
public export %hint
upcastLteRefl : {n : Nat} -> LTE n n
upcastLteRefl {n = 0}   = LTEZero
upcastLteRefl {n = S k} = LTESucc (upcastLteRefl {n = k})

public export
interface UpcastableTo (0 from : Type) (0 to : Type) where

-- Within-family integer ladders. `Float`/`BFloat` within-family
-- ladders used to live here too, but are now derived via
-- `LosslessTo → UpcastableTo` (see #410 F1). The integer ladders
-- stay per-family because `LosslessTo` doesn't currently witness
-- integer→integer; if/when needed, fold them into the LosslessTo
-- machinery (a future "Bridge LosslessTo within integer families"
-- row).
public export
{m, n : Nat} -> LTE m n => UpcastableTo (IntN m) (IntN n) where

public export
{m, n : Nat} -> LTE m n => UpcastableTo (UInt m) (UInt n) where

----------------------------------------------------------------------
-- IsFloating / IsIntegral — op-level dtype kind gates
--
-- Empty capability markers (like `UpcastableTo` / `Compatible`) that
-- classify a dtype by *kind*, used to constrain operations at the type
-- level rather than only the backend. `IsFloating dt =>` on an op means
-- the op only makes sense for real-valued dtypes (softmax, the
-- activations, the losses, gradients); `IsIntegral dt =>` marks the
-- index/count dtypes. `Bool` is deliberately neither — it's a mask
-- dtype, not a number, so `softmax` / a loss / backprop on a `Bool`
-- tensor is a compile error (no `IsFloating Bool` instance).
--
-- One polymorphic instance per family, mirroring `Precision`. There is
-- no Idris-stdlib equivalent: `Num`/`Integral`/`Fractional` classify a
-- *value-level* numeric type, but our dtypes are zero-quantity kind tags
-- that are never instantiated, so a value-level instance is meaningless.
----------------------------------------------------------------------

public export
interface IsFloating (0 t : Type) where

public export
{n : Nat} -> IsFloating (Float n) where

public export
{n : Nat} -> IsFloating (BFloat n) where

public export
interface IsIntegral (0 t : Type) where

public export
{n : Nat} -> IsIntegral (IntN n) where

public export
{n : Nat} -> IsIntegral (UInt n) where

----------------------------------------------------------------------
-- FloatPrecision — explicit mantissa + exponent bit counts
--
-- A cast `from → to` between two float dtypes is **lossless** iff
-- `mantissaBits from ≤ mantissaBits to` AND
-- `exponentBits from ≤ exponentBits to` — neither precision nor range
-- shrinks. The single-`precisionRank` (storage bit-width) view is
-- enough for within-family ladders (F16→F32 via 16≤32) but loses the
-- separate dimensions across families: BF16 and F16 both have
-- bit-width 16, but BF16 has wider exponent (8 vs 5) and narrower
-- mantissa (7 vs 10) — so neither is a lossless upcast of the other.
--
-- This typeclass exposes both dimensions so a proof witness
-- `LosslessTo from to` can be derived structurally — see below.
--
-- Instances are spelled out per concrete bit-width rather than as
-- polymorphic `{n : Nat} ->` because each width has a distinct
-- (mantissa, exponent) layout (IEEE 754: F16 has 10/5, F32 has 23/8,
-- F64 has 52/11; BF16 has 7/8). No formula collapses them cleanly.
----------------------------------------------------------------------

public export
interface IsFloating t => FloatPrecision (0 t : Type) where
  ||| Explicit-stored fraction bits (excluding IEEE 754's hidden 1).
  mantissaBits : Nat
  ||| Biased exponent bits.
  exponentBits : Nat

public export
FloatPrecision (Float 16) where
  mantissaBits = 10
  exponentBits = 5

public export
FloatPrecision (Float 32) where
  mantissaBits = 23
  exponentBits = 8

public export
FloatPrecision (Float 64) where
  mantissaBits = 52
  exponentBits = 11

public export
FloatPrecision (BFloat 16) where
  mantissaBits = 7
  exponentBits = 8

----------------------------------------------------------------------
-- LosslessTo — structural "every value of `from` is exactly
-- representable in `to`" witness, across families.
--
-- Empty typeclass with per-family-pair instances. Each instance
-- carries whatever structural condition fits its (from, to) pair:
-- mantissa+exponent LTE for float→float, `n ≤ mantissaBits + 2` for
-- IntN→Float (signed-integer max value 2^(n-1) bounded by float's
-- 2^(mb+1) exact-integer range), `n ≤ mantissaBits + 1` for
-- UInt→Float (max 2^n - 1), no condition for Bool→float (0/1
-- representable everywhere), no condition for Ternary/Binary→float
-- (value sets `{-1, 0, +1}` / `{-1, +1}` exactly representable in
-- any IEEE float — these instances ship with the BitNet ternary
-- dtype row, #411).
--
-- Bridge `LosslessTo from to => UpcastableTo from to` (below)
-- threads every LosslessTo edge into the existing `tcast` /
-- `toDeviceAs` resolution surface, so users get implicit safe
-- cross-family casts (`tcast bf16_t {to=F32}` just compiles,
-- `tcast int32_t {to=F64}` likewise) — while lossy edges
-- (`tcast f32_t {to=BF16}`, `tcast int64_t {to=F32}`) still refuse
-- to type-check and require explicit `tcastUnsafe`.
--
-- The lossless edges this covers (concrete pairs at the float
-- widths idris-ml ships — F16, F32, F64, BF16):
--
-- Float / BFloat → Float / BFloat:
--   * F16  → F32   (mantissa 10→23, exponent 5→8)
--   * F16  → F64   (10→52, 5→11)
--   * F32  → F64   (23→52, 8→11)
--   * BF16 → F32   (7→23, 8→8)
--   * BF16 → F64   (7→52, 8→11)
--   * F16  → BF16  — actually lossy (exponent 5→8 grows but
--                   mantissa 10→7 shrinks); refuses to derive.
--   * BF16 → F16   — actually lossy (mantissa 7→10 grows but
--                   exponent 8→5 shrinks); refuses to derive.
--
-- IntN → Float / BFloat (max IntN n value is 2^(n-1); fits exactly
-- if 2^(n-1) ≤ 2^(mb+1), i.e. n ≤ mb + 2):
--   * I8   → F16   (8 ≤ 12), F32, F64, BF16 (8 ≤ 9)
--   * I16  → F32 (16 ≤ 25), F64 (16 ≤ 54)
--   * I32  → F64   (32 ≤ 54)
--   * I64  → none of the floats we ship (54 too small for 64-bit)
--
-- UInt → Float / BFloat (max value 2^n - 1; fits exactly if
-- 2^n ≤ 2^(mb+1), conservatively n ≤ mb + 1):
--   * U8   → F16   (8 ≤ 11), F32, F64, BF16 (8 ≤ 8)
--
-- Bool → any Float / BFloat: trivially representable (0/1 always
-- exact).
-- Bool → IntN m: trivially representable if m ≥ 2 (IntN 2 covers
-- -2..1, fits {0, 1}). Not enforced via LTE today — IntN starts at
-- 8 bits in practice, so trivially satisfied.
-- Bool → UInt m: trivially representable if m ≥ 1.
--
-- Lossy edges (no LosslessTo instance; explicit `tcastUnsafe`
-- required) at the float widths we ship:
--   * F32 / F64 → BF16 / F16        (mantissa shrinks)
--   * F64 → F32                     (mantissa shrinks)
--   * BF16 ↔ F16                    (each direction shrinks one dim)
--   * I64 → any float we ship       (mantissa overflow)
--
-- The point: idris-ml refuses silent lossy mid-graph casts that
-- PyTorch's autocast would silently introduce. Lossy edges have to
-- be code-visible.
----------------------------------------------------------------------

public export
interface LosslessTo (0 from : Type) (0 to : Type) where

-- Float / BFloat → Float / BFloat: mantissa-bits + exponent-bits
-- both non-decreasing. Covers all four cross-product combinations
-- (F→F, BF→BF, F→BF, BF→F) since both families have FloatPrecision.
public export
{from, to : Type} ->
FloatPrecision from                               => FloatPrecision to =>
LTE (mantissaBits {t=from}) (mantissaBits {t=to}) =>
LTE (exponentBits {t=from}) (exponentBits {t=to}) =>
LosslessTo from to where

-- IntN n → Float / BFloat: signed-integer max value 2^(n-1) bounded
-- by float's 2^(mb+1) exact-integer range → `n ≤ mb + 2`.
public export
{n : Nat} -> {to : Type} ->
FloatPrecision to                   =>
LTE n (S (S (mantissaBits {t=to}))) =>
LosslessTo (IntN n) to where

-- UInt n → Float / BFloat: max value 2^n - 1 ≤ 2^(mb+1) → `n ≤ mb + 1`.
public export
{n : Nat} -> {to : Type} ->
FloatPrecision to               =>
LTE n (S (mantissaBits {t=to})) =>
LosslessTo (UInt n) to where

-- Bool → Float / BFloat: trivially lossless (0 and 1 representable
-- in every IEEE float).
public export
{to : Type} -> FloatPrecision to => LosslessTo Bool to where

-- Bool → IntN m (m ≥ 2): IntN 2 covers -2..1 which contains {0, 1}.
-- Bool → UInt m (m ≥ 1): UInt 1 covers {0, 1}.
public export
{m : Nat} -> LTE 2 m => LosslessTo Bool (IntN m) where
public export
{m : Nat} -> LTE 1 m => LosslessTo Bool (UInt m) where

-- Ternary {-1, 0, +1} → any Float / BFloat: all three values are
-- exactly representable in every IEEE float (the mantissa needs
-- to hold integers up to 1, which it does even in F16). No FloatPrecision
-- gate needed.
public export
{to : Type} -> FloatPrecision to => LosslessTo Ternary to where

-- Binary {-1, +1} → any Float / BFloat: same reasoning.
public export
{to : Type} -> FloatPrecision to => LosslessTo Binary to where

-- Ternary → IntN m (m ≥ 2): IntN 2 covers -2..1 which contains {-1, 0, 1}.
-- No UInt edge (Ternary has -1, which UInt can't represent).
public export
{m : Nat} -> LTE 2 m => LosslessTo Ternary (IntN m) where

-- Binary → IntN m (m ≥ 2): IntN 2 covers -2..1 which contains {-1, +1}.
-- No UInt edge (Binary has -1).
public export
{m : Nat} -> LTE 2 m => LosslessTo Binary (IntN m) where

-- Bridge: every LosslessTo instance is an UpcastableTo. Threads the
-- cross-family lossless edges into the existing `tcast` /
-- `toDeviceAs` surface.
public export
LosslessTo from to => UpcastableTo from to where

----------------------------------------------------------------------
-- Compatible — (device, dtype) admissibility
--
-- Empty capability marker. `Compatible D T where` declares "device
-- D can store dtype T." Constrains tensor-construction smart
-- constructors so a backend without F32 support refuses to compile
-- against `Tensor [..] D F32 ...` at the spelling site.
----------------------------------------------------------------------

public export
interface Compatible (0 ex : Executor) (0 t : DType) where
