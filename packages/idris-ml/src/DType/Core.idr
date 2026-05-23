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
||| `Compatible (0 d : Device) (0 t : DType)` is the empty capability
||| marker — device d can store dtype t. Instance head IS the proof.
|||
||| Design memo: `docs/develop/precision-parameter.md`.
module DType.Core

import public Data.Nat
import Device.Core


----------------------------------------------------------------------
-- `DType` kind alias
--
-- Kind-level slot for Tensor's `t` parameter. 0-quantity, no runtime
-- enforcement — exists for documentation at every kind-binder site.
-- Same trick as `Device.Core`'s `Device = Type`.
----------------------------------------------------------------------

public export
0 DType : Type
DType = Type


----------------------------------------------------------------------
-- Dtype families
--
-- Each family is a `Nat`-parameterized type constructor. Users
-- write `Float 32` directly, or use the aliases (`F32`) below.
-- Constructors are present to mirror the Device tag convention
-- (`CPU`/`CUDA n`/`MPS`); user code never constructs values of
-- these types — they exist only as type-level tags.
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
-- Exported for ergonomics. `Tensor [4] CPU F64 WithGrad` reads
-- better than `Tensor [4] CPU (Float 64) WithGrad`.
----------------------------------------------------------------------

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
-- runtime declare them. Backend asymmetry (e.g. tape has no fp32
-- arena) is expressed by which per-dtype C symbols exist — link-time
-- error if you try to use an unsupported (backend, dtype) pair.
--
-- Method signatures mirror the existing top-level `prim__create*`
-- bindings in `Tensor.idr`; instances are defined where those
-- primitives are accessible.
----------------------------------------------------------------------

public export
interface RuntimeDType (0 t : Type) where
  ||| Allocate a scalar tensor. (value, requires_grad) → handle.
  primCreateScalar  : Double -> Int -> AnyPtr

  ||| Allocate a general rank-N tensor from a contiguous double buffer.
  ||| (data, shape, rank, requires_grad) → handle.
  primCreate        : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr

  ||| Allocate a 1D tensor. (n, data, requires_grad) → handle.
  primCreate1d      : Int -> AnyPtr -> Int -> AnyPtr

  ||| Allocate a 2D tensor. (rows, cols, data, requires_grad) → handle.
  primCreate2d      : Int -> Int -> AnyPtr -> Int -> AnyPtr

  ||| Allocate + register a 1D parameter (rg=1 implicit, registered).
  primCreateParam1d : Int -> AnyPtr -> AnyPtr

  ||| Allocate + register a 2D parameter.
  primCreateParam2d : Int -> Int -> AnyPtr -> AnyPtr

  ||| Allocate + register a 3D parameter.
  primCreateParam3d : Int -> Int -> Int -> AnyPtr -> AnyPtr

  ||| Allocate + register a 4D parameter.
  primCreateParam4d : Int -> Int -> Int -> Int -> AnyPtr -> AnyPtr

  ||| Allocate a 1D per-sequence state tensor (refcounted on mlx).
  primCreateState1d : Int -> AnyPtr -> AnyPtr

  ||| Allocate a 2D per-sequence state tensor.
  primCreateState2d : Int -> Int -> AnyPtr -> AnyPtr


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
-- require an explicit `tcast` op.
--
-- Used as `=>` constraint on `toDeviceAs` and `tcast`'s lossless
-- variant. `toDevice` (same-dtype across devices) doesn't need it.
----------------------------------------------------------------------

public export
interface UpcastableTo (0 from : Type) (0 to : Type) where

public export
{m, n : Nat} -> LTE m n => UpcastableTo (Float m) (Float n) where

public export
{m, n : Nat} -> LTE m n => UpcastableTo (BFloat m) (BFloat n) where

public export
{m, n : Nat} -> LTE m n => UpcastableTo (IntN m) (IntN n) where

public export
{m, n : Nat} -> LTE m n => UpcastableTo (UInt m) (UInt n) where


----------------------------------------------------------------------
-- Compatible — (device, dtype) admissibility
--
-- Empty capability marker. `Compatible D T where` declares "device
-- D can store dtype T." Constrains tensor-construction smart
-- constructors so a backend without F32 support refuses to compile
-- against `Tensor [..] D F32 ...` at the spelling site.
----------------------------------------------------------------------

public export
interface Compatible (0 d : Device) (0 t : DType) where
