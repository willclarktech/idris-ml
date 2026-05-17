||| Precision tags for type-safe tensor numeric precision.
|||
||| Mirrors the `Device.Core` pattern: `Precision` is a kind-level
||| slot, `F32` and `F64` are *types* (not values of a sum), and
||| `Compatible (0 d : Device) (0 p : Precision)` is the empty
||| capability interface that gates which (device, precision) pairs
||| can be inhabited.
|||
||| Design memo: `docs/develop/precision-parameter.md`.
|||
||| The motivating concrete demo: `Compatible (MlxDev MGpu) F32`
||| exists, `Compatible (MlxDev MGpu) F64` does not, so any user
||| spelling `Tensor [..] (MlxDev MGpu) F64` gets a compile-time
||| "no implementation" error — PyTorch's runtime
||| `RuntimeError: Float64 not supported on Metal` lifted to compile
||| time.
module Precision.Core

import Device.Core


----------------------------------------------------------------------
-- `Precision` kind alias
--
-- `Precision` is a 0-quantity alias for `Type`. Tensor's `p` phantom
-- (added by the precision-type-parameter work) is declared as
-- `(0 p : Precision)`, which is exactly `(0 p : Type)` underneath
-- but reads as "p is a precision tag" at every kind-binder site. No
-- type-system enforcement: nothing stops a caller writing
-- `Tensor [4] CPU Bool`. But construction (`tparam1d` etc.) requires
-- `Compatible d p =>`, so non-precision `p`s can be declared but
-- never inhabited.
--
-- Same trick as `Device.Core`'s `0 Device : Type`.
----------------------------------------------------------------------

public export
0 Precision : Type
Precision = Type


----------------------------------------------------------------------
-- Built-in precision tags
----------------------------------------------------------------------

||| 32-bit IEEE 754 floating point. Native precision for MLX
||| (Metal GPU constraint) and the only precision supported by
||| `(MlxDev MGpu)`.
public export
data F32 : Type where MkF32 : F32

||| 64-bit IEEE 754 floating point. Default precision for every
||| backend today; the only precision supported by `CPU` / `TapeDev`
||| / `TorchDev` / `(MlxDev MCpu)`.
public export
data F64 : Type where MkF64 : F64


----------------------------------------------------------------------
-- Compatible — (device, precision) admissibility
--
-- Empty marker interface. The instance head IS the proof: an
-- instance `Compatible D P where` declares that device `D` supports
-- precision `P`. No methods (no dispatch needed — capability check
-- only).
--
-- Used as a constraint on tensor-construction smart constructors
-- (`tparam1d`, `tinput*`, `tconstScalar`, etc.) and on `toDevice`'s
-- destination. Construction-site placement keeps error messages
-- pointed at the user's spelling site.
--
-- See `docs/develop/precision-parameter.md` for the design rationale.
----------------------------------------------------------------------

public export
interface Compatible (0 d : Device) (0 p : Precision) where
