# Dtype type parameter

Design memo for adding a dtype type parameter to `Tensor` so that
`Tensor [4,8] MlxGpu F64` fails to typecheck (Metal GPU does not support
f64) and `Tensor [4,8] MlxGpu F32` runs end-to-end with f32 storage and
no f32↔f64 boundary conversion. Pairs with a derived
lossless-upcast partial order so the compiler can also reject
silently-lossy assignments like `Tensor … F64 → Tensor … F32`.

## Why

Today every `Tensor` is implicitly f64. `Tensor.idr:958` declares

```idris
record Tensor (dims : Vect rank Nat) (0 d : Device) (0 g : GradMode) where
  constructor MkTensor
  tensorPtr : AnyPtr
  paramId   : Maybe String
```

with no dtype slot. The three C backends hardcode `double` throughout
(805 occurrences across tape/torch/mlx/backend.h: 493/100/147/65). The
MLX backend internally runs **float32** because Metal GPUs dropped f64
support in mlx 0.31, then bridges f32↔f64 at every FFI boundary —
`mx_to_doubles` / `mx_from_doubles` in `backend_mlx.cpp:192-211`. Tape
and torch are end-to-end f64.

Two problems compound:

1. **Mismatched expressivity.** PyTorch users routinely choose dtype per
   tensor (`torch.float32` for activations, `torch.float64` for verified
   numerics, `torch.bfloat16` for throughput). Our user has one knob
   (`MLX_DEVICE` env) selecting CPU vs GPU stream and no way to express
   dtype at all. `Tensor [..] (MlxGpu) F64` blows up with a libtorch-style
   runtime `RuntimeError` deep inside C++; no compile-time recourse.

2. **Wasted boundary conversion on MLX.** Every Idris→MLX tensor load
   walks a `double*` buffer and casts to `float`, then constructs the f32
   `mx::array`. Reverse walks the f32 array casting to `double`.
   Eliminating that bridge on the demo path is a small clarity win and
   a non-trivial throughput one.

The proven template for opening a `Tensor` parameter is already in the
codebase: the Device-opening work turned `Device` from a closed sum into
an open type-level kind alias with a `UserDeviceCore` typeclass. Mirror
that pattern for dtype, with a richer typeclass layering since dtypes
have a partial-order structure (precision-ranked upcasts) that devices
don't.

## What changes

A new 0-quantity phantom parameter on `Tensor`:

```idris
record Tensor (dims : Vect rank Nat) (0 d : Device) (0 t : DType) (0 g : GradMode) where
  constructor MkTensor
  tensorPtr : AnyPtr
  paramId   : Maybe String
```

A kind alias `0 DType : Type; DType = Type` (identical trick to `Device`),
with dtype families as `Nat`-parameterized type constructors and aliases
for the common widths.

### Dtype families

```idris
data Float  : Nat -> Type where MkFloat  : Float n
data BFloat : Nat -> Type where MkBFloat : BFloat n
data IntN   : Nat -> Type where MkIntN   : IntN n   -- "Int" is reserved
data UInt   : Nat -> Type where MkUInt   : UInt n
data Bool   : Type        where MkBool   : Bool

F32  = Float 32
F64  = Float 64
F16  = Float 16
BF16 = BFloat 16
I8   = IntN 8
I16  = IntN 16
I32  = IntN 32
I64  = IntN 64
U8   = UInt 8
```

Five separate type constructors, not a closed sum. Each family is its
own ladder for within-family lossless upcasts; **cross-family**
conversion is never auto-derived — converting a `UInt 8` to an `F16`,
or a `BF16` to an `F32`, always requires explicit `tcast` (the compiler
can't decide whether a `UInt 8 → F16` is what the user wanted, even
though the bit-level fit is lossless).

### Three layered typeclasses

```idris
public export
interface IsDType (0 t : Type) where
  dtypeName  : String     -- "f32", "bf16", "i32"
  dtypeBytes : Int        -- 4, 2, 4

public export
interface IsDType t => Precision (0 t : Type) where
  precisionRank : Nat     -- bit width, used by UpcastableTo derivation

public export
interface UpcastableTo (0 from : Type) (0 to : Type) where
```

- `IsDType` — capability marker, "t is a valid tensor element type."
  One polymorphic instance per family
  (`IsDType (Float n)`, `IsDType (IntN n)`, ...). Bool gets a special
  unparameterized instance.
- `Precision` — rank-aware subset. Every `Nat`-parameterized family has
  a `Precision` instance with `precisionRank = n`. Bool deliberately has
  no `Precision` instance — no bit-width precision concept.
- `UpcastableTo` — lossless conversion witness. **Derived** per-family
  from an `LTE m n` constraint:

  ```idris
  {m, n : Nat} -> LTE m n => UpcastableTo (Float m) (Float n) where
  {m, n : Nat} -> LTE m n => UpcastableTo (BFloat m) (BFloat n) where
  {m, n : Nat} -> LTE m n => UpcastableTo (IntN m) (IntN n) where
  {m, n : Nat} -> LTE m n => UpcastableTo (UInt m) (UInt n) where
  ```

  Idris's auto-search synthesises the `LTE m n` proof from the Nat
  constructors at the call site. `UpcastableTo F32 F64` resolves
  (because `LTE 32 64` is solvable). `UpcastableTo F64 F32` does not
  (no `LTE 64 32` proof exists). `UpcastableTo BF16 F32` does not (no
  cross-family instance).

### `Compatible` capability interface

```idris
public export
interface Compatible (0 d : Device) (0 t : DType) where
```

Empty body — the instance head IS the proof. Initial instance set:

| Device         | F64 | F32 | F16 | I32 | … |
|----------------|-----|-----|-----|-----|---|
| `CPU`          | ✓   | ✗   | ✗   | ✗   |   |
| `TapeDev`      | ✓   | ✗   | ✗   | ✗   |   |
| `TorchDev`     | ✓   | ✗   | ✗   | ✗   |   |
| `MlxDev MCpu`  | ✓   | ✓   | ✗   | ✗   |   |
| `MlxDev MGpu`  | ✗   | ✓   | ✗   | ✗   |   |

The single missing F64 cell on `MlxDev MGpu` is where the demo error
lives. The F16/I32 columns are empty in the demo scope; they fill in
when those dtypes' C-side support lands. Tape/torch CPU could in
principle grow F32 (would require a parallel `float*` arena in tape;
mechanical refactor in torch), but neither is motivated by the demo.

## Key design decisions

### Empty `Compatible` instead of method-bearing interface

The Device-opening work used five sliced interfaces (`UserDeviceCore`,
`UserDeviceLinear`, `UserDeviceNN`, `UserDeviceConv`, `UserDeviceTape`)
because backends legitimately implement different op subsets — a BYO
backend without conv simply omits the `UserDeviceConv` instance and
conv-using code refuses to typecheck against it.

Dtype admissibility is not like that. If MLX-GPU supports `F32` add, it
supports `F32` matmul, conv, softmax, everything — the underlying mlx
library does not have op-specific dtype restrictions. A single empty
marker interface is the right shape; methods would be pure ceremony.

### Constraint on constructors, not every op

`(Compatible d t, IsDType t) =>` goes on the ~15 smart constructors in
`Tensor.idr` (`tparam1d`, `tparam2d`, `tinput*`, `tconstScalar`, etc.)
and on `toDevice`'s destination. **Not** on every elementwise op.

Reasoning: once a `Tensor dims d t g` exists, its type carries `t`.
Every downstream op consumes that input type — admissibility was
checked at the construction site. Putting `Compatible d t =>` on every
op signature would be redundant noise in error messages and force
constraint solving at every use site.

The error a user sees when they spell `Tensor [..] (MlxDev MGpu) F64`:

```
While processing right hand side of demo
...
No implementation for Compatible (MlxDev MGpu) F64
```

That message points at the construction call site. Exactly where the
user can fix it.

### `Precision` is family-general, not float-only

An earlier draft made `Precision` IEEE-float-only — `Precision (Float n)`
but no `Precision (IntN n)`. Wrong: integer families have the same
within-family bit-width upcast structure (`Int 16 → Int 32` is exactly
as lossless as `Float 16 → Float 32`). Every `Nat`-parameterized family
gets a `Precision` instance.

The illusion of "Precision is float-only" came from worrying about
cross-family upcasts — sharing `precisionRank = 32` between `Float 32`
and `IntN 32` does not make them mutually upcastable, because
`UpcastableTo` instances are scoped per family. The shared rank doesn't
leak across.

### Cross-family upcasts always require explicit cast

The compiler can't generally know whether a cross-family conversion is
semantically appropriate. `UInt 8` values 0–255 fit losslessly in F16's
mantissa, but the user might be storing them as ordinal labels, not
numeric magnitudes — converting to F16 is "lossless on the bit-pattern"
but a type confusion. Similarly `BFloat 16 → Float 32`: bit-pattern
lossless, but the user might want F32 with strict IEEE compatibility,
not a bf-extended representation.

So: cross-family conversions always go through `tcast` (or a more
specific named cast like `tcastUintToFloat`). The compiler's role is
to *reject* implicit cross-family, not to be a numerical-correctness
oracle.

### `MlxDev` as a parameterized family, not opaque siblings

A naïve split would introduce two unrelated types:

```idris
data MlxGpu : Type where MkMlxGpu : MlxGpu  -- ❌
data MlxCpu : Type where MkMlxCpu : MlxCpu  -- ❌
```

Reject this. The two MLX devices are related: same backend library,
same C symbols, only stream selection differs. The existing `CUDA Nat`
device is already parameterized (`CUDA 0`, `CUDA 1` are both `CUDA n`
instantiated at different `n`); the MLX split should mirror that
precedent.

```idris
data MlxStream : Type where
  MGpu : MlxStream
  MCpu : MlxStream

data MlxDev : MlxStream -> Type where
  MkMlxDev : MlxDev s

MlxGpu : Type
MlxGpu = MlxDev MGpu

MlxCpu : Type
MlxCpu = MlxDev MCpu
```

One `UserDeviceCore`/Linear/NN/Conv/Tape instance set, parameterized
over `s`. Functions polymorphic over the MLX stream become expressible:
`f : Tensor dims (MlxDev s) t g -> ...` works for either stream — that's
the readability win over opaque siblings.

Stream selection at the C boundary uses a companion typeclass
`HasMlxStream` mirroring `HasDeviceIndex` from the Device work.

### F32 FFI on the Idris side

The `IsDType` typeclass methods factor allocation/read/write by `t`:

```idris
interface IsDType (0 t : Type) where
  dtypeName  : String
  dtypeBytes : Int
```

These metadata methods drive a separate set of FFI primitives:
- `precAlloc  : Int -> AnyPtr` (allocate N elements of dtype t)
- `precSet    : AnyPtr -> Int -> Double -> AnyPtr` (write index, cast inside)
- `precCreate : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr`
- `precItem   : AnyPtr -> Double` (read scalar, cast out)

Idris-side scalar surface stays `Double`; the F32 instance casts at the
FFI shim. Per-op kernels don't change — only the create/read boundary.

(These primitives may grow into their own `DTypeFFI` typeclass, parallel
to `IsDType`, with one instance per (backend × dtype) pair. Design
deferred — initial implementation keeps them as free `%foreign` decls
selected via type-class dispatch.)

### Runtime dtype tag on the C tensor handle (not parallel symbol variants)

C-side choice between two strategies:

- **(A) Parallel symbol variants**: each op gets per-dtype symbols
  (`tensor_add_f32`, `tensor_add_f64`, ...). ~500 ops × N dtypes × 3
  backends = thousands of symbols.
- **(B) Runtime dtype tag**: each backend's tensor handle gains a
  `dtype` field; kernels branch on it internally.

Pick (B). PyTorch's `at::Tensor` already carries a runtime dtype and
dispatches internally; mlx's `mx::array` carries the dtype in its
header. The Idris-side compile-time `Compatible` check is the new
value; the C-side just respects whichever dtype it's told.

### What's not in scope

- **F32 on tape.** `backend_tape.c`'s `double*` arena is 5.8K LOC of
  pointer arithmetic; a parallel f32 arena is real work. Tracked
  separately if a user materializes.
- **F32 on torch.** Mechanical refactor (thread a `dtype` argument
  through `tensor_create`) but unmotivated by the demo.
- **BF16 / F16.** bf16 is GPU-bound; CPU implementations are slow or
  absent. Defer to the CUDA support story.
- **Integer tensors as first-class element types.** Today, integer
  indices live as raw `Int` buffers passed alongside float tensors. The
  scaffolding supports `IntN n` aliases (I8/I16/I32/I64), but no C
  backend implements them yet. Adding them is a separate slot.
- **Mixed-precision autocast / `GradScaler`.** Lives under the PyTorch
  design survey TODO (row 38).
- **Param-registry serialization across dtypes.** Loading a `(MlxGpu)
  F32` checkpoint into an `(MlxCpu) F64` model would need a runtime
  cast at load. Flag in user docs; defer the implementation.
- **Performance.** This work is a type-system / correctness story. Per
  `feedback_vm_perf_noise.md` and the "explicitly not planned: mixed
  precision/quantization (performance optimisation)" caveat in
  `TODO.md:62`, do not justify on f32 throughput numbers.

## Risks

**Elaborator hang from a 4th `Tensor` parameter.** The Idris-2 type
checker has known sensitivity to multiplicative shape arithmetic
(`feedback_idris2_tvar_nat_mult.md`), which the `TVec`/`TMat` aliases
work around. Adding a 4th type parameter is structurally identical to
the existing `(0 g : GradMode)` — same shape, same 0-quantity, no
shape-arithmetic interaction — so risk is low. A half-day spike at the
start of the Tensor-propagation work confirms before committing to full
propagation. Fallback: dtype lives only in `Compatible`/`UpcastableTo`
constraint scope, not as a `Tensor` slot.

**Auto-search depth on `LTE`.** `UpcastableTo` derivation needs Idris to
synthesise `LTE m n` proofs at auto-search time. Successfully exercised
in the `DType.Test` smoke file for proof depths up to `LTE 8 64` and
`LTE 32 64` (well under the default search depth of 50). If a future
dtype lands with bit-width > ~50, search depth may need to be raised
with `%search_timeout` or the derivation reworked to use a
constant-time decidable predicate.

**`Data.Nat` re-export.** `DType.Core` re-exports `Data.Nat` via
`import public` so consumers get `LTE` in scope automatically. Without
this, `UpcastableTo` constraints in consuming modules failed to resolve
even though the instance heads are correct (auto-search can't find LTE
constructors that aren't in scope). Verified by the smoke test.

**`LayerLike` propagation churn.** ~15 layer files take a `t` binder.
Mechanical but tedious. Escape: omit `t` from `LayerLike` entirely, let
op-site `Compatible` constraints carry it.

**MLX stream selection refactor.** Current `backend_mlx.cpp` uses a
global `mx::set_default_device`. Per-call stream selection means every
`mx::add` / `mx::matmul` call needs an explicit stream argument. mlx
supports this via `StreamOrDevice` overloads — mechanical edits, no
design surprise.

**The GPU-is-slower reality.** Per `project_mlx_gpu_environment.md`,
mlx GPU loses on every workload at this codebase's scales due to
kernel-launch wall. The demo runs but won't beat CPU. Doc deliverable
must say this explicitly.

## Adjacent design constraints (cross-references)

- `feedback_no_backcompat.md` — no users yet, no backwards-compatibility
  shims. The old unparameterized `MlxDev` is retired outright.
- `feedback_pytorch_precedent_test.md` — PyTorch's `at::Tensor` is the
  precedent for runtime dtype tagging. Compile-time dtype parameter on
  top is the dependent-types delta.
- `feedback_typeclass_zero_arg_method_eval.md` — any new `IsDType` or
  `Compatible` method bound to a side-effecting C call must be
  `PrimIO`-typed, not unit. The current sketch has no side-effecting
  methods.
- `project_mlx_gpu_environment.md` — mlx GPU is slower than CPU on this
  codebase's example sizes. Demo is correctness, not speed.
- `docs/develop/design-decisions.md` — "Type-safe device placement" /
  "Type-level grad-mode" entries are the natural neighbours; a new
  "Open `t` parameter" entry slots in after the MLX f32 work lands.
