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


## Lessons learned

Documented after the dt-parameter refactor landed (commit `02dc04b`).

### The polymorphic-vs-concrete-slot mismatch

The first attempt at threading `(0 dt : DType)` through the library
took a "loose migration" shape: the Tensor record gained the
polymorphic slot, but the LayerLike interface's methods and the
library smart constructors hardcoded F64 in their bodies. The
reasoning seemed reasonable — only F64 worked at the C side anyway,
so why expose dt polymorphism in the interface?

The result was an elaborator memory blowup. Each Tensor reference in a
layer's `applyVar` body allocated a fresh `dt` unification variable
(because the record is polymorphic). Idris-2 kept those metavars alive
across the module to support cross-method elaboration. With hundreds
of references per layer file (Layer.Dnc the worst), the kept-alive
metavar state pushed Chez Scheme's resident set above **30 GB on a
single idris-ml build**. Running four parallel idris2 builds during
iteration drove iTerm2 (and its spawned processes) to **99 GB total**,
triggering an out-of-memory event.

### The fix: full polymorphism

Switching to fully polymorphic dt in every interface method and smart
constructor signature collapsed the metavar accumulation. Each
function now binds `dt` once at its signature; all internal Tensor
references reuse that one bound variable. The same idris-ml build
that had been at 30+ GB now completes inside the normal memory budget.

The principle: **never mix a polymorphic record slot with a concrete
hardcoded value in the methods that operate on it.** Plumb the
parameter all the way through, even when only one value of the
parameter is supported by the current C side. Callers pick the
concrete value at the leaf use site (examples set `dt = F64`); the
library stays polymorphic.

Filed as a gotcha in `docs/develop/gotchas.md` under "Polymorphic
type-parameter slot vs concrete value in method body."

### Demo outcome

`Example.DTypePitch` is the type-system pitch demo. Positive cases
(`Tensor [4] CPU F64 WithGrad`, `Tensor [4] MlxCpu F32 WithGrad`,
`Tensor [4] (MlxDev MGpu) F32 WithGrad`, etc.) compile cleanly because
the corresponding `Compatible` instances exist.

The deliberately missing `Compatible (MlxDev MGpu) F64` instance means
uncommenting the demo's `failMlxGpuF64` line produces:

```
Can't find an implementation for Compatible (MlxDev MGpu) F64.
```

PyTorch's runtime `RuntimeError: Float64 not supported on Metal`
lifted to compile time. The error points at the user's spelling site,
not at an op deep inside the layer chain.

### Deferred for follow-up

- F32 runtime implementation on MLX (smart constructors currently
  allocate F64 buffers; the `(MlxDev MGpu) F32` type rejects the
  bad pair statically but doesn't yet route F32 data through the C
  side at runtime).
- C-side stream selection (`MlxDev MGpu` should set the Metal stream;
  `MlxDev MCpu` should set the CPU stream — currently both forward to
  the global `MLX_DEVICE` env var).
- F32 on tape and torch backends (the C arenas are double-only;
  adding f32 storage is a separate workstream).
- The `Reinforce` test's pre-existing `Data.List.index : IO (Vect ...)`
  bug, surfaced (not caused) by the dtype refactor.


## Per-build-mode dtype selection (2026-05-17)

Once the type system supports two valid configurations — `CPU` + `F64`
everywhere except mlx-GPU, and `MlxDev MGpu` + `F32` on mlx-GPU — the
question is how to switch between them in the example surface.

Idris-2 has no runtime-env-to-type-level escape hatch. `DType` and
`Device` are type parameters fixed at elaboration time, before `main`
runs. So `System.getEnv "MLX_DEVICE"` at runtime can't drive a
type-level dtype choice on `Tensor`'s `dt` slot.

The mechanism that works: a Makefile-generated source file
`packages/idris-ml-examples/src/BuildConfig.idr`, sed-substituted
from a version-controlled template `BuildConfig.idr.in`:

```idris
public export ExampleDevice : Type
ExampleDevice = @DEVICE@        -- CPU or MlxDev MGpu

public export ExampleDType : DType
ExampleDType = @DTYPE@          -- F64 or F32
```

The generation rule mirrors the existing `.backend-stamp` pattern in
the Makefile (line ~313): a `.buildconfig-stamp` records the active
`$(PRIMARY):$(MLX_DEVICE)` tuple, regeneration fires only when the
tuple changes (so no-op rebuilds don't churn TTC files and trigger
unnecessary example recompiles).

Every tensor-using example imports `BuildConfig` and references
`ExampleDevice` / `ExampleDType` instead of hardcoded `CPU` / `F64`.
Switching modes is `make BACKEND=mlx MLX_DEVICE=gpu install` — zero
example source edits required. The library stays fully polymorphic in
`dt` and `d`; the examples pin both at the leaf.

**Layer creators device-polymorphisation.** A precondition for the
example migration: 11 `*LayerAny` creators (`linearLayerAny`,
`conv2dLayerAny`, etc.) used to hardcode `CPU` in their return types.
Each got `CPU` swapped for a free type variable `d` (Idris auto-binds
as `{0 d : Device}`). Bodies are unchanged — they use unsuffixed
`prim__paramRegister` / `prim__createParam2d` calls routed to the
primary backend via Phase-1's symbol-rename + alias mechanism, so they
work for whichever device tag the caller pins.

**4-lane test matrix.** `make test-examples` previously iterated
`tape mlx torch`. Now it iterates `tape mlx mlx-gpu torch`. The
`mlx-gpu` lane is a virtual entry: the loop parses it as `b=mlx` with
`lane_env=MLX_DEVICE=gpu`, exported to the recursive inner Make so
BuildConfig regenerates for F32 mode. Wall-clock cost: ~13 min →
~30-60 min, dominated by Idris VM time (not the C-side; mlx GPU and
CPU are similar at example scales per
`project_mlx_gpu_environment.md`).

**Special-case examples that don't migrate.** `DTypePitch.idr` —
its rejection demo (`failMlxGpuF64`) requires hardcoded `F64` and
`(MlxDev MGpu)` to demonstrate the type-level rejection; using
`ExampleDType` would auto-resolve to F32 and lose the pedagogy.
Skipped from the migration script. Verified to still build under both
modes.

**Per-lane expect thresholds.** The lane-specific
`test-examples.expect.mlx-gpu` file is supported (Makefile picks it
up if it exists, otherwise falls back to `test-examples.expect`).
Not shipped yet — calibration requires a run on real Metal hardware
that exposes the GPU stream cleanly. Add when an mlx-gpu CI run on
real M-series surfaces F32-precision diffs from the F64 reference
thresholds.
