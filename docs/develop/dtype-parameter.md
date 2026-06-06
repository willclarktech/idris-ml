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
or a `BF16` to an `F32`, always requires explicit `tcastUnsafe` (the
compiler can't decide whether a `UInt 8 → F16` is what the user wanted,
even though the bit-level fit is lossless).

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
`UserDeviceLinear`, `UserDeviceNN`, `UserDeviceConv`, `UserDeviceTraining`)
because backends legitimately implement different op subsets — a BYO
backend without conv simply omits the `UserDeviceConv` instance and
conv-using code refuses to typecheck against it.

Dtype admissibility is not like that. If MLX-GPU supports `F32` add, it
supports `F32` matmul, conv, softmax, everything — the underlying mlx
library does not have op-specific dtype restrictions. A single empty
marker interface is the right shape; methods would be pure ceremony.

### Constraint on constructors, not every op

`Compatible ex dt =>` rides the construction boundary: the `dtCreate*`
family in `Tensor.idr` (the lowest-level `(d, dt) → handle` mint),
every smart constructor that calls them (`tparam1d`, `tparam2d`,
`tconstScalar`, `tparamScalar`, `tcast`/`tcastUnsafe`, `bulkToTensor`,
etc.), every layer constructor (`linearLayer`, `lstmLayer`, …), and
`toDevice`'s destination. **Not** on a plain elementwise op like
`tadd` — once a `Tensor dims ex dt g` exists, its type already carries
`dt`; admissibility was checked when it was minted, and re-checking on
every op would be redundant error-message noise.

One deliberate exception: the `LayerLike` forward path (`applyVar`,
`applyVarBatch`, and therefore `applyVarAny` / `forwardVar` /
`forwardVarBatch` / `forwardVarTraced`) also carries `Compatible ex dt`.
The recurrent cells (RNN/LSTM/GRU/NTM/DNC) lazily construct their
initial zero state *inside* `applyVar` on the first step
(`prevOut = Nothing ⇒ tzeroState1d`), so the dispatch surface mints a
tensor and inherits the construction gate. The cell already proved
`Compatible` when its parameters were built, so this never rejects a
layer that constructed successfully — it just threads the same evidence
through the forward call.

Wired in 2026-05-21; before that, `Compatible` had instances but no
constructor referenced it, so an unrepresentable `(device, dtype)`
compiled and crashed at runtime.

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

So: cross-family conversions always go through `tcastUnsafe` (or a
more specific named cast like `tcastUintToFloat`). The compiler's role
is to *reject* implicit cross-family, not to be a numerical-correctness
oracle.

**Update 2026-06-01 (#410 A0.5):** the "bit-pattern lossless" cases
above (`BF16 → F32`, `UInt 8 → F16` etc.) are now witnessable via a
new `LosslessTo from to` typeclass that captures *only* the structural
"every value of the source dtype is exactly representable in the
target dtype" property. The two-dimensional structural check —
mantissa-bits AND exponent-bits both non-decreasing — is enough to
detect that `BF16 → F32` is lossless (7→23 mantissa, 8→8 exponent)
while `F32 → BF16` is not (mantissa shrinks 23→7). The witness lives
in `packages/idris-ml/src/DType/Core.idr`'s `FloatPrecision` typeclass
+ `LosslessTo` definition; a negative compile-test gate
(`packages/idris-ml/test/neg/LossyDirectionRejected.idr`, gated by
`make check-lossy-cast-gate`) confirms F32→BF16 refuses to type-check.
The semantic-confusion concern still holds — `LosslessTo` is purely
structural — so we don't auto-derive `UpcastableTo` from `LosslessTo`
at the framework level by default. The follow-up row #412 wires that
bridge for the cases where bit-pattern losslessness IS the wanted
semantics (the mixed-precision-training case: F32 master → BF16
compute on forward, BF16 grad → F32 master on backward).

### `FloatPrecision` + `LosslessTo` — structural cross-family witness

Added 2026-06-01 (#410 A0.5). `FloatPrecision dt` refines `IsFloating`
with explicit mantissa-bits and exponent-bits per dtype:

```idris
interface IsFloating t => FloatPrecision (0 t : Type) where
  mantissaBits : Nat
  exponentBits : Nat

FloatPrecision (Float 16)  where mantissaBits = 10 ; exponentBits = 5
FloatPrecision (Float 32)  where mantissaBits = 23 ; exponentBits = 8
FloatPrecision (Float 64)  where mantissaBits = 52 ; exponentBits = 11
FloatPrecision (BFloat 16) where mantissaBits = 7  ; exponentBits = 8
```

`LosslessTo from to` is a definitional pair of `LTE` proofs on the
two dimensions:

```idris
LosslessTo : (0 from : DType) -> (0 to : DType) ->
             FloatPrecision from => FloatPrecision to => Type
LosslessTo from to =
  ( LTE (mantissaBits {dt=from}) (mantissaBits {dt=to})
  , LTE (exponentBits {dt=from}) (exponentBits {dt=to})
  )
```

Auto-resolves via Idris's hint search for the safe edges:
`LosslessTo (BFloat 16) (Float 32)` is provable (7≤23, 8≤8), and
`LosslessTo (Float 32) (BFloat 16)` is not (the mantissa LTE proof
`LTE 23 7` has no inhabitant). The two-dimensional view catches what
the single-`precisionRank` view misses: `BF16` and `F16` both have
bit-width 16, but neither is a lossless upcast of the other (each
shrinks one dimension while growing the other).

**Why ternary / binary need no `FloatPrecision`-style metadata**:
their value sets `{-1, 0, +1}` and `{-1, +1}` are finite and exactly
representable in any IEEE float, any BFloat, and any IntN with ≥2
bits. The `LosslessTo` typeclass shape (per-pair instances, each
with its own structural condition or empty if always-lossless) is
open enough to admit those instances without extending the
`FloatPrecision` framework. Ternary / Binary `LosslessTo` instances
will ship with the BitNet b1.58 row (#411).

**Lossy edges still require explicit `tcastUnsafe`**: the mixed-
precision training case (F32 master → BF16 compute) IS lossy — the
mantissa shrinks 23→7 bits. The cast is intentional, code-visible at
the layer boundary inside `LinearMixed.applyVarMixed`, and uses
`tcastUnsafe`. The autograd-aware backward path (commit `66eca8f`)
propagates a BF16 gradient back through the cast and accumulates an
F32 grad into the master weight; the F32→BF16 lossiness applies only
to the forward, not to the gradient flow.

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

Documented after the dt-parameter refactor landed (commit `28f36d3`).

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

### Test files pin a concrete dtype at the leaf

Library code stays polymorphic in `dt`; examples pin both `d` and
`dt` via `BuildConfig`'s `ExampleDevice`/`ExampleDType` (see "Per-
build-mode dtype selection" below). Tests live one layer further out:
each test function is its own leaf with no upstream caller to infer
`dt` from, so the dtype slot has to be a concrete type literal in
the test's body, not a free variable.

`Test.GradMode` originally read:

```idris
weakenGradFlipsRequiresGrad : IO Bool
weakenGradFlipsRequiresGrad = do
  let t = the (Tensor (the (Vect 0 Nat) []) CPU dt WithGrad) (MkTensor ptr Nothing)
  ...
```

`dt` is unbound — the function's signature is `IO Bool` so there's
no implicit slot Idris can pick up. Build failed with "Undefined name
dt." Fix on 2026-05-18: pin to `F64` directly (the test exercises
grad-mode flipping, not dtype polymorphism, so concreteness is fine):

```idris
let t = the (Tensor (the (Vect 0 Nat) []) CPU F64 WithGrad) (MkTensor ptr Nothing)
```

The choice of `F64` matches the default `CPU`-lane convention; the
test runs identically on tape and torch (the only test-backed lanes
today). When mlx-GPU lanes start running the Idris-side unit tests,
this concrete pin will need to gain `BuildConfig`-style indirection.

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

- ~~F32 runtime implementation on MLX~~ — *Partially landed
  2026-05-18 (mlx-runtime-fp64 plan).* The original plan's bullet
  was inverted: mlx is hardcoded `mx::float32`, and what was
  missing was actually **fp64** support for the type-level
  `Compatible (MlxDev MCpu) F64` claim. RuntimeDType + per-dtype
  FFI symbols + cascade through the entire layer stack now route
  F64 to `mx::float64` at allocation. Outstanding: ~72 hardcoded
  `mx::float32` constants in fused-op kernels and the tape-replay
  pool produce wrong math when mixed with fp64 inputs. Tracked in
  a separate Medium-Priority TODO row ("Audit mlx fused-op +
  constant-pool dtype handling").
- C-side stream selection (`MlxDev MGpu` should set the Metal stream;
  `MlxDev MCpu` should set the CPU stream — currently both forward to
  the global `MLX_DEVICE` env var).
- ~~F32 on tape backend — the C arena is `double*` throughout;
  tape's `_f32` symbols are abort stubs since 2026-05-18.~~ —
  **landed 2026-05-23**. Tape now ships F32 as a real training
  dtype: dedicated `float*` storage (`tape_arena_f32_from_doubles` /
  `tape_persistent_f32_from_doubles`), `tape_load_d` / `tape_store_d`
  dtype-aware element accessors, and a `.inc`-stamped F32 elementwise
  kernel pair (`SCALAR=double` for F64, `SCALAR=float` for F32). The
  per-rung gradcheck oracle ladder (T29) — elementwise / matmul /
  softmax / optimizer step — is GREEN, and Phase 3b extended F32
  routing to every remaining public `tensor_*` (scalars / reshape /
  losses / BLAS-heavy linalg / norm+conv+pool / lookups / recurrent
  cells). Asymmetric `data=F32` / `grad=F64` design choice: the
  67-case backward switch stays dtype-agnostic for grad reads/writes;
  only data reads in ~12 backward cases that touch input data needed
  `tape_load_d`. Inference-only dtypes (BF16/F16/I8/I16/I32/I64/U8/Bool)
  ship via the `double` lingua franca in `tape_round_to_dtype` with
  half-precision routed through the bit helpers lifted from
  `safetensors.c` into `shared_utils.{c,h}`. `Compatible TapeDev <dt>`
  is open for all 10 dtypes. See the "Tape dtype parity" entry in
  `CHANGELOG.md`.
- The `Reinforce` test's pre-existing `Data.List.index : IO (Vect ...)`
  bug, surfaced (not caused) by the dtype refactor.


## Runtime dtype dispatch landed (2026-05-18)

The "type-level only" qualifier from the original rollout has been
substantially reduced. `RuntimeDType` — a separate capability
interface alongside `IsDType` — carries per-dtype FFI primitives;
instances bind to per-dtype C symbols (`tensor_create_scalar_f32` vs
`_f64`, etc.); smart constructors and the entire layer stack require
`RuntimeDType dt =>`. End-to-end on tape and torch:

- `Tensor [..] CPU F64` on tape → `tensor_create_*_f64` →
  fp64 `double*` arena. Bit-identical loss across multiple examples
  vs the pre-cascade behaviour.
- `Tensor [..] CPU F64` on torch → `tensor_create_*_f64` →
  `kFloat64` everywhere. Bit-identical to tape.
- `Tensor [..] CPU F64` on mlx → `tensor_create_*_f64` →
  `mx::float64` at allocation, BUT downstream fused-op kernels mix
  fp32 constants → produces wrong math today. Phase 6 audit pending.
- `Tensor [..] (MlxDev MGpu) F32` still rejected at compile time
  (Metal has no fp64). The reject is the original design intent.

### Design: `RuntimeDType` as a runtime tag carrier

The bridge Idris ↔ C is a single `dtypeTag : Int` method on
`RuntimeDType` (one instance per concrete dtype, e.g. `F32`/`F64`/
`BF16`/...). The `dtCreate*` free functions in `Tensor.idr` pass
this tag to the device's `primCreate*Streamed` method, which
switches on it to pick the concrete C-side dtype. This is how the
type-level `(d, dt)` pair drives both backend dispatch (via `d`) and
dtype dispatch (via this tag) without a 2-D typeclass.

The tag uses a **kind-major layout** (closed 2026-05-23, replaced
the original grow-as-needed `F32=0, F64=1, BF16=2, ...` scheme that
caused the d4255db zero-init footgun):

```
0   invalid (zero-init traps loudly at every backend's default arm)
1   Bool
4   U8                              (family 1 — U)
8   I8     9 I16   10 I32   11 I64  (family 2 — I)
13  F16   14 F32   15 F64           (family 3 — F)
17  BF16                             (family 4 — BF)
20-23 reserved                       (family 5 — TF; TensorFloat-32)
24-31 reserved                       (sub-byte: U4/I4/NF4/ternary/MX)
```

`family = tag >> 2`, `lane = tag & 3`; for numeric families
`bit_width = 8 << lane` ∈ {8, 16, 32, 64}. Sub-byte families
(6 + 7) use named lanes because their semantics aren't pure
`(family, bit-width)` — NF4 has a learned mapping table, ternary is
{−1, 0, +1}, MX has per-block scale metadata.

Adding a new dtype is local: one `RuntimeDType` instance for the
Idris-side type, one `case` arm in each backend's dispatch
(`st_for_dtag` on torch, the mlx F32/F64 fast-paths or
`mlx_dtype_unsupported`, tape's `tape_tag_from_dtag`). Backend
asymmetry is expressed via `Compatible ex dt` — a type-level gate
that prevents constructing `Tensor [..] (MlxDev MGpu) F64` at
compile time. The wire tag isn't a persistent format
(`safetensors.c` uses the string dtype name on disk), so renumbers
land as one atomic paired commit without on-disk migration.

### The cascade (~25 functions, 17 files)

Adding `RuntimeDType dt =>` to one smart constructor (`tparam2d`)
cascaded through:

- `Tensor.idr` smart constructors (5): `tconstScalar`, `tparam1d`,
  `tparam2d`, `tparamScalar`, `tzeroState1d`
- Bulk helpers (3): `bulkToTensor`, `bulkToTensor2d`,
  `vectorToTensorPersistent` (and their `toTDP` caller)
- Layer constructors (16): one per layer × 2 (concrete + Any
  wrapper) for Linear/Rnn/Lstm/Gru/Conv1d/Conv2d/BatchNorm/
  LayerNorm/Embedding/Ntm/Dnc/Transformer; plus `mkLinearWith`,
  `mkLinearVec`, `mkBlock`, `mkBlocks`
- Per-layer `apply*` functions (12): one per layer
- Layer interface + composition (5): `LayerLike.applyVar` + `applyVarBatch`,
  `applyVarAny`, `applyVarBatchAny`, `forwardVar`, `forwardVarBatch`,
  `forwardVarTraced`
- Training functions (7): `perPointLoss`, `perPointLossTensor`,
  `epochVarTensorBatch`, `recurStep`, `decodeStep`, `encodeStep`,
  `forwardTwoPhase`
- Curriculum (3): `runChunk`, `trainStage`, `runCurriculum`
- Example call sites: `bulkToTensor` etc. need `{dt=ExampleDType}`
  explicit at every call (bulk-fixed via sed across 8 example files)

Total commit: 19 files, ~110 lines of `RuntimeDType dt =>`
additions + ~30 explicit `{dt=ExampleDType}` annotations at call
sites.

### Verification

Cross-backend bit-identity on the fp64 backends:

| Example | tape loss (2 epochs) | torch loss (2 epochs) | bit-identical |
|---|---|---|---|
| supervised | 1.5936567368116856 | 1.5936567368116856 | ✓ |
| rnn | 0.6054955984956504 | 0.6054955984956504 | ✓ |
| lstm | 0.6944339289046904 | 0.6944339289046904 | ✓ |

Same `Tensor [..] CPU F64 WithGrad` Idris code → bit-identical
output on both backends. This is the precision rollout's design
promise delivered: the type system claim about precision is
actually honored at runtime.


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


## Ternary / Binary: sub-native dtypes (2026-06-01)

`Ternary` (dtag 25, values {-1, 0, +1}) and `Binary` (dtag 24,
reserved, values {-1, +1}) are the first dtypes whose intrinsic
storage is sub-byte — 2 bits and 1 bit respectively. They land
under #411 alongside BitNet b1.58 inference work. The IsDType
instances use `dtypeBytes = 0` as a sentinel: any arithmetic that
multiplies `numel * dtypeBytes` and reaches zero is a guaranteed
loud crash, forcing callers onto the sub-byte-aware path
(`tape_packed_bytes()` on tape, framework-native int8 storage on
torch/mlx).

The `LosslessTo` instances (shipped under #411 B1) capture the fact
that {-1, 0, +1} fits exactly into every IEEE float, BFloat, and
IntN m≥2 — there's no `FloatPrecision`-style metadata
predicate to satisfy. Empty-body instances:

```idris
{to : Type} -> FloatPrecision to => LosslessTo Ternary to where
{m : Nat} -> LTE 2 m => LosslessTo Ternary (IntN m) where
-- + symmetric for Binary
```

The `LosslessTo → UpcastableTo` bridge from F1 (#412) threads these
edges into `tcast`-typeclass resolution so `tcast` on a Ternary
tensor works on every backend without an `UpcastableTo` instance
per dtype-pair.

### Per-backend physical storage diverges

See `design-decisions.md` "Per-backend ternary storage —
BitNet b1.58 (#411)" for the full rationale. The summary table:

| Backend | Physical storage | Bits/value |
|---|---|---|
| tape   | packed 2-bit (4 values / byte) | 2 |
| torch  | int8 with values in {-1, 0, +1}  | 8 |
| mlx    | int8 with values in {-1, 0, +1}  | 8 |

The Idris-side `Tensor [o, i] ex Ternary g` type is the same
everywhere. The 4× tape-vs-others byte-count difference is invisible
above the FFI boundary and parallels the existing tape-side
F64-lingua-franca for BF16 / F16 (where the asymmetry goes in the
opposite direction: tape uses more bytes than the intrinsic dtype
demands; torch/mlx use the native width).

The pattern for future sub-native dtypes (NF4, FP4, MX): if the
target on-disk format is sub-byte AND a backend's native tensor
type lacks the dtype, use the nearest framework-native dtype and
document the asymmetry in the per-backend storage table above.
Don't force-fit framework-foreign storage shapes onto torch / mlx
just for byte-symmetry with tape — the engineering cost of bypassing
framework op dispatch is real.
