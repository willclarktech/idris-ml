# Precision type parameter

Design memo for adding a precision type parameter to `Tensor` so that
`Tensor [4,8] MlxGpu F64` fails to typecheck (Metal GPU does not support
f64) and `Tensor [4,8] MlxGpu F32` runs end-to-end with f32 storage and no
f32↔f64 boundary conversion.

## Why

Today every `Tensor` is implicitly f64. `Tensor.idr` declares

```idris
record Tensor (dims : Vect rank Nat) (0 d : Device) (0 g : GradMode) where
  constructor MkTensor
  tensorPtr : AnyPtr
  paramId   : Maybe String
```

with no precision slot, and the three C backends hardcode `double`
throughout. The MLX backend internally runs **float32** (Metal GPU dropped
f64 support in mlx 0.31) and bridges f32↔f64 at every FFI boundary —
`mx_to_doubles` / `mx_from_doubles` in `backend_mlx.cpp:192-211`. Tape and
torch are end-to-end f64 (tape's arena is a `double*`; torch's tensors are
constructed with `torch::kFloat64`).

Two problems compound:

1. **Mismatched expressivity.** PyTorch users routinely choose dtype per
   tensor (`torch.float32` for activations, `torch.float64` for verified
   numerics, `torch.bfloat16` for throughput on supported hardware). Our
   user has one knob (`MLX_DEVICE` env var) selecting CPU vs GPU stream and
   no way to express dtype at all. The `Tensor [..] (Mlx GPU) F64` case
   blows up with a libtorch-style runtime `RuntimeError` deep inside C++;
   no compile-time recourse.

2. **Wasted boundary conversion on MLX.** Every Idris→MLX tensor load
   walks a `double*` buffer and casts element-by-element to `float`, then
   constructs the f32 `mx::array`. The reverse direction walks the f32
   array casting to `double`. For tiny scalars this is invisible; for any
   real workload it doubles the per-FFI cost. Eliminating that bridge on
   the demo path is a small clarity win and a non-trivial throughput one.

The proven template for opening a `Tensor` parameter is already in the
codebase: the Device-opening work (commits `b44eaf9`..`9e20307`) turned
`Device` from a closed sum into an open type-level kind alias with a
`UserDeviceCore` typeclass. Mirror that pattern for `Precision`.

## What changes

A new 0-quantity phantom parameter on `Tensor`:

```idris
record Tensor (dims : Vect rank Nat) (0 d : Device) (0 p : Precision) (0 g : GradMode) where
  constructor MkTensor
  tensorPtr : AnyPtr
  paramId   : Maybe String
```

A kind alias `0 Precision : Type; Precision = Type` (identical trick to
`Device`), with `F32` and `F64` as standalone empty types. A capability
interface that gates which (device, precision) pairs are admissible:

```idris
public export
interface Compatible (0 d : Device) (0 p : Precision) where
```

Empty body — the instance head IS the proof. Initial instance set:

| Device         | F64 | F32 |
|----------------|-----|-----|
| `CPU`          | ✓   | ✗   |
| `TapeDev`      | ✓   | ✗   |
| `TorchDev`     | ✓   | ✗   |
| `MlxDev MCpu`  | ✓   | ✓   |
| `MlxDev MGpu`  | ✗   | ✓   |

The single missing cell — `MlxDev MGpu × F64` — is where the demo error
lives. `MlxDev MCpu × F32` is the cheap symmetric path. `CPU × F32` and
`TorchDev × F32` are deferred as future work (tape would need a parallel
`float*` arena; torch would just need to thread a `dtype` argument
through `tensor_create` — neither motivated by the demo).

## Key design decisions

### Empty `Compatible` instead of sliced precision interfaces

The Device-opening work used five sliced interfaces (`UserDeviceCore`,
`UserDeviceLinear`, `UserDeviceNN`, `UserDeviceConv`, `UserDeviceTape`)
because backends legitimately implement different op subsets — a BYO
backend without conv simply omits the `UserDeviceConv` instance and
conv-using code refuses to typecheck against it.

Precision capability is not like that. If MLX-GPU supports f32 add, it
supports f32 matmul, conv, softmax, everything — the underlying mlx library
does not have op-specific dtype restrictions. There is no realistic backend
that supports `f32` for some ops and `f64` for others on the same device.
A single empty marker interface is the right shape; methods would be
pure ceremony.

Anti-pattern avoidance per `feedback_no_backcompat.md` and the no-over-
engineering line: do not add ceremony unless a real backend needs it.

### Constraint on constructors, not every op

`(Compatible d p, Precision p) =>` goes on the ~15 smart constructors in
`Tensor.idr` (`tparam1d`, `tparam2d`, `tinput*`, `tconstScalar`, etc.) and
on `toDevice`'s destination. **Not** on every elementwise op.

Reasoning: once a `Tensor dims d p g` exists, its type carries `p`. Every
downstream op consumes that input type — admissibility was checked at the
construction site. Putting `Compatible d p =>` on every op signature
would be redundant noise in error messages and force constraint solving
at every use site.

The error a user sees when they spell `Tensor [..] (MlxDev MGpu) F64`:

```
While processing right hand side of demo
...
No implementation for Compatible (MlxDev MGpu) F64
```

That message points at the construction call site. Exactly where the
user can fix it.

### `MlxDev` as a parameterized family, not opaque siblings

A naïve split would introduce two unrelated types:

```idris
data MlxGpu : Type where MkMlxGpu : MlxGpu  -- ❌
data MlxCpu : Type where MkMlxCpu : MlxCpu  -- ❌
```

Reject this. The two MLX devices are related: they wrap the same backend
library, share every C symbol, and differ only in stream selection. The
existing `CUDA Nat` device is already parameterized (`CUDA 0`, `CUDA 1` are
both `CUDA n` instantiated at different `n`); the MLX split should mirror
that precedent.

```idris
public export
data MlxStream : Type where
  MGpu : MlxStream
  MCpu : MlxStream

public export
data MlxDev : MlxStream -> Type where
  MkMlxDev : MlxDev s

public export
MlxGpu : Type
MlxGpu = MlxDev MGpu

public export
MlxCpu : Type
MlxCpu = MlxDev MCpu
```

`MlxGpu` and `MlxCpu` are ergonomic aliases, not new types. Functions
polymorphic over the MLX stream become expressible:

```idris
f : Tensor dims (MlxDev s) p g -> ...
-- works for either stream
```

This forecloses the bug class of an op that "works on both MLX devices"
being written twice with copy-paste drift, and removes the
parallel-instance-sets duplication that the opaque-siblings shape forces.

Stream selection at the C boundary uses a companion typeclass
`HasMlxStream` (mirroring `HasDeviceIndex` from the Device work — see
`docs/grad-mode-and-device-typing.md`'s "Custom devices" section). The
instance head binds `{s : MlxStream} ->` so the method body can observe
the tag at runtime; the C entrypoint reads `deviceName` (`"mlx:gpu"` /
`"mlx:cpu"`) and calls `mx::set_default_stream` per op. The `MLX_DEVICE`
env var becomes the fallback when no device-specific stream is set.

### Idris-side scalar surface stays `Double`

The 1004 `Double` references across the codebase fall into two buckets:

- **Tensor-touching** — FFI primitives that cross the precision boundary
  (`prim__createScalar : Double -> Int -> AnyPtr`, `prim__allocDoubles`,
  `prim__setDouble`, `tensor_item`). ~30 sites, all in `Tensor.idr`.
- **Non-touching** — loss scalars, learning rates, eval metrics, sampler
  output, scheduler state, hyperparameter values. ~970 sites scattered
  across the library and examples.

Only the first bucket is precision-relevant. Don't rename the second
bucket; it's pure Idris-side arithmetic that never enters a tensor.

The boundary surface gets a `Precision` typeclass that factors
allocation/read/write by `p`:

```idris
public export
interface Precision p where
  precAlloc  : Int -> AnyPtr                      -- malloc N elements
  precSet    : AnyPtr -> Int -> Double -> AnyPtr  -- write idx, cast inside
  precCreate : AnyPtr -> AnyPtr -> Int -> Int -> AnyPtr
  precItem   : AnyPtr -> Double                   -- read scalar, cast out
```

The Idris-side function signatures take/return `Double`; the `Precision
F32` instance casts at the FFI shim. This keeps the user surface stable
and confines the f32 plumbing to ~four primitives per backend that
implements f32.

### Runtime dtype tag on the C tensor handle (not parallel symbol variants)

The C-side choice between two strategies:

- **(A) Parallel symbol variants**: each op gets per-precision symbols
  (`tensor_add_f32`, `tensor_add_f64`, ...). ~500 ops × 2 precisions × 3
  backends = 3000 symbols.
- **(B) Runtime dtype tag**: each backend's tensor handle gains a `dtype`
  field; kernels branch on it internally. One symbol per op per backend.

Pick (B). PyTorch's `at::Tensor` already carries a runtime dtype and
dispatches internally; mlx's `mx::array` carries the dtype in its header.
Adding F32 to a backend means:

- Tape: would need a parallel `float*` arena. Not in scope for the demo.
- Torch: replace hardcoded `torch::kFloat64` with a `dtype` argument
  threaded through `tensor_create`. Not in scope for the demo.
- MLX: trivially supports both because mlx itself does — `mx::array`
  accepts any dtype.

The Idris-side compile-time `Compatible` check is the new value; the
C-side just respects whichever dtype it's told. Backends that don't yet
support a precision stub their f32 entrypoints with `abort()` — the
`Compatible` instance gating means those entrypoints are statically
unreachable.

### What's not in scope

- **F32 on tape.** The tape backend's `double*` arena is 5.8K LOC of
  pointer arithmetic; a parallel f32 arena is real work. Tracked
  separately if a user materializes.
- **F32 on torch.** Mechanical refactor (thread a `dtype` argument
  through `tensor_create` and friends), but unmotivated by the demo.
- **BF16 / F16.** bf16 is GPU-bound; CPU implementations are slow or
  absent. Defer to the CUDA support story per the original TODO row.
- **Mixed-precision autocast / `GradScaler`.** Lives under the PyTorch
  design survey TODO (row 38), separate concern.
- **Param-registry serialization across precisions.** Loading a
  `(MlxDev MGpu) F32` checkpoint into an `(MlxDev MCpu) F64` model would
  need a runtime cast at load. Flag in user docs; defer the
  implementation.
- **Performance.** This work is a type-system / correctness story. Per
  `feedback_vm_perf_noise.md` and the "explicitly not planned: mixed
  precision/quantization (performance optimisation)" caveat in
  `TODO.md:62`, do not justify on f32 throughput numbers.

## Risks

**Elaborator hang from a 4th `Tensor` parameter.** The Idris-2 type
checker has known sensitivity to multiplicative shape arithmetic
(`feedback_idris2_tvar_nat_mult.md`), which the `TVec` / `TMat` aliases in
`Tensor.idr:1016-1021` work around. Adding a 4th type parameter is
structurally identical to the existing `(0 g : GradMode)` — same shape,
same 0-quantity, no shape-arithmetic interaction — so risk is low. A
half-day spike at the start of the Tensor-propagation work confirms
before committing to full propagation. Fallback: precision lives only in
`Compatible` constraint scope, never as a `Tensor` slot; weaker
guarantees but unblocks the demo.

**`LayerLike` propagation churn.** Adding `(0 p : Precision)` to
`LayerLike` propagates to ~15 layer files' `applyVar`/`applyVarBatch`
signatures. Mechanical but tedious. Escape valve: omit `p` from
`LayerLike` entirely, let op-site constraints carry it; equivalent
guarantee, less churn.

**MLX stream selection refactor.** Current `backend_mlx.cpp` uses a
global `mx::set_default_device` in `mlx_backend_init`. Per-call stream
selection means every `mx::add` / `mx::matmul` / etc. call needs an
explicit stream argument. mlx supports this via `StreamOrDevice`
overloads on every op — no design surprise, just mechanical edits.

**The GPU-is-slower reality.** Per `project_mlx_gpu_environment.md`,
mlx GPU loses on every workload at this codebase's scales due to
kernel-launch wall. The demo runs but won't beat CPU. Doc deliverable
must say this explicitly; do not advertise the demo as a speed win.

## Adjacent design constraints (cross-references)

- `feedback_no_backcompat.md` — no users yet, no backwards-compatibility
  shims. The old unparameterized `MlxDev` is retired outright;
  examples migrate to `MlxCpu`.
- `feedback_pytorch_precedent_test.md` — PyTorch's `at::Tensor` is the
  precedent for runtime dtype tagging. Compile-time precision parameter
  on top is the dependent-types delta.
- `feedback_typeclass_zero_arg_method_eval.md` — any new `Precision` or
  `Compatible` method bound to a side-effecting C call must be
  `PrimIO`-typed, not unit. The current sketch has no side-effecting
  methods.
- `project_mlx_gpu_environment.md` — mlx GPU is slower than CPU on this
  codebase's example sizes. Demo is correctness, not speed; doc
  deliverable must lead with that.
- `docs/develop/design-decisions.md` — "Type-safe device placement" /
  "Type-level grad-mode" entries are the natural neighbours; a new
  "Open `p` parameter" entry slots in after the MLX f32 work lands.
