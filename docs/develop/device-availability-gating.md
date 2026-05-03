# Device availability gating — design exploration

**Status: design settled, not yet implemented.** Captures the design for
the TODO row "Env-driven hardware-availability gating for backends".

Two pieces of groundwork are already in place:
- The `Compatible (device, dtype)` capability is now wired into every
  tensor constructor (2026-05-21), so the "empty-capability-marker +
  per-backend instances + constraint on the constructors" pattern is
  proven in the codebase. `Linked` (below) mirrors it exactly.
- The runtime half is settled as **EAFP** (attempt construction, catch
  the backend's exception), not the LBYL pre-probe an earlier draft of
  this doc proposed. See "The runtime gate is EAFP, not LBYL" below.

## Problem

Devices are scoped by *backend* (`TapeDev`, `TorchDev d`, `MlxDev s`). That
scoping is correct — you can't multiply a `TorchDev TMps` tensor by a
`MlxDev MGpu` tensor even though both live on the same physical Apple GPU,
because the handles are foreign to each other's backend.

But backend-scoping loses the **hardware commonality** across backends:

- On an Apple Silicon laptop with torch+mlx linked, the user should be able
  to reach `TorchDev TCpu`, `TorchDev TMps`, `MlxDev MCpu`, `MlxDev MGpu`.
- On a Linux box with 2 CUDA GPUs and torch-only linked, the user should
  reach `TorchDev TCpu`, `TorchDev (TCuda 0)`, `TorchDev (TCuda 1)` — and
  `TorchDev TMps` / any `MlxDev` should be prohibited.

Today nothing gates this. The only capability gate is `Compatible d t`
(dtype admissibility, `DType/Core.idr`). A program can spell
`TorchDev (TCuda 1)` on a CPU-only host and compiles fine, then SIGABRTs
deep in libtorch at runtime.

## Reframe: "env-driven availability" is two facts at two times

The single phrase hides two facts that live at different lifecycle points:

1. **Backend linkage** — is `torch` / `mlx` even compiled into this
   `libidrisml`? A **build-time** fact. The Makefile already knows it
   (`BACKEND=torch,mlx`) and `BuildConfig.idr` already bakes build-time env
   into generated source. On a torch-only build, `MlxDev _` should be
   *unspellable*.

2. **Hardware presence** — is there an MPS chip? how many CUDA GPUs? A
   **runtime** fact. Only the host answers "is `cuda:1` real" — and the
   cheapest honest way to ask is to *attempt* the allocation and see if it
   throws (EAFP), not to maintain a parallel probe. `TCuda : Nat ->
   TorchHwDev` is parameterized by a Nat that can't be enumerated at
   compile time anyway.

## The crux

Idris types fix at elaboration, so a runtime fact cannot drive a
compile-time gate — the same wall `BuildConfig` hit ("can't drive
type-level selection from a runtime env var"; types fix at elaboration).
So the gate location must be chosen *per axis*. That choice is the design.

## Options considered

**A — Compile-time `Available d` capability (mirror `Compatible`).** Empty
interface `Available (0 d : Device)`; instances emitted by a
BuildConfig-style generated module that probes the host *at build time*.
References to an unavailable device fail to compile.
- Elegant, zero runtime cost, consistent with `Compatible`.
- Breaks on `TCuda n`: build-time can't know the runtime GPU count, forcing
  `{n} -> Available (TorchDev (TCuda n))` (gates nothing) or a hardcoded
  bound. Also can't reflect a build moved to a different machine.

**B — Runtime guard via IO smart constructor (EAFP).** Keep types open;
route device-pinned construction through
`mkTensorOn : ... -> IO (Either DeviceError (Tensor ...))` that *attempts*
the construction and reports the backend's own failure as `Left`. No
pre-probe — see "The runtime gate is EAFP, not LBYL" below for why.
- Honest for runtime-discovered hardware; handles `TCuda` count cleanly.
- Loses the "fails to compile" guarantee (that's the linkage half's job).

**C — Hybrid (recommended).** Gate each fact where it actually lives:
- *Linkage* → compile-time empty capability `Linked d`, emitted by a
  generated `HwConfig.idr` from the `BACKEND` list. A torch-only build has
  no `Linked (MlxDev _)` instance → mlx devices are unspellable. Cheap,
  honest half. Mirrors the now-shipped `Compatible` machinery.
- *Hardware presence* → EAFP runtime construction (option B), answering
  the genuinely-runtime question ("is this *linked* device backed by real
  hardware right now") by attempting the allocation and catching, e.g.
  `cuda:1` on a 1-GPU box.

C matches the two motivating examples exactly: "enabled torch and mlx" is
linkage (compile-time); "2 CUDA GPUs" / "Apple Silicon" is hardware
presence (runtime).

## The runtime gate is EAFP, not LBYL

An earlier draft of the hardware-presence half was look-before-you-leap:
probe `mps_is_available()` / `cuda_device_count()` *first*, then
construct. That's the wrong shape here:

- The probe answers "is the chip present," not "can *this* allocation
  succeed right now" (OOM, driver state, paravirt-MPS quirks). You still
  need the failure path after the probe passes — so the probe is mostly
  redundant.
- It's a second source of truth that can drift from what the backend
  actually does on construction, plus a TOCTOU gap between probe and use.

**Easier-to-ask-forgiveness is feasible here, and strictly better.** The
fear that drove LBYL — "spelling `TorchDev (TCuda 1)` on a CPU-only host
SIGABRTs deep in libtorch" — is an *uncaught*, not *uncatchable*,
exception:

- libtorch device failures are catchable `c10::Error` C++ exceptions.
  `torch_mps_eager_init` (`backend_torch.cpp`) already does
  `try { construct on MPS } catch (...) {}`.
- `tensor_to_device` (`backend_torch.cpp`, ~line 756) is **unguarded**:
  `to_tensor(h)->to(device)`. On `cuda:1` with one GPU, `.to()` throws,
  and with nothing catching it the exception propagates across the
  C→Chez FFI boundary, where it becomes `std::terminate`/SIGABRT. The
  crash is a *missing `try/catch`*, not an inherent abort.

So the runtime gate is: **attempt the construction; the C++ shim wraps it
in `try/catch`, converts any backend exception to a null-handle return;
Idris lifts null → `Left DeviceError`.** One source of truth (the
backend's real allocation), no TOCTOU, no probe to maintain, and it
catches every failure mode uniformly (absence, bad device index, OOM,
driver).

The one genuine LBYL holdout is a *true* uncatchable abort/SIGSEGV — e.g.
the MPS lazy-init race documented at `backend_torch.cpp` ~line 21. But
that's a startup race, not an availability error, and it's already closed
by the eager warm-up at dylib load. EAFP covers everything that throws;
the warm-up covers the one thing that doesn't. The two are orthogonal.

## Recovering the cross-backend "commonality"

Do **not** unify `TorchDev TMps` and `MlxDev MGpu` at the type level — you
genuinely can't mix their tensors, so unifying would lie. Express the
shared-hardware fact as runtime data instead:

```idris
data HardwareClass = HostCpu | AppleGpu | Nvidia Nat | Other String

-- per-device method (open, so BYO backends map their own)
hardwareClass : HardwareClass

availableDevices : IO (List SomeDevice)   -- existential-wrapped tags
-- Apple Silicon torch+mlx build → [TorchCpu, TorchMps, MlxCpu, MlxGpu]
-- Linux torch-only, 2 GPUs       → [TorchCpu, TorchCuda 0, TorchCuda 1]
```

`hardwareClass` lets discovery group by physical hardware ("TMps and MlxGpu
both map to `AppleGpu`") for reporting, without ever letting their tensors
meet. `Other String` (namespaced `user/<name>`) is the BYO escape hatch.

`availableDevices` is itself built on the EAFP primitive: take the
`Linked`-witnessed candidate list and, for each, *attempt* a tiny (1-element)
allocation and keep the ones that don't throw. No separate `is_available`
surface to maintain or drift — the same "attempt construction, catch the
backend's exception" path that powers `mkTensorOn` powers discovery. (A
backend may still expose a native fast-count like `cuda_device_count` purely
as an optimisation to bound the candidate list before the attempts, but the
*decision* always comes from a real allocation, never a standalone probe.)

## BYO backend story

Every piece of the hybrid is an **open interface** or an **extensible
value-level list**, so a BYO backend plugs in exactly as it already does
for `UserDeviceCore` / `UserDeviceTransfer` / `Compatible`:

| Piece | Built-in | BYO backend |
|---|---|---|
| `Linked d` | generated `HwConfig.idr` *withholds* the instance when the backend isn't in `BACKEND` | author self-declares `Linked MyDev where` (they're compiling it in, so it's available by definition) |
| runtime availability | construction shim wraps the alloc in `try/catch` and returns null on failure | nothing extra — if the backend's own construction throws on bad hardware, EAFP gating works for free; a backend whose alloc never fails simply never reports `Left` (degrades to no gating) |
| `hardwareClass` | `AppleGpu` / `Nvidia n` / `HostCpu` | map to `Other "user/<name>"` (or a built-in class if it shares hardware) |
| discovery | `builtinDevices : List SomeDevice` from the generated module | compose `builtinDevices ++ [MkSomeDevice MyDev]`; discovery attempts a 1-element alloc per candidate and keeps the survivors |

The only built-in-specific magic is the generated module *withholding*
`Linked` instances for non-compiled backends. A BYO author always provides
their own, so there's no friction — no registry to register into, just an
instance to write and a witness to append.

## Open questions (from the TODO row)

- **(a) env → elaboration.** Resolved by precedent: observe `IDRISML_HARDWARE`
  (e.g. `cpu,mps` or `cpu,cuda:0,cuda:1`) at build time, bake into a
  generated `HwConfig.idr` exactly like `BuildConfig.idr`. This only covers
  the *linkage* + *build-host* gate; runtime-discovered hardware is answered
  by option B's EAFP attempt, not a build-time fact.
- **(b) graceful fallback** when a user references a hardware variant not on
  the build: with `Linked` it's a clean type error at the spelling site;
  with the EAFP runtime gate it's a `Left DeviceError` the caller pattern-
  matches. Decide per call whether to skip (tests) or hard-fail with a clear
  message.
- **(c) `toDevice` when the destination isn't on the host.** `toDevice`
  already round-trips through host memory for cross-backend moves; the
  destination construction (`primCreateFromHost` / `primIntraMigrate`) must
  route through the same `try/catch` shim so a move to an absent device
  surfaces as `Left DeviceError` instead of aborting. This means a gated
  `toDevice` returns `IO (Either DeviceError (Tensor ...))` (or a partner
  `toDeviceChecked`), wired to the same primitive as `mkTensorOn`.
