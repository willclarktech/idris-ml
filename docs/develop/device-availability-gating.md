# Device availability gating — design exploration

**Status: both halves implemented (2026-05-21).** Tracks the TODO row
"Env-driven hardware-availability gating for backends".

- **Done — compile-time linkage gate.** The `Linked ex` capability
  (`Executor.Core.Kind`) is wired into the construction + forward path
  alongside `Compatible`, with instances emitted per build by the generated
  `HwConfig` module (one per backend in `BACKEND`). Naming a device whose
  backend isn't linked fails to compile — e.g. `tconstScalar {ex =
  MlxExecutor MGpu}` on a tape build is rejected for lack of `Linked
  (MlxExecutor MGpu)`.
  This mirrors the `Compatible (device, dtype)` machinery (also wired
  2026-05-21). Consequence: inherently-cross-backend modules (Transfer,
  MlxStreamDemo) can no longer compile under a single-backend build, so
  they live outside the always-compiled examples ipkg and build only via
  their multi-backend targets.
- **Done — runtime hardware-presence gate (EAFP).** Attempt construction,
  catch the backend's exception — not the LBYL pre-probe an earlier draft
  proposed (see "The runtime gate is EAFP, not LBYL" below). Shipped:
  - `tensor_to_device` in `backend_torch/device.cpp` now wraps `.to()` in
    `try/catch`, returning a NULL handle on any backend exception. All
    torch device-pinning (scalar/create/param/state/createFromHost/
    intraMigrate) routes through this one shim, so the guard is
    comprehensive. (tape/mlx `tensor_to_device` are `return t` — never
    throw.)
  - `prim__handleIsNull` (`Ml/Tensor/Handle.idr`) reads wrap-v2 slot 2;
    `ExecutorError`
    + `attemptOn : IO (Tensor ..) -> IO (Either ExecutorError (Tensor ..))`
    (`Ml/Tensor/Core.idr`) lift a null handle to `Left`. `attemptOn`
    composes with *any* construction action, so there's no per-constructor
    duplication; `toExecutorChecked` applies it to `toExecutor`.
  - `HardwareClass` + the opt-in `HardwareClassed` interface,
    `SomeExecutor` descriptor, and `someExecutor` / `availableExecutors`
    for EAFP discovery (attempt a 1-element alloc per candidate, keep
    survivors).
  - `builtinExecutors : List SomeExecutor` — the build's candidate list, so
    `availableExecutors builtinExecutors` needs no caller-supplied list. It's
    the value-level mirror of the generated `Linked` instances: a second
    generated module (`HwExecutors.idr` from `HwExecutors.idr.in` + the
    Makefile `HWDEVICES_IDR` recipe in `mk/genconfig.mk`), emitting one
    `someExecutor {ex} {dt}`
    per admissible cell of each linked backend. It lives *downstream* of
    `Tensor` (where `someExecutor` is defined), not in `HwConfig` (which the
    Executor barrel re-exports upstream of `Tensor`) — so it's a separate
    generated module, re-exported by the notebook prelude. torch lists all
    three hw variants (TCpu/TMps/TCuda 0) and EAFP filters to what's
    present.
  - On tape (and mlx stream switches) construction never fails, so the
    gate degrades cleanly to "always `Right` / always available". The
    null→`Left` path is exercised only on torch (CUDA/MPS absence); that
    path is verified-by-construction here — the torch/mlx C++ can't be
    compiled in this tape-only environment, so it relies on the CI lanes.
- **Deferred follow-up.** Multi-GPU `TCuda n` enumeration: `builtinExecutors`
  lists a single `TCuda 0` candidate today; enumerating `TCuda 0..k-1`
  bounded by a native `cuda_device_count` (called at list construction,
  before the EAFP attempts) would surface every CUDA GPU on a multi-GPU
  box. The decision still comes from the EAFP attempt — the count only
  bounds how many candidates to generate.

## Problem

Devices are scoped by *backend* (`TapeExecutor`, `TorchExecutor d`,
`MlxExecutor s`). That
scoping is correct — you can't multiply a `TorchExecutor TMps` tensor by a
`MlxExecutor MGpu` tensor even though both live on the same physical Apple
GPU, because the handles are foreign to each other's backend.

But backend-scoping loses the **hardware commonality** across backends:

- On an Apple Silicon laptop with torch+mlx linked, the user should be able
  to reach `TorchExecutor TCpu`, `TorchExecutor TMps`, `MlxExecutor MCpu`,
  `MlxExecutor MGpu`.
- On a Linux box with 2 CUDA GPUs and torch-only linked, the user should
  reach `TorchExecutor TCpu`, `TorchExecutor (TCuda 0)`, `TorchExecutor
  (TCuda 1)` — and
  `TorchExecutor TMps` / any `MlxExecutor` should be prohibited.

Today nothing gates this. The only capability gate is `Compatible ex t`
(dtype admissibility, `Ml/DType/Core.idr`). A program can spell
`TorchExecutor (TCuda 1)` on a CPU-only host and compiles fine, then
SIGABRTs deep in libtorch at runtime.

## Reframe: "env-driven availability" is two facts at two times

The single phrase hides two facts that live at different lifecycle points:

1. **Backend linkage** — is `torch` / `mlx` even compiled into this
   `libidrisml`? A **build-time** fact. The Makefile already knows it
   (`BACKEND=torch,mlx`) and `BuildConfig.idr` already bakes build-time env
   into generated source. On a torch-only build, `MlxExecutor _` should be
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

**A — Compile-time `Available ex` capability (mirror `Compatible`).** Empty
interface `Available (0 ex : Executor)`; instances emitted by a
BuildConfig-style generated module that probes the host *at build time*.
References to an unavailable device fail to compile.
- Elegant, zero runtime cost, consistent with `Compatible`.
- Breaks on `TCuda n`: build-time can't know the runtime GPU count, forcing
  `{n} -> Available (TorchExecutor (TCuda n))` (gates nothing) or a
  hardcoded bound. Also can't reflect a build moved to a different machine.

**B — Runtime guard via IO combinator (EAFP).** Keep types open; route
device-pinned construction through
`attemptOn : IO (Tensor ..) -> IO (Either ExecutorError (Tensor ..))`
(shipped name; the design draft called it `mkTensorOn`) that *attempts*
the construction and reports the backend's own failure as `Left`. A
combinator over the construction *action* composes with every existing
smart constructor instead of duplicating each. No pre-probe — see "The
runtime gate is EAFP, not LBYL" below for why.
- Honest for runtime-discovered hardware; handles `TCuda` count cleanly.
- Loses the "fails to compile" guarantee (that's the linkage half's job).

**C — Hybrid (recommended).** Gate each fact where it actually lives:
- *Linkage* → compile-time empty capability `Linked ex`, emitted by a
  generated `HwConfig.idr` from the `BACKEND` list. A torch-only build has
  no `Linked (MlxExecutor _)` instance → mlx devices are unspellable. Cheap,
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
fear that drove LBYL — "spelling `TorchExecutor (TCuda 1)` on a CPU-only
host SIGABRTs deep in libtorch" — is an *uncaught*, not *uncatchable*,
exception:

- libtorch device failures are catchable `c10::Error` C++ exceptions.
  `torch_mps_eager_init` (`backend_torch/mps_init.cpp`) already does
  `try { construct on MPS } catch (...) {}`.
- `tensor_to_device` (`backend_torch/device.cpp`) was **unguarded**
  before the shipped fix: `to_tensor(h)->to(device)`. On `cuda:1` with
  one GPU, `.to()` throws, and with nothing catching it the exception
  propagates across the C→Chez FFI boundary, where it becomes
  `std::terminate`/SIGABRT. The crash was a *missing `try/catch`*, not
  an inherent abort.

So the runtime gate is: **attempt the construction; the C++ shim wraps it
in `try/catch`, converts any backend exception to a null-handle return;
Idris lifts null → `Left ExecutorError`.** One source of truth (the
backend's real allocation), no TOCTOU, no probe to maintain, and it
catches every failure mode uniformly (absence, bad device index, OOM,
driver).

The one genuine LBYL holdout is a *true* uncatchable abort/SIGSEGV — e.g.
the MPS lazy-init race documented in `backend_torch/mps_init.cpp`. But
that's a startup race, not an availability error, and it's already closed
by the eager warm-up at dylib load. EAFP covers everything that throws;
the warm-up covers the one thing that doesn't. The two are orthogonal.

## Recovering the cross-backend "commonality"

Do **not** unify `TorchExecutor TMps` and `MlxExecutor MGpu` at the type
level — you
genuinely can't mix their tensors, so unifying would lie. Express the
shared-hardware fact as runtime data instead:

```idris
data HardwareClass = HostCpu | AppleGpu | Nvidia Nat | Other String

-- per-device method on the opt-in `HardwareClassed` interface
-- (open, so BYO backends map their own)
hardwareClass : HardwareClass

-- shipped: a concrete discovery *descriptor*, not an existential tag.
-- The (d, dt) is captured at `someExecutor` construction (where a
-- compatible dtype is known); the descriptor keeps only what discovery
-- needs, so it's dtype-agnostic and existential-free.
record SomeExecutor where
  deviceLabel : String          -- deviceName {ex}
  hwClass     : HardwareClass    -- hardwareClass {ex}
  probe       : IO Bool          -- attempt a 1-element alloc; True = usable

availableExecutors : List SomeExecutor -> IO (List SomeExecutor)
-- Apple Silicon torch+mlx build →
--   [TorchExecutor TCpu, TorchExecutor TMps, MlxExecutor MCpu, MlxExecutor MGpu]
-- Linux torch-only, 2 GPUs →
--   [TorchExecutor TCpu, TorchExecutor (TCuda 0), TorchExecutor (TCuda 1)]
```

`hardwareClass` lets discovery group by physical hardware ("TMps and MGpu
both map to `AppleGpu`") for reporting, without ever letting their tensors
meet. `Other String` (namespaced `user/<name>`) is the BYO escape hatch.

The descriptor form (vs an existential-wrapped device tag) was chosen
deliberately: you can't mint more tensors from a `SomeExecutor` — which is
exactly what discovery wants, since use sites name the concrete device
themselves. It also sidesteps the per-device dtype problem (MGpu/TMps are
F32-only) by baking the right dtype into the `probe` at `someExecutor`
construction.

`availableExecutors` is itself built on the EAFP primitive: take a candidate
list and, for each, *attempt* a tiny (1-element) allocation and keep the ones
that don't throw. No separate `is_available` surface to maintain or drift —
the same "attempt construction, catch the backend's exception" path that
powers `attemptOn` powers discovery. The candidate list comes from the
build's generated `builtinExecutors` (mirroring the generated `Linked`
instances — see the "Done" bullet above); callers can still pass their own
list, e.g. `builtinExecutors ++ [someExecutor {ex = MyDev} {dt = F64}]` for
a BYO backend. (A
backend may still expose a native fast-count like `cuda_device_count` purely
as an optimisation to bound the candidate list before the attempts, but the
*decision* always comes from a real allocation, never a standalone probe.)

## BYO backend story

Every piece of the hybrid is an **open interface** or an **extensible
value-level list**, so a BYO backend plugs in exactly as it already does
for `UserExecutorCore` / `UserExecutorTransfer` / `Compatible`:

| Piece | Built-in | BYO backend |
|---|---|---|
| `Linked ex` | generated `HwConfig.idr` *withholds* the instance when the backend isn't in `BACKEND` | author self-declares `Linked MyDev where` (they're compiling it in, so it's available by definition) |
| runtime availability | construction shim wraps the alloc in `try/catch` and returns null on failure | nothing extra — if the backend's own construction throws on bad hardware, EAFP gating works for free; a backend whose alloc never fails simply never reports `Left` (degrades to no gating) |
| `hardwareClass` | `AppleGpu` / `Nvidia n` / `HostCpu` | map to `Other "user/<name>"` (or a built-in class if it shares hardware) |
| discovery | `builtinExecutors : List SomeExecutor` from the generated module | compose `builtinExecutors ++ [someExecutor {ex = MyDev} {dt = ...}]`; discovery attempts a 1-element alloc per candidate and keeps the survivors |

The only built-in-specific magic is the generated module *withholding*
`Linked` instances for non-compiled backends. A BYO author always provides
their own, so there's no friction — no registry to register into, just an
instance to write and a witness to append.

## Open questions (from the TODO row)

- **(a) env → elaboration.** Resolved by precedent: observe the build's
  `BACKEND` list at build time, bake it into a generated `HwConfig.idr`
  exactly like `BuildConfig.idr` (the shipped recipe lives in
  `mk/genconfig.mk`). This only covers the *linkage* gate; hardware
  presence is answered by option B's EAFP attempt, not a build-time fact.
- **(b) graceful fallback** when a user references a hardware variant not on
  the build: with `Linked` it's a clean type error at the spelling site;
  with the EAFP runtime gate it's a `Left ExecutorError` the caller pattern-
  matches. Decide per call whether to skip (tests) or hard-fail with a clear
  message.
- **(c) `toExecutor` when the destination isn't on the host.** *Resolved.*
  `toExecutor` round-trips through host memory for cross-backend moves; the
  destination construction (`primCreateFromHost` / `primIntraMigrate`)
  routes through torch's guarded `tensor_to_device` shim, so a move to an
  absent device returns a NULL handle. `toExecutorChecked` wraps
  `toExecutor` in `attemptOn` and returns
  `IO (Either ExecutorError (Tensor ...))`,
  wired to the same `prim__handleIsNull` primitive as `attemptOn`.
