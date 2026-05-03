# Device availability gating — design exploration

**Status: proposal, not implemented.** Captures the design discussion for
the TODO row "Env-driven hardware-availability gating for backends". Pick
this up when the row is scheduled; nothing here is committed architecture.

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
   **runtime** fact. Only a host probe answers "is `cuda:1` real," and
   `TCuda : Nat -> TorchHwDev` is parameterized by a Nat that can't be
   enumerated at compile time.

## The crux

Idris types fix at elaboration, so a runtime probe cannot drive a
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

**B — Runtime guard via IO smart constructor.** Keep types open; add a
runtime probe and route device-pinned construction through
`mkTensorOn : ... -> IO (Either DeviceError (Tensor ...))` that fails when
the host lacks the hardware.
- Honest for runtime-discovered hardware; handles `TCuda` count cleanly.
- Loses the "fails to compile" guarantee.

**C — Hybrid (recommended).** Gate each fact where it actually lives:
- *Linkage* → compile-time empty capability `Linked d`, emitted by a
  generated `HwConfig.idr` from the `BACKEND` list. A torch-only build has
  no `Linked (MlxDev _)` instance → mlx devices are unspellable. Cheap,
  honest half.
- *Hardware presence* → runtime probe + IO smart constructor (option B),
  now only answering the genuinely-runtime question ("is this *linked*
  device backed by real hardware right now"), e.g. `cuda:1` on a 1-GPU box.

C matches the two motivating examples exactly: "enabled torch and mlx" is
linkage (compile-time); "2 CUDA GPUs" / "Apple Silicon" is hardware
presence (runtime).

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

## BYO backend story

Every piece of the hybrid is an **open interface** or an **extensible
value-level list**, so a BYO backend plugs in exactly as it already does
for `UserDeviceCore` / `UserDeviceTransfer` / `Compatible`:

| Piece | Built-in | BYO backend |
|---|---|---|
| `Linked d` | generated `HwConfig.idr` *withholds* the instance when the backend isn't in `BACKEND` | author self-declares `Linked MyDev where` (they're compiling it in, so it's available by definition) |
| runtime hardware probe | per-backend C probe (`mps_is_available`, `cuda_device_count`) | implement the method; if unprobeable, return `True` → degrades to no gating |
| `hardwareClass` | `AppleGpu` / `Nvidia n` / `HostCpu` | map to `Other "user/<name>"` (or a built-in class if it shares hardware) |
| discovery | `builtinDevices : List SomeDevice` from the generated module | compose `builtinDevices ++ [MkSomeDevice MyDev]`, then filter by the runtime probe |

The only built-in-specific magic is the generated module *withholding*
`Linked` instances for non-compiled backends. A BYO author always provides
their own, so there's no friction — no registry to register into, just an
instance to write and a witness to append.

## Open questions (from the TODO row)

- **(a) env → elaboration.** Resolved by precedent: observe `IDRISML_HARDWARE`
  (e.g. `cpu,mps` or `cpu,cuda:0,cuda:1`) at build time, bake into a
  generated `HwConfig.idr` exactly like `BuildConfig.idr`. This only covers
  the *linkage* + *build-host* gate; runtime-discovered hardware still needs
  option B's probe.
- **(b) graceful fallback** when user references a hardware variant not on
  the build: with `Linked` it's a clean type error at the spelling site;
  with the runtime probe it's an `Either DeviceError`. Decide per call
  whether to skip (tests) or hard-fail with a clear message.
- **(c) `toDevice` when the destination isn't on the host.** `toDevice`
  already round-trips through host memory for cross-backend moves; gating
  must reject a move whose *destination* device fails the availability check
  before it allocates the foreign handle. Likely the IO smart-constructor
  guard covers this if `toDevice`'s destination construction routes through
  it.
