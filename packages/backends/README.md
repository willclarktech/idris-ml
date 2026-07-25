# packages/backends/

C/C++ backends + the shared training port that lets them share infrastructure.

## Layout

```
packages/backends/
├── backend.h                  # The single public C ABI declared to Idris.
├── shared_utils.{c,h}         # Backend-agnostic helpers (compile-once,
│                              #   unified symbols: index arrays, RSS,
│                              #   dropout RNG, _wall_ms, bf16/f16
│                              #   bit-cast helpers).
├── idx.{c,h}                  # IDX-format dataset loader (compile-once,
│                              #   unified symbols). Lives outside the
│                              #   per-backend dispatch surface (no
│                              #   rename header, no backend.h decl):
│                              #   the Idris side reaches it via plain
│                              #   `%foreign "C:idx_*,libidrisml"`
│                              #   bindings, not via UserExecutor*
│                              #   methods.
├── shared/
│   └── training/
│       ├── port.h             # BackendPort dispatch table (50 slots).
│       ├── param_registry.c   # FFI param_*. Routes through port.
│       ├── optimizer.c        # FFI optimizer_*. Trampolines through port
│       │                      #   for per-backend math; shared helpers
│       │                      #   (zero_grad / polyak / clip / *_return
│       │                      #   wrappers) stay shared.
│       ├── ffi_shims.c        # The *_return value-coercion helpers
│       │                      #   (tensor_backward_return, idrisml_seq, ...).
│       └── dtype_streamed.c   # 11 dtag-dispatched create + cast wrappers.
├── backend_tape/              # Tape backend — per-typeclass-slice modular tree.
│   ├── arena.{c,h}            # Bump allocator + tape_load_d/store_d.
│   ├── tensor.{c,h}           # Tensor struct + dtype tag.
│   ├── tape.{c,h}             # TapeEntry + OP_* enum + tape_append.
│   ├── broadcast.{c,h}        # Numpy-style broadcast helpers.
│   ├── core/                  # Lifecycle, elementwise, scalar ops.
│   ├── linear/                # Linalg, shape, reduction, concat, index, sort.
│   ├── nn/                    # Activation, softmax, norm, mask, loss,
│   │                          #   attention, recurrent.
│   ├── conv/                  # Conv1d/2d + pools + transpose + grouped.
│   └── training/              # Tape's adapter + the bits not shared.
│       ├── adapter.c          # Binds the port slots.
│       ├── optimizer.c        # TapeOptimizer + per-element math.
│       ├── dtype_dispatch.c   # tape_create_*_dtag (port creator impls).
│       ├── param_create.c     # tensor_create_param_*d.
│       ├── diagnostics.c      # DEBUG_PARAM_GRADS / DEBUG_LSTM_TRAJ.
│       ├── per_dtype_aliases.c # Per-dtype _f32/_f64 creator ABI aliases.
│       ├── profiling.c        # Per-OP timing arrays + report.
│       ├── shims.c            # backend_reset_for_eval + mlx_compile stubs.
│       ├── host_io.c          # tensor_to_doubles + size queries.
│       └── autograd/          # Backward driver + dispatch table + helpers.
├── backend_torch/             # Torch backend — same modular slice tree
│                              #   (core/linear/nn/conv/training + device.cpp,
│                              #   mps_init.cpp). Adopts shared param_registry,
│                              #   ffi_shims, dtype_streamed.
├── backend_mlx/               # mlx backend — same modular slice tree
│                              #   (+ init.cpp, stream.h, precision.h,
│                              #   tape.cpp). Adopts shared param_registry +
│                              #   ffi_shims.
├── safetensors.c              # Serialization (consumes shared registry).
├── refc_shims.c               # RefC compatibility shims.
└── rename_<b>.h               # Auto-generated symbol-rename header.
```

cJSON (JSON for safetensors metadata) is vendored at the repo root
(`vendored/cJSON/`). Criterion tests are colocated with the code they
test — `backend_<b>/<slice>/test_*.c` for backend-specific suites,
plus the cross-cutting infra + `backend.h`-contract suites in
[`packages/idris-test-c/`](../idris-test-c/).

## The shared training port

`shared/training/port.h` defines `struct BackendPort` — 50 function-pointer
slots covering tensor introspection (numel / requires_grad / has_grad),
per-element data + grad read/write, bulk zero/load, backward driver, the
full optimizer surface (5 constructors + free + setters + step + 7
serialization slots), wall clock, and 11 dtag-dispatched create/cast
methods.

Each backend defines exactly one `g_active_port` instance (per-backend
renamed via `g_active_port_<b>` so multi-link doesn't collide). The
`shared/training/*.c` TUs dereference `g_active_port` to do their work —
they hold no backend-specific code.

### Why a function-pointer struct rather than weak symbols + dlsym

1. Multi-link dylibs (`BACKEND=tape,torch,mlx`) compile the shared TUs
   once per backend with that backend's rename header; each gets its own
   `g_active_port_<b>` symbol. Function-pointer dispatch handles this
   naturally — weak-symbol fallback would require ifdefs per backend.
2. The dispatch table is a single allocator-free struct literal;
   debugging adapter mis-wiring is trivial (compare struct contents).

### Per-TU opt-in

Each shared TU has its own backend list in the Makefile:

```make
SHARED_BACKENDS_param_registry := tape torch mlx
SHARED_BACKENDS_optimizer      := tape
SHARED_BACKENDS_ffi_shims      := tape torch mlx
SHARED_BACKENDS_dtype_streamed := tape torch
```

A backend not in a list keeps its own monolithic implementation of that
TU's FFI surface (must not also export the shared symbols, or the link
sees duplicates). The granularity exists because some shared TUs can't
serve every backend cleanly:

| Shared TU | Tape | Torch | Mlx | Why not |
|---|---|---|---|---|
| `param_registry` | ✓ | ✓ | ✓ | — |
| `ffi_shims` | ✓ | ✓ | ✓ | — |
| `dtype_streamed` | ✓ | ✓ | — | mlx's `stream_tag` is mlx-only state the shared trampoline drops |
| `optimizer` (trampolines) | ✓ | — | — | libtorch/mlx fast paths (`at::_foreach_adam`, mlx vectorized) are incompatible with the shared flat-buffer Optimizer struct + per-element scalar math |

The shared optimizer.c **always** compiles for backends in its list —
the trampolines call into `g_active_port.optimizer_step` etc., which
each backend's adapter binds to its own math. The struct is per-backend
(`TapeOptimizer`, torch's `OptWrapper`, mlx's `Optimizer`).

## Backend rename mechanism

`scripts/codegen/gen-rename-headers.py` scans `backend.h` for function
declarations and emits `rename_<b>.h` for each backend with
`#define <sym> <sym>_<b>` lines. `g_active_port` is added manually via
the script's `EXTRA_EXPORTS` list (the regex only matches function
declarations).

The per-TU compile rule force-includes the rename header
(`-include rename_<b>.h`), so every backend-specific TU exports
suffixed symbols and the dylib supports multi-link with no collisions.

Shared TUs (`shared/training/*.c`, `safetensors.c`) compile per
backend in their respective opt-in lists. `shared_utils.c` and `idx.c`
are the exceptions — compile-once with no rename, intentionally
unified (`_wall_ms`, the bit-cast helpers, index-array helpers,
IDX-format dataset loader).

## Adding a new op to an existing backend

1. Pick the slice the op belongs to (`core/elementwise`, `linear/linalg`,
   `nn/softmax`, …).
2. Tape: add a new `<op>.c` file with forward + backward + a
   `TAPE_REGISTER_OP(OP_<X>, tape_backward_<op>)` constructor. The
   per-TU build picks it up via `find`. Add `OP_<X>` to the enum in
   `backend_tape/tape.h` and the entry to `op_name[]` in
   `backend_tape/training/profiling.c`.
3. Torch / mlx: add the op to the matching slice in `backend_torch/`
   / `backend_mlx/` (libtorch / mlx own the autograd, so no backward
   needed).
4. Add a Criterion test colocated at `backend_<b>/<slice>/test_<op>.c`
   — the build's `find` picks it up.
5. Add the export to `backend.h` and run `make rename-headers` to
   refresh the rename headers.

## Adding a new backend

1. Create a `backend_<name>/` tree implementing every FFI function in
   `backend.h`.
2. Add a section to the Makefile for the new backend's compile rule
   (CC, CFLAGS, LDFLAGS, primary vs secondary symbol handling).
3. Decide which shared TUs the backend can adopt and add it to those
   `SHARED_BACKENDS_<tu>` lists.
4. Define `g_active_port_<name>` at the bottom of the backend file (or
   in a separate adapter TU) — populate the slots whose shared TUs the
   backend opted into; leave the others `nullptr`.
5. Run `make rename-headers` to add the backend to the rename script.
6. Mirror the tape Criterion suites (colocated `test_*.c` files) once
   the backend passes the `backend.h`-contract suites in
   `packages/idris-test-c/`.

## F64 byte-identical regression

Every commit through the modularization preserved F64 numerics
byte-for-byte:

- `make test-unit-c-{tape,torch,mlx}` runs the per-backend Criterion
  suites: explicit `ASSERT_NEAR`-style checks against hard-coded
  expected values for SGD / RMSprop / Adam / Polyak / clip /
  cross-backend transfer, plus per-op forward + backward and the
  lifted shared TUs.
- `make example-supervised` at seed=42 yields
  `loss=0.1380880098682826` (re-pinned 2026-07-29 when `Nn.linear`
  adopted PyTorch's `nn.Linear` init — `U(±1/√fan_in)` weight *and*
  bias, replacing `N(0, 1/√fan_in)` with zero bias — which necessarily
  changes every from-scratch run's starting weights. The prior pin
  `0.13666947626094297` remains the correct value for the pre-init-change
  tree and is what the fused softmax-xent work was verified
  bit-identical against; see `nn/loss/softmax_xent.c`'s FP-contraction
  notes. Init draws now come from the host-buffer `Uniform` path, so this
  value is also expected to hold across tape/torch/mlx.)

When changing a hot path, the regression bar is "test-unit-c +
example-supervised seed=42 unchanged". Any deviation needs an
explicit explanation (and probably a rollback unless the change is a
documented bug fix).
