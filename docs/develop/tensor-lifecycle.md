# Tensor lifecycle — wrapped-handle FFI ABI

Reference doc for the idris-ml tensor lifecycle model. Companion docs:
- `tensor-lifecycle-plan.md` — the phased rollout (P0' → P6') with status
- `design-decisions.md` "Tensor lifecycle: wrapped-handle FFI ABI" — the
  rationale entry
- `gotchas.md` "Wrapped-handle ABI on mlx" — the operational do/don't

## The model

Every Tensor's existence at the Idris level is represented by a Chez
vector — the *wrapped handle*. The vector wraps the raw C-side
`Tensor*` (or whatever the backend's handle type is) and is
registered with a top-level Chez guardian. The wrap IS the Tensor's
runtime identity: the Idris-Chez compiler cannot elide the wrap
without eliding the Tensor value itself.

```
Idris value (AnyPtr)  ─────►  Chez vector #(tensor-handle-v2 "TAG" raw)
                                                │
                              registered with ──┴──►  idris-tensor-guardian
                              raw is what the C code sees
                              TAG is the backend identity ("tape"/"torch"/"mlx"/"primary")
```

The wrap is a 3-slot Chez vector:

- slot 0: literal sentinel symbol `'tensor-handle-v2` — `-v2` marks
  the current layout; a stale consumer reading slot 1 expecting a raw
  pointer would see a string instead and crash obviously rather than
  silently corrupt;
- slot 1: backend tag string — `"tape"` / `"torch"` / `"mlx"` for
  per-backend wraps in `Device/*.idr`, or `"primary"` for the
  unsuffixed wraps in `Tensor.idr` that call link-time-aliased
  unified C symbols (which alias to whichever backend is primary in
  this build);
- slot 2: the raw C tensor pointer — what `foreign-procedure` calls
  actually consume.

C-level Tensor lifecycle is refcount-driven (where the backend
supports it — see "Backend asymmetry" below). The drain function
in `Tensor.idr` reads slot 1, builds the symbol name
`tensor_release_handle_<tag>` at runtime (or unified
`tensor_release_handle` for `"primary"`), and calls it on slot 2.
The lookup is cached per-tag in a Chez hashtable so the hot-path
eval loop doesn't re-resolve every drained handle.

- Wrap creation: `+1` via `tensor_retain_handle_<tag>` (suffixed)
  or unified `tensor_retain_handle` (for `"primary"`) immediately
  after the C-side constructor.
- Wrap death (Chez declares it unreachable, drain pops it): `-1`
  via the corresponding `tensor_release_handle_<tag>`.
- Long-term C holders (tape entries, param registry) take their own
  retains symmetrically.

When refcount hits 0, the Tensor is freed; its backing storage
(e.g. mlx `mx::array` → Metal MTLBuffer) is reclaimed.

**Why per-backend dispatch matters**: with the legacy unified-symbol
design (pre-2026-05-19), the drain always called the link-time-
aliased unified `tensor_release_handle` — which resolves to *one*
backend's release (the primary). In multi-backend builds that meant
mlx-allocated tensors had their refcount maintained by mlx's
allocator (which does decrement) but were released through torch's
no-op stub (when torch was primary), leaking `mx::array` storage
permanently. At process exit, mlx's static destructors then race
the still-live arrays against Metal teardown and SIGSEGV. The
v2 layout's tag-based dispatch closes this — each tensor returns
to its own backend's release.

### Why this design

The non-obvious invariant is **owner = value**. The earlier attempts
(see appendix) all kept the *owner identity* parallel to the *value*:
a Chez shadow object registered with the guardian, while the Tensor
record carried the raw pointer separately. Idris-Chez codegen does
live-range narrowing on let-bound records — once a record's only use
is `.tensorPtr` extraction, the record (and its shadow) can be elided
while the raw pointer survives. Shadow drops → guardian queues it →
drain frees the Tensor → in-flight raw pointer dangles. Crash.

The wrapped-handle ABI side-steps this by making the wrap the only
representation of the Tensor at the Idris level. There is no parallel
"raw pointer" path — every Idris-visible AnyPtr is a Chez vector;
only the FFI's Scheme glue ever sees the raw pointer, and only inside
a synchronous foreign call (during which Chez GC can't fire).

### Backend asymmetry

The wrap-and-retain plumbing fires uniformly across backends, but
refcount semantics only matter on mlx:

| Backend | `tensor_retain_handle` | `tensor_release_handle` | Notes |
|---------|------------------------|-------------------------|-------|
| mlx     | refcount++             | refcount--, delete on 0 | The reason this design exists. |
| tape    | no-op stub             | no-op stub              | Arena owns lifetimes; refcount unused. |
| torch   | no-op stub             | no-op stub              | `at::Tensor` shared_ptr handles lifetime. |

The Idris-side wrap layer is identical across backends — same vector
allocation, same guardian registration, same FFI conventions. Only
the C-side bookkeeping differs. This means new mlx FFIs and new tape
FFIs both use the same wrap-on-return template; the linter doesn't
need a per-backend mode.

## The wrapped-handle ABI

Every `%foreign` declaration that touches Tensors binds to a Scheme
wrapper, not directly to a C function. The wrapper does the wrap +
unwrap so Idris doesn't have to know about them.

### Template

For a per-backend FFI (suffixed C name `_tape` / `_torch` / `_mlx`):

```scheme
;; Tensor -> Tensor -> Tensor  (e.g. tensor_add_torch)
(lambda (a0 a1)
  (let ((raw_r ((foreign-procedure "tensor_FOO_torch" (void* void*) void*)
                (vector-ref a0 2) (vector-ref a1 2))))
    (let ((wr (vector 'tensor-handle-v2 "torch" raw_r)))
      ((top-level-value 'idris-tensor-guardian) wr)
      ((foreign-procedure "tensor_retain_handle_torch" (void*) void) raw_r)
      wr)))

;; Tensor -> Int -> Tensor  (e.g. tensor_select_mlx_streamed)
(lambda (a0 a1)
  (let ((raw_r ((foreign-procedure "tensor_FOO_mlx_streamed" (void* int) void*)
                (vector-ref a0 2) a1)))
    (let ((wr (vector 'tensor-handle-v2 "mlx" raw_r)))
      ((top-level-value 'idris-tensor-guardian) wr)
      ((foreign-procedure "tensor_retain_handle_mlx" (void*) void) raw_r)
      wr)))

;; Tensor -> Double (no wrap on primitive return)
(lambda (a0)
  ((foreign-procedure "tensor_FOO_tape" (void*) double) (vector-ref a0 2)))
```

For an unsuffixed FFI in `Tensor.idr` (link-time-aliased to primary):

```scheme
;; Tensor -> Tensor (e.g. tensor_add, aliased to primary)
(lambda (a0 a1)
  (let ((raw_r ((foreign-procedure "tensor_add" (void* void*) void*)
                (vector-ref a0 2) (vector-ref a1 2))))
    (let ((wr (vector 'tensor-handle-v2 "primary" raw_r)))
      ((top-level-value 'idris-tensor-guardian) wr)
      ((foreign-procedure "tensor_retain_handle" (void*) void) raw_r)
      wr)))
```

Rules:
- Each Tensor argument (classifier `T`) is unwrapped via
  `(vector-ref a<i> 2)` before being passed to the C function.
- A Tensor return (classifier `T`) is wrapped in a fresh
  `(vector 'tensor-handle-v2 "TAG" raw)`, registered with the
  guardian, and retained — in that order — using the
  backend-tagged retain symbol.
- The TAG must match the C function's suffix: `_tape` → `"tape"`,
  `_torch` → `"torch"`, `_mlx` (or `_mlx_streamed`) → `"mlx"`,
  unsuffixed → `"primary"`. The generator's `backend_tag_of` in
  `scripts/lifecycle/ffi_manifest.py` enforces this.
- Raw-pointer arguments (`R`, e.g. malloc'd double buffers) and
  primitive arguments (`int`, `double`, `string`) pass through
  unchanged.
- Primitive returns (`int`, `double`, `string`, `void`) flow back
  unchanged — no wrap.
- A handful of "first creators" (`tensor_create_scalar`,
  `tensor_create_*`, `mnist_get_image`, etc.) include a
  `(when (not (top-level-bound? 'idris-tensor-guardian)) ...)`
  guardian lazy-init at the head of the lambda. These are the only
  FFIs that might run before any other Tensor allocation has
  initialized the guardian.

### Three properties that make this work

1. **Foreign calls in Chez aren't interrupted by GC.** While C code
   runs inside a `foreign-procedure` invocation, Chez's GC can't
   fire. So C-side code can safely dereference the raw pointer
   extracted from the wrap; no concurrent free is possible.

2. **The wrap is a heap-allocated Chez object.** Its live range is
   tracked by Chez's normal liveness analysis on let-bindings and
   function arguments — the same mechanism the compiler is allowed
   to optimise, but only via reachability, not via field-projection
   elision. The vector can't be reduced to "just the raw pointer"
   except through an explicit `vector-ref` call inside the FFI
   wrapper.

3. **The guardian gives an Idris-liveness signal.** When the wrap
   becomes unreachable from Idris-level bindings (Chez decides this),
   the guardian queues it. A drain pass pops the dead queue and
   releases the C-side retain.

## The drain mechanism

The guardian is a single top-level Chez object, lazily initialized.
Drain is a single Idris-callable primitive (`drainManagedHandles`)
that loops `(guardian)` calls until it returns `#f`, calling
the matching `tensor_release_handle_<tag>` on each popped vector's
raw pointer at slot 2 (tag read from slot 1; lookup cached per-tag
in a Chez hashtable for hot-loop drain).

Drain triggers:
- **`withNoGrad` exit** (`packages/idris-ml/src/Tensor.idr`): the
  `withNoGrad` combinator force-runs a Chez major GC + drain on its
  way out. This is the primary lifecycle pump for eval-phase
  workloads, where ops bypass `tape_append` and per-op refcount
  bookkeeping doesn't fire.
- **Unit tests** (`Test.ManagedHandle`): `forceMajorGc` +
  `drainManagedHandles` verify the pattern works in isolation.

No mid-block drain trampoline is currently wired. Phase 5'
investigated this and found the existing drain points keep memory
bounded at ~49MB on the long mlx eval workloads (`ntm-copy`,
`ntm-associative-recall`, `mountain-car-cont`) that originally
motivated the lifecycle work. See `tensor-lifecycle-plan.md`
Phase 5' status for the decision rationale.

## Discipline for new FFIs

### The two-file workflow

1. **Add the FFI to the manifest.** Open
   `scripts/lifecycle/ffi_manifest.py` and add a new `MANIFEST` entry
   for your C symbol's base name (no `_tape`/`_torch`/`_mlx` suffix):

   ```python
   "tensor_my_op": (("T", "T", "i"), "T"),   # (arg classes, return class)
   ```

   Classifiers: `T` = wrapped Tensor handle, `R` = raw AnyPtr (not a
   Tensor — malloc'd buffer, pair handle, etc.), `i` = Int, `d` =
   Double, `s` = String, `v` = void.

2. **Run the converter (or hand-edit using its output as the
   template).** Existing `%foreign "C:tensor_my_op,libidrisml"`
   declarations in `Tensor.idr` + `Device.idr` +
   `Device/{Mlx,Tape,Torch}.idr` will be rewritten in place:

   ```sh
   python3 scripts/lifecycle/ffi-convert-to-scheme.py \
     packages/idris-ml/src/Tensor.idr \
     packages/idris-ml/src/Device.idr \
     packages/idris-ml/src/Device/{Mlx,Tape,Torch}.idr
   ```

3. **Verify with the linter:**

   ```sh
   make check-ffi-wrap-template
   ```

   Or just push — CI runs it in the `check-paired-defaults`
   preflight before the long matrix burns minutes.

### What the linter catches

- `%foreign "C:cname,libidrisml"` where `cname`'s base is in MANIFEST
  → "should have been converted to wrap-on-return scheme template."
- `%foreign "scheme:..."` whose body's `foreign-procedure` call is in
  MANIFEST but is missing `(vector-ref a<i> 1)` for a T arg, missing
  the `(vector 'tensor-handle ...)` wrap on a T return, missing the
  `tensor_retain_handle` call, or missing the
  `idris-tensor-guardian` registration.
- Foreign-procedure typespec mismatches against the manifest's
  classifiers.

Bespoke Scheme helpers that don't call any MANIFEST symbol
(`drainManagedHandles`, `forceMajorGc`, `initManagedHandles`) are
auto-exempt — no annotation needed.

### Don'ts

- **Don't pass `tensorPtr` to non-FFI Scheme code that expects a raw
  pointer.** On mlx primary builds, `t.tensorPtr` is a Chez vector,
  not a raw pointer. Either route through an FFI (which knows to
  unwrap) or write your own `(vector-ref ... 1)` extraction.

- **Don't introduce a parallel "raw pointer" field.** This was the
  failure mode of every earlier attempt. The `Tensor` record has one
  pointer field (`tensorPtr`), and it holds the wrap.

- **Don't add `Tensor*` arguments to C helper functions called from
  Scheme outside the FFI template.** Any code path that takes a
  Tensor handle must go through a wrapped `%foreign` declaration so
  the unwrap is done in one place.

- **Don't manually drain inside an FFI wrapper.** Drain reentrancy
  isn't currently guarded; calls land in a single thread and the
  ordering of guardian → drain → retain matters. Use the
  `withNoGrad`-exit drain or call `drainManagedHandles` from Idris.

## `tensor_free` in the refcount world

Pre-refcount, `tensor_free` was the C-level "release this handle"
primitive — it called `delete t` directly and removed the Tensor from
`all_tensors`. In the wrapped-handle ABI (Phase 4'), that role is
covered by the guardian → drain → `tensor_release_handle` chain, so
`tensor_free` is now legacy. Backends handle it as follows:

| Backend | `tensor_free` body | Why |
|---------|--------------------|-----|
| mlx     | `tensor_release_internal(t)` after probing `all_tensors` for liveness | Refcount sweep at next `tape_reset` reclaims. Force-delete would dangle tape entries. |
| tape    | no-op (`(void)h`) | Arena owns lifetimes; `tape_reset` clears the whole arena. |
| torch   | no-op (`(void)h`) | `at::Tensor` `shared_ptr` handles lifetime; small documented leak for test-only paths. |

The mlx behavior here was tightened on 2026-05-18 (`<commit>`) after
test_backend.c surfaced an intermittent Bus error on tape_reset.
The pre-existing code force-deleted intermediates, leaving dangling
`Tensor*` pointers in tape entries; the next `tape_reset` walked
them and crashed. The new behavior drops one refcount and lets the
sweep at `tape_reset` do the deletion when the count actually hits 0
— consistent with the doc model above. The `all_tensors` probe
defends against the caller passing an already-swept handle (e.g.
across an `optimizer_step` that called `tape_reset` internally).

### Test-harness implication: read grads before `param_clear`

`param_clear` on mlx is part of the refcount lifecycle: it releases
the per-param retain and runs `tape_reset` (which itself sweeps
refcount=0 tensors). The vector backing `param_registry` is actually
emptied. So any code path that reads `param_grad_item_at(...)` after
`param_clear` on mlx hits an empty vector — historically undefined,
in practice "got 0.000000" or worse.

Tape's `param_clear` is count-only (`param_count_val = 0` — see
`backend_tape.c`) and leaves the registry array intact. Reads after
clear accidentally still work on tape, masking the bug class.

The fix lives at the test layer, not the backend. Capture analytical
grads **before** any `param_clear` inside an FD block:

```c
tensor_backward(loss);

/* Capture analytical grads before FD scaffolding —
   each FD block below calls param_clear. */
double analytic_x = param_grad_item_at(0, 0);
double analytic_y = param_grad_item_at(1, 0);

/* FD checks (each subblock free to call param_clear) */
{
    param_clear();
    /* ... fresh non-grad tensors for FD ... */
    ASSERT_NEAR("x grad", analytic_x, fd, FD_TOL);
}
```

Applied across `test_mm_backward`, `test_bmm_backward`, and
`test_layer_norm_2d_backward` on 2026-05-18. The pattern is the same
on every backend; mlx just surfaces the latent bug visibly.

## Appendix: history of attempts

The wrapped-handle ABI was the third design attempted; the previous
two foundered on the same compiler-elision contract. Each attempt
left commits and a learning that's worth not re-discovering.

### Attempt 1: state-only refcount (commit `8ffd48b`)

Surgical refcount for per-sequence transient state Tensors only —
the ones produced by `Layer/Ntm.idr` and `Layer/Dnc.idr`'s
`zeroState1d`/`zeroState2d` helpers. A new `is_state` flag on
the mlx `Tensor` struct gated `tensor_retain_handle` /
`tensor_release_handle` (no-op when `is_state == 0`). Per-FFI
wrap was conditional via `prim__wrapHandle`'s `tensor_is_state`
check.

What worked: `dnc-copy` / `dnc-recall` passed on mlx. The
narrow surface (one flag, a handful of allocation sites)
side-stepped the audit-all-89-Tensor-allocation-sites problem.

What didn't: intermediate Tensors inside long `withNoGrad`
blocks weren't lifecycle-managed. `ntm-copy`,
`ntm-associative-recall`, `mountain-car-cont` still hit the
Metal MTLBuffer ceiling on Apple Virtualization VMs (Tart, GHA
macOS runners) before `no_grad_end`'s sweep fired.

### Attempt 2: wrap-everywhere via `prim__wrapHandle` in `MkTensor` (reverted)

The dormant Phase 2.2 design: `MkTensor` (smart constructor)
called `prim__wrapHandle` to register a shadow with the
guardian and retain. Every Tensor record carried a parallel
`managedShadow : AnyPtr` field aliasing the wrap.

What didn't: Idris-Chez does live-range narrowing on let-bound
records. When a record's only use was `.tensorPtr` extraction,
the compiler elided the record (and the shadow). Shadow drops
→ guardian queues it → drain releases the Tensor → in-flight
raw pointer dangles. "Exception: invalid memory reference"
during dnc-copy's eval phase. Various follow-up fixes
(`prim__wrapHandle` retains; revert `refcount` constructor
default) produced different downstream crashes — all
symptoms of the same root cause.

**Lesson**: owner identity can't live parallel to the value.

### Attempt 3: per-FFI `RetainGuard` (reverted)

A C++ RAII guard on the C side: each Tensor FFI's entry built a
`RetainGuard` (retain on construction, release on
destruction) around each Tensor argument.

What didn't: a guard's retain-then-release cycle is net-zero
*unless* the Tensor enters the FFI with refcount > 0. For a
freshly-created Tensor at refcount=0, the guard pushed it to 1
on entry and back to 0 on exit, triggering a free — *after* the
FFI had returned the freshly-created pointer to the caller, who
was about to wrap it. "Exception: invalid memory reference."

**Lesson**: transient retains can't bridge a value's first
allocation to its first long-term holder.

### Attempt 4: wrapped-handle ABI (the current design)

Every Tensor-touching `%foreign` declaration binds to a Scheme
wrapper that wraps on return and unwraps Tensor args. The wrap
is the value's only Idris-level identity. No parallel shadow
field. No conditional wrapping based on `is_state`.

Rolled out in phases:
- P0' (commit `e47975a`): 4 FFIs converted, unit tests green
- P1' (commit `0ec6a99`): mechanical sweep across all 5 wrap-handle
  files (~600 FFIs total — Tensor.idr + Device.idr +
  Device/{Mlx,Tape,Torch}.idr)
- P3'-a (commit `78bc19b`): retire `prim__wrapHandle` /
  `prim__unwrapHandle` / `managedShadow` field / smart-constructor
  function — the wrap doing all the work means the parallel layer
  was redundant
- P4' (commit `9664726`): structural linter +
  `scripts/lifecycle/ffi_manifest.py` as single source of truth
- P5' (commit `7fc29c7`): perf baselines (within VM noise of
  pre-sweep) + drain-cadence tuning declined (memory bounded at
  ~49MB on the originally-failing mlx examples without mid-block
  drain)

Outstanding follow-ups: `is_state` gate retirement +
`tensor_create_managed_state_*` collapse (task #87); rnn epoch-count
hang (task #86); ntm-copy mid-run UAF at ~450 epochs (task #88).
