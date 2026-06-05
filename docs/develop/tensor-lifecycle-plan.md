# Tensor lifecycle: wrapped-handle FFI ABI

Plan for the next iteration of the tensor-lifecycle refactor on mlx, building on the state-only refcount work in commit `0c0cfe5` and the failed wrap-everywhere + per-FFI-RetainGuard experiment (see `tensor-lifecycle.md`).

## Goal

A single uniform Tensor lifecycle model that's robust to Idris-Chez compiler optimizations. Every Tensor's existence in Idris is represented by a Chez vector (the *wrapped handle*); that vector is what Idris-side code passes around and what every FFI takes/returns. The wrap is inseparable from the value — Idris cannot elide it without eliding the value itself. Refcount on the C side tracks all current holders (Idris bindings, tape entries, param registry, init-time constants). Free at refcount=0.

The three mlx failures (`ntm-copy`, `ntm-associative-recall`, `mountain-car-cont`) clear because intermediate accumulation inside large `withNoGrad` blocks gets bounded by periodic Chez GC + drain that's now safe — the wrap retains Tensors as long as Idris is using them, and drain only frees ones Idris has actually dropped.

## Why the previous experiments failed

**`is_state`-only refcount (commit `0c0cfe5`)** correctly handles per-sequence state Tensors but doesn't fix intermediate accumulation, which is what kills the failing examples.

**Wrap-everywhere refcount via `prim__wrapHandle`** failed because Idris-Chez codegen does live-range analysis on let-bound records. When the only use of a `MkTensor` record is `.tensorPtr` extraction, the compiler can elide the record (and therefore the guardian shadow that held its retain). The raw pointer survives, the shadow doesn't, drain frees the Tensor while the raw pointer is still in use.

**Per-FFI `RetainGuard`** failed because a `RetainGuard` is a transient holder (retain on entry, release on exit) — net zero contribution. If a Tensor enters an FFI at refcount=0 (no existing holder), the guard's `0 → 1 → 0` cycle causes a free at FFI exit. The Idris caller's next FFI hits a dangling pointer.

Both failures share a root cause: **the Tensor's owner identity lives parallel to the Tensor's value (raw pointer)**, and the compiler can drop the owner while keeping the value. Any robust design must make owner = value.

## Design

### The wrapped handle

Every Tensor on mlx is identified at the Idris level by a Chez vector:

```scheme
#(tensor-handle raw_ptr)
```

`raw_ptr` is the C-side `Tensor*`. The first element is a tag (currently the symbol `tensor-handle`) for debug introspection. The vector is registered with `idris-tensor-guardian`. Holding the vector retains the Tensor; dropping it eventually drains and releases.

From Idris's type-system perspective, the vector is still `AnyPtr`. The change is purely at the value-representation layer:

- Idris-level: an `AnyPtr` value that's actually a Chez vector.
- Scheme-level: a vector object on the heap, allocated by the FFI's Scheme glue at the moment of Tensor creation, kept alive by reachability from Idris bindings.
- C-level: the raw pointer extracted from the vector via `vector-ref` in the Scheme glue; never exposed to Idris.

### FFI conventions

Every `%foreign` declaration touching Tensors now binds to a Scheme wrapper, not directly to a C function. The wrapper:

1. `vector-ref`s each Tensor-arg to extract the raw pointer.
2. Calls the C function via cached `foreign-procedure` with raw pointers.
3. For Tensor-returning FFIs: wraps the C return in a fresh Chez vector, registers it with the guardian, retains the C-side refcount, returns the vector.
4. For primitive-returning FFIs (Double, Int): returns the C value directly.

Template:

```scheme
;; Tensor -> Tensor -> Tensor
(lambda (wa wb)
  (let ((raw_r ((foreign-procedure "tensor_FOO" (void* void*) void*)
                (vector-ref wa 1) (vector-ref wb 1))))
    (let ((wr (vector 'tensor-handle raw_r)))
      ((top-level-value 'idris-tensor-guardian) wr)
      ((foreign-procedure "tensor_retain_handle" (void*) void) raw_r)
      wr)))

;; Tensor -> Int -> Tensor   (only the Tensor arg gets vector-ref'd)
(lambda (wt dim)
  (let ((raw_r ((foreign-procedure "tensor_FOO" (void* int) void*)
                (vector-ref wt 1) dim)))
    (let ((wr (vector 'tensor-handle raw_r)))
      ((top-level-value 'idris-tensor-guardian) wr)
      ((foreign-procedure "tensor_retain_handle" (void*) void) raw_r)
      wr)))

;; Tensor -> Double   (no wrapping on return)
(lambda (wt)
  ((foreign-procedure "tensor_FOO" (void*) double) (vector-ref wt 1)))
```

The guardian is self-initialized in the wrapper (first call creates it). `foreign-procedure` calls are cached by Chez per call site, so the overhead per FFI is one `vector-ref`, one C call, one `vector` allocation, one guardian registration, one retain — measured target <100 ns per FFI.

### What changes on the C side

Minimally — most C code stays the same:

- Tensor struct gains a `refcount` field if not already present (it already does in `tensor-lifecycle.md`'s Phase 2.1 work).
- `tensor_retain_handle` / `tensor_release_handle` already exist; behavior unchanged.
- `tensor_create_*` returns Tensors with refcount=0 (caller's Scheme wrapper takes the first retain via `tensor_retain_handle` immediately after).
- `tape_append` retains result + args; `tape_reset` releases. Unconditional — applies whether `rg=true` or `rg=false`. (For `rg=false` ops, the result still needs lifecycle tracking; the tape entry is the holder. The tape may or may not contain a replay-meaningful entry — `requires_grad=false` skips replay either way.)
- `param_register` retains; `param_clear` releases each.
- Init-time constants (NTM `buildNonDiagMask`, etc.) get a permanent retain at creation.
- The "delete non-persistent at tape_reset" sweep goes away. Refcount drives all freeing.

### What changes on the Idris side

- The `Tensor` record's `tensorPtr` field semantically becomes "wrapped handle, opaque to Idris." The data layout is still `AnyPtr`, just the runtime value is now a Chez vector.
- `MkTensor wr pid = MkTensorRaw wr pid wr` — the `managedShadow` field is just `wr` itself (or we drop the field entirely since the shadow IS the tensorPtr now).
- The old `prim__wrapHandle` is gone. No conditional wrap. No `is_state`-driven branching.
- `prim__unwrapHandle` is gone (every FFI internally unwraps in its Scheme glue).
- `withNoGrad`'s drain at exit still fires; the drain mechanism (guardian + `tensor_release_handle`) is unchanged.
- Per-tape-step Chez GC + drain (foreign-callable trampoline) is still wired and now safe to fire mid-block because the wrap correctly retains across FFI calls.

### What goes away entirely

- `is_state` flag and the wrap-conditionally-on-state machinery.
- `tensor_create_managed_state_*` (collapses into `tensor_create_state_*`; every state Tensor has the uniform lifecycle).
- The "delete non-persistent at tape_reset" sweep.
- `tensor_no_grad_end`'s persistent-only sweep.
- `prim__wrapHandle` and `prim__unwrapHandle`.

## Phased rollout

### Phase 0' — Convert ONE FFI, validate dnc-copy

**Goal:** prove the mechanism works before scaling.

Convert just `prim__add` (`%foreign "C:tensor_add,libidrisml"` → `%foreign "scheme:..."` with the wrap-on-return template). All other FFIs continue with the old ABI. The mixed state has to coexist: when `prim__add` returns a wrapped handle and the next op is e.g. `prim__mul` (still old ABI taking raw `AnyPtr`), the wrap has to either flow through OR the next op needs to be converted too.

Actually that won't work cleanly — once a value is wrapped, every downstream FFI must understand wrapped values. So Phase 0' has to convert ALL the FFIs in the chain that dnc-copy uses on its hot path, not just `tensor_add`.

Refined Phase 0' scope: convert `tensor_add`, `tensor_mul`, `tensor_softmax`, `tensor_sigmoid`, `tensor_mv`, `tensor_cosine_similarity`, `tensor_narrow`, `tensor_reshape*`, `tensor_cat*`, `tensor_unsqueeze`, `tensor_linear`, `tensor_clamp_min`, `tensor_neg`, `tensor_sub`, `tensor_exp`, `tensor_log`, `tensor_sqrt`, `tensor_add_scalar`, `tensor_mul_scalar`, `tensor_item`, `tensor_create_scalar`, `tensor_create_param_*`, `tensor_create_state_*`, `prim__paramRegister`, plus the small fan-out around DNC's specific ops. Probably ~30-40 FFIs.

Then run `make BACKEND=mlx example-dnc-copy DNC_COPY_ARGS='--epochs 5 --max-len 3 --batch 1'` and expect it to pass. If yes: design works on the hot path, proceed to Phase 1'. If no: lifecycle log + a minimal repro narrows the bug.

This phase is more invasive than I'd hoped — coexistence isn't a free lunch — but it's still a contained scope (under 50 FFIs) before committing to the full 165.

### Phase 1' status (2026-05-16, in-progress)

The plan's original Phase 1' scope ("convert ~10 hot FFIs") proved insufficient: mixed-ABI doesn't compose, so a partial conversion of the dnc-copy hot path leaves *every* example crashing as soon as a converted FFI's wrapped return reaches an unconverted FFI expecting raw. Compounding that, the Device.idr typeclass-dispatch refactor added ~436 additional FFI declarations (Device.idr + Device/{Mlx,Tape,Torch}.idr), bringing the actual FFI surface to ~608 across 5 files (`scripts/lifecycle/ffi-convert-to-scheme.py` enumerates the manifest).

**Mechanical sweep approach.** Convert *all* Tensor-touching FFIs to the wrap-on-return Scheme template in one commit. Non-Tensor FFIs (`tensor_alloc_doubles`, `optimizer_*`, `mnist_load/count/label`, `param_grad_item*`, `tensor_no_grad_*`, `backend_*`, `get_*_rss_mb`) keep the `%foreign "C:..."` form. The converter is the operative definition of which FFI gets which treatment — see its `MANIFEST` and the small `SKIP` set for `tensor_retain_handle` / `tensor_release_handle` / `tensor_is_state` (used inside the wrap template itself).

**Performance trade-off in the wrapper template.** A naive `(when (not (top-level-bound? 'idris-libidrisml-loaded)) (load-shared-object …) …)` init check on every wrapper invocation costs measurable wall time on examples with high FFI counts. Removed: Idris-2's chez codegen already emits `(load-shared-object "libidrisml.dylib")` at top of the compiled `.ss` (line 14 of `supervised.ss`) for *any* `%foreign "C:..."` declaration in the source, and we retain ~40 of those (the non-Tensor ones above). So libidrisml is loaded before any Scheme wrapper runs. The guardian lazy-init is kept only on FFIs that can be the very first to touch the wrap layer (`tensor_create_scalar` + `tensor_create_*` + `tensor_one_hot` + `tensor_causal_mask` + `mnist_get_image` — see `INIT_FFI` in the converter).

**Working after sweep (on tape):** `example-supervised`, `example-mnist`, `example-lstm`. Unit tests (`Test.ManagedHandle`) green on both `BACKEND=tape` and `BACKEND=mlx` primary builds.

**Working after sweep (on mlx):** `example-supervised`, `example-dnc-copy` (training runs end-to-end on `--epochs 100 --max-len 5 --batch 1`; previously segfaulted), `example-ntm-copy` (training runs).

**Phase 1' follow-ups — status update (2026-05-16):**

- `example-rnn` hang at epochs 11-14: **resolved**. 24+ trials (3 runs × 4 epoch counts × 2 backends) all complete with deterministic losses; full `make test-examples` matrix shows rnn passing on all 3 backends. The fix landed implicitly in either Phase 3'-a (commit `4a38a5f`, removed the `managedShadow` field → Tensor record went 3 fields → 2 fields, shifting Chez GC timing) or Phase 4' canonicalization (commit `c3460ce`, normalized 3 stale Phase 0' FFI bodies that used named arg vars + a spurious `idris-libidrisml-loaded` lazy-init). The plan note's hypothesis ("GC-timing / drain-cadence interaction") is consistent with either explanation; bisecting would be expensive (build-cache invalidation) and the resolution is durable across stress tests.

- Post-main mlx `Exception: invalid memory reference`: still fires on some mlx examples after training completes, manifesting as `make test-examples` "crashed (rc=2)" entries for the 4 CI-skipped mlx targets (`ntm-copy`, `ntm-associative-recall`, `dnc-recall`, `mountain-car-cont`) plus occasionally on `a2c`, `ppo`, `dnc-copy`. Pre-existing C++ static-destructor issue (commit `2df9442`); the wrapped-handle ABI did not clear it and is unlikely to without changes to mlx's own teardown order on Apple VMs.

- `make test-examples` matrix sweep: completed on commit `d12b3bb`. 71/79 ok; 8 fail. The 8 break down: 4 are CI-skipped pre-existing (mlx destructor), 3 are likely the same post-main destructor reaching examples not yet in the skip list, 1 is `ppo:tape` mid-run UAF (separate issue — also reports `backend=mlx` in its banner despite being the tape iteration, suggesting test-examples loop is also susceptible to dylib cross-contamination on this VM).

### Phase 1' — Hot-path validation + perf baseline

After Phase 0' passes, run on the broader set the test gate already covers on mlx (supervised, rnn, lstm, gru, transformer, gpt, mnist, seq-classify, dnc-copy, reinforce, q-learning, etc.). Anything that uses an unconverted FFI from the converted side will fail loudly (raw pointer passed to wrap-expecting glue, or vice versa).

Triage: each failure either points to (a) an FFI we missed converting, or (b) a coexistence pattern that needs both sides converted at once.

Once mlx passes broadly with the ~40-FFI core converted, measure:
- `make bench-compare` ratios before/after on a hot example (e.g. lstm or transformer).
- `make bench-ops-compare` for the raw FFI overhead delta.
- Append a `tensor-lifecycle-wrapped-handle-baseline` entry to `perf-log.jsonl`.

Acceptable target: <10% wall-clock regression on the hot examples. If higher, investigate cache misses on `foreign-procedure` calls, vector allocation pressure on the Chez heap, etc.

### Phase 2' — Convert the remaining ~125 FFIs

Mechanical pass through `Tensor.idr`. Two passes:

1. **Tensor-returning FFIs** (`AnyPtr -> ... -> AnyPtr`): wrap-on-return template.
2. **Tensor-consuming FFIs returning primitive** (`AnyPtr -> ... -> Double` / `Int` / `()`): vector-ref args, return primitive.

Edge cases:

- **`TensorPair*` returns** (`tensor_lstm_gates_pair`): the C struct contains two raw pointers. Scheme glue: call C, get pair pointer, extract `first` and `second` (via `tensor_pair_first` / `tensor_pair_second` already exposed), wrap each, return as a Scheme pair or a 2-element vector. Idris-side `prim__pairFirst` / `prim__pairSecond` become Scheme accessors that take the wrapped pair and return the wrapped half.
- **FFIs taking raw buffers (not Tensors)** like `tensor_create_param_2d(int rows, int cols, double* data)`: the `data` is a raw buffer pointer from `prim__allocDoubles`, not a Tensor. Don't `vector-ref` it. Wrap only the Tensor return.
- **`prim__paramRegister(name, t) -> AnyPtr`**: returns the same Tensor pointer it was passed. In the wrapped ABI, returns the same WRAPPED handle. The wrap+retain happened upstream when `t` was created; `paramRegister` doesn't need to wrap again. Just `vector-ref` the input, call C, return the input wrap unchanged.
- **`prim__createScalar(double, int) -> AnyPtr`**: takes no Tensor inputs but returns one. Wrap-on-return template applies.

After this pass, every Tensor-touching FFI in `Tensor.idr` should be on the new ABI.

### Phase 3' — Retire the old wrap machinery

Split into two sub-phases. Phase 3'-a is the Idris-side cleanup; Phase 3'-b is the deeper C-side surgery that interacts with Phase 5'.

**Phase 3'-a (Idris-side smart-constructor wrap) — DONE.** Commit removes:
- `prim__wrapHandle` and `prim__unwrapHandle`. No callers remained (every FFI's Scheme wrapper handles wrap/unwrap internally).
- The `managedShadow` field on the `Tensor` record (it aliased `tensorPtr` under the wrapped-handle ABI — the wrap IS the value).
- The `MkTensor`/`MkTensorRaw` split — the data constructor is now `MkTensor` directly; the smart-constructor function is gone. Pattern matches in `weakenGrad`, `retypeGrad` simplified accordingly.

**Phase 3'-b (C-side is_state + state-helper collapse) — partially done.**

The plan's *full* P3'-b (retire `is_state` gate, remove the tape_reset sweep, remove `tensor_no_grad_end`'s persistent-only sweep, remove the `is_state` field) requires intermediate-Tensor drain triggers during training to be safe — without those, removing the sweep lets intermediates accumulate across epochs. Phase 5' deferred the drain trigger; the full P3'-b inherits that gating.

**What landed (P3'-b-min)** — the part that's safe without drain triggers:

- mlx: `tensor_create_state_*` and `tensor_create_managed_state_*` collapsed into a single function. The merged version uses `is_state=1` semantics (refcount-managed, survives sweep via `is_state` rather than `persistent`). Callers previously using `persistent=1` (NTM mask, BatchNorm running stats, transformer PE, DNC mask) now go through the same refcount-driven path as the per-sequence transient state — alive via the Idris-side wrap held by the model record / per-sequence binding.
- `tensor_create_managed_state_*` declarations removed from `backend.h`, definitions removed from `backend_tape.c`, `backend_torch.cpp`, `backend_mlx.cpp`. Per-backend rename headers regenerated.
- Idris-side: `prim__createManagedState1d`/`prim__createManagedState2d` removed. `Layer/Ntm.idr` and `Layer/Dnc.idr`'s `zeroState1d/zeroState2d` helpers now call `prim__createState1d`/`prim__createState2d`.
- `scripts/lifecycle/ffi_manifest.py`: removed the `tensor_create_managed_state_*` entries from MANIFEST and INIT_FFI.

**P3'-b-rest landed.** The drain-trigger gate was resolved by embedding the drain into `prim__nativeTrainStep`'s Scheme wrapper — minor GC + guardian drain runs after each training step, so refcounts from dead Idris wraps get released in time for the next sweep.

- Removed `is_state` and `persistent` fields from the mlx `Tensor` struct.
- Made `tensor_retain_internal`/`tensor_release_internal` unconditional. Release no longer deletes — it just decrements; the sweep handles deletion.
- `tape_reset` sweep now refcount-driven: `if (t->refcount > 0) keep; else delete`.
- `tensor_no_grad_end` sweep same — refcount-driven, no more flag checks.
- `param_register` retains; `param_clear` releases each entry. No more persistent flag.
- Removed `tensor_is_state` (declaration + tape/torch stubs + manifest SKIP entry).
- Removed `no_grad_state_created` tracking (refcount handles it now).
- Removed scattered `persistent = 1` assignments (dropout mask, embedding idx, param_3d/4d, view_1d/2d).
- mlx `Tensor` constructor: `refcount(0)` (was 1). Tape_append retains result + arg1 + arg2 unconditionally.

**Side effect on `make test-examples`**: 76/79 ok (was 72/79 on P3'-b-min). 4 mlx examples newly passing: `dnc-copy`, `dnc-recall`, `a2c`, `ppo`. Remaining 3 mlx failures are exactly the originally-hardest cases from the `refcount-baseline` JSONL: `ntm-copy`, `ntm-associative-recall`, `mountain-car-cont`. Same `[malloc] Unable to allocate N bytes` error, peak RSS stays bounded at 49MB → it's the Metal buffer-count ceiling on the Apple VM, not memory size. These need eval-phase drain triggers (the `withNoGrad`-exit drain isn't frequent enough for the very long eval loops these examples run).

**Verification of P3'-b-min:**
- `make BACKEND=tape test` + `make BACKEND=mlx test`: 25/25 unit tests green (including `Test.ManagedHandle`).
- `make check-ffi-wrap-template`: clean (604 FFI decls, 2 fewer than before).
- `make test-examples`: 72/79 ok, 7 fail — same 7 mlx failures as commit `d12b3bb` plus `ppo:tape` *now passes* (was failing on `d12b3bb`).

Side effect: ppo:tape went FAIL → PASS. Suggests the previous `ppo:tape` UAF (task #89) was related to the duplicate state-creation paths in some indirect way (e.g. a tape-iteration build state interacting with managed-state symbol resolution). Worth re-examining #89 in light of this.

Validate after each removal: `make BACKEND=tape,torch,mlx test` + `make BACKEND=mlx example-dnc-copy`.

### Phase 4' — Linter — DONE

Landed alongside a refactor that extracts the canonical manifest into
`scripts/lifecycle/ffi_manifest.py` — both the converter and the linter
now read from it.

- `scripts/lifecycle/check-ffi-wrap-template.py` runs structural checks
  across all 5 wrap-handle files (Tensor.idr + Device.idr +
  Device/{Mlx,Tape,Torch}.idr). Per FFI decl:
  - `%foreign "C:cname,..."` — error if base(cname) is in MANIFEST
    (missing conversion).
  - `%foreign "scheme:..."` — find the first foreign-procedure call
    whose name is in MANIFEST; verify (a) the typespec matches the
    manifest's arg/return classifiers, (b) every T arg at position i is
    unwrapped via `(vector-ref a<i> 1)`, (c) T returns wrap +
    register with `idris-tensor-guardian` + retain, (d) non-T returns
    do *not* contain a stray `(vector 'tensor-handle …)`.
  - Bespoke scheme helpers that don't call any MANIFEST symbol
    (`drainManagedHandles`, `forceMajorGc`, `initManagedHandles`) are
    exempt automatically — no annotation needed.

- `make check-ffi-wrap-template` — local invocation.
- `.github/workflows/test.yml` — added to the `check-paired-defaults`
  preflight job so a PR fails before the long matrix burns CI minutes.

Found and fixed: 3 stale Phase 0' hand-edits in Tensor.idr (`prim__item`,
`prim__requiresGrad`, `prim__setRequiresGrad`) used named arg vars (`wt`,
`rg`) instead of the canonical `a<i>` naming + one carried a stale
`idris-libidrisml-loaded` lazy-init block. Bit-identical training loss
pre/post canonicalization on supervised:tape.

606 FFI decls scanned clean across the 5 files.

### Phase 5' — Perf measurement + drain cadence tuning — DONE (scope reduced)

**Perf measurement (the measure half).** Baselines on `lstm:tape/mlx`,
`transformer:mlx`, `dnc-copy:mlx/tape` show the wrapped-handle ABI is
within the VM noise floor (~15-20% per the saved feedback memory) vs
the pre-sweep `4d350d9+dirty` baseline. `transformer:mlx` improved
slightly (37.09 → 31.63 ms/ep, -15%) which we treat as noise per the
same threshold. No example showed a measurable regression. The
mlx-CPU-stream kernel-launch wall (~30-140 ms/ep depending on op
density) dominates over any per-FFI wrap cost. See
`docs/develop/perf-changes.md` 2026-05-16 entry.

**Drain cadence tuning (the tune half) — declined as not needed.** The
plan called for a mid-block drain via foreign-callable trampoline from
`tape_append`'s no_grad branch, motivated by the original 3 failing
mlx examples (`ntm-copy`, `ntm-associative-recall`, `mountain-car-cont`)
leaking inside long `withNoGrad` blocks. Under the wrapped-handle ABI
alone (Idris-side `withNoGrad`-exit drain only), all three of these
examples now show bounded memory:
- `ntm-associative-recall`: peak=49MB stable across 700+ iters
- `mountain-car-cont`: peak=49MB stable, training to completion
- `ntm-copy` (500 epochs): peak=49MB stable across 400+ epochs

The `withNoGrad`-exit drain + per-FFI wrap-and-retain are sufficient
to keep Tensor count bounded. Mid-block drain is no longer
load-bearing; revisit if/when a workload actually needs it.

**Follow-up identified:** ntm-copy at ~450 epochs trips a mid-run UAF
(`Exception: invalid memory reference. Some debugging context lost`)
which is *not* a memory leak (memory stays at 49MB throughout). Either
a long-tail FFI lifecycle bug or the known post-main mlx VM issue
firing earlier than usual. Separate ticket.

### Phase 6' — Documentation — DONE

- New consolidated reference: `docs/develop/tensor-lifecycle.md`. Structure: The model → The wrapped-handle ABI → The drain mechanism → Discipline for new FFIs → Appendix (history of attempts 1-4). Replaces the old `tensor-lifecycle-spike.md`, which is deleted (content preserved in the appendix + commit history that the appendix points at).
- `docs/develop/design-decisions.md` "Tensor lifecycle: wrapped-handle FFI ABI" — already in place from Phase 1' (commit `218c4de`); Phase 6' refreshed the drain-mechanism paragraph (Phase 5' deferred the mid-block trampoline) and the per-FFI-churn paragraph (~600 FFIs not ~165; converter + CI linter in place).
- `docs/develop/gotchas.md` "Wrapped-handle ABI" — refreshed to point at `ffi_manifest.py` as the manifest source of truth, link to the converter + linter, and drop "once landed" language now that both are in CI.
- `CLAUDE.md` Architecture section — updated `Tensor` signature to current state (open `d : Device` kind alias, `g : GradMode` parameter), added a dedicated "Tensor lifecycle (wrapped-handle ABI)" paragraph linking to the new reference.
- Cross-references from the deleted spike doc updated in: `packages/backends/backend.h`, `packages/idris-ml/test/src/Test/ManagedHandle.idr`, `packages/idris-ml/src/Layer/Ntm.idr`, `packages/idris-ml/src/Layer/Dnc.idr`.

## Verification (cumulative)

- **Unit tests**: `make test` green at every commit.
- **Examples**: `make BACKEND=mlx test-examples` — expect *all* mlx examples passing by end of Phase 2'.
- **Cross-backend**: `make BACKEND=tape,torch,mlx test-examples` — no regressions on tape/torch (their stubs are unchanged).
- **Performance**: `make bench-compare` within 10% of pre-refactor baseline. `make bench-ops-compare` within 20%. Hard regressions trigger cadence tuning.
- **Linter**: `make check-ffi-wrap-template` clean.
- **Numerics**: `forwardVarTraced` produces bit-identical output on at least one example pre/post-refactor.

## Risks

- **Mixed-ABI coexistence in Phase 0'.** Converting one FFI at a time doesn't compose — the wrapped handle returned by a converted FFI can't flow into an unconverted FFI. Phase 0' has to convert a self-contained subset that dnc-copy uses end-to-end. Realistically ~30-40 FFIs.
- **Vector allocation pressure.** Every FFI allocates a Chez vector for the return. With ~5K FFIs/epoch, ~5K vectors/epoch on the Chez heap. Chez's GC handles this — vectors of size 2 are cheap — but watch for measurable allocation cost.
- **`foreign-procedure` cache hits.** Each Scheme wrapper calls `(foreign-procedure ...)` to dispatch to the C function. Chez caches per call site; first call resolves dlsym, subsequent calls hit the cache. The first epoch is slower; subsequent should be fast.
- **TensorPair handling.** Need to think through the two-Tensor return convention. Probably extract first and second on the Scheme side, wrap each separately, return a Scheme pair.

## Out of scope

- **Tape / torch backends.** Their lifecycles work differently (arena, libtorch shared_ptr). The wrapped-handle ABI is mlx-only for now. Tape/torch FFIs keep the raw-pointer ABI; their `tensor_is_state` / `tensor_retain_handle` stubs stay no-op. Idris-side code doesn't care because the wrap layer is purely an mlx-side implementation detail (the `AnyPtr` type doesn't distinguish).
- **Removing the Tensor record's `managedShadow` field.** It becomes redundant with `tensorPtr` once wrap is the only thing flowing through `tensorPtr`. Probably remove it in Phase 3' but not critical.
- **Idris-side language-level enforcement.** The wrap is a runtime contract, not a type-system property. A motivated user could `believe_me` past it. We're not chasing static guarantees here; the linter catches the only way to break it accidentally.
