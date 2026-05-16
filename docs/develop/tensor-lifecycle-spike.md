# Tensor lifecycle refactor — Chez GC spike findings

Date: 2026-05-15. Spike artifacts at `/tmp/refcount-spike/`.

## Pattern validated

Chez Scheme's `make-guardian` works for tracking when Idris-side handles become unreachable. Pattern:

```scheme
(define g (make-guardian))
;; Register: associate a wrapper object with the guardian
(g wrapper)
;; Drain (after GC): yields dead wrappers, one at a time, until #f
(let loop () (let ((dead (g))) (when dead (cleanup dead) (loop))))
```

A wrapper is a Chez vector containing the raw C pointer. When the only references to the wrapper come from Idris-side variables, the wrapper becomes unreachable when those variables go out of scope. After Chez GC runs, the guardian moves the wrapper to its dead queue.

Spike confirmed:
1. Pattern compiles via `%foreign "scheme:..."` in Idris 2 Chez codegen.
2. Top-level Chez bindings persist (`top-level-value` for the guardian).
3. Wrappers go through Idris-side as `AnyPtr` and survive being passed through let-bindings.
4. Dropping the Idris reference + forced GC promptly queues the wrapper as dead.

## Finding: Chez does NOT automatically GC under foreign-side pressure

The 60K-allocation test allocated 60,000 wrapper vectors without ever forcing GC:
- Drained 0 wrappers (Chez never auto-GC'd despite ~5 MB of wrapper heap pressure)
- After explicit `(collect 4)`: drained 60,000 (all reclaimed)

**Implication**: We must call `(collect)` at strategic Idris-driven points. Chez's heuristics don't trigger on wrapper-volume alone in tight loops.

## Strategic GC points

For the failing-mlx-examples case (eval-phase tight loops with 60K–240K Tensors per block):
- `tensor_no_grad_begin` — entering eval block, force minor GC + drain
- `tensor_no_grad_end` — outermost end, force major GC + drain
- Periodically inside no_grad — e.g. when `all_tensors.size()` crosses a threshold

For training (already periodically drains via `tape_reset`):
- May not need additional GC calls; `tape_reset` already runs at every optimizer step

## Cost of explicit GC

Major GC (`collect 4`) is the slow one. Minor GC (`collect 0`) is faster, ms-scale on small heaps. Per-call overhead must be amortized — drain after every FFI op is too expensive; drain after every N ops or per-batch is reasonable.

## Design implications

1. **Wrapper-as-AnyPtr is the type**: Idris-side variables hold wrapper vectors, not raw pointers. Wrappers carry the raw ptr internally.
2. **Unwrap before C FFI**: every existing `%foreign "C:..."` declaration receives raw ptrs. Idris-side wrappers extract them before the C call.
3. **Wrap after C FFI**: every C call returning a `Tensor*` is followed by `prim__makeManaged` to wrap.
4. **Forced GC at no_grad entry/exit, plus threshold-based mid-block.**

## Implementation status (2026-05-15)

**Phase 0 — baseline measured**: mlx state — dnc-copy / dnc-recall pass with the no_grad_end sweep helper (commit 524840d); ntm-copy / ntm-associative-recall / mountain-car-cont fail with `[malloc] Unable to allocate N bytes` from `MetalAllocator` (per-process Metal resource limit on the paravirtualized Tart VM). Baseline entry in `perf-log.jsonl` tagged `refcount-baseline`.

**Phase 2.1 — C-side refcount machinery, landed dormant**: mlx Tensor gains `int refcount{1}`; static `tensor_retain_internal` / `tensor_release_internal` helpers do refcount math + remove-and-delete on zero; C-exported `tensor_retain_handle` / `tensor_release_handle` declared in `backend.h`, real impl on mlx, no-op stubs on tape and torch (multi-link symmetry). Nothing yet calls them — refcount stays at 1 across each example, baseline behavior preserved.

## Phase 2.2 — Idris-side guardian plumbing (next session)

In `Tensor.idr`, add these `%foreign` declarations:

```idris
%foreign "scheme:(lambda (dummy) (if (top-level-bound? 'tensor-guardian) 0 (begin (set-top-level-value! 'tensor-guardian (make-guardian)) 1)))"
prim__initManagedHandles : Int -> PrimIO Int

%foreign "scheme:(lambda (raw) (let ((w (vector 'tensor raw))) ((top-level-value 'tensor-guardian) w) w))"
prim__wrap : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (w) (vector-ref w 1))"
prim__unwrap : AnyPtr -> AnyPtr

%foreign "scheme:(lambda (dummy) (let loop ((n 0)) (let ((d ((top-level-value 'tensor-guardian)))) (if d (begin ((foreign-procedure \"tensor_release_handle\" (void*) void) (vector-ref d 1)) (loop (+ n 1))) n))))"
prim__drainManagedHandles : Int -> PrimIO Int

%foreign "scheme:(lambda (dummy) (collect 4) 0)"
prim__forceMajorGc : Int -> PrimIO Int
```

The `foreign-procedure` call inside drain resolves `tensor_release_handle` from the dlopened libidrisml — exported in commit 0bcd... (Phase 2.1).

Call `prim__initManagedHandles` once at startup (or lazy-init on first wrap). Call `prim__drainManagedHandles + prim__forceMajorGc` at `tensor_no_grad_end` (replacing the current C-side sweep) and at every Nth FFI in heavy loops.

## Phase 2.3 — Bulk-wrap FFI returns (MUST be atomic)

**Critical finding from a partial-refactor attempt**: refactoring a single FFI is NOT safe. Tested `prim__createScalar` + `prim__item` alone — crashed with "Exception: invalid memory reference" because downstream FFIs that still expect raw pointers receive Chez-vector-wrapped handles instead and dereference them as raw `void*`.

**Therefore the refactor must touch all 165 `%foreign "C:..."` declarations in `Tensor.idr` in one commit**, OR isolate the wrapped surface behind a strict type-level boundary that prevents mixing. The latter is cleaner but a much bigger refactor.

### Atomic-bulk-wrap plan

For every `%foreign "C:..."` in `Tensor.idr` matching the pattern `prim__X : ... -> AnyPtr` or `prim__X : ... -> AnyPtr -> ...`:

1. Rename the `%foreign` declaration's symbol to `prim__XRaw`.
2. Add an Idris-side wrapper `prim__X args = prim__wrapHandle (prim__XRaw (prim__unwrapHandle a1) (prim__unwrapHandle a2) ...)` — wrapping the return, unwrapping each AnyPtr input.
3. Signatures stay the same on the outside (`AnyPtr -> ... -> AnyPtr`); wrap/unwrap is transparent at the type level.

### Edge cases

- **AnyPtr inputs, non-AnyPtr return** (e.g. `prim__item : AnyPtr -> Double`): unwrap inputs, no wrap on output.
- **Mixed args** (e.g. `prim__addScalar : AnyPtr -> Double -> AnyPtr`): unwrap only the AnyPtr arg; pass Double through.
- **TensorPair returns**: the C-side TensorPair contains two raw pointers. We need a Scheme-side variant of `prim__wrapHandle` that wraps both. Likely a separate `prim__wrapHandlePair` primitive.
- **`prim__free` / `prim__deviceTo` / etc**: these don't return Tensor handles; leave alone.
- **Index-style FFIs** (e.g. `prim__numel : AnyPtr -> Int`, `prim__size : AnyPtr -> Int -> Int`): unwrap input, return Int.

### Scripting approach

Write `scripts/bulk-wrap-ffi.py`:
1. Parse `Tensor.idr` for `%foreign "C:..."` blocks. Each block is the `%foreign` line + the next non-comment line (the type declaration).
2. Identify which args are `AnyPtr` and whether return is `AnyPtr` (or contains `AnyPtr`).
3. Generate the renamed-raw version + the Idris wrapper.
4. Emit the rewritten file. Diff-review and commit atomically.

Alternative: write the transformation as a careful manual edit, FFI block by FFI block. Tedious but tractable (~3-4h of focused work).

### Phase 2.3 validation

After bulk wrap, run the full test suite + `bash /tmp/mlx_baseline.sh`. Two outcomes expected:
- All previously-passing examples still pass (refactor is semantics-preserving with no functional change beyond when freeing happens).
- The 3 previously-failing mlx examples (ntm-copy, ntm-recall, mountain-car-cont) need additional integration: explicit `forceMajorGc + drainManagedHandles` at strategic points (no_grad_end, periodically). Phase 2.3 lands the wrapping; Phase 2.3.5 wires the drain triggers.

## Phase 2.3 attempts — design pivot + key learnings (2026-05-15 session 2)

### Pivot: smart-constructor approach, NOT per-FFI bulk wrap

Per-FFI bulk wrap (rename 165 `prim__X` → `prim__XRaw` + add Idris-side wrappers) ran into a thorny problem: many FFIs accept or return `AnyPtr` values that are NOT Tensor handles (data buffers, shape buffers, `TensorPair*`). Wrap/unwrap can't be blindly applied — needs per-FFI type discrimination. ~35 of the 165 FFIs touch non-Tensor pointers. Hand-coding the discrimination per-FFI is brittle.

Pivot **landed in commit `1594699`**: the `Tensor` record's data constructor renamed to `MkTensorRaw`; a new function-of-the-same-name `MkTensor` becomes the smart constructor. Adds a `managedShadow : AnyPtr` field — a Chez vector wrapping `tensorPtr`, registered with the guardian. The shadow is kept alive by the Tensor record; when the record becomes unreachable, the shadow is collected, the guardian queues it, drain releases.

**Key advantage**: zero FFI changes. The 185 `MkTensor X Y` call sites continue to work unchanged. The two pattern-match sites in `Tensor.idr` (`weakenGrad`, `retypeGrad`) were updated to match on `MkTensorRaw` and discard the shadow field.

**Status of commit 1594699**: dormant. Shadows are registered for every Tensor record, but nothing yet drains the guardian, so no Tensor* ever actually gets released by the new pathway. Baseline preserved exactly.

### What I tried next (REVERTED — preserved only as commits in this doc's history-of-lessons)

Wired the C-side retain/release into the lifecycle:
1. `tape_append` retains result/arg1/arg2.
2. `tape_reset` releases the same.
3. `withNoGrad` exit calls `forceMajorGc + drainManagedHandles`.
4. `param_register` retains; `param_clear` releases.
5. Constructor changed `refcount(1)` → `refcount(0)` (the constructor doesn't represent a holder; only real refs do).
6. Removed the old `tape_reset` "delete non-persistent" walk.

**Result**: increasingly subtle crashes. First: "invalid memory reference" during dnc-copy's eval phase. Diagnosis: `prim__wrapHandle` was registering a shadow but not retaining, so each shadow-drain release decremented refcount below balanced; the second wrap of the same ptr (via `retypeGrad` etc.) compounded the underflow.

**Attempted fix**: bake `tensor_retain_handle` into `prim__wrapHandle` so wrap+retain are atomic. Result: different crash — "broadcast_shapes (32,32) and (100) cannot be broadcast." Symptom of premature free: a Tensor was freed and its `mx::array` slot reused by another Tensor of different shape.

### The harder-than-expected lesson

The refcount approach requires:
- EVERY allocation of a `Tensor*` to be paired with a corresponding retain by some long-term holder (Idris handle, tape entry, param registry, persistent state, replay cache, ...) BEFORE any external code can drop the C-side caller's transient reference.
- EVERY long-term holder to release exactly once when it drops the Tensor.
- ZERO un-balanced retains or releases anywhere — including in `tensor_lstm_gates_pair`'s TensorPair construction, `tensor_one_hot`'s temporaries, and ~89 other `new Tensor(...)` sites.

The mlx backend has ~89 `new Tensor(...)` call sites and an unknown number of internal references (closures, replay state, etc.). Auditing every one is a multi-session effort, and ANY missed site causes corruption. There's no "halfway-correct" intermediate state — partial refcount is worse than no refcount.

### What to do next session

Two viable paths:

1. **Continue refcount, exhaustive audit**: spend a focused session enumerating every Tensor* lifecycle in the mlx backend, audit each for retain/release balance, instrument with sanity checks. Estimated 1-2 days of careful work. Validates option A at scale.

2. **Different design — heap-tracked allocator**: instead of refcount, track Tensor* by allocation-epoch. Periodically sweep epoch ≤ N for non-tape, non-persistent Tensors. Same idea as the current `tape_reset` no-grad-block sweep (commit 524840d) but more aggressive. Less elegant than refcount but with fewer places to get wrong.

3. **Pragmatic concession**: keep the dormant Phase 2.2 + smart constructor as a foundation. Skip the failing 4 examples locally (TODO row exists). Revisit when there's time for the exhaustive audit.

The dormant commits land harmlessly; the test suite still passes (`make test` direct), the gauntlet baseline is preserved, no production behavior changes.

## Phase 2.4 — Tape + torch backends

Wire actual refcount semantics on tape (arena needs to skip refcount-0 entries) and torch (intermediates-vector becomes ref-counted).

## Phase 2.5 — Post-refactor measurements

Re-run `bash /tmp/mlx_baseline.sh` — expect all 5 mlx examples to pass.
Run `make bench-compare` and `make bench-ops-compare`; compare to baseline.
Append `refcount-after` entry to `perf-log.jsonl`.

## Phase 3-4 — Documentation

Promote this file to `tensor-lifecycle.md` (drop the `-spike` suffix). Entries in design-decisions.md, gotchas.md, CLAUDE.md, perf-changes.md.
