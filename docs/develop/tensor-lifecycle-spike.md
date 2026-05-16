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

## Phase 2.3 — Bulk-wrap FFI returns

For every `%foreign "C:..."` in `Tensor.idr` returning `AnyPtr` (~164 of them): rename to `prim__fooRaw`, add Idris-side wrapper `prim__foo = (\args => prim__wrap (prim__fooRaw (unwrapEach args)))`. Signatures stay `AnyPtr -> ... -> AnyPtr` because wrap/unwrap is transparent at the type level.

Pattern is uniform → script with `sed`/`awk`. Tricky bit: per-input unwrap (need to detect `AnyPtr` args vs `Double`/`Int` args).

## Phase 2.4 — Tape + torch backends

Wire actual refcount semantics on tape (arena needs to skip refcount-0 entries) and torch (intermediates-vector becomes ref-counted).

## Phase 2.5 — Post-refactor measurements

Re-run `bash /tmp/mlx_baseline.sh` — expect all 5 mlx examples to pass.
Run `make bench-compare` and `make bench-ops-compare`; compare to baseline.
Append `refcount-after` entry to `perf-log.jsonl`.

## Phase 3-4 — Documentation

Promote this file to `tensor-lifecycle.md` (drop the `-spike` suffix). Entries in design-decisions.md, gotchas.md, CLAUDE.md, perf-changes.md.
