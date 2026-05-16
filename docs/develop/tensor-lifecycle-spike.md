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
