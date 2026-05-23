# Gotchas Reference

Comprehensive reference for all known pitfalls in the idris-ml codebase. Organized into four categories. See also [design-decisions.md](design-decisions.md) for rationale behind key choices.

> **Note: Path C migration deleted several V1-era gotchas.**
> The V1 entries below referencing `Variable d` (shape-erased), `nameLayer`/`autoName`, `applyDeltas`,
> `toDoubleNetwork`, `Endofunctor.emap`, `DenseOptimizer`, `NtmMemBuf`, `WeightBuf`, scalar-tape
> internals, and the V1 `LayerLike` interface are **no longer applicable** post-migration. They are
> preserved as historical context — see [path-c-migration.md](path-c-migration.md) for what
> superseded them. Top-of-file sections (Idris 2 / Chez Scheme traps, Training & Numerics, NTM /
> DNC / MLX-specific gotchas) remain accurate for V2 code.

## Idris 2 / Chez Scheme Traps

These are compiler/runtime pitfalls that produce confusing errors or silent misbehavior.

### `total` is a keyword

Idris 2 reserves `total` as a totality annotation keyword. Never use it as a variable/parameter name — produces a cryptic "Couldn't parse declaration" error at the definition clause. Use `numEpochs`, `totalEpochs`, etc. instead.

### Build flags

Forgetting `--source-dir src` or `-p contrib` produces confusing import errors. Examples aren't in the package, so manual flags are needed:

```bash
idris2 --source-dir src -p contrib -o <name> src/Example/<Name>.idr
```

### Top-level `build/ttc/` cache goes stale on where-clause body changes

Idris 2's interface-hash dependency tracking invalidates downstream TTCs only when a module's public interface changes. When you edit the body of a where-clause local inside a public function (e.g. `logEpoch` inside `runTrainingIO`), the interface hash is unchanged and `build/ttc/<ver>/Example/*.ttc` are considered fresh — but they have the old inlined code baked into their Chez-compiled `.so`. Result: library changes install correctly (`~/.idris2/.../idris-ml-0/...`) but single-file `idris2 -o` example builds reuse stale code.

Symptom: you edit a library internal, `make install` succeeds, `make example-foo` succeeds, but the binary runs old behavior. `rm -rf build/ttc` makes the change take effect.

Mitigation: the Makefile has a `build/.library-cache-stamp` sentinel depending on every library `.idr` file. When any is newer than the stamp, the recipe wipes `build/ttc`. `install` depends on the stamp, so every `make example-<name>` / `make check-examples` / `make test-examples` path gets the fresh-cache guarantee transparently. If invoking `idris2` directly outside the Makefile, run `rm -rf build/ttc` after editing library internals.

### Temporary test files

Idris2 requires source files to be in `--source-dir`. Never put test files in `/tmp` — they won't compile. Instead, add temporary test files to `src/Example/` and remove them after debugging.

### Elementwise `(*)`

`Tensor`'s `Num` instance uses elementwise multiply. For matrix-vector products, use `matrixVectorMultiply` or `vectorMatrixMultiply` from Math.idr.

### Arena chunk size must exceed largest single allocation

The arena allocator uses chunked linked-list allocation. If a single `arena_alloc` request matches the chunk size exactly, subsequent allocations after `arena_reset` can hit chunk boundary corruption. Fixed by increasing `ARENA_INIT_SIZE` from 1MB to 4MB. The trigger was embedding output for batch=32 × seqLen=64 × dModel=64 = 131072 doubles = exactly 1MB.

### Large Nat type-level reduction hangs the compiler

Idris 2 represents `Nat` as Peano numbers at the type level — `2304` becomes `S (S (... (S Z)...))` with 2304 constructors. Type unification walks all of them. This causes the type-checker to **hang indefinitely** when:

- An identity layer (same input/output dim) is used at a large dimension. For example, `DropoutState 2304` or `BatchNormState 16 576` requires proving `2304 = 2304`, which means reducing `16 * (12 * 12)` to a chain of 2304 `S` constructors.
- The network chain (`~>`) gets long (10+ layers), compounding unification cost.

**Practical thresholds observed:**
- Dims ≤ 512: fine (dropout at `AfterPool2 = 32 * (4*4) = 512` compiles instantly)
- Dims ~ 2304: hangs (dropout at `AfterPool1 = 16 * (12*12) = 2304` never completes)
- Dims ~ 9216: hopeless (batch norm at `AfterConv1 = 16 * (24*24) = 9216`)

**Workarounds:**
- Place identity layers (dropout, batch norm) only at smaller dimensions (after pooling, before FC)
- Avoid identity layers at conv output dimensions (which can be thousands)
- For batch norm specifically, consider fusing it into the conv layer (conv-bn fusion) rather than making it a separate network layer

**Root cause:** Idris 2 lacks opaque/machine-backed type-level naturals (like GHC's `TypeLits`). This is the single largest practical limitation for type-safe tensor shapes at scale. See the Idris 2 issue tracker for discussion.

### Pattern-matching `Nat` literals in a case arm OOMs the elaborator at large values

Same Peano-explosion class as the entry above, in the *pattern-compilation* path rather than the type-unification path. Symptom: a case arm like

```idris
case r of
  Left (TokVocabMismatch 12345 30522) => check "..." True   -- BOOM
  Left err                            => putStrLn (show err) >> pure False
  Right _                             => ...
```

OOM-kills `idris2` (SIGKILL during compilation of the surrounding function). The literal patterns `12345` and `30522` get unfolded to Peano `S (S (... Z))` during case-tree construction.

**Workaround**: match the constructor with pattern variables, compare values at runtime via `==`:

```idris
case r of
  Left (TokVocabMismatch claimed onDisk) =>
    if claimed == 12345 && onDisk == 30522
      then check "..." True
      else do { putStrLn ("FAIL: …"); pure False }
  ...
```

`(==)` on `Nat` is fast because `Nat` is stored as `Integer` at runtime — the equality check is O(1) integer comparison, not recursive Peano walk. Only the *type-level* / *pattern-compilation* paths suffer.

**Investigated and ruled out as a nixpkgs build issue (2026-05-26)**: the nixpkgs idris2 v0.8.0 derivation patches `bootstrap-stage2.sh` to replace `MAKE all` with `MAKE idris2-exec`, which seemed (per the now-replaced "Opaque type-level Nats" TODO row) like it might be the binding constraint. It isn't — the stdlib `.ttc` files are built by the *separate* `mkPrelude.nix`-derived `prelude` / `base` / `contrib` / etc. packages, each invoking the stage-2 `idris2-unwrapped` binary via `IDRIS2=...`. Reverting the patch would only add redundant rebuild work; it does NOT change the stdlib quality. The Nat-pattern OOM is an Idris-2-the-language thing in v0.8.0, not a nix packaging defect — wait for v0.9.0 or use the `==` idiom in pattern arms.

### `Data.Nat` stdlib functions compile to recursive Peano walks at runtime

`Nat` is stored as a GMP `Integer` at runtime (`%builtin Natural`) — checking equality and adding 1 is O(1). But the stdlib `Data.Nat` functions are *defined* by pattern matching on `Z` / `S k` constructors, so the Chez codegen emits recursive decrement code regardless of the underlying representation. Functions affected: `Data.Nat.lte`, `gte`, `lt`, `gt`, `compare`, `divNat`, `modNatNZ`, `divCeilNZ`, and anything that calls them.

For `divNat n 2` with `n = 256`, this means **128 recursive `(cond ((equal? arg-0 0) ...) (else (let ((e-0 (- arg-0 1))) (divNat e-0 ...))))` calls**, plus an `lte` call per iteration that itself recurses. A single `posEncVal` call with `dim ∈ [0, 256)` doing `div dim 2` + `modNatNZ dim 2` ran ~400 Nat-recursive operations.

**Found 2026-05-14** in `Layer/Transformer.idr`'s positional encoding (`posEncVal`). 32K `posEncVal` calls per forward × 32 forwards/epoch × ~400 Nat ops each = **3.9 billion `Data.Nat.lte`/`divC-39`/`modC-39` calls per epoch** on GptLarge. Came out as the headline hotspot in a Chez source-level profile (`docs/develop/chez-profiling.md`). Cost: ~8 of 9 wall seconds per epoch, depending on backend.

**Workaround** — cast to `Int` once at the function entry, use `Int div`/`Int mod` thereafter. Single CPU instructions:

```idris
posEncVal : Nat -> Nat -> Nat -> Double
posEncVal dModel pos dim =
  let dimI = the Int (cast dim)              -- O(1): just unwraps Integer
      i = cast {to=Double} (dimI `div` 2)    -- single Int div, not Nat recursion
  in if (dimI `mod` 2) == 0 then sin ... else cos ...
```

Keep the public `Nat` interface — the cast is essentially free. Bit-identical numerics across all backends.

**Comparison gotcha — fast vs slow Nat comparisons**:

```idris
(<) : Nat -> Nat -> Bool       -- Ord_Nat instance — Integer compare, FAST
Data.Nat.lte : Nat -> Nat -> Bool   -- recursive Peano walk, SLOW
```

The Ord-instance comparators (`<`, `<=`, etc.) route through `compare_Ord_Integer`, which is one CPU instruction. The explicit `Data.Nat.lte` / `lt` / `gte` / `gt` functions don't. Use the operators, not the named functions.

**When to care:** any code that does `Nat` arithmetic (div/mod especially) inside a hot loop. Shape arithmetic in layer forward passes is the canonical case. ML-style inner loops touch this every time they index by position, head, channel, etc. — audit with the chez profile recipe (`docs/develop/chez-profiling.md`) if a wall-time hotspot is unexplained.

**Upstream:** the cleaner fix is in the Idris 2 stdlib — define `Data.Nat.divNat` etc. as `fromInteger (cast n `div` cast m)` directly, which preserves the type and proof obligations but compiles to a single GMP `div`. File-level rewrite of ~8 stdlib helpers. Not yet attempted.

### Polymorphic type-parameter slot vs concrete value in method body — exponential metavar accumulation

If a record carries a polymorphic type-parameter slot (`record Tensor ... (0 dt : DType) ...`) but the interface methods that operate on values of that record hardcode a concrete value in the slot (`applyVar : ... -> Tensor [i] d F64 g -> ...`), Idris-2's elaborator allocates a fresh unification variable for `dt` at every Tensor reference and keeps it alive across the module to support cross-method elaboration. The variable always unifies to F64 trivially — but the metavar state isn't released until the module compiles.

For a small module this is invisible. For a layer with hundreds of Tensor references (`Layer/Dnc.idr` is the canonical case — DNC's memory + temporal-link matrix + read/write heads = many nested record-update chains), the kept-alive metavars accumulate. Observed on this codebase: 33+ GB resident in Chez Scheme on a single `idris-ml` build, climbing as Layer.Dnc elaboration proceeded. Four parallel idris2 builds during iteration drove the host (iTerm2 + spawned processes) to 99 GB.

**Symptom:** `idris2 --build` builds that exceed ~10 GB resident on a codebase that previously fit in <5 GB, with the slowdown concentrated on the most reference-dense layer files. Examples build fine; the cost is on the library where the polymorphic slot meets the concrete methods.

**Fix:** make the methods polymorphic in the slot too. Interface bodies should bind the type parameter and use it, not hardcode a concrete value:

```idris
-- ❌ Polymorphic record slot, concrete method body:
record Tensor (dims : Vect rank Nat) (0 d : Device) (0 dt : DType) (0 g : GradMode)
interface LayerLike (l : Nat -> Nat -> Device -> DType -> GradMode -> Type) where
  applyVar : ... -> l i o d F64 g -> Tensor [i] d F64 g -> ...

-- ✓ Polymorphic record slot, polymorphic method body:
record Tensor (dims : Vect rank Nat) (0 d : Device) (0 dt : DType) (0 g : GradMode)
interface LayerLike (l : Nat -> Nat -> Device -> DType -> GradMode -> Type) where
  applyVar : {0 dt : DType} -> ... -> l i o d dt g -> Tensor [i] d dt g -> ...
```

The polymorphic version binds `dt` once per function call instead of per Tensor reference, so each call site allocates one metavar regardless of how many Tensor references the function body contains. Build memory drops back to baseline.

**When you'll hit it:** any refactor that adds a new 0-quantity phantom to `Tensor` and tries to migrate the library "loosely" — leaving methods concrete because they only support one value of the new parameter today. Resist that. Plumb the parameter all the way through methods even when only one instantiation exists; let callers pick the concrete value at the leaf use site.

**Watch signal:** during the refactor, if a single `idris2 --build` exceeds ~10 GB RSS on a module that historically built in <2 GB, that's the warning. Stop and check whether method bodies hardcode the new parameter while the record is polymorphic.

Discovered during the dt-parameter refactor on 2026-05-17. Documented in detail in `docs/develop/dtype-parameter.md` "Lessons learned."

### Tensor Foldable reversal

The `foldr` instance for `Tensor` processes elements in reversed order (head into accumulator first). `toList` produces elements backwards. Use direct `Vect` traversal instead when element order matters (e.g., packing into C buffers, extracting prediction values for argmax).

Pattern for correct-order extraction:
```idris
tensorVals : {n : Nat} -> Vector n Variable -> List Double
tensorVals (VTensor xs) =
  let go : Vect k (Scalar Variable) -> List Double
      go [] = []
      go (STensor v :: rest) = prim__item v.tensorPtr :: go rest
  in go xs
```

This caused a subtle bug in the Transformer example where `toList` reversed prediction logits, making the loss function (which uses `vecStackTensor` in forward order) show near-zero loss while the argmax (which used `toList` in reversed order) gave wrong classes.

### Zero-arg FFI CSE trap

Idris 2 compiles zero-argument `%noinline` definitions as constants evaluated once at load time. `tapeGeneration` must take a dummy argument (the tape index) passed through to `prim__tapeGen` to prevent the Chez backend from caching the result. This also applies to any other FFI wrapper reading mutable state. Even making it `foo _ = expr` doesn't help — the argument must be passed THROUGH to the FFI call: `foo dummy = cast (prim__ffi (cast dummy))`.

### `PrimIO Bits64` FFI returns corrupt state in tight loops — use `PrimIO Int`

The `primPerfOpCount` FFI (introduced commit `26a0d56` for the #393 op-submission counter) originally declared `PrimIO Bits64`. The C side returns `long`, which is `int64_t` on macOS — fits both `Bits64` and `Int`. Calling the typeclass-routed `perfOpCount {d=ExampleDevice}` in a tight loop (once per decode step in HfLlama's `genLoop`) reliably crashed tape F32 inference with `Exception: invalid memory reference` / `Illegal instruction: 4` after ~8 iterations (#401). Switching the FFI declaration to `PrimIO Int` (and the `Tensor.idr` wrapper to `IO Int`) eliminated the crash — same C function, same workload, full 13-step decode. The exact failure mode of Idris-2's chez codegen for `unsigned-64` returns in a typeclass-dispatched `PrimIO` is unresolved (the other documented hypotheses — typeclass dispatch, FFI marshalling — weren't independently isolated), but the working code uses `Int`. Lesson: for FFI counters / sizes / handle indices returning values that fit in `Int64`, default to `PrimIO Int` not `PrimIO Bits64`. Reserve `Bits64` for cases where the unsigned semantics genuinely matter and the call frequency is low.

### Wrapped-handle ABI — new Tensor FFIs must use the wrap-on-return template

Every `prim__` FFI that touches Tensor handles binds to a Scheme wrapper, not directly to a C function. The wrapper extracts the raw pointer from each Tensor arg via `(vector-ref a<i> 2)` before the C call, and (for Tensor returns) wraps the C result in a fresh Chez vector + registers it with `idris-tensor-guardian` + retains via `tensor_retain_handle_<backend>`. The vector IS the Tensor's runtime identity — Idris-Chez codegen can't elide it without eliding the value itself. See `docs/develop/tensor-lifecycle.md` for the full model and `docs/develop/design-decisions.md` "Tensor lifecycle: wrapped-handle FFI ABI" for the rationale.

**Wrap layout v2** (since 2026-05-19): the wrap is `(vector 'tensor-handle-v2 "TAG" raw_ptr)` — slot 0 is a sentinel symbol, slot 1 is the backend tag string (`"tape"` / `"torch"` / `"mlx"` for per-backend wraps in `Device/*.idr`, or `"primary"` for unsuffixed wraps in `Tensor.idr` that call the link-time-aliased unified C symbol), slot 2 is the raw pointer. Retain is symmetric: per-backend wraps call `tensor_retain_handle_<tag>` (suffixed), primary wraps call unified `tensor_retain_handle` (which aliases to primary's). The drain function in `Tensor.idr` reads slot 1, builds the matching `tensor_release_handle_<tag>` symbol name at runtime, and dispatches — *this is what makes multi-backend builds correct*. Before v2 the drain always used the link-time-aliased unified release (typically the primary's no-op), so mlx-allocated tensors leaked their refcount and tripped SIGSEGV during exit-time mlx static destructor teardown.

**For new FFIs**: add the C symbol to `scripts/lifecycle/ffi_manifest.py`'s `MANIFEST` with arg/return classifiers (`T` = wrapped Tensor handle, `R` = raw AnyPtr, `i`/`d`/`s`/`v` = primitive/void), then run `python3 scripts/lifecycle/ffi-convert-to-scheme.py <files>` (or hand-edit using its output as a template). The generator derives the backend tag from the C name's suffix (`_tape`/`_torch`/`_mlx`/unsuffixed) — keep the naming convention to get the right tag.

**Do NOT** pass `tensorPtr` to non-FFI Scheme code that expects a raw pointer — it's a Chez vector. Either route through an FFI (which knows to unwrap) or write your own `(vector-ref ... 2)` extraction.

**Linter**: `make check-ffi-wrap-template` runs structural checks across all 5 wrap-handle files (Tensor.idr + Device.idr + Device/{Mlx,Tape,Torch}.idr). It catches missing conversions (raw `%foreign "C:..."` for a manifest symbol), missing `(vector-ref a<i> 2)` unwraps on T args, missing `(vector 'tensor-handle-v2 "TAG" …)` wrap on T returns, mismatched tag (e.g. a `_torch`-suffixed C name wrapped as `"mlx"`), missing `tensor_retain_handle_<tag>` retain calls, and the legacy `'tensor-handle` sentinel anywhere (which would silently corrupt slot-1 readers expecting raw pointers). Wired into the CI `check-paired-defaults` preflight — violations fail the build before the long matrix burns minutes.

### A new device-pinning C shim must `try/catch` → NULL, or it re-opens the SIGABRT

The EAFP availability gate (`docs/develop/device-availability-gating.md`) rests on one contract: any C entry point that pins a tensor to a hardware variant (`.to(device)` on torch, a stream switch on mlx, future CUDA/Metal allocators) must wrap the operation in `try/catch` and return a NULL `TensorHandle` on failure — *not* let the backend's exception propagate. An uncaught `c10::Error` (or any C++ exception) crossing the C→Chez FFI boundary becomes `std::terminate`/SIGABRT, which the Idris side cannot recover from. With the catch, the failure surfaces as a NULL handle that `prim__handleIsNull` (reads wrap-v2 slot 2; a NULL `void*` comes back from a `%foreign` as the Chez fixnum `0`) turns into `Left DeviceError` via `attemptOn`. Today all torch device-pinning routes through the single guarded `tensor_to_device` (`backend_torch.cpp`); if you add a *new* device-pinning symbol, guard it the same way or the EAFP gate silently doesn't cover it. tape/mlx `tensor_to_device` are `return t` (never throw), so the gate degrades to "always succeeds" there — correct, but it means the null→`Left` path is only exercisable on a torch build with absent CUDA/MPS (the macOS-only CI lanes can't hit CUDA absence; verify on a real CUDA box via `scripts/test_cuda_colab.sh`).

### `let x = ffiCall` inside `IO do` is hoisted to a module-level constant

Idris-2's Chez codegen treats `let x = expr` inside a `do` block as a pure binding. When the surrounding function's arguments are erased / fully-applied at module load, the *entire* `do` block including the let-bindings can be hoisted into a `csegen-NN` module-level constant — but the **side-effecting lambda body that follows it stays linked to the same allocated buffers**. Every call to the constant re-runs the side effects on the same memory.

Concretely, this `makeVec4` form crashed with `pointer being freed was not allocated` on the second call:

```idris
makeVec4 (a, b, c, d) = do
  let buf  = primAllocHost {d} 4          -- side effect
  let buf1 = prim__setDouble buf 0 a      -- side effect
  -- ... more sets
  let sh   = primAllocIntHost {d} 1       -- side effect
  let sh1  = primSetIntHost {d} sh 0 4
  let ptr  = primCreateFromHost {d} buf4 sh1 1 1
  primIO (\w => MkIORes (primFreeIntHost {d} sh1)  w)
  primIO (\w => MkIORes (primFreeHost    {d} buf4) w)
  pure (MkTensor ptr Nothing)
```

The generated Scheme bound `u--buf`, `u--sh`, `u--ptr`, etc. as `let` values **outside** the `(lambda (world-0) …)` that runs the frees. So `csegen-NN = makeVec4 someTypeclassDict 'erased someConstant` allocates the buffers and creates the tensor *once at module load*, and every test that called `csegen-NN ext-0` re-ran the frees on the same pointers.

**Fix**: route every side-effecting step through `primIO`:

```idris
makeVec4 (a, b, c, d) = do
  buf  <- primIO (\w => MkIORes (primAllocHost {d} 4) w)
  buf1 <- primIO (\w => MkIORes (prim__setDouble buf 0 a) w)
  -- ...
  ptr  <- primIO (\w => MkIORes (primCreateFromHost {d} buf4 sh1 1 1) w)
  primIO (\w => MkIORes (primFreeIntHost {d} sh1)  w)
  primIO (\w => MkIORes (primFreeHost    {d} buf4) w)
  pure (MkTensor ptr Nothing)
```

`primIO` binds the result through the `%World` token, which threads through the rest of the IO chain and prevents the let-hoisting. Now each call to the function re-runs the alloc, fill, create, free sequence cleanly. The fix lives in `packages/idris-ml/test/src/Test/Transfer.idr` and `packages/idris-ml-examples/src/Example/Transfer.idr`; the symptom is a libsystem_malloc abort on the *second* invocation of any code that follows this `let`-chain-with-side-effects pattern. Related: `feedback_typeclass_zero_arg_method_eval.md`, "Zero-arg FFI CSE trap" above.

### FFI side-effect threading

`let _ = ffiCall` is dropped by the compiler since the result is unused. FFI functions with side effects must return a value that is used in subsequent computation. `prim__gradAdd` returns the handle (`AnyPtr`), enabling handle threading through the backward pass. Dense optimizer steps use `prim__seq result st.v` to force evaluation: `let result = prim__rmspropVcStep ... in { v := prim__seq result st.v } st`. Without this, the optimizer call is silently eliminated and raw gradients are applied as deltas (lr/clip/momentum have zero effect).

### `withNoGrad (pure (pure-typed FFI body))` is a footgun

If a function with hidden FFI side effects has a *pure* type, the FFI calls fire during strict argument evaluation — *before* `pure` constructs its IO action, and therefore *before* `noGradBegin` runs. The bracket was effectively a no-op on the eval path.

The library fix (commit `8a32a86`'s parent and earlier): every Tensor-handle-touching smart constructor + every `applyVar` / `forwardVar` is now `IO`-typed. FFI bodies fire when the IO action is sequenced via `<-`, which happens *inside* the bracket. The helper `ioRerun : (() -> a) -> IO a = primIO (\w => MkIORes (f ()) w)` defers a pure body to IO without using the prelude's private `MkIO` constructor; `Lazy a` was rejected because it memoizes (we need re-evaluation per call).

### Per-FFI `ioRerun` overhead amplifies on mlx small-op training

The `ioRerun : (() -> a) -> IO a = primIO (\w => MkIORes (f ()) w)` helper that defers FFI-bearing pure bodies to IO adds one closure construction and one `MkIORes` allocation per FFI call. Tape's per-op cost is so low this is invisible. Torch absorbs it (libtorch's per-op is already heavy enough). mlx-cpu and mlx-gpu pay it visibly: small-net training (rnn/lstm/gru/ntm) regressed ~5× vs pre-IO-refactor on mlx (`perf-changes.md` 2026-05-17 entry). The matmul-bench compute-bound regime (N ≥ 2048) is invisible to the overhead because each op is ms-scale compute; mlx-gpu still hits 4.3 TFLOPS at N=4096. Treat the regression as the cost of correctness; if it ever matters, the lever is streamlining `ioRerun`'s shape (drop the closure or the IORes box).

### Large-model inference needs explicit `releaseAllPersistent` at end of `main`

Inference programs that complete with hundreds of MB of live tensor handles hit a 14-22 minute post-main C-side cleanup tail on the CPU lanes (torch-cpu, mlx-cpu). The work is libtorch's per-`at::Tensor` destructor cascade (`~at::Tensor → ~Storage → CPUAllocator-free`) walking the ~146 params + ~600 forward intermediates accumulated across an 8-token no-cache greedy decode. The GPU lanes (`torch-mps`, `mlx-gpu`) don't show this — MTLBuffer release is async.

Fix: call `releaseAllPersistent {d=ExampleDevice}` after `runGenerate` and before `pure ()`. On torch this `free_intermediates()` + walks `param_registry_arr` deleting each `at::Tensor*`; cascade runs inside `main` where it's timed + bounded. Measured on HfLlama-1.2B BF16 torch-cpu (commit `81e3caa`): **wall 23m22s → 1m21s**.

Pair with `drainManagedHandles + forceMajorGc` immediately before (the standard `withNoGrad` cleanup pair) so any guardian-tracked wraps from the run are popped before the explicit release. mlx-cpu's `releaseAllPersistent` is a no-op today — `mlx_sweep_generation` is static-scoped in `autograd.cpp` and the simpler `param_clear + mx::clear_cache` regressed the mlx-cpu wall; exposing the sweep + walking `all_tensors` is the proper fix and a deferred follow-up.

### Long eval loops need per-sequence `withNoGrad`, not per-batch

Even with the IO refactor, wrapping a 100-sequence × 20-step eval in a single outer `withNoGrad` can OOM mlx on Tart/GHA VMs: forward passes allocate Metal buffers that Chez has no visibility into, so Chez GC doesn't fire before the Metal MTLBuffer ceiling. `withNoGrad`'s exit does `forceMajorGc + drainManagedHandles`, but once-at-end is too late.

Fix: push `withNoGrad` *inside* the loop. NTM eval: `evalOne dp = withNoGrad $ do { ... }` (per-sequence). RL eval: `withNoGrad (evalEp …)` inside `evalN`'s recursive call (per-episode). Tape and torch don't need this — only mlx hits the cap — but the per-sequence pattern is cheap on both, so it lives in the example code uniformly.

### Grad-mode per-epoch eval + the mlx generation sweep = use-after-free

A `cfg.metrics` callback that runs `forwardVar` in grad mode (no `withNoGrad`) builds autograd-tape entries during evaluation that are never consumed or reset. On mlx the per-epoch generation sweep (`tensor_epoch_end`) then deletes `rc==1` tensors created since `epoch_begin` — including eval intermediates the lingering tape still references — so the *next* epoch dereferences freed memory. It surfaces non-deterministically as bogus mlx reshape-size aborts (`[reshape] Cannot reshape array of size N into shape …`, where N drifts run-to-run because the corruption lands at whatever op the reused slot hits first), not a clean error at the eval site. tape/torch are immune: their `tensor_epoch_end` is a no-op (no Metal buffer ceiling to bound). This bit transformer + gpt once the per-epoch generation free landed.

Fix lives in the library: `Train.idr`'s `logEpoch` wraps `cfg.metrics` in `withNoGrad` so eval builds no tape, and `forceMetrics` forces the result strings *inside* that bracket (the metric strings are lazy `show (primItem …)` / `argmaxAtPtr …` thunks that would otherwise dangle when the bracket-exit drain frees the eval tensors — same lazy-FFI footgun as the per-sequence case above). Example `metrics` callbacks therefore need not (and should not) manage no-grad themselves.

### The mlx generation sweep must never `delete` a tensor with a live holder — husk it

Deeper root cause behind the entry above, fixed 2026-05-21. `mlx_sweep_generation` (`backend_mlx.cpp`) used to `delete` every `rc==1` tensor created since the block/epoch start, on the theory that `rc==1` meant "wrap-only, dead intermediate." But `rc==1` *is* the Idris guardian wrap's own retain, and a wrap still at `rc==1` at sweep time is one that has **not yet been drained** (a drained wrap would have dropped the tensor to `rc==0`). Its eventual `drainManagedHandles` calls `tensor_release_handle`, an **unguarded `((Tensor*)h)->refcount--`** (no `all_tensors` probe, unlike `tensor_free`). Once the `delete`d slot's address is recycled by a later allocation — which is the *common* case under training churn — that decrement lands on a *different live tensor*, knocking it toward a premature free → macOS freelist-guard `EXC_BREAKPOINT` (`Trace/BPT trap: 5`, `_xzm_xzone_malloc_freelist_outlined`).

Symptom signature that points here: intermittent SIGTRAP heap corruption that **Guard Malloc suppresses** (it never recycles freed addresses, so the stale decrement can't alias a live object), shows up far more on **F32 than F64** (smaller objects → denser size-class reuse → higher alias probability), and scales with **run length** (more epochs ⇒ more sweeps and more accumulated un-drained wraps released in a batch at the next `withNoGrad`). `withNoGrad` masked it because it does `forceMajorGc + drainManagedHandles` *before* the sweep, so its `rc==1` survivors are genuinely reachable; the training epoch bracket (`tensor_epoch_end`) and `withGenFree` do not GC+drain first, so they hit the bug.

Fix: **husk instead of delete.** For an `rc==1` block-local tensor the sweep releases the heavy `mx::array` buffers (this reclaims the MTLBuffer, the only thing the live-handle ceiling actually cares about) but keeps the lightweight `Tensor` object alive, its address pinned, until the wrap drains it to `rc==0`, when a later sweep frees it via the `rc==0` path. This preserves the no-GC buffer reclamation that `withGenFree`'s per-step RL loops depend on, needs no Idris/wrap-layout changes, and is sound regardless of which holder (wrap or tape) the `rc==1` represents. General rule: **a refcounted Tensor object must not be freed while any holder's release can still touch it** — drop the buffer, keep the husk.

**The husk must release buffers without *allocating* (fixed 2026-05-22).** The first husk used `t->data = t->grad = mx::array(0.0f)`, which *allocates* two fresh scalars per swept tensor. On Apple Silicon every mlx buffer — even 4–8 bytes — routes through `MetalAllocator` (unified memory), so under the paravirt-Metal per-process MTLBuffer ceiling (Tart/GHA VMs) those per-sweep allocations throw `[malloc] Unable to allocate N bytes` (N=4 f32, N=8 f64 — one scalar) *mid-training*. The throw fires before `main` returns (`g_mlx_past_main == false`), so the post-main `set_terminate` mitigation can't swallow it → `Abort trap: 6`. Hit NTM/DNC/mnist/mountain-car/ppo on mlx + mlx-gpu; the old `delete`-sweep never allocated so never tripped it. Fix: assign a single process-wide `static const mx::array g_husk_empty` instead — mx::array is copy-on-write (a shared_ptr to its buffer), so the assignment is a refcount bump, not an allocation, yet still drops the husk's heavy buffer.

**The husk leaked handles on long grad-mode runs — because the drain pump was never primed (fixed 2026-05-22).** The husk's whole contract is "keep the lightweight object until the wrap drains it to `rc==0`, then the next sweep frees it." But the guardian drain helper `idris-drain-once` was installed *only* by `initManagedHandles`, which production calls **nowhere** (only `test/src/Test/ManagedHandle.idr`). So in every real run the drain epilogues were dormant no-ops: `withNoGrad`'s `drainManagedHandles` returned 0 (its prim guards `(top-level-bound? 'idris-drain-once)` → unbound), and the `(when (top-level-bound? 'idris-drain-once) …)` loop in each `native_train_step_<b>` wrapper never ran. The C sweep still dropped *buffers* (so MTLBuffer count stayed bounded — why `withNoGrad` eval looked fine), but the husk *objects* could never reach `rc==0`, so they accumulated in `all_tensors` forever (~1650/epoch on ntm-copy, linear; the mountain-car DQN run showed `handles=106324` within one episode). Harmless at smoke scale (≤30 epochs) but OOMs 10000-epoch convergence runs.

This reframed the "(a) GC+drain vs (b) generation-tagged wraps" question recorded earlier: the husk is *sound*; the bug was that draining never happened. Fix is **(a)** — make the drain actually run — in two parts. (1) **Prime the drain universally** at first tensor creation: the guardian lazy-init carried by the `INIT_FFI` create wrappers (`GUARDIAN_LAZY_INIT` in `scripts/lifecycle/ffi_manifest.py`, propagated by `prime-drain-lazy-init.py`) now also installs `idris-drain-once`, so every entry point (training, eval, notebook, manual loops) drains. (2) **Reclaim epoch-scope intermediates** in `Train.idr`'s `epochEnd`: mlx-gated `forceMajorGc + drainManagedHandles` *before* `primEpochEnd`, mirroring `withNoGrad`. The per-step `(collect 0)` minor GC in `nativeTrainStep` runs *inside* the epoch fn while intermediates are still reachable, so it can't collect them; the post-return major GC + drain makes their dead wraps unreachable and releases them (`rc 1→0`), and the existing sweep frees the husks. **(b)** generation-tagged wraps (v3 layout) is unnecessary — no ABA risk once draining works as the husk was designed to assume; no wrap-layout change. Verified: `handles=` flat across 100 epochs on ntm-copy (was ~1650/epoch). tape/torch skip the per-epoch GC (gated); their `tensor_release_handle_*` are no-op stubs so universal draining there is harmless (and stops the tiny wrap-vectors leaking in the never-drained guardian — a latent all-backend win).

### `fst`/`snd` re-evaluation trap

When a function with FFI side effects returns a tuple and the caller accesses fields via separate `fst`/`snd` projections (e.g., `fst result`, `snd result`, `fst result` again), Idris 2 compiled to Chez Scheme may re-evaluate the function call for each projection instead of sharing the result. This causes FFI side effects (tape appends, buffer allocations) to execute multiple times. Fix: use `case f args of (a, b, c) => ...` to destructure in a single pattern match. This was a 3x re-evaluation bug in the NTM forward pass — the LSTM controller was called 3 times per timestep instead of once.

### `prim__seq` for evaluation ordering

When two FFI side-effect chains must execute in order but have no data dependency, use `prim__seq a b` (Scheme `(lambda (a b) b)`) to force `a` to evaluate before `b` is used. Chez Scheme evaluates function arguments strictly.

### `foreign-set! 'void*` corrupts memory

Do NOT store C pointers in `foreign-alloc`'d arrays via `foreign-set! 'void*`. It corrupts memory in Chez Scheme — causes "invalid memory reference" crashes with large tape sizes. Use C helper functions (`ext_meta_set`) instead. Similarly, storing C `void*` pointers in Scheme vectors via `vector-set!` works initially but values silently become `#f` (possibly GC-related). Use C-side arrays for pointer storage.

### Chez Scheme output buffering

Stdout is fully buffered when redirected to file/pipe (e.g. background tasks). Use `stdbuf -oL ./build/exec/<name>` to force line-buffering for long-running training.

### C shared library required

`build/libidrisml.dylib` must exist before running any example. Build with `make build/libidrisml.dylib`. The library is loaded by the generated Chez Scheme code at startup. Idris 2 copies the dylib to `build/exec/<name>_app/` at compile time — the Makefile targets also copy it explicitly to ensure the latest version is used. When building manually (not via `make`), you must copy the dylib to the app dir after rebuilding: `cp build/libidrisml.dylib build/exec/<name>_app/`.

### Linear-quantity arguments are consumed by field projections

If you write `f : (1 _ : T) -> IO U` where `T` is a record, the body must use the argument exactly once. Idiomatic field-access syntax inside the body **counts as a use per projection** — `t.foo` and `t.bar` consume `t` twice, fails linearity. Solution: pattern-match-destructure the argument at the binder so each field is a separate ω-quantity binding:

```idris
-- FAILS: two uses of linear `t`
weakenGrad t = do
  primIO (prim__setRequiresGrad t.tensorPtr 0)
  pure (MkTensor t.tensorPtr t.paramId)

-- OK: ptr and pid are ω-bindings, can be used multiple times
weakenGrad (MkTensor ptr pid) = do
  primIO (prim__setRequiresGrad ptr 0)
  pure (MkTensor ptr pid)
```

This trapped `weakenGrad` and every layer's `unfreezeLayer` impl during the GradMode refactor.

### `traverse` doesn't compose with linear functions

`traverse : (Traversable t, Applicative f) => (a -> f b) -> t a -> f (t b)` declares its function argument at unrestricted (ω) quantity. A linear function `(1 _ : a) -> f b` doesn't unify with that type — you get "Mismatch between: (1 _ : T) -> IO U and ?a -> ?f ?b" at the `traverse` call site.

For `Maybe (Tensor ... NoGrad)`, `Vect k (Tensor ... NoGrad)`, etc., you have to write manual recursion that destructures the constructor and applies the linear function inline:

```idris
prev' <- case prev of
  Nothing => pure Nothing
  Just p  => Just <$> weakenGrad p

-- For Vect, manual recursion:
freezeLinearVec : Vect k (LinearState i o d g) -> IO (Vect k (LinearState i o d NoGrad))
freezeLinearVec [] = pure []
freezeLinearVec (l :: ls) = do
  l' <- freezeLayer l
  ls' <- freezeLinearVec ls
  pure (l' :: ls')
```

Verbose but unavoidable until / unless Idris ships a `LinearTraversable` interface.

### Polymorphic record-field types let you cast 0-quantity parameters without `believe_me`

For a record like `Tensor (dims : Vect rank Nat) (0 d : Device) (0 g : GradMode)` where `g` is 0-quantity (erased), the auto-generated constructor `MkTensor` is polymorphic in `g`:

```
MkTensor : {0 dims : Vect rank Nat} -> {0 d : Device} -> {0 g : GradMode} ->
           AnyPtr -> Maybe String -> Tensor dims d g
```

Destructure-then-reconstruct gives a **fully type-checked cast** between values at different `g`:

```idris
retypeGrad : Tensor dims d g1 -> Tensor dims d g2
retypeGrad (MkTensor ptr pid) = MkTensor ptr pid
```

No `believe_me`. The constructor accepts any `g`, the destructure binds `ptr`/`pid` at the run-time-relevant fields (unrestricted ω), and the reconstruction picks the new `g` from the expected return type. Runtime is identity (same `tensorPtr`, same `paramId`). The technique generalizes to any record where the type parameters you want to cast are 0-quantity.

### Polymorphic function fields in records require explicit higher-rank syntax

When a record stores a function value (e.g. `RnnState.activation`) that needs to operate at any `g`, the field type itself must be polymorphic — *not* the record's `g`. Concretely:

```idris
-- Field is at the record's g; activation fixed once at construction
record RnnState (i o : Nat) (0 d : Device) (0 g : GradMode) where
  ...
  activation : TVec o d g -> TVec o d g     -- ❌ fixed
  ...
```

vs.

```idris
record RnnState (i o : Nat) (0 d : Device) (0 g : GradMode) where
  ...
  activation : {0 g' : GradMode} -> TVec o d g' -> TVec o d g'  -- ✓ usable at any g
  ...
```

The fixed-`g` version forces `activation` to be specialized at construction, so after `freezeLayer` retypes the state to `NoGrad`, the stored activation function no longer matches. The polymorphic-`g'` version transports unchanged — standard activations like `ttanh` are already polymorphic post-Phase-3, so they unify with this field type automatically.

### Higher-order type parameter doesn't propagate erasure annotations

When you write a `data` declaration whose first parameter is a layer-kind type constructor like `Nat -> Nat -> (0 _ : Device) -> (0 _ : DType) -> (0 _ : GradMode) -> Type`, applying that parameter inside a constructor body **loses the `(0 _ : ...)` multiplicity annotations**. Idris-2's unifier can't propagate erasure from the parameter type to the application site, and you get:

```
Mismatch between: Type -> Type -> GradMode -> Type
and (0 _ : Device) -> (0 _ : DType) -> (0 _ : GradMode) -> Type
```

The natural-looking `LayerLike l => LayerLikeMixed (\i, o, d, _, ct, g => l i o d ct g)` type-level-lambda instance head also doesn't work — the lambda body's argument types lose multiplicity the same way.

Workaround: **wrap the higher-order type through an existing existential**. Instead of `LayerLikeMixed (LambdaOver l)`, define a concrete `data AsMixed` whose constructor takes an `AnyLayer i o d dt g` (which already has the multiplicity-annotated args bound correctly) and produces an `AsMixed i o d dt dt g`. Pattern-matching `MkAsMixed (MkAnyLayer l @{dict} layer)` recovers the inner `LayerLike` dict + layer, and your instance methods delegate. See `Layer/MixedCore.idr` (`AsMixed`) and the design-decisions "LayerLikeMixed bridge" section.

Take-away: layer-kind-to-layer-kind bridges in Idris-2 default to **concrete wrapper data types**, not type-level lambdas.

### Named auto-implicit collision when type parameters unify

If your typeclass method signature has two auto-implicit constraints on different type parameters (e.g. `{auto rdtP : RuntimeDType pDt}` + `{auto rdtC : RuntimeDType cDt}`), and a downstream instance collapses those type parameters (`pDt = cDt`), the two dicts BOTH type-match an inner call's single `RuntimeDType dt` constraint and Idris's resolver can't pick. You'll see:

```
Error: Multiple solutions found in search of:
    RuntimeDType pDt
Possible correct results:
    i_con (implicitly bound at ...)
    i_con (implicitly bound at ...)
```

Fix: **name the auto-implicits in the interface signature** (`{auto rdtP : ...}` + `{auto rdtC : ...}`), then at the call site bind by name and pass explicitly, using `@{%search}` to mark the positions where auto-resolution should still run:

```idris
applyVarMixed {rdtC} {cmpC} (MkAsMixed (MkAnyLayer l @{dict} layer)) input = do
  (layer', out) <- applyVar @{dict} @{%search} @{%search} @{rdtC}
                                    @{%search} @{cmpC} layer input
  ...
```

This pins the slots that need disambiguation while letting unconflicted constraints (`UserDeviceTraining`, `UserDeviceCore`, `Linked`) auto-resolve normally. Pattern recurs anywhere a typeclass collapses two type parameters that an inner call disambiguates by type. See `Layer/MixedCore.idr` `LayerLikeMixed AsMixed where`.

## Training & Numerics

Gradient flow, numerical stability, and training patterns.

### `paramId` is required for gradient flow

`Tensor`s without a `paramId` (i.e., `Nothing`) are invisible to the C-side optimizer and won't receive updates. Always pass a paramPrefix to `*LayerAny` constructors:

```idris
ll <- linearLayerAny {i=2} {o=3} "ll0"   -- registers "ll0_weights" + "ll0_bias"
```

For multi-network examples (A2C / PPO / SAC), pick distinct paramId prefixes per network (`"actor_"`, `"critic_"`, `"q1_"`, `"q1tgt_"`, ...) and create per-network optimizers via `nativeAdamGroup "actor_" ...`. The V1 "double `nameLayer` creates stale handles" bug class is structurally impossible in V2 since each layer is named exactly once, at construction.

### `logSoftmax` + `nllLoss`

Separate softmax + cross-entropy creates autograd intermediate gradients of 1/pp (up to 1e6) that destabilize recurrent training. Use `logSoftmaxLayer` + `nllLoss` instead. Note: the aligned NTM uses sigmoid + BCE instead, which doesn't have this issue.

### `pow` zero-base NaN

`pow(0, k)` backward for the exponent computes `0^k * log(0) = 0 * -Inf = NaN`. Fixed by returning 0 when base is 0.

### Detached max in `logSoftmax`

The max subtraction for numerical stability uses a detached constant (`fromDouble . cast`), not a reference to the max Variable. Otherwise the max element receives incorrect gradients.

### Gradient clipping

`adam` clips per-parameter; `adamGlobalClip` clips by global L2 norm (preserves gradient direction). Use `adamGlobalClip` for attention/recurrent models where parameters must coordinate — per-parameter clipping distorts direction and causes periodic loss spikes. Default maxNorm is 50.0 (Collier & Beel); 5.0 was too aggressive.

### Weight initialization

`linearLayer`/`rnnLayer` default to Xavier uniform. Biases are always zero. Init strategies compose a variance method with a distribution sampler: `xavier uniform` (default), `xavier normal`, `he normal`, `xavierGain 1.4 uniform`, etc. Use `linearLayerWith (fixedRange 1.0)` for the old `U(-1,1)` behavior. Use `linearLayerWithBias initFn biasStd` for custom bias init (normal with given std). NTM head FCs use `xavierGain 1.4 uniform` + `normal(0.01)` bias, output FC uses `he uniform` + `normal(0.01)` bias (matching PyTorch reference). NTM memory initialized to `sigmoid(xavier_random)` ≈ values in [0,1] (matching PyTorch's `sigmoid(FC_bias)`). Read output uses kaiming uniform. `Sampler.idr` provides `uniform` and `normal` (Box-Muller); `Init.idr` provides `xavier`, `xavierGain`, `he`, `lecun`, `fixedRange`.

### Hyperparameter tuning

Fix algorithmic issues first (bounded activations, correct clipping, efficient backward pass), then use `scripts/sweep.sh` for systematic grid search. Never manually loop over hyperparameters — see `design-decisions.md` for rationale.

### Periodic GC for long training

NTM training (50K+ epochs) OOMs without periodic forced GC. `forceGC` (exported from Variable.idr) calls Chez `(collect (collect-maximum-generation))` with `(heap-reserve-ratio 1.0)` every 10 epochs in NTM training loops. The `heap-reserve-ratio 1.0` minimizes retained heap (default ~2.0 retains 2x live data), and max-generation collection is more thorough. The FFI lambda must take 0 args — `%World` is erased in Chez Scheme's PrimIO calling convention.

### `getRssMB` peak RSS tracking

`getRssMB` (exported from Variable.idr) returns peak RSS in MB via C `get_rss_mb` (`getrusage(RUSAGE_SELF).ru_maxrss`). Takes a dummy `Nat` arg to prevent CSE (pass epoch number at call sites). Returns peak (high-water mark) RSS, not current — it only goes up. Division to MB done in C to avoid 64-bit return value issues. Used in training loop logs and bench output.

### `getCurrentRssMB` current RSS

`getCurrentRssMB` (exported from Variable.idr) returns current resident memory in MB via `mach_task_info` on macOS. Unlike `getRssMB` (peak), this reflects actual current usage and can decrease after GC. Returns -1 on non-macOS platforms.

### Curriculum learning

Available via the Curriculum module for staged training. The PyTorch-aligned NTM (LSTM controller + RMSprop) does not require curriculum — it converges directly with two-phase training. Curriculum was previously required for feedforward controllers (ajithcodesit finding).

## NTM-Specific

NTM architecture, training protocol, and convergence behavior.

### NTM dimension calculations

`ReadParamWidth m = (m + ShiftKernelSize) + 3` (key of width m + 3-element shift kernel + 3 dynamic params: β, g, γ). `WriteParamWidth m = ReadParamWidth m + m` (addressing params + add vector of width m). The LSTM controller input is `m + inputSize` (read output + input). The output FC input is `h + m` (hidden + read output). The `ntmLayer` constructor takes `{inputSize, outputSize, n, m, h}` as implicit args.

### NTM head parameters

β (key strength), g (interpolation gate), γ (sharpening) are dynamic — extracted from head FC outputs (fed by LSTM cell state). β uses softplus, g uses sigmoid, γ uses `1 + softplus(x)` (unbounded, [1, ∞)). Add vectors are raw linear (no activation). See `forwardReadHeadUnbounded`/`forwardWriteHeadInterp` in Memory.idr.

### NTM state flow

`readHeadOutput` from the previous timestep concatenates with current input to form LSTM input (width `m + inputSize`). LSTM cell state feeds head FCs, hidden state + read output feeds output FC. Memory, addressing weights, and read output all update each step.

### NTM batch size

Copy task converges well with batch=16 (uniform encode-then-decode structure, consistent gradient signal across sequences). Recall task requires batch=1 (online learning) — variable item counts (2-6), random query positions, and content-based retrieval create a complex optimization landscape with many local minima. Batch averaging dilutes the per-sequence addressing signal that the NTM needs to learn distinct write slots and query-triggered retrieval. All reference implementations (Graves 2014, Collier & Beel 2018, vlgiitr) use batch=1 for recall and train for 100K+ iterations. The snipsco/ntm-lasagne implementation found recall gets stuck in local minima even at 500K iterations with larger batches. Default: `NtmCopy.idr` uses batch=1 (seed=42 converges at ~9300 epochs; seed=123 does not converge — batch=1 is seed-sensitive), `NtmAssociativeRecall.idr` uses batch=1.

### NTM two-phase training

Copy/recall use `epochTwoPhaseDenseBce` — encoding inputs fed with outputs discarded, then zero inputs fed during output phase with loss on targets. The C-backed `bceWithLogitsVar` (tag 26) fuses sigmoid + BCE into a single tape entry per output vector, replacing ~7 scalar ops per element. No output activation layer needed.

### No tanh memory bounding

Interpolation write uses raw interpolation (no tanh) to match the PyTorch reference. The Collier & Beel tanh recommendation was for the original erase+add write mechanism, not interpolation write. Tanh caused cumulative degradation during output phase (near-zero write weights still applied tanh every timestep). `tanhBound` in Layer.idr is only used for LSTM gates, not NTM memory. The C kernel `interp_write_compute` supports both modes via `raw_mode` flag (1=raw, 0=tanh); Idris always sets raw_mode=1.

### NTM initial addressing

Read/write addressing weights are initialized to zeros and read output to Kaiming uniform (non-learnable, matching PyTorch reference). `syncLayerBuffers` projects addressing weights onto the probability simplex via `projectWeights` (clamp to [0, epsilon], renormalize) to prevent NaN from `pow(negative, non-integer)` in `focus`.

### NTM early stopping

NTM examples (copy/recall) use windowed-average convergence checking instead of best-loss patience. Parameters: `esThreshold` (default 0.01), `esWindow` (default 1000 epochs), `esPatience` (default 3 consecutive checks). Every 100 accumulated epochs, computes interval average loss, then averages the last `esWindow/100` intervals. Stops when this window average < threshold for `esPatience` consecutive checks. CLI flags: `--es-threshold`, `--es-window`, `--es-patience`. The LSTM example still uses the old best-loss patience mechanism.

### Controller output clipping (removed)

Previously `applyLayerVar` clamped raw NTM controller output to [-20, 20] via `clampVar`. Removed to match PyTorch reference which has no output clamping. The LSTM controller + RMSprop + value clip ±10 provide sufficient stability without artificial clamping.

### NTM-Copy convergence is highly seed-sensitive at 5K epochs

The aligned NTM-Copy model has high variance in convergence rate across seeds at moderate epoch counts. Both the PyTorch reference and the Idris tape backend show ~1/4 pass rate at 5K epochs (only specific seeds hit 99%+ accuracy at that budget). This is the model itself, not a backend bug.

Measurements at seed=42/7/99/123, batch=1, 5K epochs, threshold-disabled (`acc_short / acc_full`):

| Seed | tape         | PyTorch ref     |
|------|-------------:|----------------:|
| 42   | 75% / 59%    | **100% / 100%** |
| 7    | 82% / 74%    | 74% / 60%       |
| 99   | **99.8% / 99.8%** | 76% / 57%  |
| 123  | 75% / 62%    | 72% / 60%       |

Implication: don't read a single-seed under-budget run as a backend bug. Compare the same seed against PyTorch ref before concluding anything. Final convergence (e.g. 25K+ epochs with `WindowedPercentile` early-stop) is the right gate; 5K epoch snapshots are too noisy. The ≥4/5 multi-seed pass rate gate in the convergence plan should be applied at full convergence budgets, not at fixed-epoch checkpoints.

### NTM tape backward uses Apple Accelerate BLAS — seed=42 trajectory shifted

After commit `9311eff` (2026-05-11), `OP_MM` / `OP_BMM` / `OP_MV` /
`OP_LINEAR` / `OP_LINEAR_2D` backward kernels dispatch to
`cblas_dgemm` / `dgemv` / `dger` on Apple. The forward kernels
already used BLAS; the backward switch closes a long-standing
performance gap for matmul-heavy backwards (transformer, DNC).

`dgemm`/`dger` reduce in a different floating-point order than the
hand-rolled triple loops they replaced, so per-step gradients
differ by ~1 ULP from the pre-`9311eff` tape. NTM-Copy's
seed-sensitivity is acute enough that this flips seed=42 onto a
slower-converging branch: tape ntm-copy goes from ~4400 ep to
~7000 ep (acc_full=1.0 either way); ntm-recall from ~8500 ep /
k4=0.98 to ~18000 ep / k4=0.91. acc_short / acc_k2 stay at 1.0;
the regression is on the inherently seed-sensitive
length-generalization gates.

Implication: if you're chasing best-case NTM-Copy convergence
runtime, try multiple seeds — the BLAS reduction order may favour
a different seed than seed=42 did pre-`9311eff`. For most
workloads (transformer, DNC, supervised/RNN family) the BLAS path
is a clean win and not a tradeoff at all.

### NTM-Copy default seed is per-backend after broadcast adoption

The `Layer/Ntm.idr` `ntmInterpWriteIdris` helper now uses the tape backend's
numpy-style 2D broadcast (`(n,1)*(n,m)`) directly instead of materialising
`outer(w, ones_m)`. The change is bit-identical at single-timestep
mathematically, but multi-timestep training trajectories diverge in
ULP-level ways from the workaround chain (different reduction order in
backward sums). Combined with NTM-Copy's seed sensitivity, this flips
which seeds converge per backend:

| Seed | tape (broadcast) | torch (broadcast) | mlx (broadcast) |
|------|---|---|---|
| 42   | ✅ ~4400 ep / 1.0 | ✅ ~5300 ep / 0.99 | ❌ ~9000 ep / 0.65 |
| 99   | ⚠️ 30K-cap / 0.97 | ⚠️ slow            | ✅ ~4400 ep / 0.997 |

The `Makefile` `example-ntm-copy` target now defaults `--seed 42` for tape
and torch, `--seed 99` for mlx. Override with `NTM_COPY_ARGS="--seed N"`.
The in-Idris `defaultConfig` and the paired `torch_ref/scripts/ntm_copy.py`
default both stay at seed=42 (matches the primary tape/torch path).

## Architecture & Infrastructure

C kernels, buffer systems, optimizer internals, and the layer system.

### Any new C symbol needs `make rename-headers` regenerated

When you add a new `extern "C"` function to a backend (or to a
shared `.c` that links into a backend's dylib), the Idris-side FFI
binding likely targets the per-backend suffixed name
(`my_func_tape`, `my_func_torch`, `my_func_mlx`) — but those
suffixes are macro renames from `packages/backends/rename_<b>.h`,
auto-generated from `backend.h`. If the new function isn't in the
regenerated rename header, the dylib exports only the un-suffixed
symbol and the Idris binding fails at **link time** with:

```
Exception in foreign-procedure: no entry for "my_func_tape"
```

Fix: add the function declaration to `packages/backends/backend.h`,
then run `make rename-headers` to regenerate the per-backend rename
headers. The `make check-rename-headers` CI gate catches drift —
run it after every backend.h addition.

This bit during the #410 A3 work when `native_train_step_scaled`
was declared in `backend.h` but `rename-headers` wasn't regenerated;
`primNativeTrainStepScaled` linked against `native_train_step_scaled_tape`
which didn't exist as an exported symbol until the rename macro was
added. The fix took one `make rename-headers` command.

### Shared `optimizer.c` only links into tape — torch/mlx need per-backend ports

The `packages/backends/shared/training/optimizer.c` file is gated by
`SHARED_BACKENDS_optimizer := tape` in the Makefile, so it compiles
**only into the tape dylib**. Torch and mlx have their own
`backend_*/training/optimizer.cpp` files that implement
`native_train_step` directly against their respective libraries
(libtorch's `at::Tensor` accessors for torch; mlx's lazy arrays for
mlx).

Consequence: when you add a new train-step variant (like
`native_train_step_scaled`), you need **three implementations**, not
one. The shared file gets the tape version; each per-backend
`optimizer.cpp` gets its own port. The behavioural test then runs
on all three with the same assertion shape, catching per-backend
divergence.

Symptom of forgetting: the test links cleanly only on `BACKEND=tape`;
on torch and mlx the linker fails with `Undefined symbol:
_native_train_step_scaled` because the symbol exists only in tape's
dylib.

This bit during #410 A3 — the initial `native_train_step_scaled`
landed only in the shared file and gated the test with `#ifdef
BACKEND_TAPE`. Three commits later, torch's
`backend_torch/training/optimizer.cpp` and mlx's
`backend_mlx/training/optimizer.cpp` got their own ports and the
gate came off.

### FFI manifest entry required for wrap-on-return

Any C function whose Idris-side `%foreign` binding accepts or returns a
`TensorHandle` (raw `void*` from Idris's perspective, but a Chez
`tensor-handle-v2` vector after the lifecycle machinery wraps it) MUST
have an entry in `scripts/lifecycle/ffi_manifest.py`'s `MANIFEST` dict.
The entry is the `(arg_types, return_type)` shape that drives the
auto-generated Scheme wrapper.

What goes wrong if you forget: the Idris-side declaration stays as
`%foreign "C:tensor_xxx_<backend>,libidrisml"`. The Chez codegen wires
the AnyPtr args straight through, so the C function receives the
**wrapped** Chez vector (a pointer into the Scheme heap pointing at a
3-slot vector `#(tag tag-string raw-ptr)`) as its `TensorHandle` arg
instead of the **unwrapped** raw `Tensor*`. Cast `(Tensor*)hq` in C
then reads garbage at offset 0 (Chez vector header) and crashes with
`Exception: invalid memory reference. Some debugging context lost`.

How to fix: add the entry, then run
`python3 scripts/lifecycle/ffi-convert-to-scheme.py`. The converter
rewrites the `%foreign` lines in `Device/{Tape,Torch,Mlx}.idr` and
`Tensor.idr` to the wrap-on-return Scheme template that unwraps each
`T`-typed input via `(vector-ref a<i> 2)` and re-wraps the return
value + registers it with `idris-tensor-guardian`. `make
check-ffi-wrap-template` enforces this and fails CI if it would change
anything.

Surfaced 2026-05-30 wiring `tensor_sdpa_2d` for #399 Commit B — the
C function received the wrapped vector and crashed on the first
`q->shape[0]` access. Added to MANIFEST, regenerated the wraps, and
the symbol resolved correctly on the next build.


### Per-backend-set build tree (`build/<BUILD_KEY>/`)

All build artifacts (ttc cache, installed library prefix, dylib, example
executables, stamps) live under `build/$(BUILD_KEY)/` where
`BUILD_KEY := <backend-list>-mlx<MLX_DEVICE>-torch<TORCH_DEVICE>` (e.g.
`tape-mlxcpu-torchcpu`, `torch-mlxcpu-torchmps`,
`tape-torch-mlxcpu-torchmps`). Each distinct
`(BACKEND, MLX_DEVICE, TORCH_DEVICE)` tuple gets its own warm cache.

Implications:
- `make clean` removes ALL backend sets' trees + `build-cov/` + legacy
  `.idris2/`. `make clean-set` removes just `$(BUILD)` (active set
  only). `make clean-all` cascades to `clean-models` + removes
  `vendored/` + `data/`.
- Disk usage scales with sets exercised. Each single-backend tree is
  ~200-300 MB; full triple (`tape,torch,mlx`) is larger. Run `du -sh
  build/*` to inspect; `clean-set BACKEND=<key>` to prune a single set.
- Trees are gitignored (`build/` recursive ignore).
- The generated `.idr` files (`HwConfig.idr`, `HwDevices.idr`,
  `BuildConfig.idr`, `TestConfig.idr`) still live at their fixed
  `packages/<pkg>/src/...` paths. Cross-set switches *do* rewrite them
  (their content depends on the active set); the per-set ttc cache
  absorbs the cascade — those four files re-elaborate (~4 s total),
  but downstream modules with matching interface hashes don't. The
  cmp-then-mv pattern in the recipes avoids unnecessary mtime bumps
  within a single set's reruns.
- `LIBRARY_SRCS` (the `.library-cache-stamp` dependency list)
  *excludes* the generated `.idr` files. Including them would defeat
  the per-set cache: cross-set rewrites would look like "library
  source changed" and wipe the active set's ttc on every switch.

### Test suite

Run `make test` for Idris unit tests, `make test-c` for C tests. Tests live in `test/src/Test/*.idr` with `Harness.idr` providing assertion helpers.

### Interface-based layer system

The `LayerLike` interface + `AnyLayer` existential wrapper eliminates all mutual recursion. Each layer type lives in its own module. `AnyLayer` stores the type constructor as a non-erased parameter (`(l : Nat -> Nat -> Type -> Type)`) for interface dispatch after pattern matching. All interface methods need explicit `{i, o : Nat}` because Idris 2 QTT erases Nat type parameters by default. Instance heads for types with extra parameters (e.g., `NtmState n m h`) use `{n, m, h : Nat} -> LayerLike (NtmState n m h)` to make those Nats accessible. Adding a new layer type = one file implementing `LayerLike`, zero edits elsewhere.

### Hybrid tape architecture (legacy — old Chez Scheme tape, not current C tape backend)

Forward pass uses Scheme `foreign-set!` for scalar tape entries (tags/arg1/arg2/vals into `foreign-alloc` arrays — no FFI crossing) and C `ext_meta_set` for tensor op meta pointers (arena-allocated structs). Backward pass runs entirely in C via `walk_backward_ext`, reading meta from `ext_meta` array. PIDs stored in Scheme vector, looked up after C backward returns indices.

### Chunked arena allocator (legacy — current C tape backend has its own arena)

Meta structs AND tensor op output buffers are arena-allocated via `arena_alloc` (`prim__tensorAllocArena`). The arena uses a linked list of chunks (never `realloc`) to prevent invalidating previously allocated pointers when the arena grows mid-forward-pass. Reset frees old chunks and resets the head chunk. Output buffers are safe to arena-allocate because values are read into Variable records during `buildOutputScalars`/`buildVarsFromBuf` before `arena_reset`. `prim__tensorAlloc` (calloc) is still used for persistent WeightBuf allocations.

### Tape-based backward pass (legacy — see C Tape Backend section for current)

`collectGrads` allocates a mutable gradient array via FFI, seeds it with the initial gradient, then `walk_backward_ext` scans the tape in reverse in C. Scalar ops propagate inline; tensor ops dispatch to C backward kernels. ConstOps with non-zero gradient are collected as (index, grad) pairs. Scheme looks up PIDs and builds `SortedMap`. The tape is reset at the end of `collectGrads` (gen++).

### Scheme-native C memory access (legacy)

Use Chez Scheme's `foreign-ref`/`foreign-set!` for reading/writing C-allocated arrays instead of calling C functions per element. This avoids the Scheme->C boundary crossing overhead. See `prim__gradAdd`/`prim__gradGet` and `prim__setDouble`/`prim__setInt32` in Variable.idr.

### C-backed softmax/logSoftmax

`softmaxVar`/`logSoftmaxVar` in Variable.idr use C kernels and record a single SoftmaxOp/LogSoftmaxOp tape entry per vector instead of ~29 scalar entries. `applyLayerVar` dispatches NormalizationLayer "softmax"/"logSoftmax" to these.

### C-backed NTM memory ops

`batchCosineSimilarityVar`, `readOpVar`, `writeOpVar`, `interpolationWriteVar` in Variable.idr use C kernels (BatchCosSimOp/ReadOpOp/WriteOpOp/InterpolationWriteOp, tags 15-18) to reduce tape entries per NTM timestep. `forwardReadHeadUnboundedVar`/`forwardWriteHeadInterpVar` in Layer.idr wire these into the Variable-specialized NTM forward pass. Generic `forwardReadHeadUnbounded`/`forwardWriteHeadInterp` in Memory.idr remain parameterized on `NormalizationFunction ty` for the Double path.

### C-backed addressing ops

`interpolateVar`, `shiftVar`, `focusVar` in Variable.idr use C kernels (InterpolateOp/ShiftOp/FocusOp, tags 21-23) replacing ~1400 scalar tape entries per head with 3 tensor ops. `shiftVar` takes a pre-softmax'd kernel (apply `softmaxVar` first). Used in both `forwardReadHeadUnboundedVar` and `forwardReadHeadUnboundedVarBuf` in Layer.idr.

### C-backed LSTM cell op

`lstmCellVar` in Variable.idr uses a C kernel (LstmCellOp, tag 24) fusing bias add + gate activations (sigmoid/tanh) + cell/hidden update into a single tape entry. Replaces ~1700 scalar entries per LSTM timestep with 1. The two matmul ops (iW*x, rW*h) remain as separate MatVecOps. `applyLayerVar` in Layer.idr dispatches to `lstmCellVar` for the Variable-specialized LSTM path.

### C-backed BCE with logits

`bceWithLogitsVar` in Variable.idr uses a C kernel (BceWithLogitsOp, tag 26) fusing sigmoid + BCE loss into a single tape entry per output vector. Forward: `(1/n) * sum_i [max(p_i,0) - p_i*y_i + log(1+exp(-|p_i|))]`. Backward: `d_p_i = (1/n) * (sigmoid(p_i) - y_i) * d_loss` (gradients to predictions only, not targets). `epochTwoPhaseDenseBce` in Backprop.idr uses this directly instead of the scalar `binaryCrossEntropyWithLogits`. Meta stored via Scheme-side `ext_meta_set` (NOT C-side `tape_meta`) to match `walk_backward_ext` dispatch.

### Persistent NtmMemBuf

NTM memory matrix kept as persistent `NtmMemBuf` C struct across timesteps. Eliminates 4x per-timestep packMatrix (2560 elements each). Buffer initialized in `nameParams`, synced after `applyDeltas` via `syncLayerBuffers`, epoch-cached tape registration via `prim__ntmMemBufEnsure`. Buffer-aware ops: `batchCosineSimilarityVarBuf`, `readOpVarBuf`, `interpolationWriteVarBuf` in Variable.idr. **Per-sequence reset**: NtmMemBuf stores `initial_vals` (snapshotted at init and after optimizer deltas). `prim__ntmMemBufReset` restores `vals` from `initial_vals` and invalidates cache (forces tape re-registration). `resetNtmMemBufs` in Layer.idr reconstructs the Network with the reset buffer, called before each sequence in `calculateLossTwoPhaseVar`/`VarBce` to prevent cross-sequence mutation.

### Bias WeightBuf

LinearLayer and LstmLayer have bias WeightBuf fields (`bBuf : Maybe AnyPtr`) alongside weight WeightBufs. `nameParams` allocates them, `syncLayerBuffers` syncs after `applyDeltas`. LinearLayer fuses MatVec+Bias in a single C kernel (`matrixVectorMultiplyVarBufBias`). LstmLayer reads bias from WeightBuf in the C LSTM cell kernel (`lstmCellVarBuf`/`lstmCellVarFromBufs`). Eliminates per-timestep bias re-registration (~160K tape entries/epoch).

### Learned LSTM h0/c0

LstmLayer has `h0Buf : Maybe AnyPtr` and `c0Buf : Maybe AnyPtr` fields for learnable initial hidden/cell states. Initialized with Xavier uniform in `lstmLayerWith`. Named as `prefix_h0`/`prefix_c0` in `nameParams`, allocated as WeightBufs. Synced via `applyDeltasAndSyncLayer`/`readFromBuffersLayer`. Matches PyTorch reference's `nn.Parameter(torch.zeros(...))` learnable initial states.

### Buffer-passing MatVec to LstmCell

`matrixVectorMultiplyVarBufOut` returns raw `(AnyPtr, Int)` buffer+tapeStart instead of Variables. `lstmCellVarFromBufs` consumes these directly via `buf_to_meta` C helper, avoiding `buildOutputScalars`+`packVec` roundtrip for 2x4o intermediate elements per LSTM timestep.

### Bulk buildOutputScalars

`prim__appendOutputConstOff` bulk-appends ConstOps from a C buffer with offset in a single Scheme FFI call (internal loop), replacing per-element `tapeAppendConst`. `buildVarsFromBuf` reads values with sequential tape indices. Used by all tensor op output paths.

### Shadow ConstOps (tag=25) (legacy)

Buffer-passing ops (`*BufOut`, `*BufIO`) create shadow ConstOps instead of regular output ConstOps. These provide gradient slots without values/pids — skipped during backward collection (`if (tag == 25) continue`). Tags set via C bulk `tape_set_shadow_tags` instead of per-element Scheme `foreign-set!`. Shadow ConstOps still occupy tape entries; full elimination requires gradient region reservation (not yet implemented).

### C-side pid filtering (legacy)

`walk_backward_ext` filters ConstOps by integer `pid_id` (C-side `tape_pid_ids` array, parallel to tape). Only collects ConstOps with `pid_id >= 0` (named parameters). Dense pid_ids assigned via Scheme `pid-to-id` hash table in `prim__tapeSetParamId`. Set in three paths: `prim__tapeSetParamId` (initial naming), `prim__tapeAppendConst` (stale re-registration), `prim__tapeEnsureBulkConst`/`prim__ntmMemBufEnsure` (WeightBuf/NtmMemBuf). Reset via `tape_pid_ids_reset` after backward.

### `out_tape_start` semantics (legacy)

Tensor op meta structs store `out_tape_start = idx + 1` (first output gradient index, NOT the op entry index). Backward kernels read `meta->out_tape_start` directly without `+1`. Set by `tensor_op_set_out(tag, meta, idx+1)` during `prim__tapeAppendTensorOp`.

### Dense optimizer (legacy)

`DenseOptimizer`/`DenseOptimizerState` in Optimizer.idr use C arrays indexed by integer pid_id instead of `SortedMap String Double`. `collectGradsDense` accumulates gradients into a pre-allocated C array during backward (no per-result FFI calls, no SortedMap inserts). The gradient array is persistent across epochs via `grad_alloc_reuse` (calloc once, memset-zero on reuse). Optimizer step functions (`rmsprop_vc_step`, `sgd_step`, `adam_gc_step`) operate in-place on the array. Dense epoch functions use `applyDeltasAndSyncNetwork` which applies deltas directly to C buffers via `buf_apply_deltas` (bypassing `emap` + `syncLayerBuffers`). NTM examples use this path via `epochTwoPhaseDense`; supervised/LSTM examples still use the original `SortedMap` path. Must call `getNumPids 0` after `autoName` to get the parameter count for `initDenseState`.

### C-bulk delta application (legacy)

`applyDeltasAndSyncLayer`/`applyDeltasAndSyncNetwork` in Layer.idr apply optimizer deltas directly to WeightBuf/NtmMemBuf C arrays via `buf_apply_deltas(vals, pid_ids, count, deltas)`. Each buffer stores a parallel `int *pid_ids` array (populated during `nameParams`). This bypasses the Scheme `emap (applyDeltasDense ...)` + `syncNetworkBuffers` traversals (~63K Variable operations). WeightBuf pid_ids stored in Scheme 6-vector slot [4]; NtmMemBuf pid_ids stored in C struct field. Cache generations are reset to force tape re-registration next epoch. **Important**: Variable.value fields are NOT updated — call `readFromBuffersNetwork` before `toDoubleNetwork` to sync C buffer values back into Variable records for evaluation.

### Multi-axis `primNarrow` on torch + mlx was a silent shape lie (fixed in `bd61bef`)

`tensor_narrow(handle, dim, start, len)` is supposed to slice along axis `dim`. From the dawn of multi-backend support through 2026-05-26, the torch and mlx kernels did `(void)dim;` and then `flatten().narrow(0, start, len)` — they ignored `dim` and flattened the input before slicing axis 0. A `primNarrow ... 1 ...` on a `[seq, hidden]` tensor that was supposed to return `[seq, len]` instead returned a rank-1 `[len]`, with values from the FIRST row only. Tape always handled rank-2 axis-1 correctly. The bug hid behind BERT's 1e-3 cross-language tolerance: HfBert.idr's per-head Q/K/V split runs `primNarrow ... 1 ...`, but the BERT roundtrip gate ran only on tape (the default backend) and the multi-head numerics happened to land within tolerance even when wrong on torch/mlx. Pinned by `linear_shape_narrow::axis1_correctness_rank2` (forward) + `axis1_backward_scatters_columns` (backward gradient scatter) in the common-backend Criterion suite — they pin a `[3, 6]` tensor, narrow axis=1 start=2 length=3, and assert both rank-2 shape and the exact middle columns. Tape had a real implementation in `backend_tape/linear/shape/narrow.c` since the C tape landed; torch + mlx fixes route through `at::Tensor::narrow(dim, start, len)` (torch) and a multi-axis `mx::slice` start/stop bound builder + shape-comparison `dim` recovery at replay time (mlx).

### Per-package param-registry collisions across multi-suite test runs

The C-side param registry accumulates across all tests in a single process. If two test suites each construct a model and assert the registry "has exactly these names", whichever runs second sees the union of both suites' names — by default the second suite's "first N" slice is the FIRST suite's names, not its own. HfBert's FFI test (Bucket 2) runs first, registers 39 BERT names; HfGpt2's FFI test then naively reads `registered = readAllParamNames` and compares against 64 GPT-2 names — the first 39 are BERT names from the prior suite, mismatch on element 0. The fix: snapshot the registry count *before* constructing your model and `drop preCount allNames` so you slice off only what your suite added. Mirrors the `param_clear()` pattern the C-side unit tests use, but works in Idris-land without a clear primitive being exposed.

### HF GPT-2 stores `c_attn` / `c_proj` / `mlp.*` as `[in, out]`, not `[out, in]`

HuggingFace's GPT-2 wraps its linear projections in `transformers.pytorch_utils.Conv1D` — which is `nn.Linear` with the weight transposed. On-disk shape for `transformer.h.{i}.attn.c_attn.weight` is `[hidden, 3*hidden]` (in×out), not `[3*hidden, hidden]` (out×in) which a normal `nn.Linear` would write. The HfGpt2 module stores the weight HF-natively (matching CONVENTIONS rule 2 — storage shapes match on disk) and computes `y = x @ W + bias` directly via `primMm + primAdd` broadcast, bypassing `tlinear2d` (which expects `[out, in]`). If you accidentally route through `tlinear2d` you'll silently apply an extra transpose; the GPT-2 oracle gate at 1e-6 catches it.

### Fused QKV (`c_attn.weight`) split via axis=1 narrow at forward, not stored split

GPT-2's `attn.c_attn.weight` is one `Tensor [hidden, 3*hidden]` storing Q‖K‖V concatenated along axis=1. HfGpt2's `applySelfAttn` materialises the post-projection `[seq, 3*hidden]` blob, then takes three `primNarrow ... 1 ...` views (`q = narrow blob 1 0 hidden`, `k = narrow blob 1 hidden hidden`, `v = narrow blob 1 (2*hidden) hidden`) — each a `[seq, hidden]` view. The per-head split is a second nested narrow inside the multi-head loop. Both axes=1 narrows exercise the `bd61bef` fix; without it torch/mlx would produce wrong-rank tensors and silent garbage attention. Storage matches HF; the splitting work happens at forward.

## C Tape Backend (backend_tape.c)

These gotchas apply to the C tape backend (`BACKEND=tape`), which implements `backend.h` with a flat Wengert list in C.

### `tensor_select` rank-0 identity

`binop_elementwise` produces scalars (rank 0, numel=1) when both inputs have numel==1. If `tensorToScalars` then calls `tensor_select` on the rank-0 result, it must return the tensor itself (identity) to preserve the tape entry. The fallback path (`make_scalar(t->data[index], t->requires_grad)`) creates a copy with NO tape entry, breaking the gradient chain. Affected: any layer with output size 1 (e.g., LSTM example's Linear<1:1>).

### Arena vs calloc for view tensors

`tensor_select` (rank-1 and rank-2) creates view Tensor structs. Use `arena_alloc` (freed on tape reset) not `calloc` (never freed). Each `tensorToScalars(n)` call creates n view tensors; over thousands of epochs this leaks GBs. Exception: `tensor_view_1d`/`tensor_view_2d` are called once in `nameLayer` and must persist — keep as `calloc`.

### Optimizer per-element buffers

RMSprop/Adam velocity and momentum buffers must be sized by total parameter ELEMENTS, not total parameter count. A [400,29] weight matrix needs 11,600 velocity slots, not 1. Index via `param_element_offset(i) + j`. SGD is unaffected (no buffers).

### Fused ops require backward rules — and prefer not to add them at all

Any fused C operation that sets `requires_grad=1` on its output MUST also append tape entries and implement backward cases. Without a backward rule, the gradient chain breaks silently — the op's result gets gradient but it's never propagated to inputs.

The corollary: **don't add architecture-specific fused C ops in the first place.** A `tensor_*` op should be something a PyTorch user would expect at the FFI surface (`F.cosine_similarity`, `nn.LSTMCell`, etc.). Per-paper fusions like NTM's read-head pipeline belong in Idris, composed from primitives. The previous `tensor_ntm_read_head` / `tensor_ntm_interp_write` fusion was rolled back; NTM now composes its addressing in `Layer/Ntm.idr` like DNC always did.

### NTM state is not a parameter

NTM memory, readAddr, writeAddr, readOutput are per-sequence state, NOT learned parameters. Do NOT register them with `prim__paramRegister` — the optimizer will corrupt them with gradient updates. Use `tensor_create_state_2d`/`tensor_create_state_1d` (persistent, `requires_grad=0`, no param registration). The decomposed addressing primitives still propagate gradients to the key, beta, g, gamma, shift inputs (which DO come from FC layers with `requires_grad=1`).

### `tensor_matmul` vector-matrix backward

`tensor_matmul` for [n]×[n,m] → [m] needs `OP_VECMAT` (not `OP_DOT`). The DOT backward only reads `grad[0]`, which is wrong for vector results. The VECMAT backward: `d_a[i] = Σ_j grad[j]*b[i,j]`, `d_b[i,j] = grad[j]*a[i]`.

### Arena never frees chunks

`arena_reset()` resets `.used` pointers but never frees chunks. Memory grows to accommodate the peak forward+backward pass, then stabilizes. For NTM with n=128, the peak is ~8MB (6 chunks). This is by design (avoids realloc invalidation) but means RSS never decreases within a run.

### `tape_reset` must call `arena_reset`

The arena holds all intermediate tensors from the forward+backward pass. `tape_reset()` must call `arena_reset()` to reclaim this memory. Without it, the arena grows by ~1.7MB/epoch indefinitely (the original bug that caused 8.5GB memory usage).

Additionally, `tape_reset` must free: (1) OP_STACK `inputs` arrays (heap-allocated `Tensor**`), and (2) grad arrays on non-persistent tensors (heap-allocated by `ensure_grad` during backward, leaked when arena tensors are reused).

### `fromDouble` persistent scalar leak

`tensor_create_scalar(value, 0)` must heap-allocate (persistent) because Idris may cache `fromDouble` results in Variables across epochs (e.g., `let data = map fromDouble ...` evaluated once, reused). Arena allocation would cause use-after-free when `arena_reset` runs between epochs. The tradeoff: ~56 bytes leaked per `fromDouble` call. For NTM training with fresh data each epoch, this is ~15KB/epoch. Over 50k epochs: ~750MB. A proper fix requires either Idris-level finalizers or an explicit ephemeral tensor pool.

### `toDoubleLayer` must use tensor handles for learned weights

After training with `NativeOptimizer`, the optimizer mutates param tensor data in-place. The scalar Variable `.value` fields are stale (from initial forward pass). `toDoubleLayer` must read from tensor handles (via `buildDoubleMatrix`/`buildDoubleVector` using `prim__item2d`/`prim__item1d`) for learned weights. Exception: non-learnable state (NTM memory, addressing) can use `map value` since those retain initial values.

## MLX Backend (backend_mlx.cpp)

### `tensor_item` BF16/F16 readback — `item<float>()` mis-sizes 2-byte storage

mlx scalar storage is `bfloat16_t` / `float16_t` (2 bytes), not `float`
(4 bytes). The pre-2026-05-31 `tensor_item_mlx_streamed` branched
`f64 → item<double>()`, *default → `item<float>()`* — which on BF16
storage read 16 useful bits + 16 bits of adjacent buffer slot as if
it were a 32-bit float, returning denormal-range garbage like
`2.3e-41` for an actual `1.1` BF16 scalar. The Supervised example on
`MLX_DTYPE=BF16` exhibited this as "loss=2.3e-41 from epoch 1"
(initial random weights, no training yet) — looked like total BF16
training failure but the math was correct; just `tensor_item` lying.
Fix in `core/lifecycle/item.cpp`: explicit branches for
`mx::bfloat16` and `mx::float16` reading via the matching
`item<bfloat16_t>()` / `item<float16_t>()`. Lesson: when adding a
storage dtype, audit every per-element reader for the right-sized
`item<T>()` / `data<T>()` template instantiation; the default arm is
not safe for 2-byte types.

### Tensor lifetime: tape vs non-tape

All `Tensor*` objects self-register in `all_tensors` via the constructor. `tape_reset()` frees non-persistent ones. Unlike the tape backend's arena (which bulk-frees by resetting a pointer), MLX individually `delete`s each tensor. This means:

- **Ephemeral data tensors** (from `tensor_create` with `requires_grad=0`): NOT on the tape, but still tracked in `all_tensors`. Freed at `tape_reset()`.
- **Persistent tensors** (params via `param_register`, state via `tensor_create_state_*`, views via `tensor_view_*`): Marked `persistent=1`, survive `tape_reset()`.
- **TensorPair structs**: Tracked in `all_pairs`, freed at `tape_reset()`.

Without this tracking, non-tape tensors (every `bulkToTensor` call, BCE constants, zero tensors) would leak ~250 objects per NTM epoch × 50K epochs = 2-4GB.

### State tensors must be persistent

`tensor_create_state_1d`/`_2d` must set `persistent=1`. Without it, NTM memory matrix, addressing weights, and read output tensors are freed at the first `tape_reset()`, causing use-after-free. The tape backend uses separate `calloc` with `persistent=1`; torch uses `from_tensor_persistent()`.

### Broadcasting gradient reduction

Binary op backwards (ADD, SUB, MUL, DIV, POW) must call `reduce_grad()` to sum gradients over broadcast-expanded dimensions. Without this, scalar × vector operations (e.g., `g * content_weights` in NTM interpolation) produce vector-shaped gradients for scalar parameters, corrupting the autograd chain.

### RMSprop optimizer must be implemented

`optimizer_step` must have a `case 1:` for RMSprop. Without it, `optimizer_step` falls through to `default: break;` (no-op) and weights are never updated. This silently affects any example using `nativeRmsprop` (NTM Copy, NTM Recall). SGD (case 0) and Adam (case 2) were implemented first; RMSprop was missing until the bug was caught.

### Conv1d circular d_kernel backward shift sign

`OP_CONV1D_CIRC` d_kernel backward must use `shift = j - half_k` (matching forward indexing), not `shift = half_k - j`. The forward computes `result[i] = sum_j(input[(i - half_k + j) % n] * kernel[j])`, so the backward needs `d_kernel[j] = sum_i(grad[i] * input[(i - half_k + j) % n])`. The inverted shift corrupts shift kernel gradients, preventing the NTM from learning memory addressing order.

### Fused OP_NORMALIZE for attention normalization

The attention weight normalization `focused = powered / sum(powered)` must use a fused `OP_NORMALIZE` op, not separate `div + sum + add` ops. The decomposed backward computes `d/d(numerator) = grad/denom` and `d/d(denominator) = -grad*numer/denom²` separately — these are huge values that nearly cancel. With peaked attention (near-converged NTM), catastrophic cancellation produces NaN. The fused formula `d_a[i] = (d_r[i] - dot(d_r, r)) / (sum(a) + eps)` avoids this.

### `mx::transpose` requires explicit axes

`mx::transpose(x)` with no axis argument reverses ALL dimensions — it does NOT swap the last two like PyTorch/NumPy. For a 2D matrix, this gives the wrong result. Use `mx::transpose(x, {1, 0})` for 2D, `mx::transpose(x, {0, 2, 1})` for batched 3D. This bug was the root cause of wrong MM backward gradients, broken NTM read head addressing, and incorrect transpose test values.

### `mx::array(double)` defaults to float32

`mx::array(3.0)` creates a float32 scalar, not float64. Reading it back with `item<double>()` returns 0.0 (reinterpreting float32 bits as float64). Always use `mx::array(value, mx::float64)` for double-precision scalars. This caused `tensor_create_scalar` to produce zero-valued tensors.

### Metal float32 transcendentals

MLX's Metal GPU computes `exp`, `sigmoid`, `tanh`, etc. in float32 even when the input array is float64. Expect ~1e-6 precision for these ops, not 1e-10. Test tolerances for transcendental functions should be 1e-5 or wider on MLX.

### Non-smooth `softplus` stable form gives wrong subgradient at x=0

`tensor_softplus` uses the numerically stable form `max(0,x) + log(1 + exp(-|x|))` to avoid `log(1+exp(x))` overflowing in float32 for `x > ~88`. The naive form is C^∞ smooth and gives correct `sigmoid(x)` backward via mlx's `vjp` everywhere; the stable form is non-smooth at exactly `x=0` (both `max` and `abs` have subgradient ambiguity). At that boundary point mlx's `vjp` picks the 0 subgradient for each → `d_softplus(0)` returns 0 instead of the expected `sigmoid(0) = 0.5`. All non-boundary inputs (|x| > 0 by even a float ulp) get the correct derivative. Test workaround: skip the `x=0` probe on mlx — see `test_backend.c` `d_softplus(0)`. Permanent fix would be a piecewise smooth forward or a registered custom backward.

### Non-contiguous views and `data<T>()`

`mx::transpose` and similar ops return views with swapped strides. The raw `data<double>()` pointer still points to the original contiguous memory layout. Index arithmetic like `data[row * cols + col]` produces wrong results on transposed views. Use `mx::flatten` to force a contiguous copy first, or use MLX's indexing API.

### Lazy eval use-after-free in `tape_reset`

`tape_reset()` must call `mx::eval()` on ALL tensors before deleting any non-persistent ones. MLX array operations are lazy — `mx::add(a, b)` captures references to `a` and `b`, not copies. If `a` is deleted by `tape_reset` while a surviving tensor's lazy graph still references `a->data`, the next `mx::eval` hits a dangling pointer. The fix: batch-eval all tensor data and grads before the delete loop.

### NTM convergence comparison

MLX NTM (post-`tensor_linear` bias-on-tape fix and 2026-05-08 NTM model alignment) converges on `ntm-copy` to acc_short=0.994, acc_full=0.999 at epoch 8200 with the standard ES gate (seed=42, batch=1). Comparable to PyTorch ref's 100%/100% at ~4600 epochs. The aligned model is highly seed-sensitive at moderate budgets — see "NTM-Copy convergence is highly seed-sensitive at 5K epochs" in the NTM-Specific section.

### Replay-based VJP: every dependency must be on the tape

The MLX backend computes gradients by replaying the forward tape inside a closure passed to `mx::vjp`. The replay reconstructs each tensor's value from its tape op's `arg1`/`arg2`/meta — it does NOT use the result tensor's `data` field. Any forward op that mutates `data` after a sub-step (e.g. `result = mx::add(result, bias->data)` after the matmul) but doesn't record the dependency on the tape will produce a replay value that differs from the actual forward, and the missing input gets zero gradient (the VJP can't see a dependency that isn't in the closure).

**The bug this caught**: `tensor_linear(W, x, bias)` was recording only `OP_MV(W, x)` but adding the bias to `result.data` directly. When `tlinear` chained (one `tlinear`'s output passed as the next `tlinear`'s bias arg, e.g. `tlinear rwT h (tlinear iwT input bT)` in the LSTM combined-gates expression), the replay computed `pool[outer] = rw @ h` only — the entire inner branch (`iw`, `input`, `b`) had no path to the loss in the VJP. Gradients for every parameter on the inner branch (LSTM `iw`/`rw`/`b`, every NTM FC weight/bias in chained-FC settings) collapsed to exactly zero, and mlx training stalled at the random-baseline loss for the aligned NTM-Copy model.

**Fix**: decompose `tensor_linear` into `tensor_mv` + `tensor_add` when a bias is provided, so each dependency lands on the tape. Two tape entries instead of one is the small per-call cost; correctness requires every input read by the forward to be reachable from the tape.

**Diagnosing this class of bug**: the `DEBUG_PARAM_GRADS` env-var hook in `optimizer_step` (mirrors the one in `backend_tape.c`) dumps per-param grad L2 norm at the first optimizer step. Any `requires_grad=1` param with `grad_l2=0` is the smoking gun — that param is in the registry but has no path to the loss in the replay graph.

### Softplus must use the numerically stable form in float32

`softplus(x) = log(1 + exp(x))` overflows in float32 for x > ~88 (where `exp(x) > 3.4e38`). Use the stable form `softplus(x) = max(0, x) + log(1 + exp(-|x|))` instead — it gives the same answer everywhere, reduces to `x` for large positive x and to `exp(x)` for large negative x, and never overflows.

The bug this caught: NTM content addressing computes `betaT = softplus(scalar)` for the sharpening factor, then `softmax(betaT * cos_sim)`. With the naive softplus, once the controller drove `scalar` past ~88 (mid-training, model already at 94% accuracy), `betaT` jumped to `+inf`, the multiply produced `±inf` softmax inputs, and the whole content-addressing path NaN'd the loss in a single epoch. Tape uses a branch-on-magnitude form; torch uses `torch::softplus` (stable). Only mlx had the naive form.

**Diagnosing it**: `DEBUG_NAN_TRAP=1` in `tensor_backward` walks the forward tape on first appearance of NaN/Inf in any param grad, prints the first NaN-producing op and its args' value ranges. Found this one in one shot: `first NaN at tape[2165] op=SOFTMAX_2D` with arg1 (a MUL output) range `[-inf, +inf]`.

### nixpkgs `python3Packages.mlx` is CPU-only — use pip mlx for Metal

The nixpkgs derivation hardcodes `MLX_BUILD_METAL=false`. From `pkgs/development/python-modules/mlx/default.nix`:

> NOTE The `metal` command-line utility used to build the Metal kernels is not open-source. To build mlx with Metal support in Nix, you'd need to use one of the sandbox escape hatches which let you interact with a native install of Xcode, such as `composeXcodeWrapper`.

Symptoms when you forget: setting `MLX_DEVICE=gpu` on a nix-mlx build hits `libc++abi: terminating due to uncaught exception of type std::invalid_argument: [set_default_device] Cannot set gpu device without gpu backend` at startup. `otool -L libmlx.dylib` on the nix package shows no Metal framework linkage; `mx::is_available(gpu)` returns 0 even when the host (Apple Silicon) and the guest (Tart VM on Apple Virtualization Framework) both can see Metal.

The pip `mlx` package automatically pulls in `mlx-metal` (a 150 MB `mlx.metallib` of precompiled shaders) and links `Metal.framework` / `QuartzCore` / `Foundation` into `libmlx.dylib`. Setup for this project:

```sh
uv venv .venv-mlx
source .venv-mlx/bin/activate
uv pip install mlx
# Then rebuild backend_mlx.cpp against this site-packages dir:
make BACKEND=mlx MLX_SITE=$VIRTUAL_ENV/lib/python3.13/site-packages/mlx backend
```

The Makefile's auto-detect logic uses `python3 -c "import mlx"` to find a site-packages dir with an `include/`, so activating the venv before `make` makes detection just work.

### `MLX_DEVICE=gpu` is usually a perf loss at idris-ml example scales

Counter-intuitive but well-evidenced (2026-05-11 Job 3 sweep): switching mlx to Metal GPU made **every** example we tested *slower* than mlx CPU stream — by 3-12×:

| Cell | mlx CPU ms/ep | mlx GPU ms/ep | slowdown |
|---|---:|---:|---:|
| supervised | 1 | 11 | 11× |
| rnn | 11 | 114 | 10× |
| lstm | 13 | 156 | 12× |
| gru | 15 | 145 | 10× |
| transformer | 33 | 111 | 3× |
| mnist (ms/ep) | 22 800 | 112 800 | 5× |
| dnc-copy | 30 | 269 | 9× |

Why: per-prim Metal kernel-launch overhead dominates at these tensor sizes. The training loops here are dozens of small ops (matmuls on <100-element tensors, scalar mults, softplus, sigmoid). Each `tensor_*` call dispatches a Metal kernel, and at this granularity the launch cost dwarfs the actual compute. The replay-VJP closure compounds it — every backward call rebuilds and re-dispatches the same kernel chain. Under Tart's paravirt-graphics layer the per-dispatch latency is probably worse than bare metal too. **Convergence remains numerically clean** on GPU; it's a pure throughput issue.

The default in `backend_mlx.cpp::mlx_backend_init` is CPU for this reason — leave it. `MLX_DEVICE=gpu` is reserved for workloads where per-op compute exceeds kernel-launch latency: large batched conv/attention on bigger images, transformer training with larger H/sequence-length, or anything where a single kernel does ≥1ms of work. None of the current examples qualify.

The lever that would unlock GPU here is `mx::compile()` — mlx's JIT API that compiles a multi-op function once and replays it as a single fused kernel. We don't use it (we go through `mx::vjp` which builds a closure but doesn't compile it). Wiring `mx::compile` into the replay path is the open question for Phase B of Job 3.

### `MLX_COMPILE=1` env var — opt-in `mx::compile` path

Set `MLX_COMPILE=1` to route `tensor_backward` through `mx::compile`
(then `mx::vjp` on the compiled function). Default is off.

On CPU: typically 8-30% faster on fixed-architecture training loops
(supervised, lstm, rnn, gru, transformer, mnist). On GPU: roughly
breaks even at our current example scales (the kernel-launch wall
dominates) — wins only materialize on bigger workloads.

Correctness: bit-identical loss on most examples; ULP-level drift
on mnist because compile reorders Conv2D backward fp accumulation
(within float32 noise — convergence accuracy unchanged).

Caveats:
- Forward closure now takes `[params..., constants...]` as explicit
  inputs (rather than capturing constants). Per-batch values like X
  and y are passed through, not baked into the compiled graph.
- mlx's own `MLX_DISABLE_COMPILE` env var globally no-ops compile
  even when `MLX_COMPILE=1`. Use for A/B if needed.
- For workloads with variable shapes per call (NTM-copy `maxLen`),
  mlx recompiles on shape change — expect higher overhead.

Full empirical table: `docs/develop/mlx-survey.md` "Empirical findings".

### Paravirt-GPU hang: every kernel fails until VM reboot

When developing on a virtualised macOS (Tart, UTM, anything backed by
`AppleParavirtGPUMetalIOGPUFamily`), a single mlx-gpu kernel that
overruns the host watchdog wedges the paravirt GPU driver. Once
wedged, every subsequent `MTLComputePipelineState` creation fails —
including for processes that didn't trigger the hang. mlx reports
`[metal::Device] Unable to load kernel <name> / Compilation failed`;
PyTorch MPS reports `Failed to created pipeline state object, Error
Domain=CompilerError Code=2 "Compilation failed"`.

The error message points at "compilation," but the metallib is fine,
the device handle stays valid, and `MTLCompilerService` is healthy
and serving other processes. **The failure is below user space.**

Diagnosis one-liner — confirm it's the paravirt hang rather than a
real code regression before debugging further:

```bash
cd packages/pytorch && uv run python -c \
  "import mlx.core as mx; mx.negative(mx.array([1.0]), stream=mx.gpu)" 2>&1
/usr/bin/log show --process python3.12 --last 5m \
  | grep -iE 'GPU Hang|Paravirt|kIOGPUCommandBuffer'
```

If the log shows
`Caused GPU Hang Error (00000003:kIOGPUCommandBufferCallbackErrorHang)`
or `AppleParavirtComputePipelineState: Failed to get compute pipeline
state info`, the GPU is wedged.

Recovery: **reboot the VM.** `mx.clear_cache()`, restarting Python,
kicking WindowServer don't reach the driver state — they're all
user-space. From inside the VM: `sudo reboot`. From the Tart host:
`tart stop <vm> && tart run <vm>`.

After reboot, the mlx-gpu baseline returns (3-12× slower than CPU
stream at this codebase's example scales, per
`project_mlx_gpu_environment.md`).

Trigger pattern observed 2026-05-18: rnn / lstm / gru / transformer
mlx-gpu sweeps completed cleanly; the hang fired during an
`example-ntm-copy` mlx-gpu run. Likely cause is a long-running kernel
in NTM's cosine-similarity / softmax inner loop overrunning the
paravirt watchdog. Bare-metal Apple Silicon (no paravirt layer)
shouldn't hit this, but in a VM the watchdog is shorter than the
typical mlx kernel needs for these workloads.

The footprint downstream: silent perf-script anomalies. Once mlx-gpu
binaries abort, `scripts/perf-sweep.sh` (post-commit `c89ff85`)
correctly marks the cell as `crashed`; earlier versions silently
billed the time-to-abort as the per-epoch measurement. If a perf-log
row's `ratio` looks impossibly good for an mlx-gpu cell, check
`notes` and the surrounding rows for the abort signature.

## Torch Backend (backend_torch.cpp)

### View tensors must be persistent

`tensor_view_2d`/`tensor_view_1d` must use `from_tensor_persistent()`, not `from_tensor()`. Views are created once at `nameLayer` time and referenced by scalar Variables for the lifetime of the model. If tracked as intermediates, `free_intermediates()` frees them after the first epoch, causing crash in `refreshValue` → `prim__item` on stale pointers.

### libtorch MPS rejects F64 at *tensor construction*, not just at op dispatch

The 2026-05-19 device-taxonomy plan originally assumed PyTorch's MPS
backend silently falls back to CPU for ops without MPS kernels — so
admitting `Compatible (TorchDev TMps) F64` would let users opt into
"slow but correct" F64 on MPS.

Reality: `tensor.to("mps")` on any F64-dtype tensor errors out:

```
TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS
framework doesn't support float64. Please use float32 instead.
```

The check is inside libtorch's `at::native::to(...)` for MPS targets
— it's not an op-level fallback gate, it's a hard rejection at the
device-bind layer. PyTorch on Apple Silicon is F32-on-MPS, period.

**What this means for the type system**: `Compatible (TorchDev TMps)
F64` deliberately does NOT exist (mirrors the `(MlxDev MGpu) F64`
rejection). Admitting it would let the type system mint a value the
runtime can't represent.

**What this means for PyTorch refs**: when `--device mps` is selected,
the harness auto-downcasts to F32 via the `get_dtype()` switch in
`torch_ref/training/runner.py`. F64 lanes (CPU / CUDA) keep their
historical precision.

### A parameter must be cast/moved *before* `requires_grad_`, or it's a non-leaf and never trains

`.to(dtype)` (and `.to(device)`) applied to a tensor that already has
`requires_grad=true` is a legitimate *differentiable* op — the result
carries a `grad_fn` (`ToCopyBackward`) and is therefore a **non-leaf**.
A non-leaf's `.grad` is never populated during `backward()`, so the
native optimizer (which reads `param.grad()` out of the registry) sees
a zero gradient and silently no-ops. Training freezes at the init loss
— no error, no warning beyond a one-line "accessing .grad of a non-leaf"
notice that floods stderr.

This bit the F32 param creators: they built an F64 leaf
(`requires_grad_(true)`) and *then* cast to F32 (`torch_cast_to`),
yielding a non-leaf F32 param. The F64 path set `requires_grad` on the
un-cast leaf, so F64 (tape / torch-cpu / CUDA) was unaffected; only the
F32 lanes (torch-mps) froze. Diagnosed 2026-05-20 via the
`example-supervised` torch-mps plateau, and it was *also* the true
cause of the "NTM Abort trap 6" crash the TODO had filed as an "MPS
kernel coverage gap" — with non-leaf params the MPS backward/optimizer
path aborts rather than no-ops.

**Rule (mirrors PyTorch's own `nn.Parameter`)**: a parameter is a leaf,
so finalize its dtype and device on the plain data *first*, then flip on
`requires_grad` last — `Parameter(data.to(dtype).to(device),
requires_grad=True)`. The C helper `make_param_leaf` enforces
cast-before-grad. Don't prohibit casting grad tensors in general (mixed
precision relies on it); the invariant is specific to leaf construction.

### "F64 leaks into the F32 build": prefer the dtype-aware (`dtCreate*`/`dtCastFrom`) creators

On torch-mps the example dtype is F32 (baked into `BuildConfig`), but
several C creators and ops produce or assume **F64** — fine on
torch-cpu/tape (everything is F64) but a hard crash on MPS, which
rejects F64 at construction. Three layers bit us on 2026-05-20:

- **Idris creators**: `primCreateState2d` / `primCreateState1d` etc. are
  F64. Use the dtype-aware `dtCreateState2d {t=ExampleDType} … (deviceStreamTag {d})`
  / `dtCastFrom {t=ExampleDType} …` instead, which route through
  `RuntimeDType` to pick F32/F64. (matmul-bench, mnist failures.)
- **C ops with hardcoded accumulators**: `tensor_scatter_add` allocated
  its output as `torch::kFloat64`; fixed to inherit `src.options()` so
  the dtype follows the input. (dnc-copy / dnc-recall.) Audit any C op
  that calls `torch::zeros(..., kFloat64)` for the same trap.

Underlying asymmetry worth knowing: the **streamed** torch creators
(`primCreate*Streamed`, used by `dtCreate*`/`tparam`) stay F32-on-CPU —
they do *not* call `prim__toDeviceTorch`. Only the **non-streamed**
creators move to MPS. So the dtype-aware path is safe; crashes happened
only where code escaped to the non-streamed F64 path or mixed an F64
C-op result into the F32 graph. Verified no-op on F64 builds:
tape/torch-cpu results were bit-identical before and after each fix.

### `.contiguous()` on a libtorch op result is never free

`at::Tensor::contiguous()` returns the same tensor when already row-major,
and **a fresh materialised copy** otherwise. A "for safety" `.contiguous()`
on a `narrow` / `expand` / `unbind` / `transpose` result silently forces
a Metal command-buffer submission to copy ~`numel * dtype_size` bytes —
that's tens of MB per call at Llama-3.2-1B scale, ~16K such calls per
8-token forward (attention QKV split × 12K + RoPE table slice × 4K).

Only legitimate use is when downstream code needs row-major byte layout:
`data_ptr<T>()` reads, `safetensors.c` host buffer copies, FFI consumers
that walk doubles by stride-1. Forward-path ops accept strided views just
fine (libtorch's view-on-view chains compose without copies).

Audit pattern: any `.contiguous()` *not* immediately followed by `.cpu()`
+ `.data_ptr<T>()` is suspicious. Three sites bit us — `narrow.cpp:11`,
`expand_mask.cpp:8`, `core/lifecycle/batch.cpp` (`tensor_unbatch`) — all
removed in commit `51df3cb`, dropping torch-mps Llama wall 11:45 → 8:43
(~26%). The stale comments at those sites said "make safe for FFI
consumers"; verified no FFI consumer downstream read the masks/views.

### State tensors need atomic cast+migrate on non-F32/F64 dtags (torch-mps)

Sibling to the "parameter must be cast/moved before `requires_grad_`"
gotcha above. Same class of bug, different code path. `make_param_leaf`
in `dtype_dispatch.cpp` does the atomic cast-then-move + `requires_grad_`
for parameters. State tensors (non-grad) went through a *different* code
path — the `default:` arm of `torch_create_state_{1d,2d}_dtag` called
`tensor_cast_dtype_f64` (cast-without-migrate), then inherited CPU
placement on MPS via the F64-fallback policy in `torch_effective_device`.

This was silent on the F32/F64-fast-path arms; bit us only when the new
`TORCH_DTYPE=BF16` opt-in started exercising the `default:` arm for state
tensors on MPS. The Llama forward then crashed with
`Expected all tensors to be on the same device, but found at least two
devices, mps:0 and cpu!` on the first state-vs-param op.

Fix (commit `ab5386a`): `make_state_persistent` helper mirroring
`make_param_leaf` (atomic cast + device migrate, sans `requires_grad_`)
+ `cast_and_migrate` for create-1d/2d siblings. Every `default:` arm now
goes through the helper. Audit pattern: any `tensor_cast_dtype_*` or
post-cast `.to(dtype)` in a creator that's not paired with `.to(device)`
in the same step is suspect on MPS for non-F32/F64 dtypes.

### C++ block-comment unclosed by `*/` in prose

The byte sequence `*/` anywhere inside a `/* ... */` block closes the
comment immediately, regardless of context — including inside a list
of dtype names like `(BF16/F16/Int*/Bool)` where the author meant the
`*` as a glob/wildcard marker and `/Bool` to continue the list. The
lexer then walks into what follows as code, producing confusing errors
("`Bool` unknown type name", `unexpected character <U+2014>`) at lines
far from the actual mistake.

Cost: ~20 min lost during the BF16 torch-mps fix. Rule: when listing
dtype names in a C/C++ comment, use commas (`BF16, F16, Int*, Bool`) or
backticks-equivalents, never slashes paired with asterisks. Same trap
applies to any prose containing `*/` (regex examples, glob patterns).

### `torch.multinomial` has no MPS kernel

`torch.multinomial` calls into a CPU-only kernel for MPS tensors.
With `PYTORCH_ENABLE_MPS_FALLBACK=1` (now PyTorch's default) the
runtime silently round-trips through CPU per call — measurable in
tight RL loops.

`torch_ref/training/runner.py` ships a `multinomial_safe(probs, n)`
wrapper that's transparent on CPU/CUDA but explicit-CPU-roundtrip on
MPS. Affected models: REINFORCE, A2C, PPO, SAC, GPT (anywhere
categorical sampling fires). New RL models that sample categorically
must use the wrapper, otherwise MPS perf measurements include a
hidden silent-fallback cost.

### `nn.Module` attribute that isn't `register_buffer`'d gets left behind on `.to(device)`

The transformer model
(`torch_ref/models/multi_head_transformer.py`) had a per-block
`TransformerBlock` with `self.causal_mask = causal_mask` set in
`__init__`. When the outer `MultiHeadTransformer.to("mps")` runs,
PyTorch's `.to()` walks the module tree and moves parameters +
registered buffers; plain `self.<name>` attribute references stay on
CPU. The forward then crashes:

```
RuntimeError: expected self and mask to be on the same device, but
got mask on cpu and self on mps:0
```

Fix: use `self.register_buffer("causal_mask", causal_mask)` so the
mask travels with the module. Same trap applies to anything
non-parameter that needs to follow the module — pre-computed
indices, attention scaling constants stored as tensors, etc.

PR review tell: any `self.<name> = <Tensor>` in `nn.Module.__init__`
that isn't a parameter or wrapped in `register_buffer` is a
`.to(device)` bug waiting to fire.

## Idris 2 / Chez Scheme additions (cross-backend device work)

### `let _ = ffiVoidCall` gets elided by the Chez codegen

When an FFI function returns `()` (Idris unit), assigning its result
to `_` in a `let` is dead-code-eliminated by the Chez codegen. The
side effect (the C function being called) never fires. Caught
during the cross-backend `toDevice` work:

```idris
-- Looks like it frees the buffer. Doesn't.
let _ = primFreeHost {d=d1} dataBuf  -- elided
```

Fix: thread the call through `primIO` so the IO monad's sequencing
forces evaluation:

```idris
primIO (\w => MkIORes (primFreeHost {d=d1} dataBuf) w)
```

Or use `ioRerun` (`Tensor.idr`) inside a `do`-block once it's
visible. The forward-pass case is the inverse: `forwardVar` returns
`IO`-typed values so the FFI body fires only on `<-` sequencing — the
existing pattern. The cleanup-buffer case is the dual: a unit-typed
FFI call needs explicit IO sequencing to fire at all.

### Returning the buffer pointer through FFI Scheme wrappers (`tensor_to_doubles_return`)

A void C function like `tensor_to_doubles(handle, buf)` writes into
`buf` and returns nothing. Calling it from Idris and then using
`buf` downstream requires Idris to know it must call the FFI before
the use site. Without that data dependency, Chez's lazy evaluation
can reorder or skip the call.

Two ways to make the dependency visible:

1. Add a C-side wrapper that returns the buffer:
   `tensor_to_doubles_return(h, buf) -> double*` — calls
   `tensor_to_doubles` then returns `buf`. Now Idris sees a value
   it must depend on.
2. Craft the Scheme FFI wrapper to call the void procedure then
   return `a1` (the buf arg) explicitly:
   ```scheme
   (lambda (a0 a1)
     ((foreign-procedure "tensor_to_doubles" (void* void*) void)
       (vector-ref a0 1) a1)
     a1)
   ```

We use (2) for `primToHost` instances in `Device/{Tape,Torch,Mlx}.idr`
to avoid adding more C boilerplate. Pattern: when wrapping a
side-effecting void C function whose Idris-side need is to thread an
output buffer downstream, return the buffer at the end of the
Scheme lambda body.

### `Tensor`'s `dims : Vect rank Nat` parameter — bind it at non-zero quantity to observe it at runtime

The `Tensor` record declares `dims : Vect rank Nat` at unrestricted
quantity:

```idris
record Tensor (dims : Vect rank Nat) (0 d : Device) (0 dt : DType) (0 g : GradMode)
```

But when a function signature mentions `Tensor dims d dt g` without
explicitly binding `dims` and `rank`, Idris auto-binds them at
quantity 0 (the function-binder default). The body then can't
observe `dims` at runtime — e.g. `product dims` fails with "dims is
not accessible in this context."

Fix: explicitly bind both at non-zero quantity:

```idris
toDevice : {0 d1, d2 : Type} -> ... =>
           {rank : Nat} -> {dims : Vect rank Nat} ->
           Tensor dims d1 dt WithGrad -> IO (Tensor dims d2 dt WithGrad)
```

Caught while building the cross-backend `toDevice` host-roundtrip
path — needed `dims` at runtime to compute `product dims` (the
allocation size) and to marshal the shape array for
`primCreateFromHost`.

The existing `toDevice` signature (pre-Phase-6) bound `dims`
implicitly because its body only forwarded a single FFI call (no
shape inspection). Once the body grew shape-dependent code, the
quantity mismatch surfaced.

### `ioRerun` is forward-declared in `Tensor.idr`

`ioRerun : (() -> a) -> IO a` is defined ~line 1153 in
`Tensor.idr`. Earlier-defined functions in the same module
(`toDevice` at ~line 1100) can't reference it — Idris reports
"Undefined name ioRerun."

Workaround: inline its body using `primIO`:

```idris
primIO (\w => MkIORes (someFFICall) w)
```

Real fix would be to move `ioRerun` earlier in the module, but
that touches a lot of forward refs in the existing layout. The
inline pattern is two lines and unambiguous.

### Multi-backend test build is the unblock for cross-backend Test.Transfer

`Test.Transfer.idr` exercises `toDevice` over the
`UserDeviceTransfer` interface. The intra-backend smoke
(`TapeDev → TapeDev`) works under any single-BACKEND build.
Cross-backend smokes (`TapeDev → TorchDev TCpu`,
`TapeDev → MlxDev MCpu`, etc.) need both backends' C symbols
linked at runtime, which means a `BACKEND=tape,torch` (multi-
backend) test build — not the current `make test` default.

Symptom of forgetting: `Exception in foreign-procedure: no entry
for "tensor_to_device_tape"` when running on a torch-only build
that references `TapeDev`-typed Idris code.

The Transfer test is currently NOT wired into `Main.idr`'s
default `tests` list for this reason. Multi-backend test target
is parked in TODO.md.
