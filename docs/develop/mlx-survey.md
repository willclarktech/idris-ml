# mlx internals survey — Job 3 Phase B

Survey of mlx 0.31.2's `compile` / `vjp` C++ API and how it could
compose with our `tensor_backward`'s replay-VJP pattern in
`backend_mlx.cpp`. Phase B step 1 deliverable per the plan.

Date: 2026-05-12. Headers walked:
- `mlx/include/mlx/compile.h` (public API)
- `mlx/include/mlx/compile_impl.h` (detail / cache API)
- `mlx/include/mlx/transforms.h` (`vjp`, `value_and_grad`, etc.)
- mlx-examples MNIST `main.py` (canonical Python idiom)
- ml-explore.github.io/mlx user-facing compile docs

## TL;DR

`mx::compile` does compose with `mx::vjp` (mlx docs say "function
transformations are composable" and explicitly demonstrate
`mx.compile(mx.grad(mx.exp))`). For a fixed-architecture training
loop, compiling the forward closure once at first backward and
reusing it across epochs is the canonical use case.

Three real integration blockers stand between us and a working
prototype:

1. **Cache-key plumbing**. The public C++ `compile` overload takes
   `std::function<...> fun` and has no `fun_id` parameter. The
   *real* compile API (`mlx::core::detail::compile` in
   `compile_impl.h`) takes a `std::uintptr_t fun_id` used as the
   cache key. The Python binding derives this from
   `reinterpret_cast<std::uintptr_t>(fun.ptr())` — Python object
   identity. We need to do the equivalent in C++ ourselves — either
   keep the `std::function` at a stable address and use its
   pointer, or hash the tape signature into a `fun_id`.

2. **Closure capture must become explicit inputs**. Today
   `tensor_backward` captures `*tape_ref`, `constants`, and
   `param_pool_indices` into the forward closure. Of these, the
   `constants` array contains *non-param* inputs to the tape (input
   data X, targets y, possibly initial RNN states) which **change
   each batch**. mlx's compile bakes captured state into the graph
   at trace time — so we'd compile against the first batch's X/y
   and silently use them forever. Fix: thread constants through as
   additional function arguments (Python's `inputs=` / `outputs=`
   keywords are syntactic sugar for exactly this; the binding code
   inserts them at the end of the arg list before each call).

3. **Variable-shape examples need a different strategy**. mlx's
   `compile` with `shapeless=false` (default) recompiles on shape
   change. For variable-`maxLen` NTM-copy, the tape ops + shapes
   change each epoch — naively, every epoch would recompile. Two
   sub-options: (a) `shapeless=true` (mlx docs flag this as risky
   for graphs with shape-conditional ops, of which we have many —
   reshape, transpose, softmax-with-axis); (b) cache a compiled
   artifact per `(tape_signature, shape_tuple)` and accept the
   per-length compile cost.

## Verified facts about `mx::compile`

### Public C++ API (`compile.h`)

```cpp
namespace mlx::core {
enum class CompileMode { disabled, no_simplify, no_fuse, enabled };

MLX_API std::function<std::vector<array>(const std::vector<array>&)> compile(
    std::function<std::vector<array>(const std::vector<array>&)> fun,
    bool shapeless = false);

MLX_API std::function<std::vector<array>(const std::vector<array>&)> compile(
    std::vector<array> (*fun)(const std::vector<array>&),
    bool shapeless = false);

template <typename F, /* enable_if capture-less */>
std::function<std::vector<array>(const std::vector<array>&)> compile(
    F&& f, bool shapeless = false) { return compile(+f, shapeless); }

MLX_API void disable_compile();
MLX_API void enable_compile();
MLX_API void set_compile_mode(CompileMode mode);
}
```

- Returns a function with the same signature → composes with any
  transform (`vjp`, `jvp`, `grad`, `vmap`) that accepts a
  `vec<array> → vec<array>` function.
- `shapeless=false` (default): recompile on input shape change.
- `shapeless=true`: compile once for all shapes (risky — docs warn
  "any graphs which are conditional on the input shapes will not
  work as expected").
- Capture-less lambda overload converts via unary `+f` (function-
  pointer-decay). Our closure heavily captures state — cannot use.
- `MLX_DISABLE_COMPILE` env var globally disables compilation.
  Useful for A/B during prototype.

### Internal API (`compile_impl.h`)

```cpp
namespace mlx::core::detail {

MLX_API std::function<std::vector<array>(...)> compile(
    std::function<std::vector<array>(...)> fun,
    std::uintptr_t fun_id,
    bool shapeless = false,
    std::vector<uint64_t> constants = {});

MLX_API void compile_erase(std::uintptr_t fun_id);
MLX_API void compile_clear_cache();
MLX_API bool compile_cache_empty();
bool compile_available_for_device(const Device& device);
}
```

Header comment: *"This is not part of the general C++ API as
calling with a bad id is a bad idea."* Using it commits us to an
mlx-internal symbol. Risk: future mlx version could remove or
rename. Mitigation: pin mlx version + write a smoke test that
asserts the symbol resolves.

The `constants` parameter is **cache-key constants** (e.g. scalar
config like axis values), not the Python `inputs=` feature.

### Composition with `vjp`

Header comment from mlx docs: *"In MLX function transformations
are composable. You can apply any function transformation to the
output of any other function transformation."* Example given:
`compiled_grad_fn = mx.compile(mx.grad(mx.exp))`.

For our pattern: `mx::vjp(compiled_forward, all_inputs, {1.0f})`
should work — mlx will trace the compiled function (which already
has the optimized graph baked in), then auto-derive the VJP.

### Performance signal

The only concrete number in mlx's compile docs: *"On an M1 Max the
times are 15.5 and 3.1 milliseconds. The compiled `gelu` is five
times faster."* — but that's for a chained elementwise gelu, the
ideal kernel-fusion case. Our DNC-class tapes have ~3K entries
mixing matmul / softmax / reshape / transpose; the achievable
speedup depends on how much mlx's fuser collapses across op
boundaries.

### Canonical idiom (mlx-examples MNIST)

```python
@partial(mx.compile, inputs=model.state, outputs=model.state)
def step(X, y):
    loss, grads = loss_and_grad_fn(model, X, y)
    optimizer.update(model, grads)
    return loss
```

Note: this compiles the **entire training step** (forward + grad +
optimizer update) into one compiled artifact. A stronger pattern
than just compiling the forward — the optimizer's parameter
updates are part of the compiled graph too. We could pursue this
later; for the first prototype, just compile the forward and let
the optimizer step run eagerly.

## Our current `tensor_backward` (relevant excerpt)

`packages/backends/backend_mlx.cpp:1665-2051`:

```cpp
void tensor_backward(TensorHandle h) {
    // Collect param pool indices and arrays
    std::vector<int> param_pool_indices;
    std::vector<mx::array> param_arrays;
    for (auto& p : param_registry) { ... }

    // Build constant pool from tape (O(tape_size))
    std::vector<std::pair<int, mx::array>> constants;
    for (int i = 0; i <= loss->tape_idx; i++) {
        auto& e = tape[i];
        add_const(e.result); add_const(e.arg1); add_const(e.arg2);
    }

    // Capture tape state for the closure
    int loss_tape_idx = loss->tape_idx;
    auto tape_ref = &tape;
    auto constants_ref = &constants;

    auto forward_fn = [&](const std::vector<mx::array>& params) -> mx::array {
        std::vector<mx::array> pool(pool_size, kF32_ZERO());
        for (auto& [idx, arr] : *constants_ref) pool[idx] = arr;
        for (int i = 0; i < (int)params.size(); i++)
            pool[param_pool_indices[i]] = params[i];

        for (int i = 0; i <= loss_tape_idx; i++) {
            auto& e = (*tape_ref)[i];
            // switch over e.op — 60+ cases applying mx::add/sub/mul/matmul/...
        }
        return pool[loss_pool_idx];
    };

    auto vjp_result = mx::vjp(forward_fn, param_arrays, {mx::array(1.0f)});
    // ... write grads back to param.grad fields ...
}
```

This closure is the right shape (`vec<array> → vec<array>`,
modulo unary loss output) and is what we'd hand to `compile`. The
two issues call out by the survey:

1. **`constants` are captured by reference and change per batch**.
   X/y/etc. would bake into the compiled graph. Refactor: thread
   them as additional `params` entries; either tell `vjp` to skip
   them via `argnums`, or just discard their grads.
2. **The lambda is fresh per `tensor_backward` call** — no stable
   address for `fun_id`. Cache the lambda + its compiled form in
   a static map keyed on tape-signature hash.

## Tape signature for caching

For the cache key, hash:
- The op sequence: `[e.op for e in tape[0..loss_tape_idx]]`
- The shape signature: `[(arg1_shape, arg2_shape, result_shape) for e in tape[...]]`
- The `scalar_arg` / `op_meta` for ops that carry them (axis values
  for sum_dim, softmax dim, etc.)

For fixed-architecture examples (MNIST, supervised, fixed-seq RNN),
this hash is constant across epochs → one compile, infinite reuse.
For variable shapes (NTM-copy maxLen), hash differs per epoch → one
compile per (architecture, maxLen) tuple. With ~10 distinct lengths
seen during training, that's 10 cached artifacts — bounded.

## Prototype scope (Phase B step 2)

Smallest end-to-end test:

1. **Refactor `tensor_backward` closure** to take *all* inputs as
   function args (params first, then constants). Bit-identical
   convergence on tape (sanity check that the refactor itself
   doesn't break things).
2. **Add a `MLX_COMPILE` env-var-gated path** in `tensor_backward`:
   - Compute tape-signature hash → `fun_id`.
   - Look up cached compiled function for this `fun_id`.
   - On miss: build the forward closure, call
     `mlx::core::detail::compile(fwd, fun_id, /*shapeless=*/false,
     /*constants=*/{})`, cache.
   - Call `mx::vjp(compiled, all_inputs, {1.0f})`.
3. **Smoke**: run `make example-supervised BACKEND=mlx
   MLX_COMPILE=1` and compare loss / acc / gradients to
   `MLX_COMPILE=0` baseline. Acceptance: bit-identical gradients
   (or ULP-close on GPU — float32 noise).
4. **Measure on GPU**: `MLX_DEVICE=gpu MLX_COMPILE=1` on the
   handful of examples we can run without recompile-thrash
   (supervised, mnist, fixed-arch RNN). Compare GPU ms/ep
   pre/post.

If the smoke passes AND GPU compiled ms/ep ≤ CPU stream ms/ep on
at least one example, Phase B's acceptance gate is met and we land
the change behind the env var. If smoke fails or no GPU
configuration wins, we accept "mlx is structurally CPU-stream-only
in this environment" as the final position and document.

## Risks

- **`detail::compile` is internal**. Pin mlx 0.31.2 in the project's
  build instructions; add a compile-time assertion that the symbol
  resolves; document the version constraint in `docs/develop/gotchas.md`.
- **Compilation cost on first call may eat the wins** for examples
  that run very few epochs. Phase A measurement showed
  ~4400 ep / 4 min on mlx for NTM-copy — plenty of room to amortize
  even a 1-second compile cost. MNIST runs 3 epochs total at fixed-
  arch — compile cost might dominate; document if observed.
- **Constants-as-inputs refactor changes our gradient flow shape**.
  Currently `mx::vjp` returns grads for `param_arrays` only. The
  refactored call returns grads for `[params, constants]` — we'd
  ignore the constants slice. Verify no off-by-one in the grad
  unpacking.
- **Tape-signature hash collisions** would silently reuse the
  wrong compiled artifact. Use a content-addressable hash (SHA-1
  truncated to 64 bits, or `std::hash` chained over the full
  signature). Document the hash invariant.
- **Variable-arch examples on GPU** (NTM-copy variable maxLen)
  may still lose to CPU due to per-length recompile overhead.
  Acceptable per the plan's acceptance gate (only need ≤ 1× CPU
  on at least one example, not all).

## Open questions for follow-up

- Does `mx::compile` work with `mx::value_and_grad` or only `mx::vjp`?
  The docs say both compose; verify in the prototype.
- Does the optimizer-step-also-compiled idiom (the Python MNIST
  pattern) buy meaningful additional speedup over forward-only
  compile? Defer until forward-only result is in hand.
- Can we expose `MLX_COMPILE` (and compile-cache-clear) at the
  Idris level for end-user control? Plumbing question; resolve
  once the C++ side is working.

## Empirical findings (2026-05-12)

Implementation landed in `backend_mlx.cpp` via a TDD progression
(see `packages/backends/test_mlx_compile.c` for the test order).
`MLX_COMPILE=1` env-var gates the path; eager (default) is unchanged.

Closure refactor: forward function takes `[params..., constants...]`
as explicit inputs instead of capturing constants by reference, so
per-batch values (X, y) aren't baked into the compiled graph at
trace time. `mx::vjp` returns grads for all inputs; we discard the
trailing `n_consts` entries.

### Results

**Important measurement caveat**: the first round of comparisons
were ad-hoc `time make ...` wrappers that bounced around the VM
noise envelope (±15-20% per `feedback_vm_perf_noise`). The
canonical numbers below are from `scripts/perf-run.sh` logged to
`perf-log.jsonl` (commit `1abc206`+); back-fill on 2026-05-12
showed most ad-hoc CPU deltas collapsed within noise. Treat the
ad-hoc column as background; the perf-run.sh column is what
ships.

| example | path | CPU eager (s) | CPU compile (s) | CPU Δ | GPU eager (s) | GPU compile (s) | GPU Δ |
|---|---|---:|---:|---:|---:|---:|---:|
| supervised 1000 ep | ad-hoc `time make` | 1.57 | 1.10 | −30% | 5.69 | 5.73 | +0.7% |
| supervised 1000 ep | perf-run.sh | 8.56 | 8.52 | −0.5% (noise) | 13.98 | 12.20 | −13% (borderline) |
| lstm 50 ep | ad-hoc | 5.12 | 3.73 | −27% | 13.94 | 16.23 | +16% |
| lstm 50 ep | perf-run.sh | 9.64 | 9.32 | −3% (noise) | 12.79 | 13.09 | +2% (noise) |
| lstm 200 ep | ad-hoc | — | — | — | 46.01 | 47.21 | +2.6% (noise) |
| lstm 200 ep | perf-run.sh | — | — | — | 44.24 | 45.15 | +2% (noise) |
| mnist 1 ep / 1000 ex | ad-hoc | 105.6 | 96.8 | −8% | 129.7 | 123.1 | −5% |
| mnist 1 ep / 1000 ex | perf-run.sh | 93.0 | 103.6 | +11% (borderline regression) | 128.9 | 131.5 | +2% (noise) |
| rnn defaults | perf-run.sh | 143.6 | 143.3 | −0.2% (noise) | — | — | — |
| gru defaults | perf-run.sh | 163.0 | 168.9 | +3.6% (noise) | — | — | — |
| transformer defaults | perf-run.sh | 110.8 | 119.7 | +8% (noise) | — | — | — |

### Reading (revised after the perf-run.sh back-fill)

- **MLX_COMPILE is roughly a wash at our example scales**, on both
  CPU and GPU. The earlier ad-hoc "consistent 8-30% CPU win" claim
  was VM noise — confirmed by the perf-run.sh back-fill where every
  CPU cell except mnist landed within the ±15-20% noise envelope,
  and mnist swung from −8% (ad-hoc) to +11% (logged). Both are
  noise.
- **Correctness is solid**: bit-identical convergence on supervised,
  rnn, lstm, gru, transformer (CPU and GPU); ULP-level drift on
  mnist-CPU (compile reorders Conv2D backward fp accumulation,
  within float32 noise — accuracy preserved).
- **No regression of the eager path** verified across all
  per-example smoke pairs.
- **The earlier overclaim is a lesson on the VM noise feedback
  memory**: I should have repeat-measured before reporting deltas
  in the ±15-20% range. Logged it back honestly.

### Decision

`MLX_COMPILE=1` shipped as opt-in. Default stays disabled until a
GPU-friendly example demonstrates the unambiguous case for flipping
the default. Skipped Stages 5 (explicit `detail::compile`
fun_id caching) and 6 (shape-change handling) — mlx's
`std::function`-identity auto-cache covers the fixed-architecture
case (which is what all our examples are), and explicit caching
would be incremental work for a marginal observability win.

Open follow-up: build a GPU-friendly example (filed in `TODO.md`
Medium Priority) to settle whether GPU compile is unambiguously
the right default.

### Follow-up update (2026-05-14): GptLarge measured — preliminary verdict needs caveats

Built `Example.GptLarge` (dModel=256, heads=8, headDim=32, blocks=4,
seq=128, batch=32; 3.17 M params) — significantly bigger than anything
in the prior matrix — and ran the 6-cell perf grid at 10 epochs each.

#### Wallclock — the honest cross-backend comparator

| backend           | wall ms/ep | gap to mlx CPU eager |
|-------------------|-----------:|---------------------:|
| mlx CPU eager     |       8500 |                  —   |
| mlx CPU compile   |       8800 |               +3.5%  |
| mlx GPU eager     |      10200 |              +20.0%  |
| mlx GPU compile   |     ~10000 |              +17.6%  |
| torch             |       9500 |              +11.8%  |
| tape              |       9700 |              +14.1%  |

**Wallclock gap mlx CPU → mlx GPU is ~20%, not 8×.** GPU is slower at
this config, but the gap is modest — and most of every backend's
wallclock (~8-10 s/ep) is the same Idris-side / Chez overhead that
shows up in the parallel "Fix tape per-tape-entry Idris/Chez overhead"
investigation. That overhead is roughly constant across backends and
masks the underlying compute differences.

#### C-totals — mostly enqueue cost, not actual compute

| backend           | C-total ms/ep | what this measures            |
|-------------------|--------------:|-------------------------------|
| tape              |          8830 | actual compute (synchronous)  |
| torch             |          1080 | mostly compute (sync per op)  |
| mlx CPU eager     |            34 | **enqueue only** — see below  |
| mlx CPU compile   |            33 | **enqueue only**              |
| mlx GPU eager     |           276 | enqueue + per-`mx::eval` sync |
| mlx GPU compile   |           254 | enqueue + per-`mx::eval` sync |

The "8× GPU vs CPU" headline from a first reading of this column is
misleading. Sanity check: 75 GFLOPs/step at GptLarge size on an M2
GPU (~10 TFLOPS) implies a compute floor of ~7.5 ms. The reported
"Backward 11 ms/ep" on GPU includes `mx::eval` forced sync — that's
real compute. But mlx CPU reports "Backward 2.5 ms/ep" which would
imply ~30 TFLOPS on a CPU stream — impossible. The mlx CPU profile is
recording **enqueue time**, not actual compute time; the real compute
fires later and isn't attributed to the C-total.

So the C-totals can be split honestly only on mlx GPU (where eval
forces sync at the timing window): real backward ≈ 11 ms/ep, real
optimizer step ≈ 250 ms/ep. The 250 ms is the per-param `mx::eval`
in the AdamW loop: 293 params × ~1 ms launch wall.

#### What we actually learned

1. **GPU is ~20% slower on wallclock at GptLarge scale** — small enough
   that it's a "not yet" rather than a "never". The model is still
   inside the regime where kernel-launch overhead doesn't get
   amortized.
2. **Optimizer per-param `mx::eval` is the most actionable loss on GPU.**
   PyTorch's `_foreach_addcmul_` / `_foreach_add_` consolidate this
   into one kernel; our backend does each param eagerly. Filed as a
   high-prio TODO. **This is the next lever** for getting a GPU-wins
   example.
3. **GPU compute itself looks healthy** at ~11 ms/ep for backward
   (FLOPS-bound, ~75 GFLOPs in ~10 TFLOPS = 7.5 ms floor + sync
   overhead). The compute path is not the bug.
4. **Idris-side per-tape-entry overhead (~8 s/ep on this hardware)
   floods the wallclock** on every backend. Until that's fixed,
   wallclock comparisons are dominated by the constant, not the
   compute. The companion TODO row covers that investigation.
5. **`mx::compile` is still in the noise at this scale** — GPU compile
   shaves 8% off the GPU C-total but doesn't change the verdict.
   Compile on CPU is a wash. Default `MLX_COMPILE=0` stays.

#### Conclusion (revised)

Not "GPU is fundamentally too slow at this scale" — closer to "GPU's
fixed costs aren't amortized yet at this config, and the most
actionable contributor is the per-param optimizer eval". With a fused
multi-tensor optimizer (the TODO row), expectation is GPU compute
should *visibly* dominate the C-total picture and we re-run this
matrix to settle the wallclock side.

`MLX_DEVICE=cpu` stays the default for now. The GPU-friendly-example
deliverable from `TODO.md` is "partial" — the example and measurements
exist, but the GPU-wins outcome is blocked on the fused-optimizer
prerequisite. Verdict re-opens after that lands.

## Sources

- [mlx compile.h](https://github.com/ml-explore/mlx/blob/main/mlx/compile.h)
- [mlx compile_impl.h](https://github.com/ml-explore/mlx/blob/main/mlx/compile_impl.h)
- [mlx transforms.h](https://github.com/ml-explore/mlx/blob/main/mlx/transforms.h)
- [mlx compile user docs](https://ml-explore.github.io/mlx/build/html/usage/compile.html)
- [Python binding for compile](https://github.com/ml-explore/mlx/blob/main/python/src/transforms.cpp)
- [mlx-examples MNIST](https://github.com/ml-explore/mlx-examples/blob/main/mnist/main.py)
