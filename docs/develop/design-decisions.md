# Design Decisions

See [ntm.md](ntm.md) for NTM-specific design decisions (head parameters, memory operations, addressing, diagnostics, convergence).

> **Note: Path C migration superseded the autograd-value design.**
> Many sections below describe V1's `Variable d` (shape-erased) and the
> machinery that supported it: per-element packing, `autoName`,
> `applyDeltas`, `toDoubleNetwork`, the V1 13-method `LayerLike`. As of
> the Path C migration (commits `fa7ed54` … `0dc8d70`), the autograd
> value is `Tensor (dims : Vect rank Nat) (0 d : Device)` with shape on
> the value, the structural Vect-of-Vect type was renamed to `Array`,
> and most V1 scaffolding is gone (`autoName` / `nameLayer` /
> `applyDeltas` / `toDoubleNetwork` / `Endofunctor.emap` / pure-Idris
> `Optimizer`). See [path-c-migration.md](path-c-migration.md) for the
> mapping. Historical sections below are preserved as design context;
> they no longer describe current code.

## Gym (Gymnasium-parity RL API)

The `idris-gym` package provides a pure-Idris reimplementation of Gymnasium's core API. Feature-complete for pure-math envs (Classic Control + Toy Text); deferred for envs that need physics engines or ROM emulators (Box2D, MuJoCo, Atari).

### Outcome: sum type, not two bools

Gymnasium v0.26+ split the old `done` flag into `terminated` (natural end) and `truncated` (artificial end) because value-function bootstrapping treats them differently: when the episode was artificially cut off by a time limit, the next state's value should still be bootstrapped. We model this as:

```idris
data Outcome = Continue | Terminated | Truncated
```

A sum type is better than two bools because it makes the invalid fourth state (`terminated=True, truncated=True`) unrepresentable. Pattern matching is lightweight, and we provide `done : Outcome -> Bool` for the common "is it over?" query.

### Spaces as values, not types

An earlier sketch had `Space` as a type-level descriptor with `ActionTy : Space -> Type` projecting into concrete action types. We rejected this because:

- `Double` bounds can't appear in a type-level `Space`: Idris doesn't permit primitive `Double` as a dependent-type index, so `Box` bounds would need to be erased metadata or encoded as rationals. Either way, the type-level encoding buys nothing over a value-level one.
- Threading a `Space` type parameter through every function (rolloutEp, epochRL, evalEp, etc.) is invasive and produces noisy signatures.
- Gymnasium itself keeps spaces at the value level (runtime Python objects).

So `Space = Discrete Nat | Box (Vect n Double) (Vect n Double) | MultiBin Nat | MultiDisc (Vect k Nat)` is a plain ADT exposed via `actionSpace`/`obsSpace` methods on the `Env` interface. Validity is a contract; the policy network's output dimension enforces it in practice.

### Discrete actions: `Nat`, not `Fin n`

`Fin n` would prevent out-of-range discrete actions at the type level, but it comes with a high cost:

- `Sampler.categoricalSample` returns `Nat`. Upgrading it to `Fin n` requires either changing the sampler (the `List` length is erased, so it doesn't just work), wrapping with `natToFin + fromMaybe` at every call site, or duplicating the sampler.
- `Env state action obs` is polymorphic in `action` so a single interface covers discrete (`action = Nat`) and continuous (`action = Double` or `Vect k Double`) envs. `Fin n` would force specialization.
- `actionSpace = Discrete n` already exposes the bound for wrappers that need it.

Actions sampled from `categoricalSample (Vect n policy outputs)` are in `{0..n-1}` by construction — the invariant holds at the source, not the interface.

### Stochastic envs: seed-in-state + pure PRNG

`FrozenLake` (slippery) and `Blackjack` (card draws) need randomness inside `step`. Three options:

- Make `step` return `IO` — breaks the zero-`unsafePerformIO` policy and poisons callers.
- Thread a `List Double` of pre-generated random numbers (the current `Reinforce.idr` pattern for action selection) — awkward when the env itself needs internal randomness.
- State carries a `Bits64` seed; `step` returns the advanced seed in the next state. Pure, zero FFI.

We chose the third. `Gym.Rng` implements SplitMix64 (Steele, Lea, Flood 2014) using Idris 2's `Bits64` primitives (`prim__shr_Bits64`, `prim__xor_Bits64`, arithmetic that wraps mod 2^64). Derived distributions: `nextDouble` (top-53-bit conversion), `nextNat`, `nextNormal` (Box-Muller).

### TimeLimit as wrapper, not interface method

The previous interface had `maxSteps : Nat` baked in. We removed it because:

- Truncation is a training decision, not a property of the physics. CartPole's physics doesn't terminate after 200 steps — that's a Gymnasium convention for CartPole-v1.
- Wrapping via `TimeLimited` makes Truncated distinct from Terminated, which matters for value bootstrapping.

The env exposes `defaultTimeLimit : Maybe Nat` as informational-only metadata. Actual enforcement happens through the `TimeLimited` wrapper.

### Wrappers as helper functions, not `Env` instances

Our first attempt implemented `Env (TimeLimited state) action obs` delegating to `Env state action obs`. This ran into a name-shadowing problem: the inner `state` from the `Env state action obs` constraint and the outer `state` from `Env (TimeLimited state) action obs` share the same name, confusing instance resolution.

Further complication: interface methods that don't mention all three type parameters (e.g. `step : state -> action -> ...` doesn't mention `obs`) can't resolve the implementation from the method call alone — Idris can't pin down `obs`. Workarounds (named implementations, explicit `{state} {action} {obs}` on every call) are verbose.

Rather than fight the interface resolver, wrappers are exported as plain functions: `timeLimitedStep`, `recordedStep`, `normalizeObs`, `clipAction`, etc. Callers thread wrapper state manually. This is simpler, compiles faster, and matches the existing `Sampler`/`Generate`-style module idiom.

### Acrobot: semi-implicit Euler, not RK4

Gymnasium uses RK4 with dt=0.2 for Acrobot. We use semi-implicit Euler with 4 substeps of dt=0.05. Implementation is ~20 lines vs ~60 for proper RK4, and the task + termination condition are identical. Trajectories diverge numerically from the Gymnasium reference; for RL training purposes this is fine, but it would break a byte-identical reference comparison.

## DNC (Differentiable Neural Computer)

Extends NTM (Graves et al. 2016). Key design choices:

1. **Separate FC layers per parameter group** — cleaner than one massive FC + slicing. Each FC (writeKeyFc, eraseFc, freeGatesFc, etc.) gets its own named parameters. Matches the paper's "interface vector" decomposition.

2. **Decomposed ops, not fused** — DNC addressing ops (allocation, link update, mode mixture) are composed from existing tensor primitives. NTM's `tensor_ntm_read_head` / `tensor_ntm_interp_write` were the same shape and have since been removed too — both architectures now compose their addressing entirely in Idris from generic primitives. Two new primitives added: `tensor_argsort` (integer indices, non-differentiable) and `tensor_cumprod` (differentiable with backward rule).

3. **Erase+add write** — DNC uses `M' = M * (1 - outer(w, e)) + outer(w, a)` with separate erase and add vectors. More expressive than NTM's interpolation write. Matches the paper.

4. **R read heads parameterized at type level** — `DncState r n m h inputSize outputSize ty` with `r : Nat`. R=1 exercises all DNC mechanisms (temporal links, allocation, multi-mode reads). R=4 matches the paper.

5. **Simplified applyGeneric for eval** — The Double-typed eval path uses content-based addressing only (no temporal links or allocation). Full DNC addressing requires tensor-level ops that only work with Variable. This is acceptable because the trained model's content addressing weights carry most of the information for eval.

6. **Numerical stability via clamping** — DNC's multi-timestep state (link matrix, usage, addressing weights) requires six clamping points to prevent forward-pass explosion: link decay clamped to [0, inf) when write weights sum > 1, link entries clamped non-negative, allocation usage clamped to [1e-6, inf) before cumprod (prevents backward gradient explosion via division), retention clamped to [1e-10, inf), read weights clamped and renormalized after mode mixture. Without clamping, NaN occurs at seqLen >= 4. NTM uses the same pattern (`focusVar` clamps weights before pow/division). Weight projection (`projectWeights`) in syncBuffers prevents addressing weight drift across gradient updates.

## Type-level grad-mode

PyTorch's silent footgun: a loss tensor that came from inside `with torch.no_grad():` has no `grad_fn`; `loss.backward()` either raises a `RuntimeError` or silently does nothing depending on the path. We turn this into a compile error.

**Approach**: 0-quantity phantom parameter `g : GradMode` on `Tensor` and `Network` — `Tensor (dims : Vect rank Nat) (0 d : Device) (0 g : GradMode)` with `GradMode = WithGrad | NoGrad`. Erased at runtime; static-only.

**Key decisions**:
1. **State records polymorphic in `g`.** Every layer's state (Linear, Lstm, Rnn, Gru, Ntm, Dnc, BatchNorm, …) carries `(0 g : GradMode)` and its param fields are typed at the same `g`. This eliminates the `believe_me` requirement that an earlier attempt — keeping state at hardcoded WithGrad while threading `g` through forwards — would have forced at every layer-impl call site.
2. **Function-valued fields use their own polymorphic `g'`.** `RnnState.activation` is `{0 g' : GradMode} -> TVec o d g' -> TVec o d g'`. Standard activations (`ttanh`, `trelu`, etc.) are already polymorphic post-Phase-3, so they store at the polymorphic field type and apply at whatever `g` the state happens to be at.
3. **No `believe_me` in user-facing surface.** Coercion between gs is one internal helper, `retypeGrad : Tensor dims d g1 -> Tensor dims d g2`, defined as pure destructure-reconstruct via the polymorphic `MkTensor` constructor. Because `g` is 0-quantity the runtime is identical and Idris's type-checker accepts the cast directly — no type-system bypass.
4. **`weakenGrad` couples Idris-side type with C-side flag.** Flips `requires_grad` in C and retypes the handle as `NoGrad`. The runtime promise and the static promise stay in sync.
5. **Linear consumption on `weakenGrad` / `freezeNetwork` / `unfreezeNetwork`.** Closes the aliasing footgun: after the call, the original Idris reference is consumed at compile time. You can't accidentally keep using the WithGrad-typed name once its C-side flag has been mutated.
6. **No public `strengthenGrad`.** Setting rg=true on an unregistered tensor flips the flag but the optimizer can't update it — a footgun without legitimate use. Unfreezing goes through `unfreezeNetwork` / `unfreezeLayer`, which only walks registered params; the per-tensor primitive isn't exposed.
7. **`nativeTrainStep` / `runBackward` gated on `WithGrad`.** Sets the load-bearing safety: the bug class "loss computed in `withNoGrad`, fed to training" becomes a compile error. CI gate `make check-gradmode-gate` asserts the negative test still fails.
8. **`forwardVar` polymorphic in `g`.** A `NoGrad` input produces a `NoGrad` output naturally. `freezeNetwork` is usable end-to-end — frozen networks remain forwardable, just not trainable.
9. **`Network` constructors polymorphic in `g`, no requirement on whole-network freezing being all-or-nothing.** Coarse approximation for transfer learning is fine for the current example set; per-parameter or per-sub-network freezing is filed as future work.

**Alternatives explored and discarded**:
- *Asymmetric `tlinear` sig* (`Tensor d WithGrad -> Tensor d g -> Tensor d WithGrad -> Tensor d g`) — works for `Linear.applyVar` but breaks `Rnn`'s accumulator pattern where the bias slot holds a `g`-typed running sum, not a parameter.
- *Hardcoded-WithGrad state records + `believe_me` casts in every layer impl* — works but undermines the "type system enforces it" claim with ~40 unaudited `believe_me`s.
- *`mx::compile`-style type-level enable/disable scopes* — Idris 2 doesn't have ambient effects of this shape; would require either threading explicit witnesses or higher-rank polymorphism. State polymorphism turned out simpler.

**Scope**: ~35 files touched, ~930 lines. Two CI gates (`check-gradmode-gate`, `check-gradmode-aliasing`) lock the design against regression.

**Perf impact** (measured `9b1cdb8` → `05a976a` on this VM):
- Clean idris-ml build: 6.25s → 7.85s (+25%, +1.6s absolute).
- Clean examples build: 21.78s → 24.41s (+12%, +2.6s absolute). The extra unification work from the 4th type parameter explains it; still small in absolute terms.
- Runtime: supervised / rnn / lstm wall-clock all within ±6% of pre-refactor — within the documented VM noise floor. Bit-identical loss values. No measurable runtime regression.

See `docs/grad-mode-and-device-typing.md` for the user-facing explainer.

## Type-safe device placement

PyTorch's most common footgun: model on CPU, data on GPU, runtime crash. Idris 2's dependent types solve this at compile time.

**Approach**: Phantom erased parameter on Variable — `record Variable (0 d : Device)`. The `0` means zero runtime cost. The existing `ty` parameter in `Tensor dims ty` / `Network i hs o ty` carries device info automatically: `Network i hs o (Variable CPU)` can't accept `Vector i (Variable (CUDA 0))`.

**Key decisions**:
1. Device on Variable only (not on Tensor itself) — eval path uses `Double` which doesn't need device tracking
2. `FromDouble (Variable d)` for all `d` — pragmatic for CPU-only phase. When GPU codepaths land, restrict to `FromDouble (Variable CPU)` only
3. `toDevice : (d2 : Device) -> Variable d1 -> IO (Variable d2)` takes runtime Device (not erased) since it performs physical data transfer
4. `LossFnTensor` parameterized by Device: `LossFnTensor d = AnyPtr -> AnyPtr -> Variable d`
5. Layer constructors unchanged — `linearLayer : (Num ty, FromDouble ty) => IO (AnyLayer i o ty)` stays generic. Idris infers `ty = Variable CPU` from context

**Scope**: 21 library files + 18 example/test files. Purely mechanical: add `{d : Device}` to signatures, `Variable` → `Variable d`.

## Tape-based autograd (Wengert list)

The autograd uses a flat tape (Wengert list) stored as five parallel Chez Scheme vectors (tags, arg1, arg2, values, paramIds) via `top-level-value`. Variables are indices into this tape. Each arithmetic operation appends an entry recording the op tag, input indices, and forward value.

The backward pass is a single reverse scan of the tape with O(1) gradient accumulation via a mutable FFI array, replacing the previous O(n log n) `SortedMap`-based approach. Only parameter entries (non-empty paramId) are collected into the output `SortedMap String Double`.

This replaced the earlier closure-based graph where each Variable carried `back : Double -> List Double` closures and `children : List Variable`. The tape approach eliminates ~2,200 heap-allocated closures per NTM forward pass, avoids pointer-chasing in topological sort, and reduces per-node size from ~120 bytes to ~40 bytes.

Benchmark (100 NTM training epochs, `src/Example/Bench.idr`):

| Version | Time | Speedup |
|---------|------|---------|
| Closure-based | 37,318 ms | — |
| Tape-based | 14,466 ms | 2.6x |

## Tape generation and staleness

After `collectGrads`, the tape is reset (size=0, gen++). Variables from the previous epoch have stale `tapeGen`. The `ensureOnTape` function detects staleness via generation mismatch and re-registers the variable as a fresh Const entry with its current `.value`. This is transparent to consumers — all code uses Variables through typeclass instances.

A stale parameter used N times in one forward pass creates N independent Const entries. Gradients accumulate correctly via `mergeWith (+)` on paramId during collection.

## FFI CSE prevention

Idris 2 compiles zero-argument `%noinline` definitions as constants evaluated once at module load time. The `tapeGeneration` wrapper must take a varying argument (the tape index) passed through to the FFI call so the compiler treats each call as distinct. Without this, `tapeGeneration` returns a stale value and `ensureOnTape` never re-registers parameters, breaking gradient computation across epochs.

## Optimizer state threading

`trainFrom` and `trainRecurrentFrom` return `(Network, OptimizerState)` to support:
- **Staged training**: print intermediate losses between training blocks
- **Optimizer switching**: e.g. SGD warm-start followed by Adam fine-tuning
- **Checkpoint/resume**: save and restore optimizer momentum state

The simpler `train`/`trainRecurrent` variants discard the state for one-shot training.

## logSoftmax + nllLoss over softmax + crossEntropy

Separate softmax + cross-entropy creates intermediate gradients of `1/pp` in the autograd graph (where `pp` is a softmax probability that can be as small as 1e-6, giving gradients up to 1e6). Even though the mathematically correct combined gradient `softmax(x) - target` is bounded in [-1, 1], the autograd graph doesn't know about this cancellation and propagates the huge intermediates backward.

The log-softmax formulation computes `x - log(sum(exp(x)))` directly, avoiding tiny probabilities entirely. The NLL loss `-(target * logProb)` has no `log` operation, so no `1/pp` gradient. This was the key fix for NTM convergence: the deep computation graph (controller -> memory addressing -> read/write operations) amplified the intermediate gradient explosions.

## Cross-entropy epsilon (1e-6)

The epsilon in `crossEntropy` prevents `log(0)` in the forward pass. Chosen at 1e-6 as a balance: small enough not to distort probabilities, large enough to keep gradients bounded (1/1e-6 = 1e6 vs 1/1e-7 = 1e7 with the old value).

## Gradient clipping: per-parameter vs global norm

The optimizer provides two clipping strategies:

**Per-parameter clipping** (`adam`): bounds each gradient to `[-maxGrad, maxGrad]` independently. Simple and sufficient for feedforward/RNN models.

**Global norm clipping** (`adamGlobalClip`): scales all gradients uniformly so the L2 norm doesn't exceed `maxNorm`. This preserves gradient *direction*, which matters for attention/recurrent models where parameters must coordinate (e.g., NTM key vectors and shift distributions). Per-parameter clipping distorts direction — it can clip the key gradient but not the shift gradient, causing the model to shift to the wrong location.

For the NTM, global norm clipping replaced per-parameter clipping to fix periodic loss spikes caused by gradient direction distortion.

## applyDeltas uses record update

`applyDeltas` updates only the `.value` field via record update syntax `{ value := v.value - d } v`, preserving the existing `tapeIdx`, `tapeGen`, and `paramId`. The stale `tapeIdx` is harmless — `ensureOnTape` will re-register the parameter with the updated value when the next epoch's forward pass runs.

## Detached max in logSoftmax

The max subtraction `x - max(x)` for numerical stability must use a detached constant (created via `fromDouble . cast`, which extracts the Double value and creates a fresh leaf Variable). If the max Variable retained its backward links, every `x - maxVal` subtraction would send a `-1` gradient back through the max, corrupting the gradient of the max element.

## FFI side-effect threading

`let _ = ffiCall` in Idris 2 is dropped by the compiler since the result is unused. FFI functions with side effects must return a value consumed by subsequent computation. In the backward pass, `prim__gradAdd` returns its `AnyPtr` handle, enabling handle threading: `g' = prim__gradAdd g idx val`, where `g'` is passed to the next call. This guarantees evaluation order without `IO`.

## Buffer-backed tensor operations (C FFI)

For linear algebra operations (matrix-vector multiply, dot product), a C shared library (`csrc/tensor.c`) provides BLAS-accelerated forward and backward kernels. On macOS, it links against Apple Accelerate (cblas_dgemv, cblas_ddot); on Linux, it falls back to plain C loops.

**Architecture**: Chez Scheme (Idris 2 backend) loads `build/libidrisml.dylib` at runtime via `load-shared-object`. An arena allocator manages per-forward metadata (weight values, tape indices for backward). The tape records a single MatVecOp or DotOp entry per operation instead of O(m*n) scalar entries.

**Key optimization**: The gradient array and metadata packing use Chez Scheme's native `foreign-ref`/`foreign-set!` for direct reads/writes to C-allocated memory, avoiding the per-element Scheme→C FFI crossing overhead. This optimization alone provided the bulk of the speedup, since `prim__gradAdd`/`prim__gradGet` are called on every tape entry during backward.

**Small-matrix fallback**: For matrices where `i * o <= 4`, the forward pass falls back to scalar operations (standard dotProduct decomposition), avoiding C path overhead for trivially small matrices (e.g., 1×1 RNN weights).

**Tensor Foldable caveat**: The `foldr` instance for `Tensor` processes elements in reversed order (head-first into accumulator). Direct Vect traversal is used instead of `toList` to pack elements in correct row-major order.

Benchmark (`src/Example/Bench.idr`, seed 123456):

| Version | Supervised (1000 ep) | RNN (1000 ep) | NTM (100 ep) |
|---------|---------------------|---------------|--------------|
| Scalar-only tape | 263 ms | 609 ms | 14,858 ms |
| C buffer + Scheme-native grad | 137 ms | 482 ms | 9,259 ms |
| Speedup | 1.9x | 1.3x | 1.6x |

## Persistent weight buffers

Weight values for LinearLayer and RnnLayer are stored in persistent C buffers (`double*` allocated once by `nameParams`, synced after `applyDeltas`). A paired Scheme vector holds paramId strings. On each forward pass, `tapeEnsureBulkConst` registers all weight tape entries in a single Scheme-level loop (epoch-cached) (reading values via `foreign-ref 'double` from the C buffer and pids via `vector-ref` from the pid vector), replacing per-element `ensureOnTape` + `foreign-set!` packing.

The backward pass uses `w_tape_start` (a single int) instead of a per-element `w_tape_idx` array, computing weight indices as `w_tape_start + i*n + j` for sequential memory access.

Buffers are skipped for tiny layers (`i * o <= 4`) that already use the scalar fallback path. After `emap (applyDeltas deltas)`, `syncNetworkBuffers` traverses the network writing updated Variable values back to each layer's C buffer.

Benchmark (`src/Example/Bench.idr`, seed 123456):

| Version | Supervised (1000 ep) | RNN (1000 ep) | NTM (100 ep) |
|---------|---------------------|---------------|--------------|
| C buffer + Scheme-native grad | 137 ms | 482 ms | 9,259 ms |
| Persistent weight buffers | 130 ms | 480 ms | 8,100 ms |
| Speedup | 1.05x | 1.0x | 1.14x |

The modest improvement reflects that weight packing was one of many costs per forward pass; other operations (input packing, tape appends, backward traversal, NTM head computations) dominate.

## Weight epoch caching

Within a single training epoch, the NTM controller's weight matrices are used on every timestep (60 times for a 10-sequence batch with 6-step sequences). Previously, `tapeAppendBulkConst` re-registered all 940 controller weights as fresh tape entries on each call — 59 of the 60 registrations per epoch were redundant since weights don't change within an epoch.

The weight buffer was extended from `#(cbuf, pids)` to `#(cbuf, pids, cached-start, cached-gen)`. The new `prim__tapeEnsureBulkConst` checks `cached-gen == tape-gen`: on a cache hit it returns `cached-start` immediately (no tape mutation), on a miss it runs the bulk-append loop and updates the cache. `prim__resetTapeReturn` increments `tape-gen`, naturally invalidating the cache between epochs.

**Correctness**: backward accumulates gradients additively (`grad_array[ws + i*n + j] += ...`). All 60 timesteps sharing the same tape indices produces identical accumulated gradients to 60 separate registrations merged by `mergeWith (+)` on paramId.

A complementary optimization refactors `walkBackward` to branch on tag: ConstOp entries (tag 0) skip `propagateEntry` entirely and only check pid for gradient collection, while non-ConstOp entries skip the pid read (non-const pids are always empty). This avoids 3 wasted reads per ConstOp entry and 1 wasted string read per non-ConstOp entry.

## Learning rate schedules

Fixed learning rates require manual two-phase tuning (high lr then low lr), which is fragile and task-specific. The one-cycle policy (Smith, 2018; adopted by fastai) automates this:

1. **Warmup phase** (25% of training): linear ramp from `lrMax/25` to `lrMax`. Gradually increases step size to escape initial random parameter space without diverging.
2. **Annealing phase** (75% of training): cosine decay from `lrMax` to `lrMax/1e5`. Smoothly reduces step size for fine convergence. Cosine is preferred over linear because it spends more time at moderate learning rates.

The `(Double -> Optimizer)` factory pattern lets the schedule change lr each epoch while preserving Adam's momentum state — creating a new `Optimizer` record just changes the lr used in `adamStep`, while `OptimizerState` (m, v, t) threads through unchanged.

This replaced the previous manual two-phase approach (`lr1` for phase 1, `lr2 = 0.3*lr1` for phase 2) which required tuning two learning rates and the split point.

## Early stopping

Training for a fixed number of epochs wastes compute when the model has already converged and risks overfitting. Early stopping monitors the training loss and halts when improvement stalls:

- **Plateau detection**: if loss doesn't improve by at least `minDelta` (0.001) for `patience` consecutive epochs, training stops. This avoids wasting time on flat loss landscapes.
- **NaN detection**: if loss becomes NaN (diverged), training stops immediately. This catches learning rate explosions early.
- **Patience=0 disables**: for benchmarks or fixed-duration experiments, setting patience to 0 runs all epochs without early stopping.

The implementation uses a tail-recursive loop with `bestLoss` and `staleCount` accumulators rather than `foldl`, since `foldl` cannot short-circuit.

## Composable weight initialization (Sampler + InitStrategy)

Init methods define a target **variance** and the distribution shape is orthogonal. Previously these were conflated — `xavierInit` returned a uniform range limit, baking in both "Xavier variance formula" and "uniform distribution." The refactored design separates them into two composable pieces:

**`Sampler.idr`** — distribution shapes (`Sampler = Double -> IO Double`):
- `uniform var`: draws from `U(-sqrt(3v), sqrt(3v))`, which has variance `v`
- `normal var`: draws from `N(0, sqrt(v))` via Box-Muller transform, which has variance `v`
- `normalSample`: standard `N(0,1)` sample (Box-Muller)

**`Init.idr`** — variance methods (`InitStrategy = Nat -> Nat -> IO Double`):
- `xavier sampler`: variance = `2 / (fanIn + fanOut)` — default, good for sigmoid/tanh
- `he sampler`: variance = `2 / fanIn` — designed for ReLU
- `lecun sampler`: variance = `1 / fanIn` — designed for SELU/lecun_normal
- `fixedRange bound`: `U(-bound, bound)` ignoring dimensions — reproduces old behavior with `fixedRange 1.0`

Usage: `linearLayerWith (xavier normal)`, `linearLayerWith (he uniform)`. Default `linearLayer` = `xavier uniform` (unchanged behavior).

The variance math is verified: for Xavier with fanIn=fanOut=10, target variance = 2/20 = 0.1. The uniform sampler computes limit = `sqrt(3 * 0.1) = sqrt(0.3) ≈ 0.5477`, which equals the old `sqrt(6/20)`. So `xavier uniform` produces identical distributions to the old `xavierInit`.

Biases are always initialized to zero (standard practice).

`Init.idr` uses `import public Sampler` so callers that `import Init` get `Sampler`, `uniform`, `normal`, and `normalSample` re-exported automatically.

Layer constructors (`linearLayerWith`, `rnnLayerWith`) use `traverse` to sample each weight independently in IO, simplifying constraints from `(Random ty, FromDouble ty, Neg ty)` to `(Num ty, FromDouble ty)` since all randomness happens in `Double` space via the `InitStrategy`.

## C-backed softmax and logSoftmax

NTM head operations create ~116 scalar tape entries per timestep for softmax alone (4 calls × ~29 entries each for content addressing and shift distributions). C-backed `softmaxVar`/`logSoftmaxVar` reduce each softmax call to a single SoftmaxOp/LogSoftmaxOp tape entry plus n ConstOp entries for outputs, cutting tape growth significantly.

The architecture follows the MatVecOp pattern: `SoftmaxMeta` struct (arena-backed) stores input values, output values (saved for backward), and input tape indices. Forward computes max-subtracted softmax in C. Backward uses the Jacobian-vector product:
- **Softmax**: `dx[j] = s[j] * (dy[j] - dot(dy, s))` where `s = softmax output`
- **LogSoftmax**: `dx[j] = dy[j] - exp(logS[j]) * sum(dy)` where `logS = logsoftmax output`

The `applyLayerVar` function dispatches NormalizationLayer "softmax"/"logSoftmax" to these C kernels automatically. NTM heads use specialized `forwardReadHeadVar`/`forwardWriteHeadVar` functions that call `softmaxVar` instead of the generic `softmax` for content addressing and shift computations.

## Hyperparameter tuning protocol

Manual hyperparameter tuning is an anti-pattern that wastes hours on random adjustments. The correct order is:

1. **Fix algorithmic issues first** — bounded gamma, global gradient clipping, efficient topoSort
2. **Use systematic search** — `scripts/sweep.sh` grid search with parallel execution
3. **Never manually loop** — if a training run fails, check the algorithmic level before adjusting hyperparameters
4. **Use schedules over manual phases** — one-cycle policy handles warmup + annealing automatically

## Generic debug module

`Debug.idr` provides reusable forward-pass diagnostics for any layer type.

**Pattern matching on the Layer GADT** (closed dispatch) rather than a typeclass: the set of layer constructors is fixed and known at compile time (LinearLayer, RnnLayer, ActivationLayer, NormalizationLayer, NtmLayer). Adding a `Debuggable` typeclass would require orphan instances and wouldn't provide additional extensibility since new layer types require modifying the `Layer` GADT anyway. This matches how `show`, `emap`, and `applyLayer` already dispatch.

**`debugApplyLayer` takes the input vector**: the NTM case needs to re-run the controller to extract head parameters (key, shift, β, g, γ, erase, add vectors) from the raw controller output. These parameters are computed inside `forwardReadHead`/`forwardWriteHead` but not returned by those functions. Re-running the controller with the same input is pure and deterministic, so produces identical results.

**Double-typed forward**: diagnostics run after training on a `Double`-typed model copy (`toDoubleNetwork`), avoiding autograd tape overhead. The `toDoubleNetwork` function converts `Variable` values via `value` and reconstructs activation/normalization functions by name string matching (e.g., `"sigmoid"` → `sigmoidLayer`). This is slightly fragile but practical since the set of activation names is small and stable.

**`splitWriteInput` helper**: Idris 2's `rewrite` requires a known goal type, which isn't available inside `let` blocks with inferred types. The write head input splitting uses a dedicated function with an explicit type signature so `rewrite plusAssociative` can resolve.

**Per-layer debug entries with key-value pairs**: `DebugEntry` uses `List (String, String)` fields rather than a structured type per layer kind. This is extensible — adding fields to any layer's debug output doesn't change the `DebugEntry` type or break existing printing code

## Curriculum module extraction

Curriculum training (multi-stage training with data regeneration, stage advancement thresholds, and two-level early stopping) was extracted from `Example/NtmCopy.idr` into a reusable `Curriculum` module.

**Motivation**: curriculum learning is a standard ML technique not specific to NTMs. The inline implementation was hardcoded to `Network W [W] W Variable` and `nllLoss`, making it unusable for other architectures.

**Parameterization**: the `Stage` record replaces the previous `CurrStage` by dropping task-specific fields (`minLen`, `maxLen`) in favor of an `IO`-based `generate` function. This lets each stage encapsulate its own data generation strategy — the module doesn't know about copy tasks, sequence lengths, or task-specific parameters. The loss function and chunk size (data refresh interval) are also parameters.

**API**: `runCurriculum` takes a list of stages, an optimizer factory, a schedule, and training hyperparameters, returning the trained model, optimizer state, and total epochs completed. This is the same interface as the inline version but generic over `Network i hs o Variable`.

## Automatic parameter naming (`autoName`)

Every learnable layer must have `nameParams`/`nameNetworkParams` called before training, otherwise gradients are silently discarded (`collectGrads` drops entries with empty `paramId`). This was the #1 gotcha — no error, no warning, just weights that never update.

Additionally, `nameNetworkParams` has a latent collision bug: it applies the same prefix to every layer in a network. For multi-learnable-layer networks, weight indices overlap (both layers produce `pfx_weight0`, `pfx_weight1`, etc.), causing gradient cross-contamination via `mergeWith (+)` in `collectGrads`.

`autoName` fixes both problems by walking the network and assigning type-based prefixes with per-type counters:

- `LinearLayer` → `ll0`, `ll1`, ...
- `RnnLayer` → `rnn0`, `rnn1`, ...
- `NtmLayer` → `ntm0`, `ntm1`, ...
- Activation/Normalization layers → skipped (no learnable params)

**Scope threading for NTM**: the NTM's own params (memory, heads) use its prefix directly (`ntm0_mem0`, `ntm0_rAddr0`). Its controller is recursively auto-named with a fresh counter under the NTM's scope (`ntm0_ll0_weight0`, `ntm0_ll1_weight0`). This eliminates the collision between controller layers that occurred with `nameNetworkParams`.

**Counter state threading**: a `SortedMap String Nat` tracks per-prefix counters. `autoNameLayer` increments the counter for the matched prefix and passes the updated map to `autoNameNetwork`, which threads it through sibling layers. NTM controllers get a fresh empty map (independent scope) so their internal `ll0`/`ll1` don't interfere with outer-level linear layers.

`nameParams`/`nameNetworkParams` remain available for users who want custom semantic names.

## PyTorch benchmark suite

The `bench/` directory contains a faithful PyTorch reimplementation of all idris-ml examples for correctness validation and performance comparison. Key design choices:

**Faithful divergences documented**: Every place where a naive PyTorch port would silently differ from idris-ml has a `NOTE:` comment. The main divergences: custom `LinearRNNCell` (no activation, vs `nn.RNN`'s forced tanh), manual `cross_entropy`/`nll_loss` (soft target vectors, vs PyTorch's class-index losses), NTM add vectors using `2*sigmoid(2*x)-1` (not plain tanh), and controller output clamping to [-20, 20].

**`uv` for Python isolation**: No system Python dependency. `uv` manages its own Python 3.12 and all packages in `bench/.venv/`, fully isolated from the system. `uv.lock` is committed for reproducible installs.

**Benchmark matches Bench.idr exactly**: Same data points, same epoch counts, same warmup. The NTM benchmark uses sigmoid (not tanh) and maxNorm=5.0, matching Bench.idr which differs from NtmCopy.idr.

**Correctness tests with --slow separation**: Quick tests (loss decreases, output shapes) run in ~25s. Full curriculum convergence tests (`@pytest.mark.slow`) are gated behind `--slow` to avoid blocking CI.

## LSTM layer

The `LstmLayer` constructor implements a standard LSTM cell (Hochreiter & Schmidhuber 1997) with learnable initial hidden and cell states (h0, c0):

```
combined = W_ih * x + W_hh * h + bias      -- (4*hidden,)
(i_gate, f_gate, g_gate, o_gate) = split4 combined
i = sigmoid(i_gate), f = sigmoid(f_gate), g = tanh(g_gate), o = sigmoid(o_gate)
c' = f * c + i * g
h' = o * tanh(c')
```

**Forget gate bias**: all biases are initialized to zero, matching PyTorch's `nn.LSTMCell` default. Previously set to 1.0 (Jozefowicz et al. 2015) but changed to match the PyTorch reference implementation. The bias vector is structured as `[i_bias, f_bias, g_bias, o_bias]`.

**Cell state extraction**: `extractCellState` pattern-matches on `LstmLayer` to return the cell state directly. This is used by `NtmLayer` to feed cell state into the read/write head FCs (matching the PyTorch reference architecture where head parameters come from the LSTM cell state, not the hidden state).

**Weight initialization fan dimensions**: `lstmLayerWith` passes `(fanIn, fanOut)` to the init strategy where `fanOut = 4 * hiddenSize` (not `hiddenSize`), because the actual weight matrices are `(4*hidden, input)` and `(4*hidden, hidden)`. Using `hiddenSize` as fan-out produces Xavier variance `2/(i+o)` instead of the correct `2/(i+4*o)`, making weights ~2.5x too large and causing exploding gates that prevent convergence.

**Learned initial states (h0, c0)**: LstmLayer has `h0Buf` and `c0Buf` fields (`Maybe AnyPtr`) for learnable initial hidden and cell states, matching PyTorch's `nn.Parameter` h0/c0. Initialized with Xavier uniform in `lstmLayerWith`. Named as `prefix_h0`/`prefix_c0` in `nameParams`, stored as WeightBufs with pid_ids for the dense optimizer path. The initial states are applied at the start of each sequence and receive gradients through the normal backward pass.

**Weight buffers**: follows the same persistent buffer pattern as `LinearLayer` — two `Maybe WeightBuffer` fields for input-to-hidden and hidden-to-hidden weight matrices. The Variable-specialized path (`applyLayerVar`) falls back to the generic path for tiny layers (`i * o <= 4`).

## Interpolation write

The PyTorch NTM reference uses interpolation write instead of the classic erase+add mechanism from the original NTM paper:

```
mem'[i][j] = w[i] * add[j] + (1 - w[i]) * mem[i][j]
```

Where `w` is the addressing weight vector and `add` is the data to write. This replaces the two-step erase+add:
```
mem'[i][j] = mem[i][j] * (1 - w[i] * e[j]) + w[i] * a[j]
```

**Trade-offs**: interpolation write has fewer parameters (no erase vector) and simpler gradients. The erase+add mechanism allows selective per-element erasing, while interpolation write replaces entire rows proportional to the addressing weight. For tasks where the model writes complete vectors (copy, recall), both work; interpolation is simpler to optimize.

**C kernel**: `interpolationWriteVar` in Variable.idr uses a C-backed implementation (InterpolationWriteOp, tag 18) with forward: `out[i*w+j] = weights[i] * add[j] + (1 - weights[i]) * mem[i*w+j]` and analytic backward gradients to weights, add vector, and memory.

## Softplus gamma (unbounded sharpening)

The PyTorch reference uses `gamma = 1 + softplus(x)` (unbounded, range [1, ∞)) instead of the previous `gamma = 1 + 4*sigmoid(x)` (bounded, range [1, 5]).

**Why the change**: the bounded version was a stability measure to prevent vanishing gradients for non-dominant memory positions (`w^gamma` for large gamma). However, the reference implementations (loudinthecloud, vlgiitr) all use unbounded softplus and converge fine. The LSTM controller + interpolation write + value clipping provide sufficient stability without artificially bounding gamma.

The old bounded version remains available as `forwardReadHead`/`forwardWriteHead` in Memory.idr. The unbounded versions are `forwardReadHeadUnbounded`/`forwardWriteHeadInterp`.

## RMSprop optimizer and value clipping

The PyTorch NTM reference uses RMSprop with value clipping instead of Adam with global norm clipping:

```
v_t = alpha * v_{t-1} + (1 - alpha) * g^2
delta = lr * g / (sqrt(v_t) + eps)
```

**Value clipping** (`clipGradValue`): clips each gradient element independently to `[-maxVal, maxVal]` before the optimizer step. This differs from global norm clipping, which scales all gradients uniformly to preserve direction.

**Why value clipping works here**: the NTM reference implementations use value clip ±10.0 with RMSprop. Unlike global norm clipping (which is better when parameters must coordinate), value clipping is simpler and pairs well with RMSprop's per-parameter adaptive learning rates.

**Implementation**: `rmspropValueClip` composes `clipGradValue` with `rmspropStep`, reusing the existing `OptimizerState` infrastructure (the `v` map stores running squared gradient averages).

**Momentum**: The PyTorch reference also uses `momentum=0.9` with RMSprop. `rmspropValueClipMomentumDense` adds a momentum buffer:

```
v_t = alpha * v_{t-1} + (1 - alpha) * g^2
m_t = momentum * m_{t-1} + g / (sqrt(v_t) + eps)
delta = lr * m_t
```

The `DenseOptimizerState.m` array (already allocated for Adam) is reused as the momentum buffer. With `momentum=0.0` the formula reduces to the standard RMSprop (no momentum). NtmCopy.idr defaults to `momentum=0.9`, configurable via `--momentum`.

## Two-phase training and binary vector data

The PyTorch-aligned NTM uses a two-phase training protocol with binary vector data, replacing the one-hot symbol format with per-timestep loss.

**Binary vector format**: data consists of binary vectors (0/1) with delimiter channels. For copy: `seq_width+1` input channels (data + delimiter). For recall: `seq_width+2` channels (data + item delimiter + query delimiter). Output width is `seq_width` (data only).

**Two-phase protocol**: each sequence has an encoding phase and an output phase:
1. **Encoding**: feed input vectors to the network, discard outputs
2. **Output**: feed zero vectors, collect outputs, compute loss against targets

Loss is computed only during the output phase, matching the PyTorch reference. This is implemented by `epochTwoPhase` in Backprop.idr, which calls `forwardTwoPhase` (defined in Layer.idr) to handle the phase split.

**`TwoPhaseDataPoint`**: new record in DataPoint.idr with `encodingInputs : List (Vector i ty)` and `targets : List (Vector o ty)`. The `Functor` instance enables the standard `map fromDouble` conversion from `Double` to `Variable`.

**Sigmoid + BCE**: the network produces raw logits (no output activation layer). `binaryCrossEntropyWithLogits` applies sigmoid internally for numerical stability, matching PyTorch's `BCEWithLogitsLoss`.

## Alignment with PyTorch reference

The NTM architecture was refactored to match the PyTorch reference in `pytorch/torch_ref/`. Key structural changes:

| Aspect | Old architecture | New (PyTorch-aligned) |
|--------|-----------------|----------------------|
| Controller | Linear+tanh+Linear (copy) / RNN (recall) | LSTM with learned h0/c0 |
| Head param source | Controller output (split) | Separate FCs from LSTM cell state |
| Output computation | Remainder after splitting head params | `output_fc(hidden ++ read_output)` |
| Write mechanism | Erase + Add | Interpolation write |
| Data format | One-hot symbols (`Fin w`) | Binary vectors with delimiter channels |
| Loss contribution | Every timestep | Output phase only |
| Output activation | logSoftmax + NLL | sigmoid + BCE (via BCEWithLogits) |
| Gamma (sharpening) | `1 + 4*sigmoid(x)` bounded [1,5] | `1 + softplus(x)` unbounded |
| Optimizer | Adam, global norm clip | RMSprop, value clip |

The new `NtmLayer` constructor takes 4 explicit sub-layers instead of a nested controller `Network`:
```
NtmLayer : LSTM controller, read FC, write FC, output FC, memory, read addr, write addr, read output
```

This makes the architecture explicit — each component has its own layer with independent weights. The LSTM cell state feeds the head FCs, while the hidden state + read output feed the output FC. This matches the reference's `output = output_fc(cat(hidden, read_output))`.

## C-backed BCE with logits

The NTM training loss (`binaryCrossEntropyWithLogits`) was previously computed via ~7 scalar autograd ops per output element: sigmoid, log, multiply, subtract, etc. With 8 output channels and ~10 output timesteps per sequence, this added ~560 scalar tape entries per data point.

`bceWithLogitsVar` (BceWithLogitsOp, tag 26) fuses the entire loss computation into a single C kernel:
- **Forward**: `(1/n) * sum_i [max(p_i,0) - p_i*y_i + log(1+exp(-|p_i|))]` — numerically stable formulation avoiding direct sigmoid + log
- **Backward**: `d_p_i = (1/n) * (sigmoid(p_i) - y_i) * d_loss` — gradients to predictions only (targets are constant)

The meta struct is arena-allocated and stores prediction/target values and tape indices. Output is a single scalar (the mean loss). The tape entry uses `ext_meta_set` (Scheme-side) to store the meta pointer, matching the pattern used by DotOp and other tensor ops — NOT `tape_meta` (C-side), which is a separate array not read by `walk_backward_ext`.

`epochTwoPhaseDenseBce` in Backprop.idr uses `calculateLossTwoPhaseVarBce` which calls `bceWithLogitsVar` directly, bypassing the generic `calculateLoss` + `binaryCrossEntropyWithLogits` scalar path.

## NTM batch size: copy=16 vs recall=1

The copy and recall tasks use different default batch sizes based on their optimization landscape characteristics.

**Copy (batch=16)**: the copy task has a uniform structure — encode N vectors, decode N vectors. Every sequence in a batch produces a similar gradient signal regardless of sequence length. Batch averaging gives smoother gradients and faster wall-clock convergence without sacrificing gradient quality.

**Recall (batch=1)**: the associative recall task has three properties that make batching harmful:
1. **Variable structure**: each sequence has 2-6 items with a random query position. Averaging gradients across structurally different sequences dilutes the specific memory addressing signal (write to distinct slots, retrieve by content match).
2. **Local minima**: the NTM must simultaneously learn content-based addressing, distinct write slots, and query-triggered retrieval. Noisy gradients from single-sequence updates help escape local minima (similar to how small-batch SGD generalizes better in deep learning).
3. **Update efficiency**: 100K iterations at batch=1 takes ~22 min with 100K gradient updates. The same iterations at batch=16 takes ~6 hours with the same number of (less effective) updates.

All reference implementations use batch=1 for recall: Graves et al. 2014, Collier & Beel 2018 (Adam lr=0.001, evaluated every 200 steps), vlgiitr/ntm-pytorch (100K iterations). The snipsco/ntm-lasagne implementation found recall gets stuck in local minima even at 500K+ iterations.

Both tasks support a `--batch` CLI flag for experimentation.

## Periodic forced GC for long NTM training

Running NTM training for 50K+ epochs causes OOM kills (SIGKILL/exit 137) at ~3000 epochs on macOS. Root cause: each forward pass creates tens of thousands of temporary Scheme Variable records and intermediate allocations on the Chez Scheme heap. After `collectGradsDense` resets the tape, these become garbage. However, Chez Scheme's generational GC doesn't collect aggressively enough — temporary objects promoted to older generations accumulate faster than major collections run. Additionally, ~160MB of `foreign-alloc` tape arrays are invisible to the GC, so it underestimates actual memory pressure.

Fix: call Chez Scheme's `(collect)` (full GC) every 10 epochs via the `forceGC` FFI wrapper in Variable.idr. Cost: `(collect)` on a ~50MB live set takes ~10-50ms; every 10 epochs at ~247ms/epoch adds <2% overhead.

FFI note: `%World` is erased in Chez Scheme's PrimIO calling convention, so the foreign lambda must take 0 arguments: `(lambda () (collect) 0)`, not `(lambda (w) (collect) ...)`. Using a 1-arg lambda causes "incorrect argument count" at runtime.

## Interface-based layer system (LayerLike + AnyLayer)

Previously, `Layer.idr` was a 1104-line file containing every layer type in a single GADT with 9 `mutual` blocks. Adding a new layer type required editing ~10 places. The `Layer` and `Network` types were mutually recursive because `NtmLayer` contained sub-`Layer`s (controller, head FCs, output FC).

The refactored system uses:
- **`LayerLike` interface** (`Layer/Core.idr`): defines methods for forward pass, naming, display, buffer sync, etc.
- **`AnyLayer` existential wrapper** (`Layer/Core.idr`): hides the concrete layer type behind the interface
- **Per-layer modules** (`Layer/Linear.idr`, etc.): each defines a record type and implements `LayerLike`
- **`Network` type** (`Layer/Core.idr`): chains `AnyLayer`s with zero knowledge of concrete layers

**Why interface + existential over GADT splitting**: The GADT approach (splitting operations into separate files) still requires pattern matching on every constructor in every operation file, and mutual recursion for NTM sub-layers. The interface approach puts all per-layer logic in one place per layer type. Network operations become simple recursive walks. Adding a layer = one file, zero edits elsewhere.

**Why NTM uses concrete sub-layer types**: `NtmState` knows its controller is `LstmState` and its FCs are `LinearState`. It calls LSTM-specific functions (`applyLstmGetBuf`, `extractCellState`) for buffer-passing. This is static composition, not dynamic dispatch.

**Idris 2 QTT challenges**:
1. Existential type parameters are erased by default. Fix: store the type constructor as a non-erased explicit parameter: `MkAnyLayer : (l : Nat -> Nat -> Type -> Type) -> LayerLike l => ...`
2. Interface method `Nat` parameters are erased. Fix: add explicit `{i, o : Nat}` to all method signatures, dispatch helpers, and instance implementations.
3. Extra type params on instances (e.g., `NtmState n m h`) are also erased. Fix: `{n, m, h : Nat} -> LayerLike (NtmState n m h)` makes them available at runtime.
4. `Endofunctor (Network i hs o)` needs `{i, o : Nat}` and `{hs : List Nat}` in the instance head.

**Performance**: Dynamic dispatch through the existential adds one dictionary lookup per layer per call. For training, the actual compute (C-backed matmul, LSTM cell, memory ops) dominates.

**Numerical equivalence verified**: The refactored code produces bit-for-bit identical loss values to the original monolithic Layer.idr at every epoch checkpoint (tested with seed=42, batch=1, 10K epochs — both converge at epoch 9300 with 100%/100% accuracy). The interface dispatch, existential wrapping, and module splitting do not affect the numerical computation path.

## Multi-backend architecture (2026-04)

### Why we moved from Scheme-embedded tape to `backend.h` C abstraction

The original autograd stored the tape in Chez Scheme's heap via `top-level-value` (147 inline Scheme FFI lambdas in Variable.idr managing tape allocation, resizing, and pid tracking). This was fast (~1 memory write per scalar op) but:

- Tied to Chez Scheme — no other Idris 2 backend could use it
- 1950 lines of tangled Scheme/C FFI code in Variable.idr
- No GPU path — Scheme can't dispatch to CUDA/Metal
- Adding new ops required editing both Scheme lambdas and C backward rules
- The tape management was duplicated between `prim__tapeGen` and `prim__tapeAppendConst` (two copies of the same 15-line Scheme init guard)

The `backend.h` C API abstracts all tensor operations behind ~80 function declarations. Any backend (libtorch, custom tape, MLX) implements these functions. Variable.idr uses `%foreign "C:func_name,libidrisml"` — one library name, backend chosen at build time.

### Three-backend strategy

**Tape backend** (`BACKEND=tape`, default): Custom C with arena allocator, Accelerate BLAS, and tape-based autograd. Zero external dependencies. Target: match or exceed PyTorch CPU performance for scalar models.

**libtorch backend** (`BACKEND=torch`): Links against PyTorch's C++ library. Tensor-level autograd, GPU support (CUDA/MPS/ROCm), native optimizers. Dependency: ~2GB libtorch install. Target: GPU training, large batch workloads.

**MLX backend** (`BACKEND=mlx`): Links against Apple's MLX C++ API. Metal GPU, lazy evaluation, ~50MB dependency. Target: Apple Silicon ML workloads. NTM ops decomposed into primitives (cosine_sim, conv1d_circular, etc.) with per-op backward rules + fused OP_NORMALIZE for numerical stability.

All three compile to `libidrisml.dylib` — same name, same API. The Idris code is identical across backends. `Makefile` selects: `make backend BACKEND=tape|torch|mlx`.

### Build-time backend selection

The `backend_supports_tensor_params()` C function returns 1 (all current backends: tape, MLX, torch) to let Idris code adapt. When 1: layer `nameLayer` creates consolidated weight tensors (`[o,i]` for Linear, `[4*o,i]` for LSTM) with scalar views sharing storage. The tensor-level forward path (`tensor_mv`) operates on consolidated tensors directly. When 0: layer `nameLayer` creates per-scalar named Variables (legacy scalar fallback).

This is a runtime-queryable capability, not a compile-time flag, so the same compiled Idris binary works with either backend.

### Performance trade-offs

| | Tape | MLX | Torch |
|---|---|---|---|
| Supervised (1000 ep) | 0s | 0s | 1s |
| RNN (2000 ep) | 18s | 34s | 23s |
| Transformer (sort) | ~14s (870 ep) | ~13s (788 ep) | ~14s (874 ep) |
| NTM-copy (converge) | 9m (16K ep, 100%) | 13m (15K ep, 91%) | untested |
| Peak RSS (NTM) | 201MB | 418MB | — |
| GPU support | No | Metal | CUDA/MPS |
| Dependencies | 0 | ~50MB MLX | ~2GB libtorch |

### Decomposed NTM addressing — no fused ops

NTM's read-head pipeline (cosine_sim → softmax → interpolate → circular shift → sharpen → read) is composed in Idris from generic primitives. Earlier versions had a fused `tensor_ntm_read_head` C op with hand-rolled backward; that was removed in 2026-05-07 (commit `<this commit>`) on the principle that paper-specific fusions don't belong at the FFI layer. PyTorch users wouldn't expect a `tensor_ntm_*` op at the boundary; they'd expect to compose it themselves from the standard primitives. NTM and DNC now follow the same pattern.

### NTM numerical stability: epsilon clamping

The sharpening step computes `pow(weights, gamma) / sum(pow(weights, gamma))`. After circular shift convolution, weights can become negative or zero. `pow(negative, gamma)` = NaN when gamma is non-integer. The Idris addressing wraps the shift output in `prim__clampMin shifted 1.0e-10` before raising to gamma. Normalization adds `1.0e-10` to the denominator to avoid divide-by-zero when all weights collapse to zero.

The decomposed normalization `x / (sum(x) + eps)` backward goes through generic `prim__div` and `prim__sum` rules. Earlier work suspected catastrophic cancellation in this path (and added a fused `OP_NORMALIZE`/numerically-stable backward in the tape op for it), but the decomposed Idris path produces bit-identical numerics on tape and converges identically to within floating-point noise on mlx/torch. The fused workaround turned out to be unnecessary once forward-pass NaN sources (un-clamped pow) and stale `tensor_mul_scalar` / `tensor_add_scalar` multi-dim backward bugs were fixed. Both have been corrected upstream; no special normalization op is needed.

### Tape backend performance: optimization roadmap

The tape backend is ~64x slower than the old Scheme tape for Supervised (5.8s vs 90ms). The bottlenecks (ranked by impact):

1. **Per-scalar tensor allocation** (2-3 `malloc/calloc` per op): The old backend did 1 memory write per op (`foreign-set!` into pre-allocated arrays). Fix: arena allocator (bump-pointer, bulk reset).

2. **Stack/unstack overhead** (STACK → RESHAPE → MV → SELECT per matmul): The old backend used persistent weight buffers and buffer-passing. Fix: fused OP_MV_SCALARS that reads directly from scalar Variable tensor pointers without intermediate stacking.

3. **FFI crossing overhead** (N+2 Scheme→C calls per vector pack): `packScalarPtrs` calls `prim__ptrArraySet` N times. Fix: single C call that accepts all handles.

4. **Tape walk cost** (O(tape_size) per backward): Acceptable for small models, but the tape isn't pruned of stale entries between epochs. The `optimizer_step` tape reset helps.

These optimizations are independent and can be applied in any order. The arena allocator gives the best ratio of impact to effort.

### Resolution: tensor-level path + metadata (2026-04-02)

Enabled `backend_supports_tensor_params=1` for the tape backend. Layer weights are now consolidated tensors — one `tensor_mv` call instead of stacking thousands of scalar Variables. Added `op_meta` field to TapeEntry for fused backward metadata (MvMeta, SoftmaxMeta, LstmGatesMeta). Fixed `tensor_lstm_gates` to set `requires_grad=1` on outputs and record OP_LSTM_GATES with cached gate activations.

Result: NTM-copy 100 epochs: 48 min → <1 sec (~2880x speedup). Small model (Supervised) regressed from 3.7s to 5.6s due to `tensorToScalars`/`vecStackTensor` FFI overhead on tiny vectors — this is acceptable since the overhead is fixed per layer and the NTM improvement dominates. Added consecutive-data cache in `tensor_stack_from_array` to skip copy when restacking selects from the same parent.

### Gradient chain fixes (2026-04-04)

Four bugs prevented convergence on the tape backend:

1. **`tensor_select` rank-0 fallback**: `binop_elementwise` produces scalars (rank 0) when both args have `numel==1`. When `tensorToScalars` called `tensor_select` on these scalars, the fallback path created a copy with `requires_grad=1` but no tape entry — severing the gradient chain. Fix: return the scalar directly (identity select on rank 0).

2. **`tensor_matmul` [n]×[n,m] backward**: Used `OP_DOT` backward rule which only reads `grad[0]`, incorrect for vector results. Added `OP_VECMAT` with correct backward: `d_a[i] = Σ_j grad[j]*b[i,j]`, `d_b[i,j] = grad[j]*a[i]`.

3. **`tensor_view_1d/2d` requires_grad=0**: View Variables into param tensors were created with `requires_grad=0`. When the NTM stacked these views (e.g., memory matrix), the resulting tensor had `rg=0` and was disconnected from autograd. Fix: inherit parent's `requires_grad` and record `OP_SELECT` tape entries for views.

4. **Optimizer per-element buffers**: RMSprop/Adam allocated one velocity/momentum slot per param tensor instead of per element. A [400,29] weight matrix had all 11,600 elements sharing one `v[]` slot — the last element's `g²` overwrote all previous. Fix: size buffers by total element count, index by `param_offset + element_j`.

Also removed fused NTM C ops (`tensor_ntm_read_head`, `tensor_ntm_interp_write`) — they were re-added at one point with hand-rolled backward rules but were finally removed for good in 2026-05-07 as part of a cleanup of architecture-specific fusions from the cross-backend FFI surface (see "Decomposed NTM addressing — no fused ops" above). NTM addressing is now composed entirely from generic primitives: `prim__cosineSimilarity`, `prim__softmax`, `prim__conv1dCircular`, `prim__pow`, `prim__matmul`, etc.

Result: LSTM converges (0.714 → 0.048 in 5k epochs, ~1.5 min). NTM-copy with short sequences (1-5) reaches 83.5% accuracy in 2k epochs (~40 min). NTM per-epoch is ~1.3s (short) / ~3.5s (full) — ~10-30x slower than the old fused-C backend. Performance optimization is the next priority.

### Zero believe_me policy (2026-04-10)

Eliminated all `believe_me` and `unsafePerformIO` from the codebase. Every type conversion is now proven correct:

**Nat arithmetic in reshape/flatten**: `(S k) * n = n + (k * n)` reduces as `Refl` in Idris 2. `Tensor.splitAt` uses this to split `Vector ((S k) * n)` into `(Vector n, Vector (k * n))` type-safely. No arithmetic lemma imports needed — the mult definition itself provides the proof.

**Erased proof fields in TransformerState**: The record carries `0 inputPrf : inputSize = seqLen` and `0 outputPrf : outputSize = seqLen * vocabSize`. Input is token indices (embedding lookup, O(1) per token — replaces one-hot + linear). These are erased at runtime (zero cost) but enable `rewrite` at layer boundaries.

**`decEq` for runtime-verified type equality**: The categorical cross-entropy loss function accepts `{n : Nat} -> Vector n Variable -> ...` (generic) but internally needs `n = SeqLen * VocabSize`. Uses `case decEq n (SeqLen * VocabSize) of Yes Refl => ...` to verify and unify at runtime. The `No` branch is unreachable by construction (the network output dimension guarantees the match).

**Pure Idris matrix ops for applyGeneric**: The Transformer's `applyGeneric` (used by `toDoubleNetwork` for evaluation) implements the full attention pipeline in pure Idris using `matrixMultiply`, `transpose`, `softmaxMatrix`, etc. from Math.idr. These work for any `Num`/`Floating` type — no C tensors needed. The C tensor path (`applyVar`) is used for training performance.

### Transformer architecture (2026-04-10)

Multi-head causal self-attention with embedding lookup, Pre-LN, and per-head Q/K/V weights. Input: `[seqLen]` token indices → embedding → `[seqLen, dModel]` → blocks → `[seqLen, vocabSize]` logits. Type parameters `seqLen`, `dModel`, `numHeads`, `headDim`, `numBlocks`, `vocabSize` encode all dimensions at compile time.

New C ops: `tensor_mm` (BLAS-backed), `tensor_transpose_2d`, `tensor_softmax_2d`, `tensor_log_softmax_2d`, `tensor_masked_fill`, `tensor_causal_mask`.

New pure Idris ops in Math.idr: `matrixMultiply`, `softmaxMatrix`, `causalMaskMatrix`, `reshapeToMatrix`, `flattenMatrix`, `scaleMatrix`, `clampMinTensor`.

### Multi-head Transformer (2026-04-03)

Standard Pre-LN architecture with multi-head attention, layer normalization, learned embeddings, and sinusoidal positional encoding.

**Sum-not-concat for head combining**: Instead of concatenating head outputs [seqLen, headDim] into [seqLen, dModel] then projecting via a single [dModel, dModel] matrix, we use per-head output projections [headDim, dModel] and sum the results. This is mathematically equivalent (`concat(heads) @ Wo = Σ_h head_h @ Wo_h` where `Wo_h` is the h-th column block), but avoids needing a 2D column-concatenation op. Uses only existing `tensor_mm` + `tensor_add`.

**Per-head separate weight matrices**: Each head has its own Q/K/V projection weights [dModel, headDim] rather than one big [dModel, dModel] projection split into heads. Avoids needing 2D column slicing (tensor_narrow on dim=1). Stored as `Vect numHeads (LinearState dModel headDim ty)`.

**LayerNormState as sub-component**: Layer normalization is not a standalone `LayerLike` layer — it's used internally by the transformer. `LayerNormState` has the same dual-storage pattern as `LinearState` (typed Vect for applyGeneric, AnyPtr tensors for applyVar). Helper functions (`emapLayerNorm`, `nameLayerNorm`, etc.) parallel `LayerLike` methods.

**New C op: `tensor_layer_norm_2d`**: Row-wise normalization with learnable scale/shift. Stores normalized values and reciprocal std devs in `LayerNormMeta` for efficient backward. Gradient verified via finite differences.

**Type safety proof: `dModel = numHeads * headDim`**: The `MHTransformerState` record requires an erased proof `0 headDimPrf : dModel = numHeads * headDim`. At construction (`mkMHTransformer`), the proof is auto-resolved (e.g., dModel=32, numHeads=4, headDim=8 → `Refl`). Zero `believe_me`.

**Example: sequence reversal** (vocab=10, seqLen=11, dModel=32, numHeads=4). Teacher-forced: `[t0..t4, SEP, t4..t0, EOS]`, predict next token. Achieves 100% accuracy in ~500 epochs. PyTorch reference converges similarly (~500 epochs).

### Transformer performance analysis (2026-04-10)

Systematic investigation of the gap between Idris (52ms/epoch) and PyTorch (16ms/epoch).

**Profiling setup**: `backend_profile_report()` instruments C-side backward and optimizer timing. Wall-clock timing from `runTraining`. Batch size 16, seqLen=11, dModel=32, 4 heads.

**Key finding: the bottleneck is Chez Scheme runtime overhead, not FFI marshaling.**

| Optimization | Wall time/epoch | FFI calls/epoch | Speedup |
|---|---|---|---|
| Baseline (scalar packing) | 160ms | ~33,800 | 1x |
| Tensor-level forward (`applyVarTensor`) | 98ms | ~4,400 | 1.6x |
| C-side one-hot encoding + `TensorDataPoint` | 58ms | ~4,400 | 2.8x |
| Batched projections (`[B*seqLen, dim]` matmuls) | 56ms | ~1,220 | 2.9x |

C backend time was **2ms/epoch throughout** — unchanged by any optimization. The 160ms → 56ms improvement came entirely from reducing Idris/Scheme overhead.

**Double-nameLayer bug** (2026-04-11): The batched forward initially appeared to have incorrect gradients (model wouldn't converge). Root cause: `nameLayer` was called twice — once explicitly on the transformer state, then again by `autoName`. This created two sets of parameter tensors. The batched forward closure captured the first (stale) set while the optimizer updated the second. C backward rules were correct throughout (verified by finite-difference gradient checks). Fix: name once, skip `autoName`, share the same named state between the model Network and the batched forward function.

**Why reducing FFI calls didn't help as predicted**: We estimated ~13μs per FFI call (56ms ÷ 4,384 calls). Reducing calls from 4,384 to 1,220 should have saved ~41ms. Actual savings: 6ms. The ~13μs figure was wrong — most of the 56ms was Chez Scheme runtime cost (GC, thunk evaluation, list allocation, closure dispatch), not FFI marshaling overhead. FFI marshaling is ~1-2μs per call; the rest is Scheme computation between calls.

**Breakdown of the remaining 50ms**:
- Chez GC pauses: ~5-15ms (triggered by per-epoch allocation of lists, closures, data structures)
- Scheme-side computation: ~20-30ms (list operations in foldl, pattern matching, numeric casting, closure evaluation)
- FFI marshaling: ~2-5ms (~1,220 calls × ~2-4μs each)
- Data generation: ~5-10ms (random number generation, list manipulation)

**Implication**: Further FFI call reduction (e.g., batching the attention loop) would save only ~2-5ms. The ~3.25x gap vs PyTorch is fundamentally the cost of Chez Scheme as a runtime. Closing it requires either:
1. An Idris→C compiler backend (bypass Chez entirely)
2. Moving the entire epoch loop into C (eliminating Scheme from the hot path)
3. Accepting the gap — the C backend itself is fast (2ms), and Chez overhead is constant per epoch regardless of model size

### Pluggable Device — sliced `UserDevice` interfaces + per-backend C-symbol rename (2026-05-13)

Tracks the TODO row "Pluggable / dependent `Device` for user-supplied backends." Before this refactor, every backend op in `Tensor.idr` was a free `%foreign` declaration bound to one symbol name (e.g. `tensor_add`), and a Makefile symlink picked which dylib backed it. User-supplied backends required forking the codebase.

**Structural choice — sliced typeclasses**. Three candidates were on the table:

1. **One big interface** — single `UserDevice` typeclass with ~160 methods covering every op in `Tensor.idr`.
2. **Sliced interfaces** — `UserDeviceCore` / `UserDeviceLinear` / `UserDeviceNN` / `UserDeviceConv` / `UserDeviceTape`, ~30 methods each. A backend that doesn't implement `UserDeviceConv` simply can't be used with conv layers, and that's a *type error*.
3. **Dictionary record** — a 160-field `record DeviceOps` passed explicitly. Cheaper at the typechecker; less ergonomic at call sites.

Variant 2 (sliced) chosen. The "ops depend on device" pitch — which is the whole reason to open `Device` up — only lands cleanly under slicing. A `UserDeviceConv d` constraint on `convLayer` means a backend without conv support is a compile-time error at the layer's use-site, not a runtime crash from a missing method. Variant 1 conflates "any backend" with "every backend implements everything"; variant 3 loses interface coherence (methods become record fields, no implicit resolution).

The risk for slicing was Idris-2 instance resolution choking at full width. A throwaway `Device.ProtoWide` module with 5 sliced interfaces × ~30 stub methods each (~150 total) plus a covering instance produced these clean-build times (`rm -rf build/ttc` between runs):

| Surface | Samples | Best-of-4 mean |
|---------|---------|----------------|
| Baseline (no `ProtoWide`) | 7.18, 7.35, 7.43, 7.43, 8.08, 8.40 | 7.32s |
| Sliced 5×30 + 5 instances added | 7.67, 7.73, 7.83, 8.02, 10.73, 13.28, 16.85, 22.03 | 7.74s |

Best-of-4 delta +0.42s (+5.7%), well under the 20% threshold; outliers >10s reflect VM load and appear on both surfaces. Variant 1 (single 160-method interface) would have a *worse* resolution profile than 5 slices of 30, so it's also ruled out.

**C-side multi-link — per-backend symbol rename via `-include` header**. The three backend dylibs export 144 colliding `tensor_*` symbol names (`nm -gj` on `libidrisml_{tape,torch,mlx}.dylib` confirms direct overlap), so naive linking into one shared library fails. Two rename strategies were tested with a toy:

1. **`-D` flag macros**: `cc -Dtensor_add=tensor_add_tape -c …` → `_tensor_add_tape` in the resulting `.o`. Linker accepts both renamed `.o`s into one `.dylib`; both symbols exported, no collision.
2. **`-include rename_<backend>.h` header**: a per-backend header full of `#define tensor_add tensor_add_tape` lines, prepended via `-include`. Same outcome; preferable for 200+ symbols since the rename surface lives in a checked-in file, not a multi-kilobyte compile line.

Adopted #2 (commit `98f17cf`). `scripts/gen-rename-headers.py` parses `backend.h` and emits `packages/backends/rename_{tape,torch,mlx}.h` with `#define <sym> <sym>_<backend>` lines for all 206 exported functions; `make rename-headers` regenerates, `make check-rename-headers` gates CI drift. Returns broadened from an initial 195 to 206 once we noticed `TensorPair*`, `OptimizerHandle`, `int*` return types missing.

**Unified-name aliases keep existing Idris `%foreign` working** (commit `9e20307`). After the rename, the dylib exports only suffixed names (`_tensor_add_tape`). Idris `%foreign "C:tensor_add,libidrisml"` declarations would fail to resolve. Solution: at link time, alias each suffixed primary-backend symbol back to its unified name. macOS uses `-Wl,-alias_list,<file>` with one `<aliasee> <aliasname>` pair per line; Linux uses one `-Wl,--defsym=<unified>=<suffixed>` per symbol. Idris-side declarations stay verbatim; the primary backend handles the dispatch via the alias.

**`BACKEND` as a comma-separated list + symlink retired** (commit `93e96f2`). Makefile refactored to use per-backend property tables (`<b>_SRC` / `<b>_CC` / `<b>_CFLAGS` / `<b>_LDFLAGS_<UNAME>`) plus an `$(eval $(call …))` loop that emits one compile rule per listed backend. Final link uses `c++` if any C++ backend is in the list, else `cc`, with the union of per-backend `LDFLAGS` for the platform. `.backend-stamp` FORCE rule re-links when `BACKEND` changes value (the dylib filename is no longer `BACKEND`-parameterised).

The pre-rollout worry about libtorch + mlx internal-symbol collisions (flagged when scoping the multi-link refactor) **did not materialise on macOS**. `BACKEND=tape,torch,mlx make backend` linked cleanly, no warnings. The 601 KB `libidrisml.dylib` exports `_tensor_add`, `_tensor_add_tape`, `_tensor_add_torch`, `_tensor_add_mlx` (and same for every other op), with the unified name aliased to whichever backend is primary. `example-supervised` produces bit-identical loss output across single-backend, dual-link, triple-link, and primary-switched (`tape,torch,mlx` vs `torch,tape,mlx`) configurations.

Verified `BACKEND` combinations on macOS (Apple Silicon, libtorch 2.x via uv venv, mlx 0.31 via nix):

| `BACKEND=` | Dylib size | Outcome |
|------------|-----------|---------|
| `tape` | 192 KB | single, primary=tape |
| `torch` | (libtorch-dependent) | single, primary=torch |
| `tape,torch` | 426 KB | dual, tape primary |
| `tape,torch,mlx` | 601 KB | triple, tape primary |
| `torch,tape,mlx` | 601 KB | triple, torch primary |

**Outstanding follow-ups** (not blocking the typeclass slice rollout):
- Linux verification: macOS testing only so far. Linux needs `BACKEND=tape,torch` smoke testing; the `--defsym` flag generation is theoretically equivalent but unverified end-to-end.
- `bench-ops-compare` rebuilds the whole dylib per iteration (one backend at a time) to copy it to `libidrisml_<b>.dylib`. Slightly slower than the old per-backend-variant build but isolates per-backend operator timing correctly.
- The dylib filename is unconditionally `libidrisml.{so,dylib}`, so example apps always link against whichever primary the current build has. Switching primary requires `make BACKEND=<new> backend && make example-<name>`.

**test-examples smoke matrix after the multi-link land** (commit `1cab7f8`-ish): 74 of 76 example × backend combinations pass cleanly. Every tape and every torch example passes; both failures are on mlx and pre-existing — `mlx:example-dnc-copy` and `mlx:example-dnc-recall` crash with `[scatter] Cannot calculate VJP with respect to indices` at `--epochs 5 --max-len 3 --batch 1 --seed 99`. Verified pre-existing by checking out the pre-rename Makefile (`ee19b03`) and rebuilding the mlx dylib — same crash reproduces. The TODO row "Re-enable 4 mlx examples on macOS CI" already tracks this class of mlx DNC issues. Convergence-config runs of these examples DID pass historically (`perf-log.jsonl` shows `dnc-copy mlx` exit-0 at `--epochs 3500`+ on commits `798c4ac` / `ede8b6b` / `94700e5`), so the smoke-config bug is a narrower mlx flakiness that the long-run config doesn't trip.

### Open `d` parameter: why `Device = Type` instead of a real sub-type (2026-05-13)

When the `Tensor` record's `d` parameter was opened from the closed sum (`CPU | CUDA Nat | MPS`) to admit any type with a `UserDeviceCore` instance, the worry surfaced: `Tensor [4] Bool` now type-checks at the record level. The binder is unrestricted; from a documentation perspective `Tensor [4] Bool` looks like a valid type.

(Aside on terminology — Idris 2 doesn't have a separate "kind" sort the way Haskell does. `Type` is a value of `Type` (with universe stratification behind the scenes). What Haskell calls "the kind of `d`" is in Idris just "the type at which `d` is bound." Outside this document we'll keep saying "kind" for everyone's sanity, but strictly it's a Haskell-ism.)

**Current safety profile**:
- *Construction* is closed: every Tensor-producing path goes through one of the `UserDeviceCore` methods (`primCreateScalar`, `primCreate`, etc.). No `UserDeviceCore` instance for `Bool` ⇒ no way to inhabit `Tensor [4] Bool`.
- *Operations* are closed: `tadd`, `tsub`, …, `forwardVar`, `applyVar` all carry `UserDeviceCore d =>`. No instance ⇒ no operation typechecks.
- *Declaration* is open: `the (Tensor [4] Bool) ...` type-checks; you just can't construct the value or call any op on it. It's a phantom that can't be made real.

So in practice, the worst a non-device `d` can do is type-check uselessly. The user gets a "no `UserDeviceCore Bool` instance" error at the first attempt to use the tensor — which is a clear-enough signal, just one step later than "no `Bool` as a device."

**Options considered**:

1. **Documentation-only `Device` kind alias** *(chosen)*: `0 Device : Type; Device = Type`. Every kind-binder reads `(0 d : Device)` instead of `(0 d : Type)`. Same kind at runtime, but the call-site documentation says "this is a device tag." No type-system enforcement. ~Zero cost.

2. **Empty marker interface `IsDevice`** *(deferred)*: Empty interface implemented for each device type. `Tensor`'s `d` stays at `Type`, but every Tensor-producing function takes `IsDevice d =>`. Pushes the error one step earlier than "no `UserDeviceCore` instance"; user gets "no `IsDevice` instance" at type-checking the declaration. Cost: one interface declaration + per-built-in impl + the constraint threaded through alongside `UserDeviceCore`. Mostly redundant with `UserDeviceCore` (any device that supports any op already implements `UserDeviceCore`).

3. **Hard kind restriction on the record** *(deferred)*: Either constrain `Tensor`'s record params (Idris 2 doesn't support auto-implicit constraints on record params directly) or make `MkTensor` private and add a smart constructor requiring `UserDeviceCore d`. Real restriction. Cost: every `MkTensor` call site in the codebase (~50 places in `Tensor.idr` + smart constructors throughout) needs updating; the existential `AnyLayer` wrapping breaks because you can't smuggle the constraint through it cleanly.

4. **Closed-sum GADT wrapping any device type** *(deferred)*: A `DeviceTag` data type with `MkDeviceTag : (0 d : Type) -> UserDeviceCore d => DeviceTag`. `Tensor` parameterised on `DeviceTag` values. Loses the "user can declare their own type" simplicity (they have to wrap it in `MkDeviceTag`).

**Why option 1 now**: the practical risk of `Tensor [4] Bool` is bounded — nothing can be done with it. The kind alias gives every kind-binder site a meaningful name without forcing constraint propagation through the record or smart-constructor changes. If we later want sharper errors at the declaration site, option 2 (IsDevice marker) is the natural next step and can be added without breaking the interface plumbing for the lifecycle + arithmetic op slice.

**Revisit triggers**: open if users in the wild file confusing "no `UserDeviceCore X` instance" errors traced back to a typo like `Tensor [4] Bool`, OR if we ever want to define library code that's polymorphic in `d` without `UserDeviceCore d` available (currently impossible — every op needs it, so every polymorphic-in-`d` site naturally has it). Option 2 (`IsDevice`) is the cheapest sharper variant; option 3 is the heaviest if we want compile-time rejection at declaration sites.

## Tensor lifecycle: wrapped-handle FFI ABI

**Problem**: a refcount-driven Tensor lifecycle needs every holder of a Tensor pointer to retain on acquisition and release on drop. The Idris side's natural holder is the `Tensor` record's `tensorPtr` field. Earlier attempts (`is_state`-only refcount, wrap-everywhere via `prim__wrapHandle` in `MkTensor`, per-FFI `RetainGuard`) all foundered on the same root cause: **the Tensor record's "owner identity" (a parallel Chez guardian shadow registered by `prim__wrapHandle`) lives separately from the `tensorPtr` field's value**. Idris-Chez's codegen does live-range narrowing on let-bound records — when a record's only use is `.tensorPtr` extraction, the record (and its shadow) can be elided while the raw pointer survives. The shadow goes to the guardian's dead queue, drain releases the Tensor, and the in-flight raw pointer becomes dangling.

**Decision**: under wrapped-handle ABI on mlx, every Tensor-touching `%foreign` primitive is bound to a Scheme wrapper instead of directly to a C function. The wrapper:

- For a Tensor-arg: extracts the raw pointer via `(vector-ref wt 1)` before calling the C function.
- For a Tensor-return: allocates a fresh Chez vector `(vector 'tensor-handle raw_r)`, registers it with `idris-tensor-guardian`, retains via `tensor_retain_handle`, returns the vector.

The Chez vector IS the Tensor's Idris-level identity. The `Tensor` record's `tensorPtr` field stores the vector, not the raw pointer. `MkTensor` no longer wraps separately — the wrap is already done by the FFI. The compiler can't elide the wrap without eliding the value, because they're the same value.

**Three properties that make this work**:

1. **Foreign calls in Chez aren't interrupted by GC.** While C code runs in a `foreign-procedure` invocation, Chez's GC can't fire. So C-side code can safely dereference the raw pointer extracted from the wrap; no concurrent free is possible.
2. **The wrap is a heap-allocated Chez object.** Its live range is tracked by Chez's normal liveness analysis on let-bindings and function arguments — the same mechanism the compiler is allowed to optimise, but only via reachability, not via field-projection elision. The vector can't be reduced to "just the raw pointer" except through an explicit `vector-ref` call inside the FFI wrapper.
3. **The guardian gives us an Idris-liveness signal.** When the wrap becomes unreachable from Idris-level bindings (Chez decides this), the guardian queues it. A drain pass (called from `withNoGrad`'s exit + periodically from inside no_grad `tape_append` via a foreign-callable trampoline) pops the dead queue and releases C-side retains. Refcount → 0 → Tensor freed → `mx::array` destroyed → MTLBuffer recycled.

**Trade-offs**:

- **Allocation cost.** Each FFI now allocates a Chez vector. For ~5K FFIs/epoch that's 5K small vectors — negligible compared to mx::array allocations on the C side, but worth measuring.
- **Library load timing.** `foreign-procedure` looks up symbols via dlsym in already-loaded libraries. If the first FFI invocation is a `%foreign "scheme:..."` (no implicit library-load hint), `libidrisml` may not be loaded yet. Mitigation: `initManagedHandles` explicitly calls `load-shared-object "libidrisml.dylib"` at first guardian creation, and each converted Scheme primitive does the same load check as a fallback.
- **Per-FFI mechanical churn.** ~165 Tensor-touching FFIs each need their Scheme glue rewritten. Mechanical, lintable, but real work.
- **Cross-backend symmetry.** The ABI applies on mlx; on tape/torch primary builds, the wrappers still execute (allocate a vector, register with guardian) but `tensor_retain_handle` is a no-op stub, so the lifecycle is inert. Slight overhead on tape/torch (one vector per FFI) without lifecycle benefit. Acceptable until tape/torch want refcount-driven freeing for their own reasons.

**Options considered**:

1. **`prim__wrapHandle` in `MkTensor`** — the dormant Phase 2.2 design. Failed on codegen elision (see above).
2. **Per-FFI `RetainGuard` (C++ RAII on the C side)** — explored in a session. Failed because a guard's retain-then-release cycle frees Tensors that had refcount=0 entering the FFI, breaking the caller's raw-pointer alias.
3. **Wrapped-handle ABI** *(chosen)* — the wrap is the value. Verified end-to-end on Phase 0' with 4 FFIs (`prim__createScalar`, `prim__item`, `prim__requiresGrad`, `prim__setRequiresGrad`) and the `Test.ManagedHandle` unit tests. Drain reclaims 50 dropped wraps after forced major GC.

**Revisit triggers**: a future Idris-Chez codegen change that enables actual GC interruption of foreign calls would invalidate property #1 and require re-thinking. If Chez ever exposes a clean way for Idris to inject true module-init code, the per-primitive lib-load fallback could be retired. If the per-FFI allocation cost shows up materially in perf measurement, consider stack-allocating the vector for short-lived intermediates (Chez doesn't expose this, but a future ABI change could).

Plan + phased rollout: `docs/develop/tensor-lifecycle-plan.md`.

