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

## Training-loop checkpointing (auto-save / keep-best / resume)

`runTrainingIO` integrates checkpointing through an optional
`TrainConfig.checkpoint : Maybe CheckpointPolicy` (default `Nothing`,
so existing call sites are unaffected). When set, the loop resumes from
`<dir>/last` before the first epoch, periodically saves + keeps the best
checkpoint after each non-NaN epoch, and reloads `<dir>/best` at the end
so the returned model is the best seen, not the last (PyTorch Lightning
semantics).

Three decisions shaped the design:

- **Single format + clean seam, no adapter abstraction.** Saves go
  through the existing safetensors primitives only. safetensors is the
  de-facto standard (HF Hub default, now under the PyTorch Foundation),
  so a multi-format port/adapter would be speculative. The save/load
  path is structured so a future format is a clean drop-in, but the
  abstraction isn't built until a concrete need appears (filed as a Low
  backlog row).

- **Resume metadata in a `trainer_state.json` sidecar, written in pure
  Idris — no C change.** The heavy state (params + optimizer m/v
  buffers) already round-trips through `safetensors.c`; the only new
  state is the scalar resume info (epoch + best metric), which rides in
  an HF-Trainer-style sidecar written via `System.File`. This kept the
  C ABI and `safetensors.c` untouched. (The safetensors `__metadata__`
  free-form map was the alternative home but would have meant a C-side
  change for marginal benefit.)

- **`CheckpointPolicy` is model-agnostic.** It carries no `model` type
  parameter; `monitor` is `Maybe (IO Double)` (an override closes over
  its own eval state, the same idiom the `metrics` callback uses),
  defaulting to the per-epoch training loss. An earlier
  `monitor : Maybe (model -> IO Double)` added a second occurrence of
  `model` to `TrainConfig`'s field types and broke `model` inference at
  record-update call sites (e.g. Reinforce's `{ metrics := … }`).
  Decoupling the policy from `model` removed that coupling entirely.

`fileCheckpoint` builds the file-backed policy and closes over the
`NativeOptimizer`. It needs only the optimizer, not the model value,
because the C-side parameter registry is **global** — `saveModel`
serializes the whole registry regardless of which model value is in
scope. `loadModel` mutates the registry buffers in place, so a resumed /
best-reloaded model's subsequent forward passes see the loaded weights
with no refresh step.

Out of scope (filed as follow-up backlog rows): bit-exact RNG-stream
continuation (the sidecar stores the seed; resume re-seeds
deterministically, so data order continues approximately, not
bit-exactly) and loading *foreign* HuggingFace checkpoints (needs key
remapping + bf16/f16 read).

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
2. **Sliced interfaces** — `UserDeviceCore` / `UserDeviceLinear` / `UserDeviceNN` / `UserDeviceConv` / `UserDeviceTraining`, ~30 methods each. A backend that doesn't implement `UserDeviceConv` simply can't be used with conv layers, and that's a *type error*.
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

**Unified-name aliases kept existing Idris `%foreign` working** (commit `9e20307`) — **now removed (the per-instance migration, 2026-05-20)**. After the rename, the dylib exports only suffixed names (`_tensor_add_tape`). Idris `%foreign "C:tensor_add,libidrisml"` declarations would fail to resolve, so the original land aliased each suffixed primary-backend symbol back to its unified name at link time (macOS `-Wl,-alias_list,<file>`; Linux `-Wl,--defsym=<unified>=<suffixed>`). This was always a transitional shim: it routed *every* unified-name FFI to the *primary* backend, so in a multi-link build a non-primary device's ops/registry/no-grad-scope silently hit the primary's symbols.

The shim has been retired. Every Tensor-touching `%foreign` now lives in a `UserDevice*` instance method bound to the suffixed name directly, dispatched by the type-level `d`: arithmetic/lifecycle/reductions/shape in `UserDeviceCore`/`Linear`/`NN`/`Conv`; autograd, the param registry (`primParamRegister`/`primParamCount`/…), optimizer creation + `native_train_step`, SafeTensors I/O, profiling, `backend_name` (→ `backendTag`), `withNoGrad`, `polyak_blend`, `mnist_get_image`, `one_hot`, and the dtype-streamed create path (`primCreate*Streamed`, branching on a `RuntimeDType.dtypeTag` of 0=f32/1=f64) all dispatch per-`d`. A repo-wide scan finds zero unified-name references to per-backend-renamed C symbols, so `BACKEND_ALIAS_FILE` / `BACKEND_ALIAS_FLAGS` and the `aliases_<p>.macos.list` rule were deleted from the Makefile.

This fixed a latent multi-device correctness bug: the param registry is a per-TU `static` in each backend, and the old unified-name `param_register_return` (hardcoded wrap tag `"primary"`) registered every param into the primary's registry regardless of device. The acceptance test `Test.MultiDeviceRegistry` (run under `make test-multi`, BACKEND=torch,tape,mlx) registers a `(TorchDev TCpu)` param and asserts torch's `param_count` grows by one while tape's is unchanged — and the mirror — proving the registries are now independent.

**`BACKEND` as a comma-separated list + symlink retired** (commit `93e96f2`). Makefile refactored to use per-backend property tables (`<b>_SRC` / `<b>_CC` / `<b>_CFLAGS` / `<b>_LDFLAGS_<UNAME>`) plus an `$(eval $(call …))` loop that emits one compile rule per listed backend. Final link uses `c++` if any C++ backend is in the list, else `cc`, with the union of per-backend `LDFLAGS` for the platform. `.backend-stamp` FORCE rule re-links when `BACKEND` changes value (the dylib filename is no longer `BACKEND`-parameterised).

The pre-rollout worry about libtorch + mlx internal-symbol collisions (flagged when scoping the multi-link refactor) **did not materialise on macOS**. `BACKEND=tape,torch,mlx make backend` linked cleanly, no warnings. The 601 KB `libidrisml.dylib` exports `_tensor_add_tape`, `_tensor_add_torch`, `_tensor_add_mlx` (and same for every other op). (Before the alias-machinery removal it also exported the unaliased `_tensor_add` pointing at the primary; that unified export is now gone — every reference is suffixed.) `example-supervised` produces bit-identical loss output across single-backend, dual-link, triple-link, and primary-switched (`tape,torch,mlx` vs `torch,tape,mlx`) configurations.

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

### 2026-05-19 follow-up: per-backend hardware parameterisation + cross-backend transfer

Two structural extensions land on top of the open-`d` model:

1. **`TorchDev` parameterised over hardware** — same shape as the existing `MlxDev s`: `data TorchHwDev = TCpu | TMps | TCuda Nat`, `data TorchDev : TorchHwDev -> Type`. Each `(TorchDev d)` cell binds to `*_torch` C symbols; the `d` parameter drives a post-create `tensor_to_device(h, "mps"|"cuda:n")` so the libtorch handle lands on the right hardware variant. `Compatible (TorchDev TMps) F64` deliberately doesn't exist — libtorch's MPS backend errors at F64 *construction*, not just dispatch, so admitting the combination would let the type system mint a runtime-unrepresentable value (mirrors the `MlxDev MGpu F64` rejection).

2. **`UserDeviceTransfer` interface + backendTag-aware `toDevice`** — every backend declares a globally-unique `backendTag : String` ("tape", "torch", "mlx") plus host-marshalling primitives (`primToHost` / `primAllocHost` / `primFreeHost` / int-buffer helpers / `primCreateFromHost` / `primIntraMigrate`). `toDevice` in `Tensor.idr` compares source and dest tags at runtime: matching → fast intra-backend `primIntraMigrate` (libtorch's `.to()` on torch; stream switch on mlx; no-op on tape); differing → host buffer round-trip. Cross-backend transfer creates a fresh handle on the destination — registry membership doesn't follow, users re-register on the dest side if they need optimizer visibility.

The "open `d`" property is preserved: BYO backends declare their own tag type with `UserDeviceCore` + `UserDeviceTransfer` instances and immediately participate in `toDevice` without modifying core library code. Collision-on-`backendTag` is a runtime concern (would route the intra fast path through a foreign backend's C symbols and crash on handle type mismatch); convention is to namespace as `"user/<name>"`. Built-in tags are reserved.

Phase 7 of the 2026-05-19 refactor collapsed the historical generic-device-tag layer (`CPU` / `CUDA n` / `MPS` top-level types in `Device.idr`, plus their `*_Unified` C-symbol forwarding instances). `Device.idr` is now a barrel re-export only; every Tensor in the codebase carries a specific backend-tagged device (`TapeDev` / `TorchDev d` / `MlxDev s`). The link-time alias machinery that exposed each primary backend's symbols under unified C names still exists but has no Idris consumers — `TODO.md` has a parked row to delete it.

### Open `dt` parameter: same `Type` alias trick, with Compatible + UpcastableTo on top (2026-05-17)

`Tensor` gained a fourth 0-quantity phantom parameter for the data
type tag:

```idris
record Tensor (dims : Vect rank Nat) (0 d : Device) (0 dt : DType) (0 g : GradMode) where
```

`DType` is the same kind-alias trick (`0 DType : Type; DType = Type`)
used for `Device`. The dtype tags are `Nat`-parameterized type
constructors: `Float n`, `BFloat n`, `IntN n`, `UInt n`, plus an
unparameterized `Bool`. Aliases for common widths (`F32 = Float 32`,
`F64 = Float 64`, `I32 = IntN 32`, etc.) ship in `DType.Core`.

Three layered typeclasses sit on top:

- `IsDType t` — capability marker, "t is a valid tensor element type."
  One polymorphic instance per family (`IsDType (Float n)`,
  `IsDType (IntN n)`, ...). Carries `dtypeName` / `dtypeBytes`
  metadata for FFI shims.
- `IsDType t => Precision t` — rank-aware subset for `Nat`-parameterized
  families. Method `precisionRank : Nat` returns the bit width and
  seeds `UpcastableTo`'s derivation.
- `UpcastableTo from to` — lossless conversion witness, **derived
  per-family** from `LTE m n` via Idris's auto-search. `Float m → Float n`
  iff `LTE m n`, same for `BFloat`/`IntN`/`UInt`. No cross-family
  instances — converting `UInt 8 → F16` or `BF16 → F32` requires
  explicit `tcastUnsafe`.

A separate empty `Compatible (0 d : Device) (0 t : DType)` interface
gates which (device, dtype) pairs are admissible. The deliberately
missing `Compatible (MlxDev MGpu) F64` instance is what makes
`Tensor [4] (MlxDev MGpu) F64 WithGrad` fail to typecheck — Metal GPU
dropped float64 support in mlx 0.31, and our `Compatible` table makes
the constraint compile-time.

**Inference-only dtype scaffolding on torch (2026-05-22).** Beyond
F32/F64, the torch backend now has a runtime path for `BF16`, `F16`,
`I8`/`I16`/`I32`/`I64`, `U8`, and `Bool` (`Compatible (TorchDev TCpu)` +
`(TCuda n)` — MPS excluded as its reduced-precision/int support is
version-dependent and untestable here; `RuntimeDType` tags reordered to
the kind-major layout on 2026-05-23, see entry below).
Scope is deliberately the **lean non-grad set** — create
(scalar/Nd/1d/2d) + `cast` — wired via per-dtype C symbols
(`tensor_create_*_<dt>_streamed`, e.g. `_bf16_`) following the existing
`_f32`/`_f64` per-symbol convention; each dtype-streamed Scheme wrapper
dispatches a 10-way `cond` on the dtag. The grad `param_*`/`state_*`
create paths stay F32/F64 — torch rejects autograd on integer/bool, and
reduced-precision *training* (BF16/F16 backward + autocast/GradScaler)
is its own deferred row. This unblocks loading pretrained (BF16) weights
for inference; it is not a training feature. Tape dtype parity (F32 as a
real training dtype + the 8 inference dtypes via the lingua franca) and
the all-backend unified tag-dispatch landed shortly after — see the
"Tape dtype parity" and "Unified FFI create/cast dispatch" entries below.

**bf16/f16 + integer SafeTensors I/O via the double lingua franca
(2026-05-22).** The on-disk side of the above: `safetensors.c` now
saves/loads bf16, f16, and the integer dtypes, not just F32/F64. The key
decision was to route everything through the *existing* `double`
serialization path rather than add a byte-level backend extractor. Save
already pulls each param into doubles (`tensor_to_doubles`); load already
reads bytes into doubles then `param_load_data` narrows to the param's
storage dtype. So the only new code is per-dtype pack/unpack *inside
`safetensors.c`* — bf16/f16 bit conversions (bf16 = high 16 bits of f32;
f16 = IEEE binary16, round-to-nearest-even), integers as plain casts.
**No backend-interface, `backend.h`, rename-header, or `ffi_manifest.py`
change** — the dtype knowledge stays in one file. (Tape later gained
its own in-process bf16/f16/integer storage via the lingua-franca path
in `tape_round_to_dtype` — see the tape-dtype-parity entry below — and
both `safetensors.c` and `tape_round_to_dtype`'s DT_BF16/DT_F16 arms
now share the bit helpers lifted into `shared_utils.{c,h}`; mlx
remains F32/F64 only since Metal has no half-precision/integer
storage.) This is byte-exact for bf16/f16 round-trips
(`bf16 → f64 → bf16` is identity) and every integer except **I64 above
2^53** — a double can't represent those, and torch's `.to(kFloat64)`
rounds before the bytes are packed, so the original lingua-franca path
shipped with that caveat. **Closed (2026-05-23)** via a byte-level
extractor pair `tensor_to_int64` / `param_load_data_int64`
(declared in `backend.h`, implemented on each backend in the natural
way: torch blits through `kInt64`; tape/mlx route through the existing
`double` view since they have no native i64 storage). `safetensors.c`'s
I64 save/load branches use the new path when the on-disk and
destination dtypes are both I64; allow_cast=1 loads that narrow I64 →
some other dtype still go through the double pivot (the destination
can't preserve >2^53 anyway). Test gate: `test_safetensors.c`'s
"Exact I64 safetensors round-trip" block seeds `2^62+1`, `-(2^62+1)`,
`2^53+1`, `-(2^53+1)` via `param_load_data_int64`, round-trips through
the file, and asserts every bit pattern survives — runs under
`make BACKEND=torch test-safetensors`. New public `registerParam` puts an arbitrary-dtype,
possibly-NoGrad tensor into the param registry by name so `saveModel`
serializes it — the path for inference-dtype weights and the future
HF-checkpoint loader; `tensor_set_requires_grad` (torch) no-ops on
non-floating dtypes since torch throws otherwise. Cross-language byte
layout is verified by `Example.DTypeSerialize` + `verify_dtypes.py`
(Idris writes → `safetensors.torch` reads).

**Op-level dtype-kind gates (2026-05-22).** `Compatible` gates *which
dtypes a backend admits*; two further empty markers in `DType.Core` gate
*which dtype kind an operation accepts*: `IsFloating` (instances `Float n`,
`BFloat n`) and `IsIntegral` (`IntN n`, `UInt n`); `Bool` is neither. The
loss fns (`tmseLoss`/`tnllLoss`/`tbceLoss`) and the gradient surface
(`runBackward`/`nativeTrainStep`) carry `IsFloating dt =>`, so "training
requires a floating dtype" (a loss on, or backprop through, a Bool/Int
tensor) is a compile error. The constraint propagates to the 5 epoch fns
+ 3 curriculum sigs (bounded — `runTrainingIO` is unaffected, its `epochFn`
returns a bare `Double`). The *activations* are deliberately left
polymorphic, and extending the gate through the layer stack was
**considered and declined** (2026-05-22): `LayerLike` is parameterized
only by the layer type, so `dt` is method-quantified on `applyVar` — a
`dt` constraint must sit on the interface method (all layers) or nowhere,
with no instance-head trick for selectivity. And a layer's `dt` is the
*activation* dtype, which is float in every real network, so interface
gating would only forbid forward passes on a non-float activation (a case
nobody hits) while the meaningful invariant (no backprop/loss on non-float)
is already enforced at the gradient surface. The structural change that
would matter for genuinely-mixed-dtype layers (ternary weights + float
activations) is a param-dtype/activation-dtype split, tracked on the
BitNet row. `DTypePitch` demos the gate as a third axis.

**One-hot is dtype-aware (2026-05-22).** `tensor_one_hot` earlier emitted a
fixed dtype (the Phase-1 F32 hardcode was a band-aid) while the Idris type
claimed the polymorphic `dt` — a lie. It now takes a `dtag` and produces
exactly the requested dtype (0/1 is exact in every dtype, so lossless):
torch switches `dtag → ScalarType`, mlx maps to `mx::Dtype`, tape ignores
it because the one-hot pattern is currently exercised only on the F64
path (tape's per-dtype storage landed later — the parity entry below
covers the eventual `tensor_one_hot` route, but the existing F64 callers
are untouched). The result type honestly equals the runtime dtype; Mnist
dropped its `dtCastFrom` workaround. `mnist_get_image` got the same `dtag`
treatment (2026-05-22) and dropped its own `dtCastFrom`. `tensor_causal_mask`
had the same fixed-dtype shape but was dead (`primCausalMask` never called;
the live mask uses `dtCreateState2d`, already dtype-aware) and was deleted.
`tensor_argsort`'s integer index output was the remaining case, **fixed
2026-05-22** with a typed integral-index API: `targsort` returns an
`I64`-dtyped tensor and `tgather`/`tscatterAdd` take an `IsIntegral` index,
so "this holds indices, not reals" is in the type. Torch `tensor_argsort`
now materializes `kLong` instead of `.to(kFloat64)` — which also kills a
latent MPS abort (Metal has no F64) and the >2^53 precision loss; `gather`/
`scatter_add` already coerce to `kLong`, so the untyped DNC path is
unaffected. The surface is *torch-only by construction* (an integer tensor
only exists where `Compatible d I64` holds — `TorchDev TCpu`/`TCuda`; Metal
has no F64/int and mlx stores F32/F64 only; tape has the integer dtypes
as inference-only storage via the lingua franca, but no integer
*kernels*, so a typed `I64` index handle is still torch-shaped in
practice), so the rejected alternative of casting indices to the *input*
float dtype (lossy on low-mantissa dtypes) was avoided.
All-backend availability follows the tape/mlx integer-*kernel* row, with
no change needed to this surface. Demoed by `Example.IndexOps` (torch-only,
`make example-index-ops`).

**Tape dtype parity (2026-05-23).** The tape backend is no longer
F64-only. F32 ships as a real training dtype — dedicated 4-byte-per-elem
`float*` storage (`tape_arena_f32_from_doubles` /
`tape_persistent_f32_from_doubles`), `tape_load_d` / `tape_store_d`
dtype-aware element accessors that branch on the per-tensor `dtype_tag`,
and a `backend_tape_kernels.inc` X-macro that stamps the elementwise
binop + unop bodies once for F64 (`SCALAR=double`) and once for F32
(`SCALAR=float`). The four-rung gradcheck oracle (T29: elementwise /
`tensor_mv` + OP_MV / softmax / optimizer-step) is GREEN, and the
per-rung pattern (paired RED→GREEN commits, no `.inc` extension) was
extended to every remaining public `tensor_*` — scalars, reshape /
concat / select / view, losses (MSE/CE/BCE/NLL), BLAS-heavy linalg
(`cblas_s{gemm,gemv,ger}` for matmul / linear / bmm / outer / vecmat),
norm + conv + pool, lookups (embedding / gather / scatter / argsort /
cumprod), and recurrent cells (LSTM / GRU / cosine_sim). End state:
every public `tensor_*` accepts F32 input without abort. **Asymmetric
`data = F32` / `grad = F64`** is a deliberate scope choice — lets the
entire 67-case backward switch stay dtype-agnostic for grad reads/writes
(`((double*)t->grad)[i]` everywhere); only data *reads* in the ~12
backward cases that touch input data needed `tape_load_d`. The torch
invariant "param dtype == grad dtype" is **not** preserved on tape;
mixed-precision (BF16/F16 trainable + GradScaler/autocast) is filed as
a separate row. The 8 inference-only dtypes (BF16/F16/I8/I16/I32/I64/U8/
Bool) ship via the `double` lingua franca in `tape_round_to_dtype`,
with bf16/f16 routed through the bit helpers lifted from `safetensors.c`
into `shared_utils.{c,h}` so on-disk I/O and in-process casts share one
rounding implementation. `Compatible TapeDev <dt>` is now open Idris-side
for all 10 dtypes; T29 gradcheck ladder + T31 inference matrix + T32
cast-storage alignment unit blocks gate the contract. Demoed by
`Example.PrecisionDemo` across all three backends (`make
example-precision-demo`). Closes the "Broaden tape backend dtype storage
beyond F64" and "Runtime cross-device + safe-vs-unsafe precision demo"
rows; see the corresponding `CHANGELOG.md` entry.

**Unified FFI create/cast dispatch (2026-05-22).** Companion to the
above: one `tensor_create_<shape>_streamed(..., int dtag)` + one
`tensor_cast_dtype_streamed` per shape per backend, with internal
dtag switching. Replaces the previous per-dtype symbol explosion
(one C symbol + `cond` arm per dtype per shape per backend); a new
dtype is now a switch arm, not a symbol fan-out. backend.h went
290 → 228 exported symbols. Per-backend dispatch: torch handles all
10 dtags; mlx handles F32/F64 and aborts the rest via
`mlx_dtype_unsupported` (Metal has no half/int storage); tape handles
every dtag for which `Compatible TapeDev <dt>` is open (i.e. all of
them, after the parity work above). The `Compatible` gate already
prevents unreachable dtags at construction; the per-backend aborts are
defence-in-depth. Landed additively (entry points → backend.h decls +
rename headers → wrapper flip → delete superseded) so every
intermediate dylib stayed resolvable; the unified streamed bases were
added to `ffi_manifest.py` MANIFEST + INIT_FFI so `check-ffi-wrap-template`
now lints the previously-exempt create/cast wrappers. See the unified
create/cast FFI dispatch entry in `CHANGELOG.md`.

**Kind-major RuntimeDType tag layout (2026-05-23).** Closes the original
grow-as-needed integer tag (`F32=0, F64=1, BF16=2, F16=3, I8=4, I16=5,
I32=6, I64=7, U8=8, Bool=9`) which mixed lingua-franca demand with
insertion order and silently meant F32 when a `dtag` was zero-initialized
(the b2d6c7d mnist incident). New layout reserves `0` as invalid (any
backend's `default:` arm aborts loudly), groups by kind family with 4
lanes for {8, 16, 32, 64}-bit variants, and leaves sub-byte families
(24-31) open for future quantization dtypes:

```
0   invalid (zero-init traps)
1   Bool
4   U8                               (family 1 — U; 5-7 reserved for U16/U32/U64)
8   I8     9 I16   10 I32   11 I64   (family 2 — I)
13  F16   14 F32   15 F64            (family 3 — F; 12=F8 E4M3 reserved)
17  BF16                              (family 4 — BF; 16/18/19 reserved for BF8/BF32/BF64)
20-23 reserved                        (family 5 — TF: TensorFloat-32 etc.)
24-31 reserved                        (sub-byte: U4/I4/NF4/ternary/MX — named lanes,
                                       not arithmetic since their semantics aren't
                                       pure `(family, bit-width)`)
```

For numeric families `bit_width = 8 << (tag & 3)`, `family = tag >> 2`.
Compact: used tags span 0..17 (fits in u5); jump-table dispatch stays
dense up through 17. The wire tag is purely a runtime FFI calling
convention (safetensors uses the string dtype name on disk), so there
is no on-disk migration; renumber lands as one atomic paired commit
across the 10 `RuntimeDType` instances in `Tensor.idr`, the per-backend
dispatch switches (torch's `st_for_dtag` + 11 create/cast switches +
`tensor_one_hot`; mlx's 11 sites + `tensor_one_hot`; tape's
`tape_tag_from_dtag` translation + 10 `_streamed` wrappers), every
`dtag` literal in `test_backend.c` / `test_safetensors.c`, and the
`backend.h` reference comments. Test gate `T33` in `test_backend.c`
asserts (new dtag → expected dtype name) across all wired dtypes, on
every backend that supports them.

Tape's internal `DT_*` enum stays dense (`F64=0..BOOL=9`) so the hot
read paths (`tape_load_d`, the 67-case backward switch) keep tight
switch density; only the ABI boundary translates via the switch in
`tape_tag_from_dtag`. Closes TODO row #21.

Full design memo and decision log: `docs/develop/dtype-parameter.md`.

**Why a kind alias rather than a real sub-type (same answer as for
`Device`)**: the alias is pure documentation — `(0 dt : DType)` reads
"this is a dtype tag" but compiles to `(0 dt : Type)` underneath. The
real constraint is delivered by `Compatible d dt =>` (and
`IsDType dt =>`) on tensor-constructing functions; a non-DType `dt`
can be declared but never inhabited.

**Why a separate `Compatible` interface rather than a method on
`UserDeviceCore`**: dtype admissibility is per-(device, dtype) pair,
not per-device. If MLX-GPU supports F32 add, it supports F32 every
op; there is no realistic backend with op-specific dtype
restrictions. So an empty marker interface — one instance per
admissible pair — is the right shape. Adding methods would be pure
ceremony.

**Why per-family `UpcastableTo` rather than a single closed sum**:
each family's bit-width ladder is its own partial order. F32→F64 is
lossless within the `Float` ladder; F32→I64 would not be (even though
the bit-width fits, the semantic interpretation changes). Cross-family
upcasts are deliberately impossible to derive; the compiler can't
decide whether a `UInt 8 → F16` is what the user wanted even when the
bit-pattern fits losslessly.

**MlxDev parameterization** lives in the same PR: `data MlxDev :
MlxStream -> Type` with `MGpu` / `MCpu` constructors and ergonomic
aliases `MlxGpu = MlxDev MGpu` / `MlxCpu = MlxDev MCpu`. Mirrors the
existing `CUDA Nat` shape. One set of `UserDevice*` instances
parameterized over `{s : MlxStream}` rather than two opaque siblings.

**Lessons**: the first attempt at threading dt was a "loose
migration" that left method bodies hardcoded to F64 while the Tensor
record's slot was polymorphic. The polymorphic-vs-concrete mismatch
caused Idris-2's elaborator to allocate a fresh dt unification
variable at every Tensor reference and keep it alive across the
module — 30+ GB Chez Scheme RSS on a single library build before the
fix. Switching to full polymorphism (every method body binds dt,
callers pin the concrete value at the leaf) collapsed memory back to
baseline. Documented in
`docs/develop/gotchas.md` "Polymorphic type-parameter slot vs
concrete value in method body" and
`docs/develop/dtype-parameter.md` "Lessons learned."

**Revisit triggers**: when F32 runtime support lands on a backend, the
`Compatible` table grows that cell and the demo's `MlxGpu F32` path
becomes runnable (not just type-checkable). When BF16/F16/CUDA support
arrives, the same machinery picks up new `Compatible` + `UpcastableTo`
entries with no further refactor.

## Tensor lifecycle: wrapped-handle FFI ABI

**Problem**: a refcount-driven Tensor lifecycle needs every holder of a Tensor pointer to retain on acquisition and release on drop. The Idris side's natural holder is the `Tensor` record's `tensorPtr` field. Earlier attempts (`is_state`-only refcount, wrap-everywhere via `prim__wrapHandle` in `MkTensor`, per-FFI `RetainGuard`) all foundered on the same root cause: **the Tensor record's "owner identity" (a parallel Chez guardian shadow registered by `prim__wrapHandle`) lives separately from the `tensorPtr` field's value**. Idris-Chez's codegen does live-range narrowing on let-bound records — when a record's only use is `.tensorPtr` extraction, the record (and its shadow) can be elided while the raw pointer survives. The shadow goes to the guardian's dead queue, drain releases the Tensor, and the in-flight raw pointer becomes dangling.

**Decision**: under wrapped-handle ABI on mlx, every Tensor-touching `%foreign` primitive is bound to a Scheme wrapper instead of directly to a C function. The wrapper:

- For a Tensor-arg: extracts the raw pointer via `(vector-ref wt 1)` before calling the C function.
- For a Tensor-return: allocates a fresh Chez vector `(vector 'tensor-handle raw_r)`, registers it with `idris-tensor-guardian`, retains via `tensor_retain_handle`, returns the vector.

The Chez vector IS the Tensor's Idris-level identity. The `Tensor` record's `tensorPtr` field stores the vector, not the raw pointer. `MkTensor` no longer wraps separately — the wrap is already done by the FFI. The compiler can't elide the wrap without eliding the value, because they're the same value.

**Three properties that make this work**:

1. **Foreign calls in Chez aren't interrupted by GC.** While C code runs in a `foreign-procedure` invocation, Chez's GC can't fire. So C-side code can safely dereference the raw pointer extracted from the wrap; no concurrent free is possible.
2. **The wrap is a heap-allocated Chez object.** Its live range is tracked by Chez's normal liveness analysis on let-bindings and function arguments — the same mechanism the compiler is allowed to optimise, but only via reachability, not via field-projection elision. The vector can't be reduced to "just the raw pointer" except through an explicit `vector-ref` call inside the FFI wrapper.
3. **The guardian gives us an Idris-liveness signal.** When the wrap becomes unreachable from Idris-level bindings (Chez decides this), the guardian queues it. A drain pass (called from `withNoGrad`'s exit) pops the dead queue and releases C-side retains. Refcount → 0 → Tensor freed → `mx::array` destroyed → MTLBuffer recycled. The plan originally called for a periodic mid-block drain via foreign-callable trampoline from `tape_append`'s no_grad branch; Phase 5' measurement found memory stays bounded at peak ~49MB on the originally-failing mlx examples (`ntm-copy`, `ntm-associative-recall`, `mountain-car-cont`) without it, so the trampoline is deferred until a workload actually needs it.

**Trade-offs**:

- **Allocation cost.** Each FFI now allocates a Chez vector. For ~5K FFIs/epoch that's 5K small vectors — negligible compared to mx::array allocations on the C side, but worth measuring.
- **Library load timing.** `foreign-procedure` looks up symbols via dlsym in already-loaded libraries. If the first FFI invocation is a `%foreign "scheme:..."` (no implicit library-load hint), `libidrisml` may not be loaded yet. Mitigation: `initManagedHandles` explicitly calls `load-shared-object "libidrisml.dylib"` at first guardian creation, and each converted Scheme primitive does the same load check as a fallback.
- **Per-FFI mechanical churn.** ~600 Tensor-touching FFIs across Tensor.idr + Device.idr + Device/{Mlx,Tape,Torch}.idr each need their Scheme glue generated. Mechanical (driven by `scripts/lifecycle/ffi-convert-to-scheme.py`), lintable (`make check-ffi-wrap-template` in CI preflight), but real work.
- **Cross-backend symmetry.** The ABI applies on mlx; on tape/torch primary builds, the wrappers still execute (allocate a vector, register with guardian) but `tensor_retain_handle` is a no-op stub, so the lifecycle is inert. Slight overhead on tape/torch (one vector per FFI) without lifecycle benefit. Acceptable until tape/torch want refcount-driven freeing for their own reasons.

**Options considered**:

1. **`prim__wrapHandle` in `MkTensor`** — the dormant Phase 2.2 design. Failed on codegen elision (see above).
2. **Per-FFI `RetainGuard` (C++ RAII on the C side)** — explored in a session. Failed because a guard's retain-then-release cycle frees Tensors that had refcount=0 entering the FFI, breaking the caller's raw-pointer alias.
3. **Wrapped-handle ABI** *(chosen)* — the wrap is the value. Verified end-to-end across all 5 wrap-handle files (~600 FFIs) in commit `0ec6a99`, with the `Test.ManagedHandle` unit tests (drain reclaims 50 dropped wraps after forced major GC) green on tape + mlx. Phase 3'-a (commit `78bc19b`) retired the `prim__wrapHandle` / `prim__unwrapHandle` / `managedShadow` / smart-constructor layer once the FFI's Scheme glue was doing all the work. Phase 4' (commit `9664726`) added a structural linter (`make check-ffi-wrap-template`) with CI gating.

**Revisit triggers**: a future Idris-Chez codegen change that enables actual GC interruption of foreign calls would invalidate property #1 and require re-thinking. If Chez ever exposes a clean way for Idris to inject true module-init code, the per-primitive lib-load fallback could be retired. If the per-FFI allocation cost shows up materially in perf measurement, consider stack-allocating the vector for short-lived intermediates (Chez doesn't expose this, but a future ABI change could).

**Priming the drain in production (2026-05-22).** Property #3 above assumed the guardian drain runs, but the drain *helper* (`idris-drain-once`) was installed only by `initManagedHandles` — called nowhere in production, only in `Test.ManagedHandle`. So in real runs `withNoGrad`'s `drainManagedHandles` and the per-step `native_train_step_<b>` drain epilogues were dormant no-ops (their `(top-level-bound? 'idris-drain-once)` guard was false). On mlx the C sweep still dropped buffers (memory looked bounded), but the husk *objects* never reached `rc==0` → a slow handle leak on long grad-mode runs (see gotchas "The mlx generation sweep must never `delete`…"). Fix: the guardian lazy-init carried by the `INIT_FFI` create wrappers (`GUARDIAN_LAZY_INIT` in `ffi_manifest.py`) now also installs `idris-drain-once` at first tensor creation, so every entry point drains — not just test code. And `Train.idr`'s `epochEnd` does an mlx-gated `forceMajorGc + drainManagedHandles` before the sweep (the per-step minor GC can't reach intermediates still in epoch-fn scope; the post-return major GC + drain can). This makes property #3's "drain reclaims the husk to `rc==0`" actually hold in production. tape/torch draining stays inert (no-op release stubs), but priming there also stops the small wrap-vectors leaking in the otherwise-never-drained guardian.

Full model + how to add new FFIs: `docs/develop/tensor-lifecycle.md`.
Plan + phased rollout: `docs/develop/tensor-lifecycle-plan.md`.

### `UserDevice` interface inclusion criterion: dispatches on `d` (2026-05-17)

When trimming `UserDeviceTape` to its useful surface (commit landed 2026-05-17), the question that needed answering was: *which methods are pulling their weight as interface methods?* Some `UserDeviceTape` methods existed as instance bindings that all forwarded to the same `*Unified` C symbol — looking like a dispatch surface but not delivering one. Worse, removing them from the interface mattered for the [Pluggable Device pitch](#pluggable-device--sliced-userdevice-interfaces--per-backend-c-symbol-rename-2026-05-13): every method a BYO author "must implement" should give them actual control over the behaviour, otherwise it's onboarding tax.

The criterion adopted: **a method belongs on `UserDevice<Slice>` iff three conditions hold**:

1. **The C state it reads or mutates is per-tensor or per-backend**, not process-global. State that's truly process-wide (the `param_register` registry behind `param_count` / `param_name` / etc., the autograd-flag toggle behind `no_grad_begin`/`end` is borderline — the flag is global but the *interpretation* differs per backend's autograd machinery, so it stays) can't deliver per-`d` dispatch even if the method dispatch looks like it should.
2. **The Idris consumer of that state uses the interface method**, not a direct `%foreign` symbol. The optimizer (`nativeTrainStep`) calls `param_grad_item_and_zero` etc. via a fixed unaliased `%foreign`, so even a backend that bound `primParamGradItemAndZero` to its own C function would never have that binding called. The Idris-side consumer has to be threaded through the typeclass for the method to deliver dispatch.
3. **Different `UserDevice<X> d` instances bind it to different C symbols** in practice (or could, given a non-trivial backend implementation). If every built-in forwards to the same `*Unified` symbol, that's a smell — either the underlying state is process-global (fails #1) or the method should be a fixed FFI surface, not an interface method.

The methods removed from `UserDeviceTape` on 2026-05-17 all failed at least two of the three: `primParamCount` / `primParamName` / `primParamGradItem*` / `primParamZeroAllGrads` / `primParamSubtractDelta` failed #1 (registry is global) and #2 (optimizer bypasses the interface); `primParamClear` / `primWriteDouble` / `primPrint` failed #1 + #3. The surviving methods (autograd flag toggles, tensor-handle shape queries, per-backend allocation + scratch buffers) all pass — they operate on backend-specific state and are the dispatch surface that can grow live as layers gain `UserDeviceTape d =>`.

This criterion replaces the looser "is it dead code?" framing the original audit was using. "Dead code" was the wrong question — these methods were dead-as-dispatch, but exported as public interface members. Removing them is an *API improvement*, not a cleanup: BYO authors implement fewer no-op methods, and the interface stops looking like a slice that doesn't deliver.

**Revisit triggers**: if a future refactor moves the param registry to be per-backend (one MLX-side, one torch-side, etc. — would address the row 11 / row 16 architecture conversation), the methods that failed #1 above could come back. The dispatch criterion would still apply: re-add only the methods whose new per-backend implementations actually get called via the typeclass, not because "every backend should expose registry queries."

### Device-availability gating: `Linked` (compile-time) + EAFP (runtime) (2026-05-21)

Backend-scoping (`TapeDev` / `TorchDev d` / `MlxDev s`) says *which backend* a tensor lives on, but said nothing about *whether that backend is compiled in* or *whether the hardware exists*. A program could spell `TorchDev (TCuda 1)` on a CPU-only host, compile fine, then SIGABRT deep in libtorch. Closed in two gates, each placed where the fact actually lives (full rationale in [`device-availability-gating.md`](device-availability-gating.md)):

1. **Linkage → compile-time.** `Linked d` is an empty capability marker (sibling to `Compatible (device, dtype)`), wired into the construction + forward path. Its instances are *not* hardcoded — the generated `HwConfig` module emits one per backend in `BACKEND`, so a tape-only build has no `Linked (MlxDev _)` and any constructor naming an mlx device fails to compile. Consequence: inherently-cross-backend modules (Transfer, MlxStreamDemo) can't compile under a single-backend build and live outside the always-compiled examples ipkg.

2. **Hardware presence → runtime, EAFP not LBYL.** We answer "is this *linked* device backed by real hardware right now" by *attempting* the allocation and catching, not by a pre-probe. `tensor_to_device` (torch) wraps `.to()` in `try/catch` → NULL handle; `prim__handleIsNull` + `attemptOn` lift NULL → `Left DeviceError`. One source of truth (the real alloc), no TOCTOU, no `is_available` surface to drift. The fear that drove an earlier LBYL draft — "spelling `cuda:1` SIGABRTs" — was an *uncaught*, not *uncatchable*, exception; the guard makes it catchable. `HardwareClass` + `HardwareClassed` recover the cross-backend silicon commonality as runtime data (for grouping/reporting only — never unifying tensor types), and `availableDevices` runs the same EAFP probe over a candidate list.

Both gates preserve the open-`d` property: a BYO backend self-declares `Linked MyDev`, gets EAFP gating for free if its construction throws on bad hardware, and maps `hardwareClass` to `Other "user/<name>"`.

### HF-aligned modules store fused tensors fused (never split-at-load) (2026-05-26)

CONVENTIONS rule 2 said "storage shapes match HF on disk". The HfGpt2 worked example pinned what that means in practice for the trickiest case: HuggingFace's GPT-2 stores its Q/K/V projections as one fused `[hidden, 3*hidden]` weight per layer (the `c_attn` `Conv1D` blob), and idris-transformers mirrors this — there's no `splitFusedQkv` step at load time, no separate Q / K / V Idris-level records. The `Gpt2AttentionState` record holds one `Gpt2Conv1D hidden (3*hidden)` field, registered as `transformer.h.{i}.attn.c_attn.weight`. The Q/K/V split happens at forward time as three `primNarrow ... 1 ...` views (zero-copy on tape, fast on torch + mlx after the `bd61bef` axis-arg fix). The multi-head split per Q/K/V is then a second nested narrow inside the per-head loop.

Two consequences worth naming:
- **The module is the rename adapter.** A `loadModel "model.safetensors"` is plain string-matching against the param registry. If on-disk has `attn.c_attn.weight` and the module registered the same name with the same shape, the bytes land in the right place with no transformation. No `param_load_with_remap`, no shape-split machinery in core.
- **HF naming wart hardware travels with the storage.** GPT-2 wraps its linears in `Conv1D` (which is `nn.Linear` *transposed* — weight is `[in, out]` rather than `[out, in]`). The HfGpt2 module stores it transposed too (`Gpt2Conv1D` record holds `Tensor [i, o]`, registered with HF's exact name) and the forward `applyConv1D2d` computes `y = x @ W + bias` directly via `primMm + primAdd`, bypassing `tlinear2d` (which expects `[out, in]`). The tied LM head in `hfGpt2ForwardLm` reuses `wte.weight` the same way HfBert's `applyMlmHead` reconstitutes the decoder from `wordEmb.weight` at HfBert.idr:769 — the tied parameter is on disk once, used twice at forward.

Cross-reference: the BERT module is the opposite-direction worked example — BERT *doesn't* fuse Q/K/V on disk, so HfBert stores them as three separate `[hidden, hidden]` linears matching `attention.self.{query,key,value}.weight`. The general rule: whatever HF does on disk, the module does in its state records.


### Typed tokenizer handle: `Tokenizer vocab` + `(n ** Vect n (Fin vocab))` (2026-05-26)

The Tokenizer.idr boundary is the canonical pattern for crossing from dynamic-shape input (file IO, network IO, subprocess output) into idris-ml's dependently-typed tensor world. Three pieces:

1. **`Tokenizer vocab` GADT pairs the vocab size with the type at construction.** `mkTokenizer : String -> (vocab : Nat) -> IO (Either TokError (Tokenizer vocab))` probes the actual on-disk vocab via the Python subprocess's `vocab` subcommand and validates against the type-level `vocab` Nat — `Left TokVocabMismatch` if they disagree. Once you hold a `Tokenizer 30522`, you can't accidentally feed its outputs to a model expecting `Tokenizer 128256`; the *type* enforces alignment that's otherwise a silent runtime drift (token IDs read but interpreted against the wrong embedding table).

2. **`tokenize` returns an existential `(n ** Vect n (Fin vocab))`.** The length `n` is runtime-determined (whatever the subprocess emits) but type-tracked via the dependent pair; each ID is `Fin vocab`, statically bounded by the type-level vocab. The lift from raw `Nat` to `Fin vocab` happens once at the FFI boundary via `natToFin`; any out-of-range ID surfaces as `Left TokIdOutOfRange` (means the Python tokenizer's vocab grew past the model's claim — exactly the alignment drift the typed boundary is supposed to catch).

3. **`detokenize`'s input is `Vect n (Fin vocab)`.** Closes the loop: decoded IDs come back as a string, but the input was already validated bounded. No way to feed raw `Nat`s back through; the type lifts that to a programmer-visible mistake.

This is the shape for any future dynamic→static crossing in idris-ml: existential dependent pair for the length, `Fin n` for bounded indices, smart constructor with on-disk validation for the parameter coupling. The KV-cache existential growing-`p` in Phase 4 is the same shape, just for sequence positions instead of vocab IDs.

### Tokenizer backing: Python subprocess for v1, Rust FFI deferred (2026-05-26)

Tokenizer.idr backs onto a Python subprocess via `scripts/hf_tokenize.py` — about ~1 s of Python startup per call. Acceptable because Python is already a dev dep (the oracle scripts use it), the call pattern is "tokenize the prompt once, generate N tokens in-process, detokenize once" so subprocess starts amortise across the model forward passes, and the API surface (`Tokenizer vocab` + `(n ** Vect n (Fin vocab))`) is Liskov-substitutable. A future row tracks the Rust-FFI upgrade in a new `packages/idris-tokenizers/` package: vendor `huggingface/tokenizers`, expose a C ABI, load via `%foreign`. Drops per-call cost from ~1 s to ~50 µs; existing callers don't change. Land trigger: a user reports the startup as a real perf issue, OR a non-HF Idris example wants tokenization (where the separate-package boundary pays for itself).

### Why a separate `packages/idris-tokenizers/` package is NOT in v1 (2026-05-26)

The deferred Rust-FFI upgrade above is filed as a *new package* (`packages/idris-tokenizers/`), but the v1 Python-subprocess shim lives *inside* `packages/idris-transformers/`. Avoid-speculative-generality: today the only consumer of tokenization is the HF-aligned model layer; a separate package's boundary cost (ipkg, test harness, install target, CI lane wiring) outweighs the benefit at this scale. Same shape as the "Pluggable Device" entry's "open-but-unused-still-costs" framing — peer packages land when a real second consumer appears, not on speculation.

### Stdlib hypothesis ruled out for the large-Nat-pattern OOM (2026-05-26)

While landing Test/Tokenizer.idr, the elaborator OOM'd on `case (Left (TokVocabMismatch 12345 30522)) => ...` — Nat literals in patterns unfold to Peano during case-tree construction (same class as the "Large Nat type-level reduction" entry in `gotchas.md`, in the pattern path rather than the unification path). Investigation followed the lead in the now-retired "Opaque type-level Nats" TODO row: the nixpkgs idris2 v0.8.0 derivation patches `bootstrap-stage2.sh` to replace `MAKE all` with `MAKE idris2-exec`, which seemed like it might be skipping a stage-2 stdlib rebuild that would otherwise pick up Nat optimisations. Traced through the full nixpkgs derivation chain (`by-name/id/idris2/unwrapped.nix`, `mkPrelude.nix`, `wrapped.nix`): the stdlib `.ttc` files are built by *separate* `mkPrelude.nix`-derived `prelude` / `base` / `contrib` / `linear` / `network` packages, each invoking the stage-2 `idris2-unwrapped` binary via `IDRIS2=...`. The `bootstrap-stage2` patch only skips an *additional* stdlib rebuild inside the idris2-unwrapped derivation; that build's output is discarded by `wrapped.nix` (which goes through the separate prelude/base derivations for `IDRIS2_PACKAGE_PATH`). Reverting the patch would add redundant build work; it does NOT change stdlib quality. The OOM is a compiler-level concern in Idris 2 v0.8.0; wait for v0.9.0 or use the `==` idiom in pattern arms. The investigation memo lives in `gotchas.md` so the next future-us doesn't re-trace the same dead-end.

### Per-backend-set build cache (2026-05-27)

All build artifacts (Idris ttc, installed library prefix, dylib, example
executables, stamps) live under `build/$(BUILD_KEY)/` where the key is
`<backend-list>-mlx<MLX_DEVICE>-torch<TORCH_DEVICE>` (e.g.
`tape-mlxcpu-torchcpu`, `torch-mlxcpu-torchmps`,
`tape-torch-mlxcpu-torchmps`).

**Why all three vars in the key:** the four generated `.idr` files —
`HwConfig.idr`, `HwDevices.idr`, `BuildConfig.idr`, `TestConfig.idr` —
embed the active build's choices. HwConfig/HwDevices content depends
on the full `BACKEND` list (one `Linked` instance per linked backend);
BuildConfig/TestConfig depend on `(PRIMARY, MLX_DEVICE, TORCH_DEVICE)`
(F32 on mlx-gpu / torch-mps, F64 elsewhere). All three vars contribute
to *which* installed library + ttc you need to find on disk.

**Why backend ordering matters:** PRIMARY is the first comma-separated
entry. `BACKEND=tape,torch` and `BACKEND=torch,tape` produce different
dylibs (tape-primary vs torch-primary, different unified-symbol
aliases), so they need distinct cache trees. A PRIMARY-only key would
collide them.

**Why the four generated `.idr` files stay at fixed `packages/<pkg>/src/`
paths instead of moving under `build/<KEY>/`:** the `.ipkg`
`sourcedir = "src"` field is fixed; relocating the file would require
parallel source trees per set. Cross-set switches *do* rewrite their
content (mtime bumps), but Phase-0 verification at
`/tmp/idris-mtime-test` confirmed Idris 2 uses **interface-hash
matching** for downstream cascade (not mtime), so the cost is just the
four files themselves re-elaborating (~4 s total) — downstream modules
with matching interface hashes don't cascade.

**Why `LIBRARY_SRCS` excludes the generated `.idr` files:** the
`.library-cache-stamp` recipe wipes the per-set ttc on any
LIBRARY_SRCS change (compensates for Idris's interface-hash check
missing where-clause body changes). Including the generated files
would defeat the per-set cache — a cross-set switch would look like
"library source changed" and wipe the just-switched-to set's warm
ttc. Their own per-set ttc + interface-hash check is sufficient.

**Why write-if-different (cmp-then-mv) on the four generated files:**
within a single set's reruns, a `make` re-evaluation regenerates the
file but the content is identical. Unconditional `>` would bump the
mtime, forcing a re-elab of that file + its 1-second cost on every
no-op rebuild. cmp-then-mv keeps mtime stable when content matches.

Measured outcome (`make install` round-trip wall-clock):
- Cold tape: ~60 s.
- Cold torch: ~250 s.
- Warm switch tape → torch (both already built): ~3 s.
- Warm switch torch → tape: ~2.5 s.

The pre-refactor wall-clock for a cross-set switch was ~60 minutes
(full re-elab of idris-ml + idris-gym + idris-transformers +
idris-ml-examples). The new cost is dominated by Make's directory
walk + Idris startup for the 5 `--install` calls.

Implementation outline:
- `BUILD := build/$(BUILD_KEY)` cascades through every `$(BUILD)/...`
  path reference.
- `IDRIS2_LOCAL := $(CURDIR)/$(BUILD)/idris2-prefix` per-set install
  prefix; `IDRIS2_PACKAGE_PATH` reads from it.
- Every `idris2 --install` / `idris2 --build` passes
  `--build-dir $(CURDIR)/$(BUILD)/ttc-<pkg>` (per-package ttc dir);
  repo-root example/test builds get
  `--build-dir $(BUILD)` (folded into `IDRIS_FLAGS`).
- Test-package builds (`test-gym`, `test-transformers`,
  `test-examples-unit`, `bench-gym`) get
  `--build-dir $(CURDIR)/$(BUILD)/test-<pkg>` so their executables
  land at `$(BUILD)/test-<pkg>/exec/<name>` instead of inside the
  test source tree.
- `test-multi` recipe split into a wrapper + `_test-multi-build`
  sub-target so the `BACKEND=torch,tape,mlx install` sub-make and the
  example-build sub-make both resolve `$(BUILD)` / `$(LIB)` /
  `$(TESTCONFIG_IDR)` under the multi-link set, not the outer make's
  BACKEND context.

Out of scope (separate rows if measured to matter):
- Sharing `backend_<b>.o` across sets (set-agnostic compile of each
  backend's `.o`, per-set link). Saves a few seconds of C compile on
  set switch; the Idris re-elab is the dominant cost this row
  attacks.
- Pruning the installed-prefix tree (per-set installed libraries can
  duplicate set-agnostic content like idris-gym). Disk is cheap.

### `TORCH_DTYPE` opt-in dtype override + BF16 torch-mps gate (2026-05-28)

The default `(ExampleDevice, ExampleDType)` cell for `BACKEND=torch
TORCH_DEVICE=mps` is `(TorchDev TMps, F32)` because MPS rejects F64 at
construction (see [`gotchas.md`](gotchas.md) "libtorch MPS rejects F64 at
tensor construction"). For inference, BF16 is more memory-efficient
(2× smaller than F32) and libtorch's MPS BF16 kernel coverage is now
sufficient for HF model forward (BERT, GPT-2, Llama-3.2-1B verified).
Reduced-precision *training* on BF16 is a separate concern (gradients,
loss scaling) and stays out of scope here.

**Design choice**: a single opt-in env knob `TORCH_DTYPE=BF16` that
overrides the per-cell default and bakes into `BuildConfig.idr` at
build time. Unset (default): cell mapping unchanged. Set to `BF16`:
forces `ExampleDType = BF16` on torch builds regardless of device.
F32 stays the default for `torch/*/mps` because F32 is the broader
gate for MPS kernel coverage (any newly-added op might lack a BF16
kernel and abort with `c10::Error`); we don't make the user opt into
a less-supported path by accident.

**Why an env knob, not a third axis in BUILD_KEY's cell mapping
matrix**: BF16 isn't a hardware variant, it's a precision/perf
trade-off the user toggles per-experiment. Treating it like
`MLX_DEVICE=cpu|gpu` (a hardware-presence fact) would conflate two
orthogonal things — and would have required 12 cell-mapping rows
instead of 6 in the BuildConfig generator. The env-knob approach
keeps the cell mapping at 6 and adds one `if` for the override.

**BUILD_KEY discriminator**: `BUILD_KEY` extends with
`$(if $(TORCH_DTYPE),-tdt$(TORCH_DTYPE),)`, so
`torch-mlxcpu-torchmps-tdtBF16` and `torch-mlxcpu-torchmps` keep
distinct cache trees. Cross-mode switches stay near-free per the
per-backend-set cache (above section).

**Compatible instance**: `Compatible (TorchDev TMps) BF16` exists
again — the deliberate exclusion at `Device/Torch.idr` lines 618-625
(comment: "MPS reduced-precision support is version-dependent and
untestable in this VM") was stale; the VM has libtorch with BF16-MPS
kernels for the relevant op set. Instance restored in commit
`ab5386a`.

**Measured outcome**: Llama-3.2-1B inference on torch-mps, 8 greedy
tokens, after the P1 `.contiguous()` removal:
`F32 8:43 → BF16 7:35` (~13% additional wall reduction, dominated by
the embedding lookup + per-layer projection storage halving). End-to-end
output text matches F32 within the first ~2 greedy tokens (greedy decode
diverges later as BF16's 8-bit mantissa accumulates rounding).

**Non-blocker resolved 2026-05-31** (commits `e2ad295` + `9856771`):
mlx-Metal BF16 + F16 storage are now both wired. The original "mlx
0.31 has no BF16 storage on Metal" claim was a misreading — the C
backend was rejecting BF16 dtags via a defensive abort whose message
said "Metal has no bf16/f16/int storage", but mlx 0.31's
`mx::bfloat16` and `mx::float16` types are first-class and work on
M3+ Metal. `Compatible (MlxDev MGpu) BF16/F16` and `Compatible (MlxDev
MCpu) BF16/F16` are now admissible; the dispatch table routes
dtag 13 → `mx::float16` and dtag 17 → `mx::bfloat16` end-to-end
(per-shape streamed creators, cast, readback). Supervised converges
5/5 eval on mlx-gpu F16 (loss 0.135) and 3/5 on mlx-gpu BF16
(loss 0.193) — BF16's 7-bit mantissa is the precision floor on
small-model classification, same as torch-mps BF16. See CHANGELOG
2026-05-31 entry for full storage-vs-precision breakdown.

### torch-mps per-op MPSGraph submission cost — canonical lane choice (2026-05-28)

Even after the P1 `.contiguous()` audit and the P2 BF16 gate, torch-mps
on small-model inference (BERT, GPT-2-tiny) stays **5-20× slower** than
mlx-gpu / torch-cpu at the same workload. Measured 2026-05-28:

| Workload | torch-cpu | torch-mps | mlx-gpu | Ratio (mps/cpu) |
|---|---|---|---|---|
| hf-bert inference  | 19 s  | 1:27 | ~10 s | 4.6× |
| hf-gpt2 inference  | 14 s  | 4:35 | ~12 s | 19.6× |
| hf-llama inference | 46 s  | 6:42 | 38 s  | 8.7× |

Root cause (confirmed via `gotchas.md` "Non-contiguous views" audit + the
explore agent's submission-count analysis): libtorch's MPS path **submits
each primitive op as a separate `MTLCommandBuffer`**, while mlx batches
the entire forward into one Metal submission via its computation graph.
HF model forward at our scale is ~660 primitive ops/layer × 16 layers
= ~10K ops per Llama forward; at ~0.5-1 ms submission overhead per op,
that's 5-10 s of pure overhead on Llama and proportionally more on
BERT/GPT-2 where per-op compute is smaller.

**Canonical lane choice for inference on Metal**:
- **Prefer mlx-gpu** for HF inference (BERT, GPT-2, Llama).
- **Use torch-mps** when graph-compile isn't available or when the model
  uses a torch-only kernel.

The fix (closing the gap) is filed as a follow-up row (TODO row +
task #393); investigation paths include `jit::trace` / MPSGraph
fusion, audit of Idris-side per-token graph rebuilds (RoPE table
slicing), and a structural graph-mode constructor stack. Until that
lands, the lane-choice paragraph above is the canonical answer to
"which Metal backend for inference at idris-ml's scale".


## Keep RoPE / RmsNorm / SwiGLU as composable rank-3 primitives, not megafused kernels

When closing the ~150× torch-mps Llama gap to PyTorch Python (commits
`6850366` SDPA + `c09d374` all-heads RoPE under #399), one option
considered was wrapping the whole attention block — including RoPE —
into one fused C kernel per backend, so per-head RoPE math + Q/K/V
projections + matmul + softmax + matmul + concat all live in one
`tensor_attention_block_*` symbol. That would have been the fastest
single-architecture path: one MTLCommandBuffer per layer on torch-mps,
no Idris-side composition cost.

Rejected. Reasons:

- **Different LLM families use different RoPE variants** (Llama-3
  split-half vs GPT-NeoX interleaved vs no-RoPE), so one fused kernel
  would either duplicate per-variant (3+ symbols per backend = ~9+
  hand-written kernels) or hard-code Llama. Either gates the perf
  improvement behind specific architectures, which is the
  general-purpose-library anti-pattern.

- **The fusion targets aren't attention-specific**. RoPE is a layer
  (we already have `Layer/RoPE.idr`); RmsNorm is a layer; SwiGLU is
  a layer. They get composed into many architectures (transformer
  variants, future state-space models, mixed designs). Keeping each
  as its own primitive lets a hypothetical "Mamba + Llama-attention
  hybrid" reuse the same pieces.

- **The right fix is closing the per-op gap, not hiding ops behind
  fused kernels**. PyTorch Python's `apply_rotary_pos_emb` uses the
  same broadcast pattern we landed (`q * cos.unsqueeze(1) +
  rotate_half(q) * sin.unsqueeze(1)`) — and runs in ~2 ms/op. Our
  rank-3 broadcast through `primMul` on torch-mps is ~10–26 ms/op.
  That's a backend-level performance issue in our wrapper (likely
  contiguity-tracking, MPSGraph cache invalidation per shape, or FFI
  marshalling overhead specific to rank-3 strided views), not a
  reason to switch off composable primitives.

The all-heads RoPE landing (`c09d374`) shipped exactly the composable
rank-3 form. It delivers **2.8× speedup on mlx-gpu** (lazy `mx::array`
graph rewards fewer ops directly: 45.5s → 16s) and a deterministic
**86% op-count drop on torch-mps** (18,410 → 2,634) while leaving the
torch-mps wall flat pending the per-op investigation. The op-count
metric is the principled one for a library aiming at general
composition; the per-op cost is a separable concern we attack in
isolation.

Library-perf principle: **prefer composable primitives + per-backend
work to close per-op gaps**, over fused kernels that buy speed by
constraining model architecture. The latter is appropriate for
production-only inference libraries (mlx-lm, vLLM); idris-ml's stance
matches PyTorch's research-oriented choice of staying composable and
investing in the underlying op dispatch.


## LayerLikeMixed bridge — type-safe mixed precision (#410)

Added 2026-06-01 alongside the type-safe mixed-precision work. The
question this section answers: when we extend `LayerLike` from a
single-dtype interface to a two-dtype-slot interface
(`paramDt`/`computeDt`) so layers like `LinearMixed F32 BF16` and
the future `BitLinear Ternary BF16` express the dtype split in their
type, how do the 15 existing single-dtype layers participate without
a system-wide rewrite?

### The interface split, not the system migration

`LayerLikeMixed` lives parallel to `LayerLike` in
`packages/idris-ml/src/Layer/MixedCore.idr`. Its method signatures
carry both `pDt` and `cDt` slots; the input/output tensors are in
`cDt`; the layer state is over `(pDt, cDt)`. Layers that genuinely
need different param/activation dtypes (`LinearMixed`, `BitLinear`)
implement `LayerLikeMixed` directly. Layers that don't (`Linear`,
`Activation`, `LayerNorm`, …) keep their existing `LayerLike`
instances and *bridge* into the mixed world via an `AsMixed` wrapper.

This is deliberately NOT a system-wide migration of `LayerLike` to
two slots. Two reasons:

1. **Most layers don't need it**. An `Activation` layer takes a
   tensor of some dtype and returns a tensor of the same dtype — no
   weight storage, no dtype split. Forcing the activation layer to
   carry a second dtype slot is interface bloat for zero semantic
   gain.
2. **The bridge is zero-cost**. `AsMixed` is a wrapper around the
   existing `AnyLayer` existential; its `LayerLikeMixed` instance
   delegates each method back to the underlying `LayerLike` after
   pattern-matching into `MkAnyLayer`. No runtime overhead; existing
   layers compose into `NetworkMixed` chains via `liftAnyLayer` /
   `liftNetwork` without code changes.

### Why `AsMixed` wraps `AnyLayer`, not a type-level lambda

The natural-looking design — a type-level lambda instance —
**doesn't work in Idris-2**:

```idris
-- ❌ This refuses to elaborate.
LayerLike l => LayerLikeMixed (\i, o, d, _, cDt, g => l i o d cDt g)
```

Idris-2 can't propagate the `(0 _ : Device)` / `(0 _ : DType)` /
`(0 _ : GradMode)` multiplicity-annotated erasure annotations
through the lambda's argument types when the constructor body
applies them. Multiple attempts (data-type parameter with `(0 l :
...)`, explicit-multiplicity implicits on the constructor, named
auto-implicits) all failed at the unification layer with `Mismatch
between: Type -> Type -> GradMode -> Type and (0 _ : Device) ->
(0 _ : DType) -> (0 _ : GradMode) -> Type`.

The working design wraps a concrete data type:

```idris
data AsMixed : Nat -> Nat -> (0 _ : Device) ->
               (0 _ : DType) -> (0 _ : DType) -> (0 _ : GradMode) ->
               Type where
  MkAsMixed : AnyLayer i o d dt g -> AsMixed i o d dt dt g
```

`AsMixed` is concrete (no higher-order type parameter), and its
constructor `MkAsMixed` only inhabits the diagonal `pDt = cDt = dt`.
Pattern matching `MkAsMixed (MkAnyLayer l @{dict} layer)` recovers
the inner `LayerLike` instance dict and the underlying layer; the
`LayerLikeMixed AsMixed` instance delegates each method to the
inner `applyVar` / `freezeLayer` / etc.

**Take-away rule**: when you need a layer-kind-to-layer-kind bridge
in Idris-2, default to a concrete wrapper that passes the
higher-order type through an existing existential, not a type-level
lambda instance. The lambda form looks clean but loses the
multiplicity annotations the constructor needs.

### Named auto-implicits + `@{%search}` slot-pinning

A second-order problem: `LayerLikeMixed`'s `applyVarMixed` has both
`{auto rdtP : RuntimeDType pDt}` and `{auto rdtC : RuntimeDType
cDt}`. The `AsMixed` bridge collapses `pDt = cDt`, so both dicts
type-match the inner `LayerLike.applyVar` call's single `RuntimeDType
dt` constraint — and Idris-2's typeclass resolver can't pick one of
the two. Same problem for `Compatible d pDt` vs `Compatible d cDt`.

Fix: name the auto-implicits in the interface signature, then pass
them explicitly at the call site, position-pinning the others with
`@{%search}` (the "do normal auto-resolution here" marker):

```idris
applyVarMixed {rdtC} {cmpC} (MkAsMixed (MkAnyLayer l @{dict} layer)) input = do
  (layer', out) <- applyVar @{dict} @{%search} @{%search} @{rdtC}
                                    @{%search} @{cmpC} layer input
  ...
```

This pins slot 4 (`RuntimeDType`) to `rdtC` and slot 6 (`Compatible`)
to `cmpC`, while letting `UserDeviceTraining`, `UserDeviceCore`,
`Linked` auto-resolve. The pattern recurs anywhere a typeclass
collapses two type parameters that the inner call disambiguates by
type.

### Why static typing over runtime autocast

PyTorch's `torch.autocast(dtype=bf16)` is a thread-local context
manager + monkey-patched op dispatch — efficient, but it gives up
the property "the tensor's type tells you its dtype." idris-ml takes
the opposite stance: the dtype shows up in `Tensor [..] d dt g`'s
fourth type parameter, mixed-precision layers carry both
`paramDt` and `computeDt` in their type, and the lossy cast (F32
master → BF16 compute) is visible inside the layer's forward as a
`tcastUnsafe` call.

The payoff is structurally stronger than PyTorch autocast: PyTorch
silently casts mid-graph in either direction; idris-ml refuses
silent lossy casts (`LossyDirectionRejected.idr` neg-gate confirms
F32 → BF16 doesn't type-check via `LosslessTo`) and forces the
lossy edges to be code-visible.

Cross-references: `packages/idris-ml/src/Layer/MixedCore.idr`
(`LayerLikeMixed`, `AsMixed`, `NetworkMixed`, `liftAnyLayer` /
`liftNetwork`); `packages/idris-ml/src/Layer/LinearMixed.idr` (first
concrete user); `packages/idris-ml/src/GradScaler.idr` (the IORef-
based state machine that pairs with `LinearMixed` in
`epochVarMixed`); `docs/develop/dtype-parameter.md` "FloatPrecision +
LosslessTo" section.


## Per-backend ternary storage — BitNet b1.58 (#411)

Added 2026-06-01 with the BitLinear forward kernel work. The
question this section answers: BitNet b1.58 ternary weights live in
the set `{-1, 0, +1}` and need only 2 bits per value. The Idris-side
dtype slot is uniformly `Ternary` (dtag 25, shipped under #411 B1).
But the three backends have very different storage stories — what
shape does "a Ternary tensor" actually take in memory on each?

### The decision

| Backend | Physical storage | Bits/value | Logical shape | Tag carried |
|---|---|---|---|---|
| tape   | packed 2-bit, four values per byte (`(i+3)/4 * o` bytes for `[o, i]`) | 2 | `[o, i]` | `dtype_tag = DT_TERNARY` |
| torch  | int8 tensor with values in {-1, 0, +1} | 8 | `[o, i]` | C-wrapper tag `DT_TERNARY`; underlying `at::Tensor` is `at::ScalarType::Char` |
| mlx    | int8 tensor with values in {-1, 0, +1} | 8 | `[o, i]` | C-wrapper tag `DT_TERNARY`; underlying `mlx::array` is `int8` |

The Idris type system sees `Tensor [o, i] d Ternary NoGrad` on every
backend. The 4× storage difference between tape and the others is
invisible at the type level — it's purely a backend implementation
choice, mirrored by the existing pattern where BF16 / F16 on tape
are F64-lingua-franca stored (8 bytes/value) while on torch/mlx
they're native 2-byte storage.

### Why asymmetric storage

The clean alternative — packed 2-bit on **all three** backends — has
significant engineering cost on torch and mlx, which we explicitly
chose not to pay:

1. **No native sub-byte dtype**. `at::Tensor` and `mlx::core::array`
   have no 2-bit storage class. Packed-2-bit ternary on those
   backends would require a parallel wrapper that records logical
   shape + ternary-tag externally, with the raw bytes living in a
   1D uint8 framework tensor. Every cross-cutting op
   (`tensor_to_doubles`, `tensor_clone`, `tensor_dtype_name`,
   safetensors save/load, refcount lifecycle) would need its own
   ternary branch.
2. **No framework GEMM**. `at::matmul` won't run on a custom-bytes
   tensor; the bitlinear forward kernel on torch/mlx would become a
   hand-rolled loop instead of dispatching through libtorch's BLAS
   / MKL / cuBLAS. For inference on real BitNet models that costs
   real wall.
3. **No framework autograd**. Custom-bytes tensors are opaque to
   torch/mlx autograd. Acceptable today (BitLinear weight is
   `NoGrad`) but would block future STE-aware training on those
   backends.

Int8 storage on torch/mlx sidesteps every one of those: int8 IS a
native framework dtype, `tensor.to(scale.dtype())` dequantises with
one framework call, `at::matmul` works, and autograd through the
dequant cast is free.

### Why tape stays 2-bit

Tape's `Tensor` is a hand-rolled C struct we own end-to-end — no
external framework constrains the storage. The bitlinear kernel
already needs hand-written matmul (tape has no autograd-tracing
framework either), so decoding 2-bit codes on the inner loop is the
natural shape. And tape is the only backend that historically
absorbs novel storage paths cheaply (BF16 / F16 via F64-rounding,
Conv's im2col, NTM's circular shift). Sub-byte arena layout fits
that pattern.

### The 4× memory cost on torch/mlx

A 700M-param BitNet model weighs:

- Tape: ~175 MB (2 bits/param + scale tensors)
- Torch / mlx: ~700 MB (8 bits/param + scale tensors)

Both fit a 24 GB box comfortably. The penalty becomes more visible
for larger BitNet variants (3B param = ~3 GB on torch/mlx vs
~750 MB on tape). When that crosses into a real ceiling, the
follow-up is a torch/mlx packed-storage path — same byte format as
tape, dequant in the kernel — filed under #411 follow-ups.

### Why this differs from the BF16 / F16 lingua-franca on tape

The dtype-lingua-franca pattern on tape (BF16/F16 stored as F64
doubles, rounded into the target dtype's precision) is asymmetric
in the *opposite* direction: tape uses *more* bytes than its
intrinsic dtype demands; torch/mlx use the native byte width. For
ternary the asymmetry flips: tape uses *fewer* bytes; torch/mlx use
more. Both asymmetries live in the same shape — "the dtype tag is
authoritative; physical storage is per-backend" — but the
arithmetic is reversed.

The pattern this entrenches: each new sub-native dtype (the next
candidates are NF4, FP4, MX) faces the same trichotomy. Default to
matching the on-disk format on the backend that owns the storage
end-to-end (tape today, a future "raw-bytes" backend tomorrow); use
the nearest framework-native dtype on torch/mlx if engineering cost
demands it; document the asymmetry in a row here.

Cross-references:
`packages/idris-ml/src/DType/Core.idr` (`Ternary`, `Binary` types
+ `dtypeBytes = 0` sub-byte sentinel);
`packages/backends/backend_tape/tensor.h` (`DT_TERNARY` enum,
sub-byte storage docstring);
`packages/backends/shared_utils.{h,c}`
(`ternary_pack` / `ternary_unpack`);
`packages/pytorch/torch_ref/models/bitlinear.py` (PyTorch oracle
+ `absmean_ternary_quant`);
TODO #411 + the upcoming per-backend `tensor_bitlinear_fwd`
kernel commits.
