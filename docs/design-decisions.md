# Design Decisions

See [ntm.md](ntm.md) for NTM-specific design decisions (head parameters, memory operations, addressing, diagnostics, convergence).

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

The `LstmLayer` constructor implements a standard LSTM cell (Hochreiter & Schmidhuber 1997) with learned hidden and cell states:

```
combined = W_ih * x + W_hh * h + bias      -- (4*hidden,)
(i_gate, f_gate, g_gate, o_gate) = split4 combined
i = sigmoid(i_gate), f = sigmoid(f_gate), g = tanh(g_gate), o = sigmoid(o_gate)
c' = f * c + i * g
h' = o * tanh(c')
```

**Forget gate bias**: biases are initialized to zero except the forget gate bias, which is set to 1.0 (Jozefowicz et al. 2015). This ensures the forget gate starts nearly open, preventing early vanishing of cell state. The bias vector is structured as `[i_bias, f_bias, g_bias, o_bias]` with f_bias = 1.0.

**Cell state extraction**: `extractCellState` pattern-matches on `LstmLayer` to return the cell state directly. This is used by `NtmLayer` to feed cell state into the read/write head FCs (matching the PyTorch reference architecture where head parameters come from the LSTM cell state, not the hidden state).

**Weight initialization fan dimensions**: `lstmLayerWith` passes `(fanIn, fanOut)` to the init strategy where `fanOut = 4 * hiddenSize` (not `hiddenSize`), because the actual weight matrices are `(4*hidden, input)` and `(4*hidden, hidden)`. Using `hiddenSize` as fan-out produces Xavier variance `2/(i+o)` instead of the correct `2/(i+4*o)`, making weights ~2.5x too large and causing exploding gates that prevent convergence.

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

## Periodic forced GC for long NTM training

Running NTM training for 50K+ epochs causes OOM kills (SIGKILL/exit 137) at ~3000 epochs on macOS. Root cause: each forward pass creates tens of thousands of temporary Scheme Variable records and intermediate allocations on the Chez Scheme heap. After `collectGradsDense` resets the tape, these become garbage. However, Chez Scheme's generational GC doesn't collect aggressively enough — temporary objects promoted to older generations accumulate faster than major collections run. Additionally, ~160MB of `foreign-alloc` tape arrays are invisible to the GC, so it underestimates actual memory pressure.

Fix: call Chez Scheme's `(collect)` (full GC) every 10 epochs via the `forceGC` FFI wrapper in Variable.idr. Cost: `(collect)` on a ~50MB live set takes ~10-50ms; every 10 epochs at ~247ms/epoch adds <2% overhead.

FFI note: `%World` is erased in Chez Scheme's PrimIO calling convention, so the foreign lambda must take 0 arguments: `(lambda () (collect) 0)`, not `(lambda (w) (collect) ...)`. Using a 1-arg lambda causes "incorrect argument count" at runtime.
