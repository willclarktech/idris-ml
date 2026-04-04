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

**MLX backend** (`BACKEND=mlx`, planned): Links against Apple's [mlx-c](https://github.com/ml-explore/mlx-c). Metal GPU, lazy evaluation, ~50MB dependency. Target: Apple Silicon ML workloads.

All three compile to `libidrisml.dylib` — same name, same API. The Idris code is identical across backends. `Makefile` selects: `make backend BACKEND=tape|torch|mlx`.

### Build-time backend selection

The `backend_supports_tensor_params()` C function returns 1 (torch) or 0 (tape) to let Idris code adapt. When 1: layer `nameLayer` creates consolidated weight tensors (`[o,i]` for Linear, `[4*o,i]` for LSTM) with scalar views sharing storage. The tensor-level forward path (`tensor_mv`) operates on consolidated tensors directly. When 0: layer `nameLayer` creates per-scalar named Variables. The scalar fallback forward path stacks scalars → C op → unstacks.

This is a runtime-queryable capability, not a compile-time flag, so the same compiled Idris binary works with either backend.

### Performance trade-offs

| | Tape | libtorch | Old Scheme tape |
|---|---|---|---|
| Supervised (1000 ep) | ~5.8s | ~7.2s | 90ms |
| RNN (1000 ep) | ~16.9s | ~28.6s | 400ms |
| NTM-copy (100 ep) | NaN (fixing) | ~19.3s | ~14.7s |
| Peak RSS (Supervised) | 68MB | 453MB | 42MB |
| Per-scalar op cost | 2-3 malloc | tensor alloc + graph node | 1 memory write |
| GPU support | No | CUDA/MPS | No |
| Dependencies | 0 | ~2GB | 0 |

The old Scheme tape was fastest because `foreign-set!` into pre-allocated arrays is essentially a memory write — no allocation, no FFI crossing. The new tape backend allocates a `Tensor` struct per op. Closing this gap requires arena allocation and reducing the stack/unstack overhead.

### NTM numerical stability: epsilon clamping

The NTM addressing pipeline computes `pow(weights, gamma) / sum(pow(weights, gamma))` for focus/sharpening. After circular shift convolution, weights can become negative or zero. `pow(negative, gamma)` = NaN when gamma is non-integer.

The fused C `tensor_ntm_read_head` (used by torch backend) clamps to `1e-10` before pow (line 594 in backend_torch.cpp). The scalar Variable-level `focusVar` (used by tape backend) did NOT have this clamping, causing NaN after the first optimizer step when gradients pushed weights negative.

Fix: add `tensor_clamp_min` to `backend.h` and use in `focusVar` before `prim__pow`. Both backends must implement this. The torch backend already had the clamp in its fused path but needs it in the standalone `tensor_clamp_min` function too.

The old Scheme backend avoided this issue through buffer-passing: addressing weights stayed in C buffers where the NTM-specific C ops applied the clamping internally. The new architecture's scalar path bypasses those C ops.

**Update (2026-04-02)**: Adding `tensor_clamp_min` to `focusVar` and `shiftVar` prevents forward-pass NaN. However, backward-pass NaN persists at NTM scale (128 memory slots × 20 width). Root cause: multiple compound ops (`tensor_cosine_similarity`, `tensor_conv1d_circular`) needed tape entries for backward. Added `OP_COSINE_SIM` and `OP_CONV1D_CIRC` backward rules. Also fixed multi-dim backward for `OP_POW` and `OP_DIV`, and `tensor_unsqueeze` to use `tensor_reshape` for tape continuity. Individual ops pass C tests in isolation. **The full NTM addressing chain also passes at N=128, W=20 in a standalone C test** (test_ntm_grad.c — 5 epochs with RMSprop, all gradients finite). **Root cause found (2026-04-02)**: `tensor_mul_scalar` and `tensor_add_scalar` only read `data[0]`, treating vector inputs as scalars. The NTM's `interpolateVar` passes N-element stacked vectors through `prim__mulScalar`. The old scalar-only code created 1-element tensors; downstream `conv1d` read out-of-bounds memory → NaN. Fix: both functions now handle multi-element tensors. NTM trains correctly on the tape backend (loss=0.6935, matching libtorch).

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

Also removed fused NTM C ops (`tensor_ntm_read_head`, `tensor_ntm_interp_write`) from the tensor path since they lack backward rules in the tape backend. The individual Variable-level addressing ops (which use `OP_COSINE_SIM`, `OP_SOFTMAX`, `OP_CONV1D_CIRC`, `OP_POW`, `OP_VECMAT` etc.) are used instead.

Result: LSTM converges (0.714 → 0.048 in 5k epochs, ~1.5 min). NTM-copy with short sequences (1-5) reaches 83.5% accuracy in 2k epochs (~40 min). NTM per-epoch is ~1.3s (short) / ~3.5s (full) — ~10-30x slower than the old fused-C backend. Performance optimization is the next priority.
