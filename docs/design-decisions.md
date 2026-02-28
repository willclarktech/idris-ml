# Design Decisions

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

## Bounded NTM head parameters

The NTM focus sharpening parameter γ (gamma) must be bounded. The original formulation `softplus(x) + 1` gives γ ∈ [1, ∞), which causes vanishing gradients for non-dominant memory positions: if a weight is 0.1, then `0.1^γ` for large γ becomes negligible (0.1^20 = 1e-20), and the gradient through `w^γ` includes a factor of `γ * w^(γ-1)` which vanishes.

The fix uses `1 + 4 * sigmoid(x)` to bound γ ∈ [1, 5]. At the upper bound, `0.1^5 = 1e-5` — small but with survivable gradients. This lets the NTM learn to sharpen attention without permanently losing gradient signal for secondary memory positions.

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

## NTM training profile

Profiling a single NTM epoch (`src/Example/Profile.idr`) reveals the time split across four phases:

| Phase | Time (ms) | Share |
|-------|-----------|-------|
| Forward (calculateLossRecurrentVar) | ~35 | 45% |
| Backward (collectGrads) | ~41 | 53% |
| Optimizer (adam step) | ~1 | 1% |
| Buffer sync | ~0.2 | <1% |

**Tape size: 214,608 entries** per epoch — each must be visited during the backward scan. 891 named parameters.

Key implications:
- **Backward dominates**: the reverse tape scan over 214K entries is the single most expensive operation. Optimizing the forward pass alone yields at most ~45% of possible gains.
- **Weight packing is negligible**: persistent weight buffers (phase 3) targeted sync/packing which together are <1.5% of epoch time, explaining the modest 5-12% speedup.
- **Tape size is 5x expected**: the NTM's scalar head computations (read head, write head, memory addressing) generate far more intermediate tape entries than the controller's BLAS-backed matvec ops. Reducing tape entries per scalar op or batching head computations into C would have the largest impact.

Build and run: `idris2 --source-dir src -p contrib -o profile src/Example/Profile.idr && ./build/exec/profile`

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

## NTM diagnostic analysis

The NTM copy task achieves high training accuracy but generalizes poorly to held-out test sequences. The diagnostic analysis module (`Debug.idr`) provides quantitative summary metrics and train/test comparison to identify failure modes.

**String-based parsing roundtrip**: debug entries store field values as formatted strings (via `showVec`, `showF`, `showMat`). The analysis functions parse these back to `List Double` via `parseVec`/`parseScalar`/`parseMat`. This avoids changing the `DebugEntry` type or carrying structured data through the debug forward pass. The parsing is lossy (4 decimal places from `showF`) but sufficient for diagnostic purposes.

**Phase-split metrics**: NTM sequences have two phases — input (write) and output (read). The `computeSummary` function splits all per-timestep metrics at `seqLen` to report separate averages for each phase. This is critical because the model should behave differently in each phase (e.g., write during input, read during output).

**Key diagnostic metrics**:
- **Gate g** (0=location, 1=content): the interpolation gate between content-based and location-based addressing. If g is low during training but high during testing, the model is falling back to content addressing on novel patterns (memorization).
- **Entropy/peak mass**: addressing weight distribution focus. Low entropy and high peak mass indicate sharp, focused addressing. Diffuse addressing (high entropy) suggests the model hasn't learned to target specific slots.
- **Monotonicity**: whether the argmax of addressing weights advances sequentially through memory slots during the relevant phase (write during input, read during output). Sequential slot access is the expected behavior for a copy task.
- **Slots used**: number of memory rows with norm > 0.01 at the end of the input phase. If slots used is much less than sequence length, the model is collapsing memory.

**Addressing lag**: the debug entry at timestep t captures addressing weights from *before* the current step (the previous head state) but g/β/γ parameters *for* the current step (computed from the controller output). The addressing weights at timestep t thus show the result of timestep t-1's computation. The final addressing weights (after the last step) are in the returned model state, not in any debug snapshot.

**Interpretation guide**:

| Observation | Diagnosis | Next step |
|---|---|---|
| Train g low, test g high | Memorization — content fallback on novel data | Add curriculum learning or location bias |
| Both g high | Never learned location addressing | Architectural change needed |
| g low, monotonic=NO | Shift broken — wrong direction | Check shift distribution learning |
| Slots used << seq length | Memory collapse | Investigate initialization / capacity |

## 3-element shift kernel

The original NTM implementation used an n-element shift vector (one per memory slot), requiring the model to learn "shift by exactly 1" as one of n equally likely options with diluted gradient signal. The original paper ([Graves et al. 2014](https://arxiv.org/abs/1410.5401)) specifies a small shift kernel (typically 3 for {-1, 0, +1}).

`ShiftKernelSize = 3` decouples the shift mechanism from the number of memory slots. The shift is implemented as a 3-element circular convolution: `w'[i] = sl * aw[i+1] + ss * aw[i] + sr * aw[i-1]`, where `(sl, ss, sr) = softmax(kernel)`. This means:
- `sr` high → addressing shifts right (slot 0→1→2), correct for sequential write
- `sl` high → addressing shifts left
- `ss` high → stay on current slot

This reduces the learning problem from "pick 1 of n directions" to "pick 1 of 3 directions" — a much simpler optimization with 3x stronger gradient signal per shift option.

Impact on dimensions:
- `ReadHeadInputWidth n w` changes from `(w + n) + 3` to `(w + ShiftKernelSize) + 3` — now independent of `n`
- Controller output size decreases (e.g., n=10, w=3: from 41 to 27), reducing total parameters

**Result**: The shift kernel change alone did not fix generalization. Across four runs (lr=0.001/0.003/0.005, seeds 123456/42), the optimizer consistently converges to content-based addressing (write g ~0.9 during output) rather than learning sequential location-based shifting. The 3-element kernel is architecturally correct (matches the paper) but insufficient — the content addressing path is a stronger local attractor than the shift path.

## Hot-start addressing on slot 0

Read and write head addressing weights are initialized to focus on slot 0 (`[1, 0, 0, ...]`) instead of the previous uniform distribution (`[1/n, 1/n, ...]`). With a clear starting position, the model only needs to learn "shift right by 1 each step" for sequential access — a clean gradient signal compared to discovering both the starting position and the shift direction simultaneously.

## NTM stability alignment with reference implementations

Aligned with reference implementations to address generalization failures:

| Change | Before | After | Source |
|--------|--------|-------|--------|
| Memory init | random [-0.1, 0.1] | constant 1e-6 | Collier & Beel: 3.5x faster convergence |
| Grad clip norm | 5.0 | 50.0 | Collier & Beel default; 5.0 too aggressive |
| Controller output | unbounded | clamped [-20, 20] | Collier & Beel: prevents extreme head params |
| Training data | 13 fixed sequences | random each chunk | All reference impls use random data |
| Curriculum | none | 3 stages (len 1-3, 1-5, 1-8) | ajithcodesit: FFN "did NOT converge" without it |

**Reference implementations** (all achieve near-perfect copy task performance):
- [loudinthecloud/pytorch-ntm](https://github.com/loudinthecloud/pytorch-ntm) — most-starred PyTorch NTM, generalizes to length 80
- [Collier & Beel 2018](https://arxiv.org/abs/1807.08518) — Best Paper ICANN 2018, controlled stability experiments
- [ajithcodesit/Neural_Turing_Machine](https://github.com/ajithcodesit/Neural_Turing_Machine) — feedforward controller (like ours), requires curriculum

**Constant memory init**: `ntmLayer` initializes memory to `1e-6` via `pure (fromDouble 1.0e-6)`. This removed the `Random ty` and `Neg ty` constraints from `ntmLayer` since random generation is no longer needed. Collier & Beel's controlled experiment showed this converges 3.5x faster than random init.

**Controller output clipping**: `applyLayerVar` clamps the raw controller output to [-20, 20] using `clampVar` (straight-through gradient: detached constant when clamped, passthrough when in bounds). This prevents extreme head parameters (β, g, γ, erase/add vectors) from destabilizing training.

**Curriculum learning**: Three stages with loss thresholds (0.15, 0.10, 0.0). Each stage generates fresh random data every 100 epochs via `Generate.randomBatchVect`. The model from each stage carries over to the next with its optimizer state. This prevents the model from memorizing fixed sequences and forces it to learn the general copy algorithm.

**Random data generation**: The `Generate` module provides a port/adapter pattern — `SequenceTask` (port) defines the interface, `copyTask` (adapter) implements copy-task-specific generation. `randomBatchVect` generates typed `Vect n` batches. `randomSymbols` generates non-blank symbols (values 1..w-1). Data is regenerated every 100 training epochs to prevent overfitting.

## Tanh memory bounding

After each memory write, all memory values are clamped to [-1, 1] via `tanhBound` (Collier & Beel recommendation). Without bounding, memory values can drift unboundedly over long sequences — the write head's add vector (tanh-bounded to [-1, 1]) accumulates across timesteps while the erase vector (sigmoid, [0, 1]) only partially clears previous values. Unbounded memory causes:
- Content addressing instability: cosine similarity becomes unreliable when magnitudes vary wildly
- Gradient scale mismatch between large and small memory values

The `tanhBound` helper uses `2 * sigmoid(2x) - 1` (mathematically equivalent to tanh) expressed with `Neg`, `Fractional`, and `Floating` constraints, avoiding a `FromDouble` dependency. Applied via `map tanhBound` on the full memory matrix after `forwardWriteHead` in all three forward paths (generic, Variable, debug).

## Learned initial addressing

Read head addressing weights, write head addressing weights, and the initial read head output vector are named as learnable parameters (via `nameParams`). Previously these were fixed at `[1, 0, 0, ...]` (addressing) and `[0, 0, 0]` (read output) — the model had to learn sequential access starting from a hardcoded position.

With learned initial addressing, the model can discover optimal starting positions through backpropagation. The existing `Functor` instances on `ReadHead`/`WriteHead` already propagate `applyDeltas` through these fields — naming is all that was needed to make them visible to gradient collection.

New named parameters (for n=10 memory slots, w=3 width):
- `rAddr0..rAddr9`: read head initial addressing weights (10 params)
- `wAddr0..wAddr9`: write head initial addressing weights (10 params)
- `rOut0..rOut2`: initial read head output vector (3 params)

Total: 23 new learnable parameters, bringing the total from ~891 to ~914.

## Curriculum module extraction

Curriculum training (multi-stage training with data regeneration, stage advancement thresholds, and two-level early stopping) was extracted from `Example/NtmCopy.idr` into a reusable `Curriculum` module.

**Motivation**: curriculum learning is a standard ML technique not specific to NTMs. The inline implementation was hardcoded to `Network W [W] W Variable` and `nllLoss`, making it unusable for other architectures.

**Parameterization**: the `Stage` record replaces the previous `CurrStage` by dropping task-specific fields (`minLen`, `maxLen`) in favor of an `IO`-based `generate` function. This lets each stage encapsulate its own data generation strategy — the module doesn't know about copy tasks, sequence lengths, or task-specific parameters. The loss function and chunk size (data refresh interval) are also parameters.

**API**: `runCurriculum` takes a list of stages, an optimizer factory, a schedule, and training hyperparameters, returning the trained model, optimizer state, and total epochs completed. This is the same interface as the inline version but generic over `Network i hs o Variable`.

## Two NTM examples: copy and associative recall

The NTM has two addressing mechanisms: location-based (circular shift) and content-based (cosine similarity). A single example cannot validate both.

**NtmCopy** (location-based): the copy task writes symbols sequentially then reads them back in the same order. The model learns shift-right-by-one each timestep — pure location addressing. Content addressing is a stronger local attractor but not required.

**NtmAssociativeRecall** (content-based): K key-value pairs are stored, then queried in shuffled order. The model must look up each query key by content similarity to retrieve the associated value. Sequential shifting cannot solve this because queries arrive in random order.

### Task encoding (W=8)

With W=8, there are 7 non-blank symbols (1-7), supporting up to K=7 key-value pairs. The encoding uses one-hot vectors of width W:

- **Store phase** (2K steps): `k1 v1 k2 v2 ... kK vK` — keys are distinct non-blank symbols, values are random non-blank symbols
- **Delimiter** (1 step): blank
- **Query phase** (2K steps): `q1 blank q2 blank ... qK blank` — queries are keys in shuffled order

Output is blank everywhere except on blank-input timesteps in the query phase, where the correct value appears. This "answer on blank" pattern matches the copy task convention.

### Curriculum

Four stages: K=2 (threshold 0.12), K=3 (0.10), K=3-4 (0.08), K=4-5 (0.0). The wider W=8 alphabet enables K=5+ pairs, forcing genuine multi-slot content-based addressing — see "Breaking the degenerate one-slot minimum" below.

## Breaking the degenerate one-slot minimum (W=4 → W=8)

A systematic sweep (36 configs) found a hard ceiling at ~91.5% test accuracy on the K=3 associative recall task with W=4. Diagnostics revealed degenerate addressing: all writes collapsed to memory slot 0, reads used fixed slots 9/8. The model never learned genuine content-based addressing.

**Why 91% is the ceiling with W=4:** With K=3 pairs and 13 timesteps, 10 are blanks (always correct = 76.9% floor). The model gets ~2/3 value predictions right by memorizing the last-written pair from slot 0. Only 972 unique sequences exist at K=3 — small enough to partially memorize.

**Why increasing K breaks the one-slot strategy:** With K=5 pairs, slot 0 can only retain ~1 pair after 5 sequential overwrites, forcing the model to actually use multiple memory slots and genuine content-based retrieval to achieve high accuracy.

**Changes:**
- **W=8** (was 4): 7 non-blank symbols, K up to 7 pairs
- **N=16** (was 10): more memory slots for higher K
- **H=40** (was 20): controller output grows from 32 to 52; needs more hidden capacity
- **4 curriculum stages** (was 2): K=2 → K=3 → K=3-4 → K=4-5, gradual progression
- **lr=0.001** (was 0.003): larger model benefits from lower base LR; one-cycle peaks at 0.025
- **maxNorm=10.0** (was 5.0): more gradient headroom for larger model
- **epochs=10000** (was 6000): 4 stages need more budget
- **patience=800** (was 500): harder task needs more patience

Dimension impact (computed from Layer.idr type functions):
- NtmInputWidth: 8→16, NtmOutputWidth: 32→52
- Controller: 16→40→52 (was 8→20→32)

No core library changes needed — the type system handles dimension changes automatically via `NtmInputWidth`, `NtmOutputWidth`, and dependent types in `Layer`/`Network`.

## Unified NTM head operations via NormalizationFunction parameter

`forwardReadHead`/`forwardWriteHead` in Memory.idr were duplicated as `forwardReadHeadVar`/`forwardWriteHeadVar` in Layer.idr, differing only in which softmax function was called (`softmax` vs `softmaxVar`). The Variable versions also redefined local copies of `sig`, `softplus`, `interpolate`, `focus`, `readOp`, `eraseMemory`, `addMemory`, `writeOp` — ~80 lines of pure duplication.

The fix parameterizes on a `NormalizationFunction ty` (= `{n : Nat} -> Vector n ty -> Vector n ty`), passed to `forwardReadHead`/`forwardWriteHead` for content addressing softmax and shift softmax. The generic `applyLayer` path passes `softmax`, while the Variable path passes `softmaxVar`. Helper functions (`sig`, `softplus`, `interpolate`, `focus`, etc.) are exported from Memory.idr rather than duplicated.

After the C-backed NTM memory ops were added (see below), the Variable path was split again into dedicated `forwardReadHeadVar`/`forwardWriteHeadVar` functions in Layer.idr that call the C kernels directly. The generic Double path still uses the unified parameterized functions from Memory.idr.

## C-backed NTM memory operations

With N=128 memory slots and W=8 width, each NTM timestep created ~12,500 scalar tape entries for content addressing, read, and memory write. These head computations dominated both forward (~45%) and backward (~53%) pass time.

Three new C kernels batch these into single tape entries:

**Batch cosine similarity** (`batchCosineSimilarityVar`): computes `scores[i] = beta * cosine_similarity(key, memory[i])` for all N rows in one C call. The `BatchCosSimMeta` struct saves dot products, row norms, and key norm for backward. Backward propagates gradients to memory rows, key vector, and beta scalar using the analytic Jacobian: `d cos(a,b)/d a_k = (b_k - (a·b/|a|²) * a_k) / (|a| * |b|)`.

**Read operation** (`readOpVar`): computes `output[j] = sum_i(weights[i] * memory[i][j])` — a transpose-matvec. Backward: `d_weights[i] = sum_j(dy[j] * mem[i][j])`, `d_mem[i][j] = weights[i] * dy[j]`.

**Write operation** (`writeOpVar`): fused erase+add in one pass: `out[i][j] = mem[i][j] * (1 - w[i] * e[j]) + w[i] * a[j]`. Backward propagates to memory, weights, erase, and add vectors with analytic gradients.

| Operation | Tape entries before (N=128, W=8) | After |
|-----------|----------------------------------|-------|
| Content addressing (cosine sim + beta) | ~6,400 | 1 |
| readOp (weighted row sum) | ~2,048 | 1 |
| eraseMemory + addMemory | ~4,096 | 2 |
| **Total per read+write head** | **~12,500** | **4** |

Benchmark (`src/Example/Bench.idr`, seed 123456, N=10 W=3):

| Version | NTM (100 ep) | Speedup |
|---------|-------------|---------|
| Scalar head ops | 4,751 ms | — |
| C-backed head ops | 2,700 ms | 1.76x |

The speedup is less than the tape reduction ratio because (a) the benchmark uses small N=10 where per-entry overhead is lower, (b) the scalar ops for interpolation, shift, and focus remain unchanged, and (c) forward packing and backward unpacking have their own costs. Larger N (e.g., N=128 for associative recall) should see proportionally greater benefit.

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
