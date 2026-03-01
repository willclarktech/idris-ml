# NTM Design Decisions

NTM-specific design decisions for the Neural Turing Machine implementation. For general autograd, optimizer, and infrastructure decisions, see [design-decisions.md](design-decisions.md).

## Bounded NTM head parameters

The NTM focus sharpening parameter γ (gamma) must be bounded. The original formulation `softplus(x) + 1` gives γ ∈ [1, ∞), which causes vanishing gradients for non-dominant memory positions: if a weight is 0.1, then `0.1^γ` for large γ becomes negligible (0.1^20 = 1e-20), and the gradient through `w^γ` includes a factor of `γ * w^(γ-1)` which vanishes.

The fix uses `1 + 4 * sigmoid(x)` to bound γ ∈ [1, 5]. At the upper bound, `0.1^5 = 1e-5` — small but with survivable gradients. This lets the NTM learn to sharpen attention without permanently losing gradient signal for secondary memory positions.

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

**Constant memory init**: `ntmLayer` initializes memory to `1e-6` via `pure (fromDouble 1.0e-6)`. This removed the `Random ty` and `Neg ty` constraints from `ntmLayer` since random generation is no longer needed. Collier & Beel's controlled experiment showed this converges 3.5x faster than random init.

**Controller output clipping**: `applyLayerVar` clamps the raw controller output to [-20, 20] using `clampVar` (straight-through gradient: detached constant when clamped, passthrough when in bounds). This prevents extreme head parameters (β, g, γ, erase/add vectors) from destabilizing training.

**Curriculum learning**: Three stages with loss thresholds (0.15, 0.10, 0.0). Each stage generates fresh random data every 100 epochs via `Generate.randomBatchVect`. The model from each stage carries over to the next with its optimizer state. This prevents the model from memorizing fixed sequences and forces it to learn the general copy algorithm.

**Random data generation**: The `Generate` module provides a port/adapter pattern — `SequenceTask` (port) defines the interface, `copyTask` (adapter) implements copy-task-specific generation. `randomBatchVect` generates typed `Vect n` batches. `randomSymbols` generates non-blank symbols (values 1..w-1). Data is regenerated every 100 training epochs to prevent overfitting.

## Tanh memory bounding

After each memory write, all memory values are clamped to [-1, 1] via `tanhBound` (Collier & Beel recommendation). Without bounding, memory values can drift unboundedly over long sequences — the write head's add vector (tanh-bounded to [-1, 1]) accumulates across timesteps while the erase vector (sigmoid, [0, 1]) only partially clears previous values. Unbounded memory causes:
- Content addressing instability: cosine similarity becomes unreliable when magnitudes vary wildly
- Gradient scale mismatch between large and small memory values

The `tanhBound` helper uses `2 * sigmoid(2x) - 1` (mathematically equivalent to tanh) expressed with `Neg`, `Fractional`, and `Floating` constraints, avoiding a `FromDouble` dependency. Applied via `map tanhBound` on the full memory matrix after `forwardWriteHead` in all three forward paths (generic, Variable, debug).

## Learned initial addressing (idris-ml only)

In idris-ml, read head addressing weights, write head addressing weights, and the initial read head output vector are named as learnable parameters (via `nameParams`). The model discovers optimal starting positions through backpropagation. The `Functor` instances on `ReadHead`/`WriteHead` propagate `applyDeltas` through these fields. After `applyDeltas`, `syncLayerBuffers` projects addressing weights onto the probability simplex via `projectWeights`.

New named parameters (for n=10 memory slots, w=3 width):
- `rAddr0..rAddr9`: read head initial addressing weights (10 params)
- `wAddr0..wAddr9`: write head initial addressing weights (10 params)
- `rOut0..rOut2`: initial read head output vector (3 params)

### PyTorch reference: non-learnable addressing

The PyTorch NTM implementation uses **non-learnable** initial addressing, matching the vlgiitr reference:
- Addressing weights reset to `torch.zeros(n)` each sequence (not `nn.Parameter`)
- Read head output initialized with `kaiming_uniform_` each sequence (not `nn.Parameter`)
- No `project_addressing()` post-step needed

This matches vlgiitr/ntm-pytorch, where addressing starts from zeros and is entirely determined by the controller's behavior — the controller must *learn* to address correctly through its own outputs rather than relying on a learned addressing prior. The previous learnable addressing approach caused the optimizer to push addressing toward degenerate solutions (e.g., always focus on slot 0), and the `project_addressing()` post-step interfered with gradient flow.

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

## LSTM controller for associative recall

The vanilla RNN controller (LinearRNNCell → tanh → Linear) completely fails to learn associative recall: 0% K=1 accuracy, all reads/writes stuck on slot 0, content_match_rate=0. The model falls into a degenerate one-slot attractor and never escapes.

**Why vanilla RNN fails**: associative recall requires remembering what was stored 4+ timesteps ago and generating matching content-addressing keys during the query phase. The RNN's single hidden state provides weak temporal credit assignment — gradients for the store phase must backpropagate through every intervening timestep's tanh squashing, decaying exponentially.

**Why LSTM fixes it**: the cell state provides a direct gradient highway through the forget gate. Information stored during the store phase can persist through the delimiter and into the query phase with minimal gradient degradation. The input/output gates learn when to write (store phase) and when to read (query phase) — a natural fit for the store→query phase transition.

**Softmax gradient scaling**: with N memory slots, the softmax gradient for a non-dominant position is O(1/N²). At N=128 this gives ~6e-5 per gradient step, vs ~4e-3 at N=16 (60x stronger). The LSTM's stronger temporal gradients are essential to compensate for this dilution at large N.

**Reference alignment**: Graves et al. 2014 uses LSTM for all tasks (copy, repeat copy, associative recall, priority sort). No reference NTM implementation uses vanilla RNN for associative recall. The copy task works with a feedforward controller because it only needs sequential shift (location-based addressing), not content-based retrieval across time.

**Implementation**: `LSTMController` wraps `nn.LSTMCell` + `nn.Linear`. LSTM's output gate already applies tanh, so no extra activation is needed (unlike RNNController which adds explicit tanh after the linear recurrence). Learnable initial states `h0`, `c0` as `nn.Parameter(torch.zeros(...))` match the RNNController pattern. The `NtmRecallConfig.controller` field selects between "lstm" (default) and "rnn".

### Reference-aligned PyTorch NTM recall

Comparison with two working reference implementations — [loudinthecloud/pytorch-ntm](https://github.com/loudinthecloud/pytorch-ntm) and [vlgiitr/ntm-pytorch](https://github.com/vlgiitr/ntm-pytorch) — revealed critical architectural and training differences. Both references solve associative recall and generalize to 20+ items.

**Key differences and fixes applied**:

| Change | Before (idris-ml aligned) | After (reference aligned) | Impact |
|--------|--------------------------|--------------------------|--------|
| Output computation | `logSoftmax(controller_output_slice)` | `logSoftmax(Linear(cat(controller_hidden, read_vector)))` | **CRITICAL** — output now has access to read result at current timestep |
| γ activation | `1 + 4*sigmoid → [1,5]` bounded | `1 + softplus → [1,∞)` unbounded | Allows sharper focusing when needed |
| Add vector | `2*sig(2x)-1` bounded [-1,1] | Raw linear (no activation) | Matches reference impls |
| Optimizer | Adam | RMSprop (lr=1e-4, momentum=0.9, alpha=0.95) | Reference default |
| Grad clipping | Global L2 norm 50 | Value clipping ±10 | Reference default |
| Batch size | 48 | 1 | Reference default |
| LR schedule | One-cycle (warmup + cosine) | Constant 1e-4 | Simpler, reference default |
| Curriculum | 3 stages (K=1,2,3) | None — direct K=2 training | Reference impls use no curriculum |

**Output architecture** (most critical): Both reference implementations compute the final output from the read vector at the current timestep: `output = sigmoid(Linear(cat(controller_output, read_vector)))`. Our original implementation returned a slice of the controller output, so the read vector only fed back as input at t+1. This forced an "answer on blank" delay and meant the output had no direct access to retrieved memory content.

The `NTMLayer` now supports `output_mode="read"` which adds an `output_fc` layer and computes `output_fc(cat(controller_hidden, read_output))`. The controller output size is reduced to just head parameters (no output slice), computed via `ntm_head_params_width(w)`.

**Configuration**: `NtmRecallConfig` defaults are now reference-aligned: `output_mode="read"`, `optimizer="rmsprop"`, `clip_mode="value"`, `batch_size=1`. The old idris-ml-aligned config is available via `output_mode="controller"`, `optimizer="adam"`, `clip_mode="norm"`.

Run with `make bench-convergence-recall-ref` for the full reference-aligned configuration.

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

## Reference implementations

Summary of NTM implementations used as references for this project:

| Implementation | Controller | Copy | Recall | Key findings |
|---------------|-----------|------|--------|-------------|
| [loudinthecloud/pytorch-ntm](https://github.com/loudinthecloud/pytorch-ntm) | LSTM | Generalizes to len 80 | Yes | Most-starred PyTorch NTM |
| [vlgiitr/ntm-pytorch](https://github.com/vlgiitr/ntm-pytorch) | LSTM | Yes | Yes | Clean codebase, good defaults |
| [Collier & Beel 2018](https://arxiv.org/abs/1807.08518) | LSTM | Yes | — | Constant mem init 3.5x faster, tanh bounding, grad clip 50 |
| [ajithcodesit/Neural_Turing_Machine](https://github.com/ajithcodesit/Neural_Turing_Machine) | Feedforward | Requires curriculum | — | FFN "did NOT converge" without curriculum |
| [clemkoa/ntm](https://github.com/clemkoa/ntm) | LSTM | Yes | Yes | Uses Adam, popular fork |
| [snipsco/ntm-lasagne](https://github.com/snipsco/ntm-lasagne) | LSTM | Yes | Yes | Lasagne/Theano, Adam for recall |

**Common patterns across references**:
- LSTM controller for all tasks (no reference uses vanilla RNN for recall)
- Separate head FCs from controller hidden (not monolithic output slicing)
- Output = FC(cat(controller_hidden, read_vector)) — direct access to read result
- RMSprop lr=1e-4 for copy; Adam common for recall (clemkoa, snipsco)
- Value clipping ±10 (loudinthecloud, vlgiitr) or global norm 50 (Collier & Beel)
- Constant memory init 1e-6 (Collier & Beel recommendation)
- No curriculum needed for LSTM controllers on copy/recall

## Why recall is harder than copy

The associative recall task is fundamentally more difficult than copy for several reasons:

1. **Content vs location addressing**: Copy only needs sequential location shifts (shift-right-by-1). Recall requires content-based lookup — the controller must generate a key vector similar to the stored key, requiring the LSTM to remember what was stored and when.

2. **Gradient dilution at large N**: With N=128 memory slots, the softmax gradient for a non-dominant position is O(1/N²) ≈ 6e-5. The correct memory slot receives only a tiny gradient signal through content addressing. At N=16 the signal is 60x stronger (~4e-3).

3. **Temporal credit assignment**: The model must connect the query key (presented at time t) with the stored value (written at time t-k). The gradient must flow backward through k timesteps of NTM operations, controller updates, and memory read/writes. LSTM's cell state provides a direct gradient highway; vanilla RNN's gradients decay exponentially through tanh.

4. **Phase transition**: The model must learn distinct behaviors for store phase (write to memory) and query phase (read from memory). The delimiter signals this transition, requiring the controller to learn phase-dependent gating.

## Known failure modes

| Failure mode | Symptoms | Root cause | Fix |
|---|---|---|---|
| One-slot collapse | All writes to slot 0, low accuracy | Content addressing too strong, location shift not learned | Increase K to force multi-slot usage; curriculum |
| Frozen addressing | Read/write weights never change | Gradients too weak to move addressing | Check gradient flow, reduce N, increase lr |
| Mode collapse | All outputs identical regardless of query | Controller ignores input, produces fixed output | Check controller gradients, verify input reaches controller |
| Content-only fallback | High g during test, low during train | Model memorizes training data, falls back to content matching on novel data | Curriculum, more training data diversity |
| Gradient explosion | NaN loss after N iterations | Unbounded head parameters, especially γ | Bound γ to [1,5], clamp controller output [-20,20] |
| Memory drift | Loss increases over long sequences | Memory values grow unboundedly | Apply tanh memory bounding after each write |

## Recall convergence results

See [ntm-convergence-results.md](ntm-convergence-results.md) for full experimental results comparing RMSprop vs Adam, different N values, and curriculum strategies.
