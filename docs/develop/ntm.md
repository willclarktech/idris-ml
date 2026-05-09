# NTM Design Decisions

NTM-specific design decisions for the Neural Turing Machine implementation. For general autograd, optimizer, and infrastructure decisions, see [design-decisions.md](design-decisions.md).

> **Note:** V1 internals referenced below (`Variable d`, `forwardVarTensor`, `applyVarTensor`, `Memory.idr`) are pre-Path-C names. The NTM design — head parameters, addressing pipeline, two-phase training, simplex projection — is unchanged post-migration; the names map to V2 as `Tensor [...] d` / `forwardVar` / `applyVar`, with the NTM-specific addressing fused into `prim__ntmReadHead` and `prim__ntmInterpWrite` (still on the tech-debt list per `TODO.md`). See [path-c-migration.md](path-c-migration.md).

## NTM head parameters: gamma sharpening

The focus sharpening parameter γ (gamma) controls how peaked the addressing distribution becomes. Two variants exist:

**Unbounded (current, PyTorch-aligned)**: `gamma = 1 + softplus(x)`, giving γ ∈ [1, ∞). Used by `forwardReadHeadUnbounded`/`forwardWriteHeadInterp` in Memory.idr and the Variable-specialized paths in Layer.idr. This matches all PyTorch reference implementations (loudinthecloud, vlgiitr).

**Bounded (legacy)**: `gamma = 1 + 4 * sigmoid(x)`, giving γ ∈ [1, 5]. Available via `forwardReadHead`/`forwardWriteHead` in Memory.idr. This was a stability measure — at γ=5, `0.1^5 = 1e-5` gives survivable gradients for non-dominant positions. The bounded version is no longer used by the default NTM examples but remains in the codebase.

The unbounded version works because the LSTM controller + interpolation write + value clipping provide sufficient stability. The LSTM's cell state gives stronger temporal gradients than the old linear/RNN controllers, so the model can learn appropriate gamma values without artificial bounds.

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

## NTM stability measures

The aligned NTM retains several stability measures from Collier & Beel and reference implementations:

| Measure | Value | Source |
|---------|-------|--------|
| Memory init | sigmoid(xavier_random) ≈ [0,1] | Matches PyTorch's sigmoid(FC_bias) |
| No tanh memory bounding | Raw interpolation write | Matches PyTorch reference; tanh was for erase+add |
| Value clipping | ±10.0 on gradients | PyTorch reference (loudinthecloud, vlgiitr) |
| Forget gate bias | 0.0 (all zeros) | Matches PyTorch nn.LSTMCell default |
| Learned h0/c0 | Xavier uniform init | Matches PyTorch nn.Parameter learnable states |

**Memory init**: `ntmLayer` initializes memory to `sigmoid(xavier_random)` — values in [0,1], matching PyTorch's `sigmoid(FC_bias)`. Head FCs use `xavierGain 1.4 uniform` weights + `normal(0.01)` bias. Output FC uses `he uniform` weights + `normal(0.01)` bias. Read output uses kaiming uniform.

**Controller output clipping (removed)**: previously clamped head FC outputs to [-20, 20]. Removed to match PyTorch reference which has no output clamping. The LSTM controller + RMSprop + value clip ±10 provide sufficient stability.

**Random data generation**: The `Generate` module provides `copyTaskBinary` and `recallTaskBinary` for binary vector data, plus `copyTask`/`associativeRecallTask` for one-hot data (legacy). Each training epoch generates fresh random data to prevent overfitting.

## No tanh memory bounding (interpolation write)

The interpolation write uses raw interpolation without tanh bounding, matching the PyTorch reference. The Collier & Beel tanh recommendation was for the original erase+add write mechanism, not interpolation write. With interpolation write, tanh causes cumulative memory degradation during the output phase — near-zero write weights still apply `tanh(mem)` every timestep, so over 20 output steps a value of 0.5 degrades to ~0.24. The C kernel `interp_write_compute` supports both modes via `raw_mode` flag for testing; Idris always sets `raw_mode=1` (raw).

## Initial addressing

The PyTorch-aligned NTM uses **non-learnable** initial addressing, matching the vlgiitr reference:
- Addressing weights initialized to zeros (projected to simplex by `syncLayerBuffers`)
- Read head output initialized with Kaiming uniform
- These are reset-per-sequence state, not learned parameters

This matches vlgiitr/ntm-pytorch, where addressing starts from zeros and is entirely determined by the controller's behavior — the controller must *learn* to address correctly through its own outputs rather than relying on a learned addressing prior.

The idris-ml `NtmLayer` stores addressing weights and read output as fields that carry state across timesteps (within a sequence) but are not named as parameters by `nameParams`. After `applyDeltas`, `syncLayerBuffers` still projects addressing weights onto the probability simplex to prevent NaN from `pow(negative, non-integer)` in `focus`.

## NTM head operations: Double vs Variable paths

Memory.idr exports two sets of head functions:

**Bounded (legacy)**: `forwardReadHead`/`forwardWriteHead` — use `gamma = 1 + 4*sigmoid(x)` and erase+add write. Parameterized on `NormalizationFunction ty`.

**Unbounded + interpolation (current)**: `forwardReadHeadUnbounded`/`forwardWriteHeadInterp` — use `gamma = 1 + softplus(x)` and interpolation write. Also parameterized on `NormalizationFunction ty`.

Layer.idr has Variable-specialized versions: `forwardReadHeadUnboundedVar`/`forwardWriteHeadInterpVar` that call C-backed kernels (`batchCosineSimilarityVar`, `readOpVar`, `interpolationWriteVar`) for performance. The generic Double `applyLayer` path passes `softmax` to the Memory.idr functions.

Helper functions (`sig`, `softplus`, `interpolate`, `focus`, etc.) are exported from Memory.idr to avoid duplication between the Double and Variable paths.

## C-backed NTM memory operations

With N=128 memory slots and W=8 width, each NTM timestep created ~12,500 scalar tape entries for content addressing, read, and memory write. These head computations dominated both forward (~45%) and backward (~53%) pass time.

Three new C kernels batch these into single tape entries:

**Batch cosine similarity** (`batchCosineSimilarityVar`): computes `scores[i] = beta * cosine_similarity(key, memory[i])` for all N rows in one C call. The `BatchCosSimMeta` struct saves dot products, row norms, and key norm for backward. Backward propagates gradients to memory rows, key vector, and beta scalar using the analytic Jacobian: `d cos(a,b)/d a_k = (b_k - (a·b/|a|²) * a_k) / (|a| * |b|)`.

**Read operation** (`readOpVar`): computes `output[j] = sum_i(weights[i] * memory[i][j])` — a transpose-matvec. Backward: `d_weights[i] = sum_j(dy[j] * mem[i][j])`, `d_mem[i][j] = weights[i] * dy[j]`.

**Write operation** (`writeOpVar`): fused erase+add in one pass: `out[i][j] = mem[i][j] * (1 - w[i] * e[j]) + w[i] * a[j]`. Backward propagates to memory, weights, erase, and add vectors with analytic gradients. (Legacy — used by `forwardWriteHeadVar`.)

**Interpolation write** (`interpolationWriteVar`): `out[i][j] = w[i] * add[j] + (1 - w[i]) * mem[i][j]`. C-backed (InterpolationWriteOp, tag 18). Used by the current `forwardWriteHeadInterpVar` in the aligned NTM. Simpler than erase+add — no erase vector, fewer gradient terms.

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

**NtmCopy** (location-based): the copy task writes binary vectors sequentially then reads them back in the same order. The model learns shift-right-by-one each timestep — pure location addressing.

**NtmAssociativeRecall** (content-based): K items (each `seqLen` binary vectors) are stored, then one item is presented as a query. The model must retrieve the *next* item by content similarity.

### Task encoding (PyTorch-aligned binary format)

Both tasks use binary vectors with delimiter channels, matching the PyTorch reference:

**Copy task** (W=8): input width = `W+1` (8 data + 1 delimiter). Encoding phase: `seq_len` random binary vectors (delimiter=0), then 1 delimiter row (delimiter=1). Output phase: `seq_len` zero inputs. Target: the original `seq_len` binary vectors (width W). Loss computed only during output phase via `binaryCrossEntropyWithLogits`.

**Recall task** (W=6, seqLen=3): input width = `W+2` (6 data + item delimiter + query delimiter). Each item is `seqLen` binary vectors bracketed by an item delimiter. Query is one item's vectors bracketed by a query delimiter. Target: the *next* item's vectors.

### Training protocol

Both examples use two-phase training (`epochTwoPhase`): encoding inputs are fed to the network with outputs discarded, then zero inputs are fed during the output phase with loss computed against targets. This matches the PyTorch reference's training loop. No curriculum is used — the LSTM controller with RMSprop converges directly.

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

## LSTM controller (default for all tasks)

The aligned NTM uses an LSTM controller for both copy and recall tasks, matching all reference implementations (Graves et al. 2014, loudinthecloud, vlgiitr).

**Why LSTM over vanilla RNN**: the cell state provides a direct gradient highway through the forget gate. For associative recall, information stored during the store phase persists through the delimiter into the query phase with minimal gradient degradation. The input/output gates naturally learn phase-dependent behavior (write during store, read during query). Vanilla RNN gradients decay exponentially through tanh squashing, making temporal credit assignment over 4+ timesteps unreliable.

**Softmax gradient scaling**: with N memory slots, the softmax gradient for a non-dominant position is O(1/N²). At N=128 this gives ~6e-5 per gradient step, vs ~4e-3 at N=16 (60x stronger). The LSTM's stronger temporal gradients are essential to compensate for this dilution at large N.

**Implementation in idris-ml**: `LstmLayer` constructor in Layer.idr. The NTM's LSTM controller input is `read_head_output ++ input` (width `m + inputSize`). Head FCs (`readFc`, `writeFc`) take the LSTM cell state (via `extractCellState`), not the hidden state. The output FC takes `hidden ++ read_output` (width `h + m`). This matches the reference architecture where head parameters and output computation use different LSTM state components.

**Implementation in PyTorch reference**: `LSTMController` wraps `nn.LSTMCell` + `nn.Linear`. LSTM's output gate already applies tanh, so no extra activation is needed. Learnable initial states `h0`, `c0` as `nn.Parameter(torch.zeros(...))`. The `NtmRecallConfig.controller` field selects between "lstm" (default) and "rnn".

### Reference-aligned architecture (idris-ml and PyTorch)

Both the idris-ml and PyTorch implementations now match the reference architecture from loudinthecloud/pytorch-ntm and vlgiitr/ntm-pytorch:

| Aspect | Architecture |
|--------|-------------|
| Controller | LSTM |
| Head param source | Separate FCs from LSTM cell state |
| Head FC init | `xavierGain 1.4 uniform` + `normal(0.01)` bias |
| Output FC init | `he uniform` + `normal(0.01)` bias |
| Memory init | `sigmoid(xavier_random)` ≈ [0,1] |
| Read output init | Kaiming uniform |
| Output computation | `output_fc(cat(hidden, read_output))` |
| Write mechanism | Interpolation write |
| γ activation | `1 + softplus(x)` unbounded |
| Optimizer | RMSprop lr=1e-4, alpha=0.95 |
| Grad clipping | Value clip ±10 |
| Data format | Binary vectors with delimiter channels |
| Loss | C-backed BCE with logits (BceWithLogitsOp, tag 26) |
| Loss phase | Output phase only (two-phase training) |
| LSTM init states | Learnable h0/c0 (Xavier uniform) |
| Forget gate bias | 0.0 (all zeros, PyTorch default) |

The most critical change was the output architecture: the output FC takes `cat(hidden, read_output)`, giving the network direct access to retrieved memory content at the current timestep. The old architecture returned a slice of the controller output, so the read vector only fed back as input at t+1.

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
| Gradient explosion | NaN loss after N iterations | Unbounded head parameters, especially γ | Value clip ±10 on gradients, numerically stable BCE/softplus |
| Memory drift | Loss increases over long sequences | Memory values grow unboundedly | Use interpolation write (bounded by design when weights sum to ~1) |

## Recall convergence results

See [ntm-convergence-results.md](ntm-convergence-results.md) for full experimental results comparing RMSprop vs Adam, different N values, and curriculum strategies.
