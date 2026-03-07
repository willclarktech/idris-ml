# idris-ml

Deep learning library in Idris 2 with compile-time tensor shape checking and automatic differentiation.

## References

- [Neural Turing Machines (Graves, Wayne, Danihelka 2014)](https://arxiv.org/abs/1410.5401) — original NTM paper
- [Implementing Neural Turing Machines (Collier & Beel 2018)](https://isg.beel.org/blog/2018/08/01/a-stable-neural-turing-machine-ntm-implementation-source-code-and-pre-print/) — stability findings: constant memory init (1e-6) converges 3.5x faster, tanh memory bounding, grad clip norm 50

## Build Commands

```bash
# Build C shared library (required before running examples)
make build/libidrisml.dylib

# Type-check all library modules
idris2 --build idris-ml.ipkg

# Type-check a single module
idris2 --source-dir src -p contrib --check src/<File>.idr

# Build an example (examples aren't in the package, so manual flags are needed)
idris2 --source-dir src -p contrib -o <name> src/Example/<Name>.idr

# Run a built example
./build/exec/<name>

# Run Idris unit tests
make test

# Run C library tests
make test-c

# Run benchmark (Supervised + RNN + NTM)
make bench

# PyTorch reference implementation (requires uv)
make ref-setup          # One-time: install Python deps
make ref-test           # Run PyTorch correctness tests
make ref-lint           # Lint Python code (ruff)
make ref-typecheck      # Type-check Python code (pyright)
make ref-convergence    # NTM convergence verification (copy + recall)
make ref-convergence-copy      # Copy task only
make ref-convergence-recall    # Recall task only

# Benchmarks (Idris vs PyTorch timing)
make bench-py           # Run PyTorch timing benchmark
make bench-compare      # Side-by-side Idris vs PyTorch comparison
```

Concrete examples:

```bash
idris2 --source-dir src -p contrib -o supervised src/Example/Supervised.idr && ./build/exec/supervised
idris2 --source-dir src -p contrib -o rnn src/Example/Rnn.idr && ./build/exec/rnn
idris2 --source-dir src -p contrib -o lstm src/Example/Lstm.idr && ./build/exec/lstm
idris2 --source-dir src -p contrib -o ntm-copy src/Example/NtmCopy.idr && ./build/exec/ntm-copy
idris2 --source-dir src -p contrib -o ntm-associative-recall src/Example/NtmAssociativeRecall.idr && ./build/exec/ntm-associative-recall
# LSTM with custom hyperparameters
./build/exec/lstm --lr 0.1 --epochs 2000 --patience 500 --seed 42
# NTM copy with custom hyperparameters
./build/exec/ntm-copy --lr 0.0001 --clip 10.0 --alpha 0.95 --epochs 50000 --patience 5000 --seed 42
# NTM associative recall with custom hyperparameters
./build/exec/ntm-associative-recall --lr 0.0001 --epochs 100000 --patience 5000 --seed 42 --min-items 2 --max-items 6
# Hyperparameter sweep (builds once, runs grid in parallel)
bash scripts/sweep.sh --parallel 4
# Quick sweep (2000 epochs for fast screening)
bash scripts/sweep.sh --parallel 4 --quick
# Sweep for associative recall task
bash scripts/sweep.sh --task recall --parallel 4
bash scripts/sweep.sh --task recall --parallel 4 --quick
# Sweep for LSTM task
bash scripts/sweep.sh --task lstm --parallel 4
bash scripts/sweep.sh --task lstm --parallel 4 --quick
```

## Architecture

### Module dependency order (leaves first)

1. **Floating** - Extended `Floating` interface adding `sqrt`
2. **Util** - Helpers: `enumerate`, `permute`, `chunks`
3. **Sampler** - Distribution samplers: `uniform`, `normal` (Box-Muller), `normalSample`
3b. **Init** - Weight initialization strategies composable with samplers: `xavier`, `xavierGain`, `he`, `lecun`, `fixedRange`
4. **Tensor** - Shape-indexed tensor: `Tensor : Vect rank Nat -> Type -> Type`
5. **Math** - Loss functions, activations, linear algebra
6. **Memory** - NTM read/write head operations
7. **Variable** - Tape-based autograd (Wengert list) with hybrid Scheme/C storage and C backward pass
8. **DataPoint** - `DataPoint`, `RecurrentDataPoint`, and `TwoPhaseDataPoint` records
8b. **Generate** - Random data generation: `SequenceTask` port, `copyTask`/`associativeRecallTask` adapters, `copyTaskBinary`/`recallTaskBinary` (binary vector format), `randomBatchVect`
9. **Endofunctor** - `emap : (ty -> ty) -> e ty -> e ty` for type-preserving maps
10. **Layer** - Layer/Network types (mutually recursive), forward pass, constructors, `autoName`
11. **Optimizer** - SGD, Adam, and RMSprop optimizers with per-parameter, global norm, or value gradient clipping
12. **Schedule** - Learning rate schedules: `constant`, `cosineAnnealing`, `oneCycle`
13. **Backprop** - Training loop: `epoch`, `train`, `trainFrom`, `epochRecurrent`, `trainRecurrent`, `trainRecurrentFrom`, `trainScheduledFrom`, `trainRecurrentScheduledFrom`, `epochTwoPhase`, `trainTwoPhaseScheduledFrom`, `epochTwoPhaseDenseBce`
14. **Curriculum** - Multi-stage curriculum training: `Stage` record, `runCurriculum` with periodic data regeneration and two-level early stopping
15. **Debug** - Generic forward-pass diagnostics: `debugForward`, `debugForwardRecurrent`, per-layer state extraction, `toDoubleNetwork`

### Core type signatures

```idris
-- Tensor.idr
data Tensor : Vect rank Nat -> Type -> Type where
  STensor : ty -> Tensor [] ty
  VTensor : Vect dim (Tensor dims ty) -> Tensor (dim :: dims) ty

Scalar = Tensor []
Vector elems = Tensor [elems]
Matrix rows columns = Tensor [rows, columns]

-- Variable.idr (tape-based autograd)
record Variable where
  constructor Var
  tapeIdx : Nat           -- index into global tape (Wengert list)
  tapeGen : Nat           -- tape generation (staleness detection)
  paramId : Maybe String  -- parameter name (Nothing = intermediate)
  value : Double          -- cached forward result

-- Layer.idr (mutual block)
data Layer : (inputSize : Nat) -> (outputSize : Nat) -> Type -> Type where
  LinearLayer : Matrix outputSize inputSize ty -> Vector outputSize ty -> Maybe AnyPtr -> Maybe AnyPtr -> Layer inputSize outputSize ty
  RnnLayer : Matrix outputSize inputSize ty -> Matrix outputSize outputSize ty -> Vector outputSize ty -> Vector outputSize ty -> Layer inputSize outputSize ty
  LstmLayer : Matrix (4 * outputSize) inputSize ty -> Matrix (4 * outputSize) outputSize ty -> Vector (4 * outputSize) ty -> Vector outputSize ty -> Vector outputSize ty -> Maybe AnyPtr -> Maybe AnyPtr -> Maybe AnyPtr -> Maybe AnyPtr -> Maybe AnyPtr -> Layer inputSize outputSize ty
  ActivationLayer : String -> ActivationFunction ty -> Layer n n ty
  NormalizationLayer : String -> NormalizationFunction ty -> Layer n n ty
  NtmLayer : {n, m, h : Nat} -> Layer (m + inputSize) h ty -> Layer h (ReadParamWidth m) ty -> Layer h (WriteParamWidth m) ty -> Layer (h + m) outputSize ty -> Matrix n m ty -> Vector n ty -> Vector n ty -> Vector m ty -> Layer inputSize outputSize ty

data Network : (inputDims : Nat) -> (hiddenDims : List Nat) -> (outputDims : Nat) -> Type -> Type where
  OutputLayer : Layer i o ty -> Network i [] o ty
  (~>) : Layer i h ty -> Network h hs o ty -> Network i (h :: hs) o ty

-- Endofunctor.idr
interface Endofunctor e where
  emap : (ty -> ty) -> e ty -> e ty
```

## Key Patterns

### Network composition with `(~>)`

```idris
-- Supervised: linear -> softmax
ll <- nameParams "ll" <$> linearLayer
let model = ll ~> OutputLayer softmaxLayer

-- NTM: LSTM controller with separate head FCs and output FC
ntm <- ntmLayer {inputSize = InputW, outputSize = OutputW, n = N, m = M, h = H}
let model = autoName $ OutputLayer ntm
```

### Forward pass returns updated network (state threading)

```idris
-- applyLayer and forward both return (updated, output) pairs
applyLayer : Layer i o ty -> Vector i ty -> (Layer i o ty, Vector o ty)
forward : Network i hs o ty -> Vector i ty -> (Network i hs o ty, Vector o ty)

-- RNN/NTM layers carry state; linear/activation layers return unchanged
let (updatedModel, output) = forward model input
```

### Training cycle

```idris
-- Backprop.idr: forward -> backward (resets tape) -> optimizer step -> apply deltas
epoch opt dataPoints lossFn model st =
  let loss = calculateLoss lossFn model dataPoints   -- 1. Forward pass (appends to tape)
      grads = collectGrads 1.0 loss                  -- 2. Backward + tape reset (gen++)
      (deltas, st') = opt.step grads st              -- 3. Optimizer computes deltas
  in (emap (applyDeltas deltas) model, st', loss)    -- 4. Update .value + return loss

-- Scheduled training with early stopping:
let makeOpt = \lr => adamGlobalClip lr 0.9 0.999 1e-8 5.0
let schedule = oneCycle 0.001 25.0 1e5 0.25 6000
(model', st', epochsDone) = trainRecurrentScheduledFrom makeOpt schedule model dps lossFn 6000 10 initState

-- Two-phase training (NTM copy/recall with binary vectors):
let opt = rmspropValueClip 0.0001 0.95 1.0e-8 10.0
let (m', s', loss) = epochTwoPhase opt dataPoints binaryCrossEntropyWithLogits model st

-- Dense optimizer (C arrays, no SortedMap — ~47% faster for NTM):
let numPids = getNumPids 0
let opt = rmspropValueClipDense 0.0001 0.95 1.0e-8 10.0
let st0 = initDenseState numPids
let (m', s', loss) = epochTwoPhaseDense opt dataPoints binaryCrossEntropyWithLogits model st0

-- Dense optimizer with C-backed BCE (fused sigmoid + BCE loss, single tape entry):
let opt = rmspropValueClipMomentumDense 0.0001 0.95 1.0e-8 10.0 0.9
let (m', s', loss) = epochTwoPhaseDenseBce opt dataPoints model st0
```

### Supervised vs Recurrent vs TwoPhase API

The library provides three training modes:

| Aspect | Supervised | Recurrent | TwoPhase |
|--------|-----------|-----------|----------|
| Data type | `DataPoint i o ty` | `RecurrentDataPoint i o ty` | `TwoPhaseDataPoint i o ty` |
| Forward | `forward` | `forwardRecurrent` | `forwardTwoPhase` |
| Train | `epoch` / `train` | `epochRecurrent` / `trainRecurrent` | `epochTwoPhaseDense` / `trainTwoPhaseScheduledFromDense` |
| Loss phase | All outputs | All outputs | Output phase only |
| Use case | Feedforward nets | RNN/LSTM sequences | NTM copy/recall |

### Parameter naming (required for gradient flow)

Every learnable layer must be named before training. Use `autoName` (preferred):

```idris
ll <- linearLayer
let model = autoName $ ll ~> OutputLayer softmaxLayer  -- ll0_weight0, ll0_bias0, ...

-- NTM: ntm0_lstm0_weight0 (controller), ntm0_readFc_ll0_weight0, ntm0_mem0, ...
let model = autoName $ OutputLayer ntm
```

Manual naming is also available for custom prefixes:

```idris
ll <- nameParams "ll" <$> linearLayer           -- names: ll_weight0, ll_bias0, ...
nameNetworkParams "ntm" $ ntm ~> OutputLayer logSoftmaxLayer  -- recursive naming
```

### Endofunctor for type-preserving transforms

`emap` maps `(ty -> ty)` over Layer/Network without changing shape types. Used by `applyDeltas` in training:

```idris
-- Apply optimizer deltas to all parameters
emap (applyDeltas deltas) model
```

### Data preparation (Double -> Variable)

Raw data is `Double`; training requires `Variable`. Convert with `map fromDouble`:

```idris
let prepared = map (map fromDouble) dataPoints  -- DataPoint i o Double -> DataPoint i o Variable
```

### Curriculum training

Multi-stage training with periodic data regeneration. Each `Stage` encapsulates a label, advancement threshold, and an `IO` data generator:

```idris
import Curriculum

stages : List (Stage W W BatchSize)
stages =
  [ MkStage "Stage 1 (len 1-3)" 0.15 (genData 1 3)
  , MkStage "Stage 2 (len 1-5)" 0.10 (genData 1 5)
  , MkStage "Stage 3 (len 1-8)" 0.0  (genData 1 8)
  ]

(trained, st', epochsDone) <- runCurriculum makeOpt schedule model
  nllLoss stages totalEpochs patience chunkSize initState
```

The module is generic over network architecture and data generation — it doesn't know about copy tasks or NTM-specific types.

### Debug / diagnostics

Convert a trained `Variable`-typed network to `Double` and run the debug forward pass to dump per-timestep internal state:

```idris
import Debug
let dblModel = toDoubleNetwork trained
let (_, _, snapshots) = debugForwardRecurrent dblModel inputs
printDiagnostics "label" snapshots
```

## Adding New Examples

When adding a new model/example to the project, follow this workflow in order:

1. **Source reference implementation** — find a paper or established implementation to use as ground truth for architecture, hyperparameters, and expected convergence behavior. Add to the References section above
2. **Write PyTorch implementation** — port the reference into `pytorch/torch_ref/models/`, add correctness tests in `pytorch/torch_ref/correctness/`, add a benchmark function in `pytorch/torch_ref/benchmark.py`, and wire it into `pytorch/torch_ref/compare.py`. Verify with `make ref-test && make ref-lint && make ref-typecheck`
3. **Write idris-ml implementation** — implement in `src/Example/`, add to `src/Example/Bench.idr`, and add a Makefile target. Verify with `make test && make bench-compare`

Commit at each step. The PyTorch implementation serves as the correctness oracle for the Idris version.

## Performance Optimization

When investigating or improving performance, follow this workflow:

### 1. Profile first

Always profile before optimizing. Use the sub-phase profiler to identify where time is actually spent:

```bash
make profile    # Per-sub-phase timing: Enc/Out/Loss/Bwd/Opt/Sync + tape histogram
```

Key files:
- `src/Example/Profile.idr` — sub-phase profiler (must match current training config)
- `docs/performance-analysis.md` — historical profile data and optimization log

Profile.idr must stay in sync with production training config (optimizer, batch size, etc.). When changing training hyperparameters, update Profile.idr to match.

### 2. Benchmark at matched settings

Compare Idris vs PyTorch at identical batch size, architecture, optimizer:

```bash
make bench-compare    # Side-by-side timing for all models at production batch size
```

Key files:
- `src/Example/Bench.idr` — Idris benchmark scenarios
- `pytorch/torch_ref/benchmark.py` — PyTorch benchmark scenarios (must mirror Idris)
- `pytorch/torch_ref/compare.py` — runs both and prints comparison table

**Important**: Always compare at the same batch size. Current production batch size is 16. The ratio from bench-compare is the ground truth for per-epoch speed — convergence comparisons can be misleading if batch sizes differ.

### 3. Tune hyperparameters with sweep

Use the sweep script for systematic grid search — never manually loop:

```bash
bash scripts/sweep.sh --task copy --parallel 4 --quick  # 2000 epochs for screening
bash scripts/sweep.sh --task copy --parallel 4           # full convergence
```

Key file: `scripts/sweep.sh` — must stay in sync with current CLI flags for each task.

### 4. Run convergence comparison

After tuning, compare end-to-end convergence at matched settings:

```bash
make ref-convergence-copy    # PyTorch convergence (batch=16 default)
./build/exec/ntm-copy        # Idris convergence (uses defaultConfig)
```

Key file: `pytorch/torch_ref/scripts/convergence.py` — PyTorch convergence defaults must match Idris defaults (batch size, lr, etc.).

### 5. Document results

Update `docs/performance-analysis.md` with:
- Fresh profile data (sub-phase breakdown + tape histogram)
- bench-compare numbers (Idris vs PyTorch ratio)
- Convergence comparison results
- What changed and why

### Current performance baseline (2026-03-06)

Per-epoch at batch=16 (NTM-copy, N=128 M=20 H=100):
- Forward (Enc+Out): ~74ms | Backward: ~15ms | Tape: 1.48M entries
- Idris/PyTorch ratio: **0.87x** (Idris faster)
- Convergence: loss ~1e-6 by 5000 epochs (88% short, 77% full accuracy)
- PyTorch alignment changes applied: C-backed BCE, zero forget bias, no output clamping, learned h0/c0, lr=1e-4, per-sequence NtmMemBuf reset

### Performance optimization history

Optimizations applied to NTM-copy (from 1.38s/epoch to 0.145s/epoch, ~10x):
1. Buffer-passing for addressing chain ops (eliminate Variable materialization)
2. Shadow ConstOps (tape compaction for intermediate outputs)
3. C-side pid filtering in backward pass
4. Dense optimizer (C arrays replace SortedMap)
5. C-bulk ConstOp creation (memset+memcpy)
6. C-bulk delta application (bypass emap+sync)
7. LSTM→FC buffer-passing + case destructuring fix (eliminate 3x re-eval)

## Conventions

- **Indentation**: 2 spaces for `.idr` files (see `.editorconfig`)
- **Naming**: PascalCase for types/constructors, camelCase for functions/variables
- **Imports**: Idris stdlib first (`Data.Vect`, `System.Random`), then internal modules alphabetically
- **Commits**: Follow [Conventional Commits](https://www.conventionalcommits.org/) — `feat:`, `fix:`, `refactor:`, `docs:`, `chore:`, etc. Keep subject concise (~50 chars), imperative present tense. Commit work regularly in meaningful chunks — one logical change per commit. Never include ads, branding, or promotional text in commit messages or PR descriptions
- **Section dividers**: `----------------------------------------------------------------------` with section titles in Layer.idr style
- **Documentation**: Always update CLAUDE.md, docs/design-decisions.md, and TODO.md when adding features, changing architecture, or making design decisions

## Gotchas

- **Temporary test files**: Idris2 requires source files to be in `--source-dir`. Never put test files in `/tmp` — they won't compile. Instead, add temporary test files to `src/Example/` and remove them after debugging
- **`total` is a keyword**: Idris 2 reserves `total` as a totality annotation keyword. Never use it as a variable/parameter name — produces a cryptic "Couldn't parse declaration" error at the definition clause. Use `numEpochs`, `totalEpochs`, etc. instead
- **Build flags**: Forgetting `--source-dir src` or `-p contrib` produces confusing import errors
- **Elementwise `(*)`**: `Tensor`'s `Num` instance uses elementwise multiply. For matrix-vector products, use `matrixVectorMultiply` or `vectorMatrixMultiply` from Math.idr
- **`paramId` requirement**: Variables without a `paramId` (i.e., `Nothing`) are invisible to gradient collection and won't receive updates. Use `autoName` (preferred) or `nameParams`/`nameNetworkParams` before training. `autoName` assigns type-based prefixes with per-type counters (`ll0`, `ll1`, `rnn0`, `lstm0`, `ntm0`, ...) and scopes NTM sub-layer names under their parent (`ntm0_lstm0_`, `ntm0_readFc_ll0_`), preventing the collision bug in `nameNetworkParams`. `setParamId` writes to both the Variable record and the tape's pid vector
- **Test suite**: Run `make test` for Idris unit tests, `make test-c` for C tests. Tests live in `test/src/Test/*.idr` with `Harness.idr` providing assertion helpers
- **Tape generation staleness**: After `collectGrads` resets the tape (gen++), Variables from the previous epoch are stale. `ensureOnTape` detects this via generation mismatch and re-registers with current `.value`. Same stale Variable used N times creates N Const entries — gradients accumulate correctly via `mergeWith (+)` on paramId
- **Mutual recursion in Layer.idr**: `Layer` and `Network` are mutually recursive (NtmLayer contains sub-Layers). `applyLayer`, `forward`, `nameParams`, `nameNetworkParams`, and `Endofunctor` instances all live in `mutual` blocks
- **NTM dimension calculations**: `ReadParamWidth m = (m + ShiftKernelSize) + 3` (key of width m + 3-element shift kernel + 3 dynamic params: β, g, γ). `WriteParamWidth m = ReadParamWidth m + m` (addressing params + add vector of width m). The LSTM controller input is `m + inputSize` (read output + input). The output FC input is `h + m` (hidden + read output). The `ntmLayer` constructor takes `{inputSize, outputSize, n, m, h}` as implicit args
- **NTM head parameters**: β (key strength), g (interpolation gate), γ (sharpening) are dynamic — extracted from head FC outputs (fed by LSTM cell state). β uses softplus, g uses sigmoid, γ uses `1 + softplus(x)` (unbounded, [1, ∞)). Add vectors are raw linear (no activation). See `forwardReadHeadUnbounded`/`forwardWriteHeadInterp` in Memory.idr
- **NTM state flow**: `readHeadOutput` from the previous timestep concatenates with current input to form LSTM input (width `m + inputSize`). LSTM cell state feeds head FCs, hidden state + read output feeds output FC. Memory, addressing weights, and read output all update each step
- **NTM two-phase training**: copy/recall use `epochTwoPhaseDenseBce` — encoding inputs fed with outputs discarded, then zero inputs fed during output phase with loss on targets. The C-backed `bceWithLogitsVar` (tag 26) fuses sigmoid + BCE into a single tape entry per output vector, replacing ~7 scalar ops per element. No output activation layer needed
- **`logSoftmax` + `nllLoss`**: Separate softmax + cross-entropy creates autograd intermediate gradients of 1/pp (up to 1e6) that destabilize recurrent training. Use `logSoftmaxLayer` + `nllLoss` instead. Note: the aligned NTM uses sigmoid + BCE instead, which doesn't have this issue
- **`pow` zero-base NaN**: `pow(0, k)` backward for the exponent computes `0^k * log(0) = 0 * -Inf = NaN`. Fixed by returning 0 when base is 0
- **Detached max in `logSoftmax`**: The max subtraction for numerical stability uses a detached constant (`fromDouble . cast`), not a reference to the max Variable. Otherwise the max element receives incorrect gradients
- **Hybrid tape architecture**: Forward pass uses Scheme `foreign-set!` for scalar tape entries (tags/arg1/arg2/vals into `foreign-alloc` arrays — no FFI crossing) and C `ext_meta_set` for tensor op meta pointers (arena-allocated structs). Backward pass runs entirely in C via `walk_backward_ext`, reading meta from `ext_meta` array. PIDs stored in Scheme vector, looked up after C backward returns indices. `foreign-set! 'void*` MUST NOT be used for storing C pointers — it corrupts memory in Chez Scheme
- **Chunked arena allocator**: Meta structs AND tensor op output buffers are arena-allocated via `arena_alloc` (`prim__tensorAllocArena`). The arena uses a linked list of chunks (never `realloc`) to prevent invalidating previously allocated pointers when the arena grows mid-forward-pass. Reset frees old chunks and resets the head chunk. Output buffers are safe to arena-allocate because values are read into Variable records during `buildOutputScalars`/`buildVarsFromBuf` before `arena_reset`. `prim__tensorAlloc` (calloc) is still used for persistent WeightBuf allocations
- **Tape-based backward pass**: `collectGrads` allocates a mutable gradient array via FFI, seeds it with the initial gradient, then `walk_backward_ext` scans the tape in reverse in C. Scalar ops propagate inline; tensor ops dispatch to C backward kernels. ConstOps with non-zero gradient are collected as (index, grad) pairs. Scheme looks up PIDs and builds `SortedMap`. The tape is reset at the end of `collectGrads` (gen++)
- **Zero-arg FFI CSE trap**: Idris 2 compiles zero-argument `%noinline` definitions as constants evaluated once at load time. `tapeGeneration` must take a dummy argument (the tape index) passed through to `prim__tapeGen` to prevent the Chez backend from caching the result. This also applies to any other FFI wrapper reading mutable state
- **FFI side-effect threading**: `let _ = ffiCall` is dropped by the compiler. FFI functions with side effects must return a value that is used in subsequent computation. `prim__gradAdd` returns the handle (`AnyPtr`), enabling handle threading through the backward pass. Dense optimizer steps use `prim__seq result st.v` to force evaluation: `let result = prim__rmspropVcStep ... in { v := prim__seq result st.v } st`. Without this, the optimizer call is silently eliminated and raw gradients are applied as deltas (lr/clip/momentum have zero effect)
- **`fst`/`snd` re-evaluation trap**: When a function with FFI side effects returns a tuple and the caller accesses fields via separate `fst`/`snd` projections (e.g., `fst result`, `snd result`, `fst result` again), Idris 2 compiled to Chez Scheme may re-evaluate the function call for each projection instead of sharing the result. This causes FFI side effects (tape appends, buffer allocations) to execute multiple times. Fix: use `case f args of (a, b, c) => ...` to destructure in a single pattern match. This was a 3× re-evaluation bug in the NTM forward pass — the LSTM controller was called 3 times per timestep instead of once
- **Gradient clipping**: `adam` clips per-parameter; `adamGlobalClip` clips by global L2 norm (preserves gradient direction). Use `adamGlobalClip` for attention/recurrent models where parameters must coordinate — per-parameter clipping distorts direction and causes periodic loss spikes. Default maxNorm is 50.0 (Collier & Beel); 5.0 was too aggressive
- **Controller output clipping (removed)**: Previously `applyLayerVar` clamped raw NTM controller output to [-20, 20] via `clampVar`. Removed to match PyTorch reference which has no output clamping. The LSTM controller + RMSprop + value clip ±10 provide sufficient stability without artificial clamping
- **Curriculum learning**: Available via the Curriculum module for staged training. The PyTorch-aligned NTM (LSTM controller + RMSprop) does not require curriculum — it converges directly with two-phase training. Curriculum was previously required for feedforward controllers (ajithcodesit finding)
- **No tanh memory bounding**: Interpolation write uses raw interpolation (no tanh) to match the PyTorch reference. The Collier & Beel tanh recommendation was for the original erase+add write mechanism, not interpolation write. Tanh caused cumulative degradation during output phase (near-zero write weights still applied tanh every timestep). `tanhBound` in Layer.idr is only used for LSTM gates, not NTM memory. The C kernel `interp_write_compute` supports both modes via `raw_mode` flag (1=raw, 0=tanh); Idris always sets raw_mode=1
- **NTM initial addressing**: Read/write addressing weights are initialized to zeros and read output to Kaiming uniform (non-learnable, matching PyTorch reference). `syncLayerBuffers` projects addressing weights onto the probability simplex via `projectWeights` (clamp to [0, epsilon], renormalize) to prevent NaN from `pow(negative, non-integer)` in `focus`
- **Hyperparameter tuning**: Fix algorithmic issues first (bounded activations, correct clipping, efficient backward pass), then use `scripts/sweep.sh` for systematic grid search. Never manually loop over hyperparameters — see `docs/design-decisions.md` for rationale
- **C shared library required**: `build/libidrisml.dylib` must exist before running any example. Build with `make build/libidrisml.dylib`. The library is loaded by the generated Chez Scheme code at startup. Idris 2 copies the dylib to `build/exec/<name>_app/` at compile time — the Makefile targets also copy it explicitly to ensure the latest version is used. When building manually (not via `make`), you must copy the dylib to the app dir after rebuilding: `cp build/libidrisml.dylib build/exec/<name>_app/`
- **Scheme-native C memory access**: Use Chez Scheme's `foreign-ref`/`foreign-set!` for reading/writing C-allocated arrays instead of calling C functions per element. This avoids the Scheme→C boundary crossing overhead. See `prim__gradAdd`/`prim__gradGet` and `prim__setDouble`/`prim__setInt32` in Variable.idr
- **`prim__seq` for evaluation ordering**: When two FFI side-effect chains must execute in order but have no data dependency, use `prim__seq a b` (Scheme `(lambda (a b) b)`) to force `a` to evaluate before `b` is used. Chez Scheme evaluates function arguments strictly
- **Tensor Foldable reversal**: The `foldr` instance for `Tensor` processes elements in reversed order (head into accumulator first). `toList` produces elements backwards. Use direct `Vect` traversal instead when element order matters (e.g., packing into C buffers)
- **Weight initialization**: `linearLayer`/`rnnLayer` default to Xavier uniform. Biases are always zero. Init strategies compose a variance method with a distribution sampler: `xavier uniform` (default), `xavier normal`, `he normal`, `xavierGain 1.4 uniform`, etc. Use `linearLayerWith (fixedRange 1.0)` for the old `U(-1,1)` behavior. Use `linearLayerWithBias initFn biasStd` for custom bias init (normal with given std). NTM head FCs use `xavierGain 1.4 uniform` + `normal(0.01)` bias, output FC uses `he uniform` + `normal(0.01)` bias (matching PyTorch reference). NTM memory initialized to `sigmoid(xavier_random)` ≈ values in [0,1] (matching PyTorch's `sigmoid(FC_bias)`). Read output uses kaiming uniform. `Sampler.idr` provides `uniform` and `normal` (Box-Muller); `Init.idr` provides `xavier`, `xavierGain`, `he`, `lecun`, `fixedRange`
- **C-backed softmax/logSoftmax**: `softmaxVar`/`logSoftmaxVar` in Variable.idr use C kernels and record a single SoftmaxOp/LogSoftmaxOp tape entry per vector instead of ~29 scalar entries. `applyLayerVar` dispatches NormalizationLayer "softmax"/"logSoftmax" to these
- **C-backed NTM memory ops**: `batchCosineSimilarityVar`, `readOpVar`, `writeOpVar`, `interpolationWriteVar` in Variable.idr use C kernels (BatchCosSimOp/ReadOpOp/WriteOpOp/InterpolationWriteOp, tags 15-18) to reduce tape entries per NTM timestep. `forwardReadHeadUnboundedVar`/`forwardWriteHeadInterpVar` in Layer.idr wire these into the Variable-specialized NTM forward pass. Generic `forwardReadHeadUnbounded`/`forwardWriteHeadInterp` in Memory.idr remain parameterized on `NormalizationFunction ty` for the Double path
- **C-backed addressing ops**: `interpolateVar`, `shiftVar`, `focusVar` in Variable.idr use C kernels (InterpolateOp/ShiftOp/FocusOp, tags 21-23) replacing ~1400 scalar tape entries per head with 3 tensor ops. `shiftVar` takes a pre-softmax'd kernel (apply `softmaxVar` first). Used in both `forwardReadHeadUnboundedVar` and `forwardReadHeadUnboundedVarBuf` in Layer.idr
- **C-backed LSTM cell op**: `lstmCellVar` in Variable.idr uses a C kernel (LstmCellOp, tag 24) fusing bias add + gate activations (sigmoid/tanh) + cell/hidden update into a single tape entry. Replaces ~1700 scalar entries per LSTM timestep with 1. The two matmul ops (iW×x, rW×h) remain as separate MatVecOps. `applyLayerVar` in Layer.idr dispatches to `lstmCellVar` for the Variable-specialized LSTM path
- **C-backed BCE with logits**: `bceWithLogitsVar` in Variable.idr uses a C kernel (BceWithLogitsOp, tag 26) fusing sigmoid + BCE loss into a single tape entry per output vector. Forward: `(1/n) * sum_i [max(p_i,0) - p_i*y_i + log(1+exp(-|p_i|))]`. Backward: `d_p_i = (1/n) * (sigmoid(p_i) - y_i) * d_loss` (gradients to predictions only, not targets). `epochTwoPhaseDenseBce` in Backprop.idr uses this directly instead of the scalar `binaryCrossEntropyWithLogits`. Meta stored via Scheme-side `ext_meta_set` (NOT C-side `tape_meta`) to match `walk_backward_ext` dispatch
- **Persistent NtmMemBuf**: NTM memory matrix kept as persistent `NtmMemBuf` C struct across timesteps. Eliminates 4× per-timestep packMatrix (2560 elements each). Buffer initialized in `nameParams`, synced after `applyDeltas` via `syncLayerBuffers`, epoch-cached tape registration via `prim__ntmMemBufEnsure`. Buffer-aware ops: `batchCosineSimilarityVarBuf`, `readOpVarBuf`, `interpolationWriteVarBuf` in Variable.idr. **Per-sequence reset**: NtmMemBuf stores `initial_vals` (snapshotted at init and after optimizer deltas). `prim__ntmMemBufReset` restores `vals` from `initial_vals` and invalidates cache (forces tape re-registration). `resetNtmMemBufs` in Layer.idr reconstructs the Network with the reset buffer, called before each sequence in `calculateLossTwoPhaseVar`/`VarBce` to prevent cross-sequence mutation
- **Bias WeightBuf**: LinearLayer and LstmLayer have bias WeightBuf fields (`bBuf : Maybe AnyPtr`) alongside weight WeightBufs. `nameParams` allocates them, `syncLayerBuffers` syncs after `applyDeltas`. LinearLayer fuses MatVec+Bias in a single C kernel (`matrixVectorMultiplyVarBufBias`). LstmLayer reads bias from WeightBuf in the C LSTM cell kernel (`lstmCellVarBuf`/`lstmCellVarFromBufs`). Eliminates per-timestep bias re-registration (~160K tape entries/epoch)
- **Learned LSTM h0/c0**: LstmLayer has `h0Buf : Maybe AnyPtr` and `c0Buf : Maybe AnyPtr` fields for learnable initial hidden/cell states. Initialized with Xavier uniform in `lstmLayerWith`. Named as `prefix_h0`/`prefix_c0` in `nameParams`, allocated as WeightBufs. Synced via `applyDeltasAndSyncLayer`/`readFromBuffersLayer`. Matches PyTorch reference's `nn.Parameter(torch.zeros(...))` learnable initial states
- **Buffer-passing MatVec→LstmCell**: `matrixVectorMultiplyVarBufOut` returns raw `(AnyPtr, Int)` buffer+tapeStart instead of Variables. `lstmCellVarFromBufs` consumes these directly via `buf_to_meta` C helper, avoiding `buildOutputScalars`+`packVec` roundtrip for 2×4o intermediate elements per LSTM timestep
- **Bulk buildOutputScalars**: `prim__appendOutputConstOff` bulk-appends ConstOps from a C buffer with offset in a single Scheme FFI call (internal loop), replacing per-element `tapeAppendConst`. `buildVarsFromBuf` reads values with sequential tape indices. Used by all tensor op output paths
- **Shadow ConstOps (tag=25)**: Buffer-passing ops (`*BufOut`, `*BufIO`) create shadow ConstOps instead of regular output ConstOps. These provide gradient slots without values/pids — skipped during backward collection (`if (tag == 25) continue`). Tags set via C bulk `tape_set_shadow_tags` instead of per-element Scheme `foreign-set!`. Shadow ConstOps still occupy tape entries; full elimination requires gradient region reservation (not yet implemented)
- **C-side pid filtering**: `walk_backward_ext` filters ConstOps by integer `pid_id` (C-side `tape_pid_ids` array, parallel to tape). Only collects ConstOps with `pid_id >= 0` (named parameters). Dense pid_ids assigned via Scheme `pid-to-id` hash table in `prim__tapeSetParamId`. Set in three paths: `prim__tapeSetParamId` (initial naming), `prim__tapeAppendConst` (stale re-registration), `prim__tapeEnsureBulkConst`/`prim__ntmMemBufEnsure` (WeightBuf/NtmMemBuf). Reset via `tape_pid_ids_reset` after backward
- **out_tape_start semantics**: Tensor op meta structs store `out_tape_start = idx + 1` (first output gradient index, NOT the op entry index). Backward kernels read `meta->out_tape_start` directly without `+1`. Set by `tensor_op_set_out(tag, meta, idx+1)` during `prim__tapeAppendTensorOp`
- **Dense optimizer**: `DenseOptimizer`/`DenseOptimizerState` in Optimizer.idr use C arrays indexed by integer pid_id instead of `SortedMap String Double`. `collectGradsDense` accumulates gradients into a pre-allocated C array during backward (no per-result FFI calls, no SortedMap inserts). The gradient array is persistent across epochs via `grad_alloc_reuse` (calloc once, memset-zero on reuse). Optimizer step functions (`rmsprop_vc_step`, `sgd_step`, `adam_gc_step`) operate in-place on the array. Dense epoch functions use `applyDeltasAndSyncNetwork` which applies deltas directly to C buffers via `buf_apply_deltas` (bypassing `emap` + `syncLayerBuffers`). NTM examples use this path via `epochTwoPhaseDense`; supervised/LSTM examples still use the original `SortedMap` path. Must call `getNumPids 0` after `autoName` to get the parameter count for `initDenseState`
- **C-bulk delta application**: `applyDeltasAndSyncLayer`/`applyDeltasAndSyncNetwork` in Layer.idr apply optimizer deltas directly to WeightBuf/NtmMemBuf C arrays via `buf_apply_deltas(vals, pid_ids, count, deltas)`. Each buffer stores a parallel `int *pid_ids` array (populated during `nameParams`). This bypasses the Scheme `emap (applyDeltasDense ...)` + `syncNetworkBuffers` traversals (~63K Variable operations). WeightBuf pid_ids stored in Scheme 6-vector slot [4]; NtmMemBuf pid_ids stored in C struct field. Cache generations are reset to force tape re-registration next epoch. **Important**: Variable.value fields are NOT updated — call `readFromBuffersNetwork` before `toDoubleNetwork` to sync C buffer values back into Variable records for evaluation
- **Chez Scheme output buffering**: Stdout is fully buffered when redirected to file/pipe (e.g. background tasks). Use `stdbuf -oL ./build/exec/<name>` to force line-buffering for long-running training
- **Periodic GC for long training**: NTM training (50K+ epochs) OOMs without periodic forced GC. `forceGC` (exported from Variable.idr) calls Chez `(collect (collect-maximum-generation))` with `(heap-reserve-ratio 1.0)` every 10 epochs in NTM training loops. The `heap-reserve-ratio 1.0` minimizes retained heap (default ~2.0 retains 2x live data), and max-generation collection is more thorough. The FFI lambda must take 0 args — `%World` is erased in Chez Scheme's PrimIO calling convention
- **`getRssMB` peak RSS tracking**: `getRssMB` (exported from Variable.idr) returns peak RSS in MB via C `get_rss_mb` (`getrusage(RUSAGE_SELF).ru_maxrss`). Takes a dummy `Nat` arg to prevent CSE (pass epoch number at call sites). Returns peak (high-water mark) RSS, not current — it only goes up. Division to MB done in C to avoid 64-bit return value issues. Used in training loop logs and bench output
- **`getCurrentRssMB` current RSS**: `getCurrentRssMB` (exported from Variable.idr) returns current resident memory in MB via `mach_task_info` on macOS. Unlike `getRssMB` (peak), this reflects actual current usage and can decrease after GC. Returns -1 on non-macOS platforms
