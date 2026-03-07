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

# Run Idris unit tests (108 tests)
make test

# Run C library tests (187 tests)
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
3b. **Init** - Weight initialization strategies composable with samplers: `xavier`, `he`, `lecun`, `fixedRange`
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
13. **Backprop** - Training loop: `epoch`, `train`, `trainFrom`, `epochRecurrent`, `trainRecurrent`, `trainRecurrentFrom`, `trainScheduledFrom`, `trainRecurrentScheduledFrom`, `epochTwoPhase`, `trainTwoPhaseScheduledFrom`
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
  LinearLayer : Matrix outputSize inputSize ty -> Vector outputSize ty -> Layer inputSize outputSize ty
  RnnLayer : Matrix outputSize inputSize ty -> Matrix outputSize outputSize ty -> Vector outputSize ty -> Vector outputSize ty -> Layer inputSize outputSize ty
  LstmLayer : Matrix (4 * outputSize) inputSize ty -> Matrix (4 * outputSize) outputSize ty -> Vector (4 * outputSize) ty -> Vector outputSize ty -> Vector outputSize ty -> Maybe WeightBuffer -> Maybe WeightBuffer -> Layer inputSize outputSize ty
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
```

### Supervised vs Recurrent vs TwoPhase API

The library provides three training modes:

| Aspect | Supervised | Recurrent | TwoPhase |
|--------|-----------|-----------|----------|
| Data type | `DataPoint i o ty` | `RecurrentDataPoint i o ty` | `TwoPhaseDataPoint i o ty` |
| Forward | `forward` | `forwardRecurrent` | `forwardTwoPhase` |
| Train | `epoch` / `train` | `epochRecurrent` / `trainRecurrent` | `epochTwoPhase` / `trainTwoPhaseScheduledFrom` |
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

## Conventions

- **Indentation**: 2 spaces for `.idr` files (see `.editorconfig`)
- **Naming**: PascalCase for types/constructors, camelCase for functions/variables
- **Imports**: Idris stdlib first (`Data.Vect`, `System.Random`), then internal modules alphabetically
- **Commits**: Follow [Conventional Commits](https://www.conventionalcommits.org/) — `feat:`, `fix:`, `refactor:`, `docs:`, `chore:`, etc. Keep subject concise (~50 chars), imperative present tense. Commit work regularly in meaningful chunks — one logical change per commit. Never include ads, branding, or promotional text in commit messages or PR descriptions
- **Section dividers**: `----------------------------------------------------------------------` with section titles in Layer.idr style
- **Documentation**: Always update CLAUDE.md, docs/design-decisions.md, and TODO.md when adding features, changing architecture, or making design decisions

## Gotchas

- **Temporary test files**: Idris2 requires source files to be in `--source-dir`. Never put test files in `/tmp` — they won't compile. Instead, add temporary test files to `src/Example/` and remove them after debugging
- **Build flags**: Forgetting `--source-dir src` or `-p contrib` produces confusing import errors
- **Elementwise `(*)`**: `Tensor`'s `Num` instance uses elementwise multiply. For matrix-vector products, use `matrixVectorMultiply` or `vectorMatrixMultiply` from Math.idr
- **`paramId` requirement**: Variables without a `paramId` (i.e., `Nothing`) are invisible to gradient collection and won't receive updates. Use `autoName` (preferred) or `nameParams`/`nameNetworkParams` before training. `autoName` assigns type-based prefixes with per-type counters (`ll0`, `ll1`, `rnn0`, `lstm0`, `ntm0`, ...) and scopes NTM sub-layer names under their parent (`ntm0_lstm0_`, `ntm0_readFc_ll0_`), preventing the collision bug in `nameNetworkParams`. `setParamId` writes to both the Variable record and the tape's pid vector
- **Test suite**: Run `make test` for 108 Idris unit tests, `make test-c` for 187 C tests. Tests live in `test/src/Test/*.idr` with `Harness.idr` providing assertion helpers
- **Tape generation staleness**: After `collectGrads` resets the tape (gen++), Variables from the previous epoch are stale. `ensureOnTape` detects this via generation mismatch and re-registers with current `.value`. Same stale Variable used N times creates N Const entries — gradients accumulate correctly via `mergeWith (+)` on paramId
- **Mutual recursion in Layer.idr**: `Layer` and `Network` are mutually recursive (NtmLayer contains sub-Layers). `applyLayer`, `forward`, `nameParams`, `nameNetworkParams`, and `Endofunctor` instances all live in `mutual` blocks
- **NTM dimension calculations**: `ReadParamWidth m = (m + ShiftKernelSize) + 3` (key of width m + 3-element shift kernel + 3 dynamic params: β, g, γ). `WriteParamWidth m = ReadParamWidth m + m` (addressing params + add vector of width m). The LSTM controller input is `m + inputSize` (read output + input). The output FC input is `h + m` (hidden + read output). The `ntmLayer` constructor takes `{inputSize, outputSize, n, m, h}` as implicit args
- **NTM head parameters**: β (key strength), g (interpolation gate), γ (sharpening) are dynamic — extracted from head FC outputs (fed by LSTM cell state). β uses softplus, g uses sigmoid, γ uses `1 + softplus(x)` (unbounded, [1, ∞)). Add vectors are raw linear (no activation). See `forwardReadHeadUnbounded`/`forwardWriteHeadInterp` in Memory.idr
- **NTM state flow**: `readHeadOutput` from the previous timestep concatenates with current input to form LSTM input (width `m + inputSize`). LSTM cell state feeds head FCs, hidden state + read output feeds output FC. Memory, addressing weights, and read output all update each step
- **NTM two-phase training**: copy/recall use `epochTwoPhase` — encoding inputs fed with outputs discarded, then zero inputs fed during output phase with loss on targets. Use `binaryCrossEntropyWithLogits` (sigmoid applied inside loss) with no output activation layer
- **`logSoftmax` + `nllLoss`**: Separate softmax + cross-entropy creates autograd intermediate gradients of 1/pp (up to 1e6) that destabilize recurrent training. Use `logSoftmaxLayer` + `nllLoss` instead. Note: the aligned NTM uses sigmoid + BCE instead, which doesn't have this issue
- **`pow` zero-base NaN**: `pow(0, k)` backward for the exponent computes `0^k * log(0) = 0 * -Inf = NaN`. Fixed by returning 0 when base is 0
- **Detached max in `logSoftmax`**: The max subtraction for numerical stability uses a detached constant (`fromDouble . cast`), not a reference to the max Variable. Otherwise the max element receives incorrect gradients
- **Hybrid tape architecture**: Forward pass uses Scheme `foreign-set!` for scalar tape entries (tags/arg1/arg2/vals into `foreign-alloc` arrays — no FFI crossing) and C `ext_meta_set` for tensor op meta pointers (arena-allocated structs). Backward pass runs entirely in C via `walk_backward_ext`, reading meta from `ext_meta` array. PIDs stored in Scheme vector, looked up after C backward returns indices. `foreign-set! 'void*` MUST NOT be used for storing C pointers — it corrupts memory in Chez Scheme
- **Chunked arena allocator**: Meta structs are arena-allocated via `arena_alloc`. The arena uses a linked list of chunks (never `realloc`) to prevent invalidating previously allocated pointers when the arena grows mid-forward-pass. Reset frees old chunks and resets the head chunk
- **Tape-based backward pass**: `collectGrads` allocates a mutable gradient array via FFI, seeds it with the initial gradient, then `walk_backward_ext` scans the tape in reverse in C. Scalar ops propagate inline; tensor ops dispatch to C backward kernels. ConstOps with non-zero gradient are collected as (index, grad) pairs. Scheme looks up PIDs and builds `SortedMap`. The tape is reset at the end of `collectGrads` (gen++)
- **Zero-arg FFI CSE trap**: Idris 2 compiles zero-argument `%noinline` definitions as constants evaluated once at load time. `tapeGeneration` must take a dummy argument (the tape index) passed through to `prim__tapeGen` to prevent the Chez backend from caching the result. This also applies to any other FFI wrapper reading mutable state
- **FFI side-effect threading**: `let _ = ffiCall` is dropped by the compiler. FFI functions with side effects must return a value that is used in subsequent computation. `prim__gradAdd` returns the handle (`AnyPtr`), enabling handle threading through the backward pass
- **Gradient clipping**: `adam` clips per-parameter; `adamGlobalClip` clips by global L2 norm (preserves gradient direction). Use `adamGlobalClip` for attention/recurrent models where parameters must coordinate — per-parameter clipping distorts direction and causes periodic loss spikes. Default maxNorm is 50.0 (Collier & Beel); 5.0 was too aggressive
- **Controller output clipping**: `applyLayerVar` clamps raw NTM controller output to [-20, 20] via `clampVar` (straight-through gradient). Prevents extreme head parameters from destabilizing training
- **Curriculum learning**: Available via the Curriculum module for staged training. The PyTorch-aligned NTM (LSTM controller + RMSprop) does not require curriculum — it converges directly with two-phase training. Curriculum was previously required for feedforward controllers (ajithcodesit finding)
- **Tanh memory bounding**: `tanhBound` (exported from Layer.idr) is applied to memory after each write via `map tanhBound`. Keeps memory values in [-1, 1], preventing drift over long sequences (Collier & Beel recommendation). Applied in all three forward paths (generic, Variable, debug)
- **NTM initial addressing**: Read/write addressing weights are initialized to zeros and read output to Kaiming uniform (non-learnable, matching PyTorch reference). `syncLayerBuffers` projects addressing weights onto the probability simplex via `projectWeights` (clamp to [0, epsilon], renormalize) to prevent NaN from `pow(negative, non-integer)` in `focus`
- **Hyperparameter tuning**: Fix algorithmic issues first (bounded activations, correct clipping, efficient backward pass), then use `scripts/sweep.sh` for systematic grid search. Never manually loop over hyperparameters — see `docs/design-decisions.md` for rationale
- **C shared library required**: `build/libidrisml.dylib` must exist before running any example. Build with `make build/libidrisml.dylib`. The library is loaded by the tape init guard in Variable.idr
- **Scheme-native C memory access**: Use Chez Scheme's `foreign-ref`/`foreign-set!` for reading/writing C-allocated arrays instead of calling C functions per element. This avoids the Scheme→C boundary crossing overhead. See `prim__gradAdd`/`prim__gradGet` and `prim__setDouble`/`prim__setInt32` in Variable.idr
- **`prim__seq` for evaluation ordering**: When two FFI side-effect chains must execute in order but have no data dependency, use `prim__seq a b` (Scheme `(lambda (a b) b)`) to force `a` to evaluate before `b` is used. Chez Scheme evaluates function arguments strictly
- **Tensor Foldable reversal**: The `foldr` instance for `Tensor` processes elements in reversed order (head into accumulator first). `toList` produces elements backwards. Use direct `Vect` traversal instead when element order matters (e.g., packing into C buffers)
- **Weight initialization**: `linearLayer`/`rnnLayer` default to Xavier uniform. Biases are always zero. Init strategies compose a variance method with a distribution sampler: `xavier uniform` (default), `xavier normal`, `he normal`, etc. Use `linearLayerWith (fixedRange 1.0)` for the old `U(-1,1)` behavior. NTM memory initialized to constant `1e-6` (Collier & Beel: 3.5x faster convergence vs random). `Sampler.idr` provides `uniform` and `normal` (Box-Muller); `Init.idr` provides `xavier`, `he`, `lecun`, `fixedRange`
- **C-backed softmax/logSoftmax**: `softmaxVar`/`logSoftmaxVar` in Variable.idr use C kernels and record a single SoftmaxOp/LogSoftmaxOp tape entry per vector instead of ~29 scalar entries. `applyLayerVar` dispatches NormalizationLayer "softmax"/"logSoftmax" to these
- **C-backed NTM memory ops**: `batchCosineSimilarityVar`, `readOpVar`, `writeOpVar`, `interpolationWriteVar` in Variable.idr use C kernels (BatchCosSimOp/ReadOpOp/WriteOpOp/InterpolationWriteOp, tags 15-18) to reduce tape entries per NTM timestep. `forwardReadHeadUnboundedVar`/`forwardWriteHeadInterpVar` in Layer.idr wire these into the Variable-specialized NTM forward pass. Generic `forwardReadHeadUnbounded`/`forwardWriteHeadInterp` in Memory.idr remain parameterized on `NormalizationFunction ty` for the Double path
- **C-backed addressing ops**: `interpolateVar`, `shiftVar`, `focusVar` in Variable.idr use C kernels (InterpolateOp/ShiftOp/FocusOp, tags 21-23) replacing ~1400 scalar tape entries per head with 3 tensor ops. `shiftVar` takes a pre-softmax'd kernel (apply `softmaxVar` first). Used in both `forwardReadHeadUnboundedVar` and `forwardReadHeadUnboundedVarBuf` in Layer.idr
- **C-backed LSTM cell op**: `lstmCellVar` in Variable.idr uses a C kernel (LstmCellOp, tag 24) fusing bias add + gate activations (sigmoid/tanh) + cell/hidden update into a single tape entry. Replaces ~1700 scalar entries per LSTM timestep with 1. The two matmul ops (iW×x, rW×h) remain as separate MatVecOps. `applyLayerVar` in Layer.idr dispatches to `lstmCellVar` for the Variable-specialized LSTM path
- **Persistent NtmMemBuf**: NTM memory matrix kept as persistent `NtmMemBuf` C struct across timesteps. Eliminates 4× per-timestep packMatrix (2560 elements each). Buffer initialized in `nameParams`, synced after `applyDeltas` via `syncLayerBuffers`, epoch-cached tape registration via `prim__ntmMemBufEnsure`. Buffer-aware ops: `batchCosineSimilarityVarBuf`, `readOpVarBuf`, `interpolationWriteVarBuf` in Variable.idr
- **Chez Scheme output buffering**: Stdout is fully buffered when redirected to file/pipe (e.g. background tasks). Use `stdbuf -oL ./build/exec/<name>` to force line-buffering for long-running training
