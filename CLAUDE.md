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

# Run Idris unit tests (91 tests)
make test

# Run C library tests (76 tests)
make test-c

# Run benchmark (Supervised + RNN + NTM)
make bench
```

Concrete examples:

```bash
idris2 --source-dir src -p contrib -o supervised src/Example/Supervised.idr && ./build/exec/supervised
idris2 --source-dir src -p contrib -o rnn src/Example/Rnn.idr && ./build/exec/rnn
idris2 --source-dir src -p contrib -o ntm-copy src/Example/NtmCopy.idr && ./build/exec/ntm-copy
idris2 --source-dir src -p contrib -o ntm-associative-recall src/Example/NtmAssociativeRecall.idr && ./build/exec/ntm-associative-recall
# NTM copy with custom hyperparameters
./build/exec/ntm-copy --lr 0.001 --max-norm 50.0 --epochs 6000 --patience 10 --seed 42
# NTM copy with diagnostics (summary metrics + train/test comparison)
./build/exec/ntm-copy --diagnose
# NTM copy with verbose diagnostics (summary + raw per-timestep dumps)
./build/exec/ntm-copy --diagnose-verbose
# NTM associative recall with custom hyperparameters
./build/exec/ntm-associative-recall --lr 0.001 --epochs 10000 --patience 800 --seed 42
# Hyperparameter sweep (builds once, runs grid in parallel)
bash scripts/sweep.sh --parallel 4
# Quick sweep (2000 epochs for fast screening)
bash scripts/sweep.sh --parallel 4 --quick
# Sweep for associative recall task
bash scripts/sweep.sh --task recall --parallel 4
bash scripts/sweep.sh --task recall --parallel 4 --quick
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
7. **Variable** - Tape-based autograd (Wengert list) with Chez Scheme FFI storage
8. **DataPoint** - `DataPoint` and `RecurrentDataPoint` records
8b. **Generate** - Random data generation: `SequenceTask` port, `copyTask`/`associativeRecallTask` adapters, `randomBatchVect`
9. **Endofunctor** - `emap : (ty -> ty) -> e ty -> e ty` for type-preserving maps
10. **Layer** - Layer/Network types (mutually recursive), forward pass, constructors, `autoName`
11. **Optimizer** - SGD and Adam optimizers with per-parameter or global norm gradient clipping
12. **Schedule** - Learning rate schedules: `constant`, `cosineAnnealing`, `oneCycle`
13. **Backprop** - Training loop: `epoch`, `train`, `trainFrom`, `epochRecurrent`, `trainRecurrent`, `trainRecurrentFrom`, `trainScheduledFrom`, `trainRecurrentScheduledFrom`
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
  ActivationLayer : String -> ActivationFunction ty -> Layer n n ty
  NormalizationLayer : String -> NormalizationFunction ty -> Layer n n ty
  NtmLayer : Network (NtmInputWidth w) hs (NtmOutputWidth n w) ty -> Matrix n w ty -> ReadHead n ty -> WriteHead n ty -> Vector w ty -> Layer w w ty

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

-- NTM: controller network nested inside NtmLayer
controllerHidden <- linearLayer {i = NtmInputWidth W, o = H}
controllerOut <- linearLayer {i = H, o = NtmOutputWidth N W}
let controller = controllerHidden ~> sigmoidLayer ~> OutputLayer controllerOut
ntm <- ntmLayer {n = N, w = W} controller
let model = nameNetworkParams "ntm" $ ntm ~> OutputLayer logSoftmaxLayer
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
```

### Parameter naming (required for gradient flow)

Every learnable layer must be named before training. Use `autoName` (preferred):

```idris
ll <- linearLayer
let model = autoName $ ll ~> OutputLayer softmaxLayer  -- ll0_weight0, ll0_bias0, ...

-- NTM: ntm0_ll0_weight0 (ctrl hidden), ntm0_ll1_weight0 (ctrl output), ntm0_mem0, ...
let model = autoName $ ntm ~> OutputLayer logSoftmaxLayer
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

### Supervised vs Recurrent API

| Aspect | Supervised | Recurrent |
|--------|-----------|-----------|
| Data type | `DataPoint i o ty` (x, y vectors) | `RecurrentDataPoint i o ty` (xs, ys lists) |
| Forward | `forward` / `forwardMany` | `forwardRecurrent` (folds over list) |
| Train | `train` / `trainFrom` | `trainRecurrent` / `trainRecurrentFrom` |
| State | Not carried between examples | Accumulated within a sequence, reset between sequences |
| Loss fn | `crossEntropy`, `meanSquaredError` | `nllLoss`, `binaryCrossEntropyWithLogits`, `crossEntropy` |

## Conventions

- **Indentation**: 2 spaces for `.idr` files (see `.editorconfig`)
- **Naming**: PascalCase for types/constructors, camelCase for functions/variables
- **Imports**: Idris stdlib first (`Data.Vect`, `System.Random`), then internal modules alphabetically
- **Commits**: Follow [Conventional Commits](https://www.conventionalcommits.org/) — `feat:`, `fix:`, `refactor:`, `docs:`, `chore:`, etc. Keep subject concise (~50 chars), imperative present tense. Commit work regularly in meaningful chunks — one logical change per commit. Never include ads, branding, or promotional text in commit messages or PR descriptions
- **Section dividers**: `----------------------------------------------------------------------` with section titles in Layer.idr style
- **Documentation**: Always update CLAUDE.md, docs/design-decisions.md, and TODO.md when adding features, changing architecture, or making design decisions

## Gotchas

- **Build flags**: Forgetting `--source-dir src` or `-p contrib` produces confusing import errors
- **Elementwise `(*)`**: `Tensor`'s `Num` instance uses elementwise multiply. For matrix-vector products, use `matrixVectorMultiply` or `vectorMatrixMultiply` from Math.idr
- **`paramId` requirement**: Variables without a `paramId` (i.e., `Nothing`) are invisible to gradient collection and won't receive updates. Use `autoName` (preferred) or `nameParams`/`nameNetworkParams` before training. `autoName` assigns type-based prefixes with per-type counters (`ll0`, `ll1`, `rnn0`, `ntm0`, ...) and scopes NTM controller names under their parent (`ntm0_ll0_`, `ntm0_ll1_`), preventing the collision bug in `nameNetworkParams`. `setParamId` writes to both the Variable record and the tape's pid vector
- **Test suite**: Run `make test` for 91 Idris unit tests, `make test-c` for 76 C tests. Tests live in `test/src/Test/*.idr` with `Harness.idr` providing assertion helpers
- **Tape generation staleness**: After `collectGrads` resets the tape (gen++), Variables from the previous epoch are stale. `ensureOnTape` detects this via generation mismatch and re-registers with current `.value`. Same stale Variable used N times creates N Const entries — gradients accumulate correctly via `mergeWith (+)` on paramId
- **Mutual recursion in Layer.idr**: `Layer` and `Network` are mutually recursive (NtmLayer contains a Network). `applyLayer`, `forward`, `nameParams`, `nameNetworkParams`, and `Endofunctor` instances all live in `mutual` blocks
- **NTM dimension calculations**: `ReadHeadInputWidth _ w = (w + ShiftKernelSize) + 3` (key + 3-element shift kernel + 3 dynamic params: β, g, γ). The shift vector is `ShiftKernelSize` (3) elements, not `n` — decoupled from memory slot count. The controller output width is `NtmOutputWidth n w = ReadHeadInputWidth n w + WriteHeadInputWidth n w + w`. Since `ReadHeadInputWidth` no longer depends on `n`, type annotations are needed in `ntmLayer` for `memory`/`readHead`/`writeHead`
- **NTM head parameters**: β (key strength), g (interpolation gate), γ (sharpening) are dynamic — extracted from controller output. β uses softplus, g uses sigmoid, γ uses `1 + 4*sigmoid(x)` to bound to [1,5]. Unbounded γ via softplus causes vanishing gradients for non-dominant memory positions. Erase vectors use sigmoid, add vectors use tanh (via `2*sig(2x)-1`). See `forwardReadHead`/`forwardWriteHead` in Memory.idr
- **NTM state flow**: `readHeadOutput` from the previous timestep concatenates with current input to form controller input (`NtmInputWidth w = w + w`). Memory, read head, and write head all update each step
- **`logSoftmax` + `nllLoss` for NTM**: Separate softmax + cross-entropy creates autograd intermediate gradients of 1/pp (up to 1e6) that destabilize recurrent/NTM training. Use `logSoftmaxLayer` + `nllLoss` instead — log-softmax avoids tiny probabilities, and NLL has no log so no 1/pp gradient
- **`pow` zero-base NaN**: `pow(0, k)` backward for the exponent computes `0^k * log(0) = 0 * -Inf = NaN`. Fixed by returning 0 when base is 0
- **Detached max in `logSoftmax`**: The max subtraction for numerical stability uses a detached constant (`fromDouble . cast`), not a reference to the max Variable. Otherwise the max element receives incorrect gradients
- **Tape-based backward pass**: `collectGrads` allocates a mutable gradient array via FFI, seeds it with the initial gradient, then scans the tape in reverse. Each entry propagates gradients to its inputs via `prim__gradAdd` (O(1) accumulation). Only parameter entries (non-empty `paramId`) are collected into the output `SortedMap`. The tape is reset at the end of `collectGrads` (gen++)
- **Zero-arg FFI CSE trap**: Idris 2 compiles zero-argument `%noinline` definitions as constants evaluated once at load time. `tapeGeneration` must take a dummy argument (the tape index) passed through to `prim__tapeGen` to prevent the Chez backend from caching the result. This also applies to any other FFI wrapper reading mutable state
- **FFI side-effect threading**: `let _ = ffiCall` is dropped by the compiler. FFI functions with side effects must return a value that is used in subsequent computation. `prim__gradAdd` returns the handle (`AnyPtr`), enabling handle threading through the backward pass
- **Gradient clipping**: `adam` clips per-parameter; `adamGlobalClip` clips by global L2 norm (preserves gradient direction). Use `adamGlobalClip` for attention/recurrent models where parameters must coordinate — per-parameter clipping distorts direction and causes periodic loss spikes. Default maxNorm is 50.0 (Collier & Beel); 5.0 was too aggressive
- **Controller output clipping**: `applyLayerVar` clamps raw NTM controller output to [-20, 20] via `clampVar` (straight-through gradient). Prevents extreme head parameters from destabilizing training
- **Curriculum learning**: NTM copy task trains in 3 stages (len 1-3, 1-5, 1-8) with loss thresholds. Fresh random data generated every 100 epochs via `Generate.randomBatchVect`. Required for feedforward controllers (ajithcodesit finding)
- **Tanh memory bounding**: `tanhBound` (exported from Layer.idr) is applied to memory after each write via `map tanhBound`. Keeps memory values in [-1, 1], preventing drift over long sequences (Collier & Beel recommendation). Applied in all three forward paths (generic, Variable, debug)
- **Learned initial addressing**: Read/write head addressing weights and readHeadOutput are named parameters (`rAddr`, `wAddr`, `rOut` prefixes in `nameParams`). After `applyDeltas`, `syncLayerBuffers` projects addressing weights onto the probability simplex via `projectWeights` (clamp to [0, epsilon], renormalize) to prevent NaN from `pow(negative, non-integer)` in `focus`
- **Hyperparameter tuning**: Fix algorithmic issues first (bounded activations, correct clipping, efficient backward pass), then use `scripts/sweep.sh` for systematic grid search. Never manually loop over hyperparameters — see `docs/design-decisions.md` for rationale
- **C shared library required**: `build/libidrisml.dylib` must exist before running any example. Build with `make build/libidrisml.dylib`. The library is loaded by the tape init guard in Variable.idr
- **Scheme-native C memory access**: Use Chez Scheme's `foreign-ref`/`foreign-set!` for reading/writing C-allocated arrays instead of calling C functions per element. This avoids the Scheme→C boundary crossing overhead. See `prim__gradAdd`/`prim__gradGet` and `prim__setDouble`/`prim__setInt32` in Variable.idr
- **`prim__seq` for evaluation ordering**: When two FFI side-effect chains must execute in order but have no data dependency, use `prim__seq a b` (Scheme `(lambda (a b) b)`) to force `a` to evaluate before `b` is used. Chez Scheme evaluates function arguments strictly
- **Tensor Foldable reversal**: The `foldr` instance for `Tensor` processes elements in reversed order (head into accumulator first). `toList` produces elements backwards. Use direct `Vect` traversal instead when element order matters (e.g., packing into C buffers)
- **Weight initialization**: `linearLayer`/`rnnLayer` default to Xavier uniform. Biases are always zero. Init strategies compose a variance method with a distribution sampler: `xavier uniform` (default), `xavier normal`, `he normal`, etc. Use `linearLayerWith (fixedRange 1.0)` for the old `U(-1,1)` behavior. NTM memory initialized to constant `1e-6` (Collier & Beel: 3.5x faster convergence vs random). `Sampler.idr` provides `uniform` and `normal` (Box-Muller); `Init.idr` provides `xavier`, `he`, `lecun`, `fixedRange`
- **C-backed softmax/logSoftmax**: `softmaxVar`/`logSoftmaxVar` in Variable.idr use C kernels and record a single SoftmaxOp/LogSoftmaxOp tape entry per vector instead of ~29 scalar entries. `applyLayerVar` dispatches NormalizationLayer "softmax"/"logSoftmax" to these
- **C-backed NTM memory ops**: `batchCosineSimilarityVar`, `readOpVar`, `writeOpVar` in Variable.idr use C kernels (BatchCosSimOp/ReadOpOp/WriteOpOp, tags 15-17) to reduce ~12,500 tape entries per NTM timestep to ~4 C-backed entries. `forwardReadHeadVar`/`forwardWriteHeadVar` in Layer.idr wire these into the Variable-specialized NTM forward pass. Generic `forwardReadHead`/`forwardWriteHead` in Memory.idr remain parameterized on `NormalizationFunction ty` for the Double path
- **Chez Scheme output buffering**: Stdout is fully buffered when redirected to file/pipe (e.g. background tasks). Use `stdbuf -oL ./build/exec/<name>` to force line-buffering for long-running training
