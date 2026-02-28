# idris-ml

Deep learning library in Idris 2 with compile-time tensor shape checking and automatic differentiation.

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

# Run C library tests
make test-c

# Run benchmark (Supervised + RNN + NTM)
make bench
```

Concrete examples:

```bash
idris2 --source-dir src -p contrib -o supervised src/Example/Supervised.idr && ./build/exec/supervised
idris2 --source-dir src -p contrib -o rnn src/Example/Rnn.idr && ./build/exec/rnn
idris2 --source-dir src -p contrib -o ntm src/Example/Ntm.idr && ./build/exec/ntm
# NTM with custom hyperparameters
./build/exec/ntm --lr 0.001 --max-norm 5.0 --epochs 6000 --patience 10 --seed 42
# Hyperparameter sweep (builds once, runs grid in parallel)
bash scripts/sweep.sh --parallel 4
# Quick sweep (2000 epochs for fast screening)
bash scripts/sweep.sh --parallel 4 --quick
```

## Architecture

### Module dependency order (leaves first)

1. **Floating** - Extended `Floating` interface adding `sqrt`
2. **Util** - Helpers: `enumerate`, `permute`, `chunks`
3. **Init** - Weight initialization strategies: `xavierInit`, `heInit`, `lecunInit`, `uniformInit`
4. **Tensor** - Shape-indexed tensor: `Tensor : Vect rank Nat -> Type -> Type`
5. **Math** - Loss functions, activations, linear algebra
6. **Memory** - NTM read/write head operations
7. **Variable** - Tape-based autograd (Wengert list) with Chez Scheme FFI storage
8. **DataPoint** - `DataPoint` and `RecurrentDataPoint` records
9. **Endofunctor** - `emap : (ty -> ty) -> e ty -> e ty` for type-preserving maps
10. **Layer** - Layer/Network types (mutually recursive), forward pass, constructors
11. **Optimizer** - SGD and Adam optimizers with per-parameter or global norm gradient clipping
12. **Schedule** - Learning rate schedules: `constant`, `cosineAnnealing`, `oneCycle`
13. **Backprop** - Training loop: `epoch`, `train`, `trainFrom`, `epochRecurrent`, `trainRecurrent`, `trainRecurrentFrom`, `trainScheduledFrom`, `trainRecurrentScheduledFrom`

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

Every learnable layer must be named before training:

```idris
ll <- nameParams "ll" <$> linearLayer           -- names: ll_weight0, ll_bias0, ...
rnn <- nameParams "rnn" <$> rnnLayer            -- names: rnn_inputWeight0, rnn_bias0, ...
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

## Gotchas

- **Build flags**: Forgetting `--source-dir src` or `-p contrib` produces confusing import errors
- **Elementwise `(*)`**: `Tensor`'s `Num` instance uses elementwise multiply. For matrix-vector products, use `matrixVectorMultiply` or `vectorMatrixMultiply` from Math.idr
- **`paramId` requirement**: Variables without a `paramId` (i.e., `Nothing`) are invisible to gradient collection and won't receive updates. Always call `nameParams`/`nameNetworkParams` before training. `setParamId` writes to both the Variable record and the tape's pid vector
- **No test framework**: No test suite exists. Verify changes by type-checking (`--check`) and running examples (TapeTest.idr for gradient smoke tests)
- **Tape generation staleness**: After `collectGrads` resets the tape (gen++), Variables from the previous epoch are stale. `ensureOnTape` detects this via generation mismatch and re-registers with current `.value`. Same stale Variable used N times creates N Const entries — gradients accumulate correctly via `mergeWith (+)` on paramId
- **Mutual recursion in Layer.idr**: `Layer` and `Network` are mutually recursive (NtmLayer contains a Network). `applyLayer`, `forward`, `nameParams`, `nameNetworkParams`, and `Endofunctor` instances all live in `mutual` blocks
- **NTM dimension calculations**: `ReadHeadInputWidth n w = (w + n) + 3` (key + shift + 3 dynamic params: β, g, γ). The controller output width is `NtmOutputWidth n w = ReadHeadInputWidth n w + WriteHeadInputWidth n w + w`. Getting these wrong causes type errors at network composition
- **NTM head parameters**: β (key strength), g (interpolation gate), γ (sharpening) are dynamic — extracted from controller output. β uses softplus, g uses sigmoid, γ uses `1 + 4*sigmoid(x)` to bound to [1,5]. Unbounded γ via softplus causes vanishing gradients for non-dominant memory positions. Erase vectors use sigmoid, add vectors use tanh (via `2*sig(2x)-1`). See `forwardReadHead`/`forwardWriteHead` in Memory.idr
- **NTM state flow**: `readHeadOutput` from the previous timestep concatenates with current input to form controller input (`NtmInputWidth w = w + w`). Memory, read head, and write head all update each step
- **`logSoftmax` + `nllLoss` for NTM**: Separate softmax + cross-entropy creates autograd intermediate gradients of 1/pp (up to 1e6) that destabilize recurrent/NTM training. Use `logSoftmaxLayer` + `nllLoss` instead — log-softmax avoids tiny probabilities, and NLL has no log so no 1/pp gradient
- **`pow` zero-base NaN**: `pow(0, k)` backward for the exponent computes `0^k * log(0) = 0 * -Inf = NaN`. Fixed by returning 0 when base is 0
- **Detached max in `logSoftmax`**: The max subtraction for numerical stability uses a detached constant (`fromDouble . cast`), not a reference to the max Variable. Otherwise the max element receives incorrect gradients
- **Tape-based backward pass**: `collectGrads` allocates a mutable gradient array via FFI, seeds it with the initial gradient, then scans the tape in reverse. Each entry propagates gradients to its inputs via `prim__gradAdd` (O(1) accumulation). Only parameter entries (non-empty `paramId`) are collected into the output `SortedMap`. The tape is reset at the end of `collectGrads` (gen++)
- **Zero-arg FFI CSE trap**: Idris 2 compiles zero-argument `%noinline` definitions as constants evaluated once at load time. `tapeGeneration` must take a dummy argument (the tape index) passed through to `prim__tapeGen` to prevent the Chez backend from caching the result. This also applies to any other FFI wrapper reading mutable state
- **FFI side-effect threading**: `let _ = ffiCall` is dropped by the compiler. FFI functions with side effects must return a value that is used in subsequent computation. `prim__gradAdd` returns the handle (`AnyPtr`), enabling handle threading through the backward pass
- **Gradient clipping**: `adam` clips per-parameter; `adamGlobalClip` clips by global L2 norm (preserves gradient direction). Use `adamGlobalClip` for attention/recurrent models where parameters must coordinate — per-parameter clipping distorts direction and causes periodic loss spikes
- **Hyperparameter tuning**: Fix algorithmic issues first (bounded activations, correct clipping, efficient backward pass), then use `scripts/sweep.sh` for systematic grid search. Never manually loop over hyperparameters — see `docs/design-decisions.md` for rationale
- **C shared library required**: `build/libidrisml.dylib` must exist before running any example. Build with `make build/libidrisml.dylib`. The library is loaded by the tape init guard in Variable.idr
- **Scheme-native C memory access**: Use Chez Scheme's `foreign-ref`/`foreign-set!` for reading/writing C-allocated arrays instead of calling C functions per element. This avoids the Scheme→C boundary crossing overhead. See `prim__gradAdd`/`prim__gradGet` and `prim__setDouble`/`prim__setInt32` in Variable.idr
- **`prim__seq` for evaluation ordering**: When two FFI side-effect chains must execute in order but have no data dependency, use `prim__seq a b` (Scheme `(lambda (a b) b)`) to force `a` to evaluate before `b` is used. Chez Scheme evaluates function arguments strictly
- **Tensor Foldable reversal**: The `foldr` instance for `Tensor` processes elements in reversed order (head into accumulator first). `toList` produces elements backwards. Use direct `Vect` traversal instead when element order matters (e.g., packing into C buffers)
- **Weight initialization**: `linearLayer`/`rnnLayer` default to Xavier uniform (was `U(-1,1)`). Biases are always zero. Use `linearLayerWith (uniformInit 1.0)` for the old behavior. NTM memory stays at `U(-0.1, 0.1)`. Custom strategies via `linearLayerWith`/`rnnLayerWith` accepting `InitStrategy` from `Init.idr`
- **C-backed softmax/logSoftmax**: `softmaxVar`/`logSoftmaxVar` in Variable.idr use C kernels and record a single SoftmaxOp/LogSoftmaxOp tape entry per vector instead of ~29 scalar entries. `applyLayerVar` dispatches NormalizationLayer "softmax"/"logSoftmax" to these. NTM heads use `forwardReadHeadVar`/`forwardWriteHeadVar` in Layer.idr which call `softmaxVar` for content addressing and shift
- **Chez Scheme output buffering**: Stdout is fully buffered when redirected to file/pipe (e.g. background tasks). Use `stdbuf -oL ./build/exec/<name>` to force line-buffering for long-running training
