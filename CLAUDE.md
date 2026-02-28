# idris-ml

Deep learning library in Idris 2 with compile-time tensor shape checking and automatic differentiation.

## Build Commands

```bash
# Type-check all library modules
idris2 --build idris-ml.ipkg

# Type-check a single module
idris2 --source-dir src -p contrib --check src/<File>.idr

# Build an example (examples aren't in the package, so manual flags are needed)
idris2 --source-dir src -p contrib -o <name> src/Example/<Name>.idr

# Run a built example
./build/exec/<name>
```

Concrete examples:

```bash
idris2 --source-dir src -p contrib -o supervised src/Example/Supervised.idr && ./build/exec/supervised
idris2 --source-dir src -p contrib -o rnn src/Example/Rnn.idr && ./build/exec/rnn
idris2 --source-dir src -p contrib -o ntm src/Example/Ntm.idr && ./build/exec/ntm
# NTM with custom hyperparameters
./build/exec/ntm --lr1 0.001 --lr2 0.0003 --max-norm 5.0 --epochs1 3000 --epochs2 3000 --seed 42
# Hyperparameter sweep (builds once, runs grid in parallel)
bash scripts/sweep.sh --parallel 4
```

## Architecture

### Module dependency order (leaves first)

1. **Floating** - Extended `Floating` interface adding `sqrt`
2. **Util** - Helpers: `enumerate`, `permute`, `chunks`
3. **Tensor** - Shape-indexed tensor: `Tensor : Vect rank Nat -> Type -> Type`
4. **Math** - Loss functions, activations, linear algebra
5. **Memory** - NTM read/write head operations
6. **Variable** - Autograd node with computational graph
7. **DataPoint** - `DataPoint` and `RecurrentDataPoint` records
8. **Endofunctor** - `emap : (ty -> ty) -> e ty -> e ty` for type-preserving maps
9. **Layer** - Layer/Network types (mutually recursive), forward pass, constructors
10. **Optimizer** - SGD and Adam optimizers with per-parameter or global norm gradient clipping
11. **Backprop** - Training loop: `epoch`, `train`, `trainFrom`, `epochRecurrent`, `trainRecurrent`, `trainRecurrentFrom`

### Core type signatures

```idris
-- Tensor.idr
data Tensor : Vect rank Nat -> Type -> Type where
  STensor : ty -> Tensor [] ty
  VTensor : Vect dim (Tensor dims ty) -> Tensor (dim :: dims) ty

Scalar = Tensor []
Vector elems = Tensor [elems]
Matrix rows columns = Tensor [rows, columns]

-- Variable.idr
record Variable where
  constructor Var
  paramId : Maybe String
  value : Double
  grad : Double
  back : Double -> List Double
  children : List Variable

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
-- Backprop.idr: forward (via calculateLoss) -> collectGrads -> optimizer step -> apply deltas
epoch opt dataPoints lossFn model st =
  let loss = calculateLoss lossFn model dataPoints   -- 1. Forward pass + loss
      grads = collectGrads 1.0 loss                  -- 2. Backprop gradients
      (deltas, st') = opt.step grads st              -- 3. Optimizer computes deltas
  in (emap (applyDeltas deltas) model, st')          -- 4. Apply parameter updates
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
- **`paramId` requirement**: Variables without a `paramId` (i.e., `Nothing`) are invisible to `gradMap` and won't receive gradient updates. Always call `nameParams`/`nameNetworkParams` before training
- **No test framework**: No test suite exists. Verify changes by type-checking (`--check`) and running examples
- **`updateParam` creates fresh Variables**: `updateParam` in Backprop.idr constructs a new `Var` with empty `back`/`children` to allow GC of the computation graph. This is intentional, not a bug
- **Mutual recursion in Layer.idr**: `Layer` and `Network` are mutually recursive (NtmLayer contains a Network). `applyLayer`, `forward`, `nameParams`, `nameNetworkParams`, and `Endofunctor` instances all live in `mutual` blocks
- **NTM dimension calculations**: `ReadHeadInputWidth n w = (w + n) + 3` (key + shift + 3 dynamic params: β, g, γ). The controller output width is `NtmOutputWidth n w = ReadHeadInputWidth n w + WriteHeadInputWidth n w + w`. Getting these wrong causes type errors at network composition
- **NTM head parameters**: β (key strength), g (interpolation gate), γ (sharpening) are dynamic — extracted from controller output. β uses softplus, g uses sigmoid, γ uses `1 + 4*sigmoid(x)` to bound to [1,5]. Unbounded γ via softplus causes vanishing gradients for non-dominant memory positions. Erase vectors use sigmoid, add vectors use tanh (via `2*sig(2x)-1`). See `forwardReadHead`/`forwardWriteHead` in Memory.idr
- **NTM state flow**: `readHeadOutput` from the previous timestep concatenates with current input to form controller input (`NtmInputWidth w = w + w`). Memory, read head, and write head all update each step
- **`logSoftmax` + `nllLoss` for NTM**: Separate softmax + cross-entropy creates autograd intermediate gradients of 1/pp (up to 1e6) that destabilize recurrent/NTM training. Use `logSoftmaxLayer` + `nllLoss` instead — log-softmax avoids tiny probabilities, and NLL has no log so no 1/pp gradient
- **`pow` zero-base NaN**: `pow(0, k)` backward for the exponent computes `0^k * log(0) = 0 * -Inf = NaN`. Fixed by returning 0 when base is 0
- **Detached max in `logSoftmax`**: The max subtraction for numerical stability uses a detached constant (`fromDouble . cast`), not a reference to the max Variable. Otherwise the max element receives incorrect gradients
- **Memoized DAG traversal**: `collectGrads` uses `topoSort` which memoizes visited nodes via `SortedSet Nat` of `nodeId`s. Each `Variable` gets a unique `nodeId` from an FFI counter. Without memoization, the DAG would be traversed exponentially
- **Gradient clipping**: `adam` clips per-parameter; `adamGlobalClip` clips by global L2 norm (preserves gradient direction). Use `adamGlobalClip` for attention/recurrent models where parameters must coordinate — per-parameter clipping distorts direction and causes periodic loss spikes
- **Hyperparameter tuning**: Fix algorithmic issues first (bounded activations, correct clipping, efficient backward pass), then use `scripts/sweep.sh` for systematic grid search. Never manually loop over hyperparameters — see `docs/design-decisions.md` for rationale
- **Chez Scheme output buffering**: Stdout is fully buffered when redirected to file/pipe (e.g. background tasks). Use `stdbuf -oL ./build/exec/<name>` to force line-buffering for long-running training
