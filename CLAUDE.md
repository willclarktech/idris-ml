# idris-ml

Deep learning library in Idris 2 with compile-time tensor shape checking and automatic differentiation.

## Build Commands

There is no `.ipkg` file. Both `--source-dir src` and `-p contrib` are always required.

```bash
# Type-check a module
idris2 --source-dir src -p contrib --check src/<File>.idr

# Build an example
idris2 --source-dir src -p contrib -o <name> src/Example/<Name>.idr

# Run a built example
./build/exec/<name>
```

Concrete examples:

```bash
idris2 --source-dir src -p contrib -o supervised src/Example/Supervised.idr && ./build/exec/supervised
idris2 --source-dir src -p contrib -o rnn src/Example/Rnn.idr && ./build/exec/rnn
idris2 --source-dir src -p contrib -o ntm src/Example/Ntm.idr && ./build/exec/ntm
idris2 --source-dir src -p contrib -o ntmdebug src/Example/NtmDebug.idr && ./build/exec/ntmdebug
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
10. **Backprop** - Training loop: `zeroGrad`, `backward`, `step`, `epoch`, `train`

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
let model = nameNetworkParams "ntm" $ ntm ~> OutputLayer softmaxLayer
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
-- Backprop.idr: zeroGrad -> forward (via calculateLoss) -> backward -> step
epoch lr dataPoints lossFn model =
  let zeroed = zeroGrad model                        -- 1. Clear gradients
      loss = calculateLoss lossFn zeroed dataPoints  -- 2. Forward pass + loss
      propagated = backward loss zeroed              -- 3. Backprop gradients
   in step lr propagated                             -- 4. Gradient descent update
```

### Parameter naming (required for gradient flow)

Every learnable layer must be named before training:

```idris
ll <- nameParams "ll" <$> linearLayer           -- names: ll_weight0, ll_bias0, ...
rnn <- nameParams "rnn" <$> rnnLayer            -- names: rnn_inputWeight0, rnn_bias0, ...
nameNetworkParams "ntm" $ ntm ~> OutputLayer softmaxLayer  -- recursive naming
```

### Endofunctor for type-preserving transforms

`emap` maps `(ty -> ty)` over Layer/Network without changing shape types. Used by `zeroGrad`, `step`, `backward`:

```idris
zeroGrad : Network i hs o Variable -> Network i hs o Variable
zeroGrad = emap {grad := 0}

step lr = emap (updateParam lr)
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
| Train | `train` / `epoch` | `trainRecurrent` / `epochRecurrent` |
| State | Not carried between examples | Accumulated across timesteps |
| Loss fn | `crossEntropy`, `meanSquaredError` | `binaryCrossEntropyWithLogits`, `crossEntropy` |

## Conventions

- **Indentation**: 2 spaces for `.idr` files (see `.editorconfig`)
- **Naming**: PascalCase for types/constructors, camelCase for functions/variables
- **Imports**: Idris stdlib first (`Data.Vect`, `System.Random`), then internal modules alphabetically
- **Commit messages**: Imperative present tense, concise (~50 chars). Examples: "Add sqrt/complement to Tensor", "Simplify Network", "Tidy Supervised". Commit work regularly in meaningful chunks — one logical change per commit
- **Section dividers**: `----------------------------------------------------------------------` with section titles in Layer.idr style

## Gotchas

- **Build flags**: Forgetting `--source-dir src` or `-p contrib` produces confusing import errors
- **Elementwise `(*)`**: `Tensor`'s `Num` instance uses elementwise multiply. For matrix-vector products, use `matrixVectorMultiply` or `vectorMatrixMultiply` from Math.idr
- **`paramId` requirement**: Variables without a `paramId` (i.e., `Nothing`) are invisible to `gradMap` and won't receive gradient updates. Always call `nameParams`/`nameNetworkParams` before training
- **No test framework**: No test suite exists. Verify changes by type-checking (`--check`) and running examples
- **`updateParam` creates fresh Variables**: `updateParam` in Backprop.idr constructs a new `Var` with empty `back`/`children` to allow GC of the computation graph. This is intentional, not a bug
- **Mutual recursion in Layer.idr**: `Layer` and `Network` are mutually recursive (NtmLayer contains a Network). `applyLayer`, `forward`, `nameParams`, `nameNetworkParams`, and `Endofunctor` instances all live in `mutual` blocks
- **NTM dimension calculations**: The controller output width is `NtmOutputWidth n w = ReadHeadInputWidth n w + WriteHeadInputWidth n w + w`. Getting these wrong causes type errors at network composition
- **NTM state flow**: `readHeadOutput` from the previous timestep concatenates with current input to form controller input (`NtmInputWidth w = w + w`). Memory, read head, and write head all update each step
