# PyTorch to idris-ml

A concept mapping for PyTorch users. This covers the mental model — for API details, use `:t`, `:doc`, and `:browse` in the REPL or Jupyter notebooks.

## The big difference

PyTorch mutates models in-place. idris-ml is purely functional — `forward` returns `(updatedModel, output)`:

```python
# PyTorch
output = model(input)           # model mutated in-place (hidden state, batch norm stats)
loss = criterion(output, target)
loss.backward()                 # gradients accumulated in-place
optimizer.step()                # weights updated in-place
optimizer.zero_grad()
```

```idris
-- idris-ml
let (model', output) = forward model input    -- new model returned (state updated functionally)
-- backward + optimizer step fused in epochNative (C-level, all-at-once)
let (model'', loss) = epochNative opt data lossFn model
```

No mutation, no `zero_grad`, no manual backward/step separation. The epoch function handles the full train step.

## Tensors

| PyTorch | idris-ml | Notes |
|---------|----------|-------|
| `torch.tensor([1,2,3])` | `VTensor [STensor 1, STensor 2, STensor 3]` | Shape in the type: `Vector 3 Double` |
| `torch.zeros(3,4)` | `the (Matrix 3 4 Double) (pure 0)` | `pure` fills with a value |
| `x.shape` | No runtime query — shape is in the type | `Vector 3 Double` means shape is `[3]`, always |
| `x + y` | `x + y` | Elementwise, same as PyTorch |
| `x * y` | `x * y` | **Elementwise** — not matmul |
| `x @ y` | `x <> y` | Infix matmul, dimension-checked at compile time |
| `x.reshape(3,4)` | `reshapeToMatrix v` | Requires `auto` proof that product of dims matches |
| Runtime `RuntimeError: shape mismatch` | Compile error: `Can't unify 8 with 5` | The point of the library |

## Models

| PyTorch | idris-ml | Notes |
|---------|----------|-------|
| `nn.Module` | `LayerLike` interface | Defines `forward`, `show`, parameter access, etc. |
| `nn.Linear(4, 8)` | `linearLayer {i=4, o=8}` | Returns `IO (AnyLayer 4 8 ty)` — IO because init is random |
| `nn.Sequential(l1, l2, l3)` | `l1 ~> l2 ~> OutputLayer l3` | Type-checks that output dims match input dims |
| `model.parameters()` | `networkParamIds model` | Returns `List String` (parameter names) |
| `model(x)` | `forward model x` | Returns `(updatedModel, output)` — not just output |
| `model.train()` | `setNetworkTraining True model` | Returns new model (no mutation) |
| `model.eval()` | `setNetworkTraining False model` | Affects dropout, batch norm |

### Layer mapping

| PyTorch | idris-ml | Constructor |
|---------|----------|-------------|
| `nn.Linear` | `linearLayer` | `IO (AnyLayer i o ty)` |
| `nn.RNN` | `rnnLayer` | `IO (AnyLayer i o ty)` |
| `nn.LSTM` | `lstmLayer` | `IO (AnyLayer i o ty)` |
| `nn.GRU` | `gruLayer` | `IO (AnyLayer i o ty)` |
| `nn.Conv1d` | `conv1dLayer` | `IO (AnyLayer (inC*len) (outC*ConvOutDim len k pad) ty)` |
| `nn.Conv2d` | `conv2dLayer` | `IO (AnyLayer (inC*(h*w)) (outC*(ConvOutDim h kH padH * ConvOutDim w kW padW)) ty)` |
| `nn.MaxPool1d` | `maxPool1dLayer` | `AnyLayer (c*len) (c*PoolOutDim len k s) ty` |
| `nn.MaxPool2d` | `maxPool2dLayer` | Similar, 2D |
| `nn.Dropout` | `dropoutLayer` | `IO (AnyLayer n n ty)` |
| `nn.BatchNorm1d` | `batchNormLayer` | `IO (AnyLayer n n ty)` |
| `nn.LayerNorm` | `layerNormLayer` | `IO (AnyLayer n n ty)` |
| `nn.Embedding` | `embeddingLayer` | `IO (AnyLayer vocabSize embedDim ty)` |
| `nn.ReLU` | `reluLayer` | `AnyLayer n n ty` (no IO — no parameters) |
| `nn.Tanh` | `tanhLayer` | `AnyLayer n n ty` |
| `nn.Sigmoid` | `sigmoidLayer` | `AnyLayer n n ty` |
| `nn.Softmax` | `softmaxLayer` | `AnyLayer n n ty` |
| `nn.LogSoftmax` | `logSoftmaxLayer` | `AnyLayer n n ty` |

Note: layers with learnable parameters return `IO` (random initialization). Stateless layers are pure values.

**Memory-augmented architectures:**

| PyTorch | idris-ml | Notes |
|---------|----------|-------|
| Custom NTM (Graves 2014) | `ntmLayer` | `IO (AnyLayer i o ty)`, LSTM controller + memory |
| Custom DNC (Graves 2016) | `dncLayer` | `IO (AnyLayer i o ty)`, extends NTM with temporal links, allocation, multi-head read |

### Model composition

```python
# PyTorch
model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 3),
    nn.Softmax(dim=-1)
)
```

```idris
-- idris-ml
l1 <- linearLayer {i=2, o=8}
l2 <- linearLayer {i=8, o=3}
let model = autoName (l1 ~> reluLayer ~> l2 ~> OutputLayer softmaxLayer)
```

The compiler checks that each layer's output dimension matches the next layer's input. Changing `o=8` to `o=10` without updating `i=8` is a compile error.

`autoName` assigns parameter names (`l1_weight0`, `l1_bias0`, ...). Without it, parameters are invisible to the gradient system and training silently does nothing.

## Training data

| PyTorch | idris-ml | Notes |
|---------|----------|-------|
| `(input_tensor, target_tensor)` | `DataPoint i o ty` | Dimensions `i`, `o` are in the type |
| `[(x1,y1), (x2,y2), ...]` | `Vect n (DataPoint i o ty)` | Length `n` is also in the type |
| Sequence data | `RecurrentDataPoint i o ty` | `.xs : List (Vector i ty)`, `.ys : List (Vector o ty)` |
| `DataLoader` | `mkIndexedLoader` / `mkGeneratorLoader` | Batched, with shuffle/repeat |

```idris
-- A data point for a 2-input, 3-class classifier
MkDataPoint (VTensor [STensor 1.5, STensor (-2.7)])   -- input: Vector 2 Double
            (VTensor [STensor 0, STensor 1, STensor 0]) -- target: Vector 3 Double (one-hot)
```

The compiler ensures `DataPoint 2 3 Double` can only be used with a model whose input is 2 and output is 3.

## Loss functions

| PyTorch | idris-ml | Notes |
|---------|----------|-------|
| `nn.CrossEntropyLoss()` | `crossEntropy` | Combines log-softmax + NLL |
| `nn.NLLLoss()` | `nllLoss` | Expects log-probabilities |
| `nn.MSELoss()` | `meanSquaredError` | |
| `nn.BCELoss()` | `binaryCrossEntropy` | Expects probabilities |
| `nn.BCEWithLogitsLoss()` | `binaryCrossEntropyWithLogits` | Numerically stable |

All loss functions have type `LossFunction ty = {n : Nat} -> Vector n ty -> Vector n ty -> ty` — prediction and target must have the same dimension.

## Optimizers

| PyTorch | idris-ml | Notes |
|---------|----------|-------|
| `optim.SGD(lr=0.01)` | `nativeSgd 0.01` | |
| `optim.Adam(lr, betas, eps)` | `nativeAdamGlobalClip lr beta1 beta2 eps maxNorm` | Includes gradient clipping |
| `optim.AdamW(lr, betas, eps, wd)` | `nativeAdamW lr beta1 beta2 eps weightDecay maxNorm` | Decoupled weight decay |
| `optim.RMSprop(lr, alpha, eps)` | `nativeRmsprop lr alpha eps clipVal momentum` | |

These are C-level native optimizers. The backward pass, gradient clipping, and parameter update happen in one fused step inside the epoch function.

## Training loop

```python
# PyTorch — manual loop
for epoch in range(1000):
    output = model(input)
    loss = criterion(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

```idris
-- idris-ml — declarative
(trained, epochs, finalLoss) <- runTraining
  (\m, d => epochNative opt d lossFn m)  -- epoch function
  (pure trainingData)                     -- data source (IO)
  (simpleConfig 1000)                     -- config (epochs, early stopping)
  model                                   -- initial model
```

`runTraining` handles: epoch loop, loss logging, NaN detection, early stopping, timing.

### Early stopping

| PyTorch | idris-ml |
|---------|----------|
| Manual patience counter | `patienceConfig totalEpochs patience` |
| Manual loss windowing | `MkTrainConfig epochs logEvery (WindowedAvg threshold window patience) noMetrics` |
| No early stopping | `simpleConfig totalEpochs` |

### Training modes

| Scenario | Epoch function | Data type |
|----------|---------------|-----------|
| Feedforward (classification, regression) | `epochNativeTensorPre` | `Vect n (TensorDataPoint i o)` |
| Recurrent (RNN, LSTM, GRU) | `epochRecurrentNativeTensor` | `Vect n (RecurrentDataPoint i o Double)` |
| Two-phase (NTM encode/decode) | `epochTwoPhaseBceNative` | `Vect n (TwoPhaseDataPoint i o Variable)` |

## Saving and loading

```python
# PyTorch
torch.save(model.state_dict(), "model.pt")
model.load_state_dict(torch.load("model.pt"))
```

```idris
-- idris-ml (SafeTensors format — interoperable with PyTorch and MLX)
ok <- saveModel "model.safetensors"
ok <- loadModel "model.safetensors"
let model' = emap refreshValue model  -- refresh cached values after load
```

SafeTensors files can be loaded in Python: `safetensors.torch.load_file("model.safetensors")`.

## Evaluation

```python
# PyTorch
model.eval()
with torch.no_grad():
    output = model(test_input)
```

```idris
-- idris-ml
let evalModel = setNetworkTraining False trained
let dblModel = toDoubleNetwork (emap refreshValue evalModel)
let (_, output) = forward dblModel testInput
```

`toDoubleNetwork` converts from `Variable` (autograd-tracked) to `Double` (pure evaluation). No autograd overhead.

## Key differences to internalize

1. **`forward` returns the model**: `(model', output) = forward model input`. The returned model has updated state (RNN hidden state, batch norm running stats). Always use the returned model for the next call.

2. **`autoName` is required**: without it, parameters have no names, the optimizer can't find them, and training silently does nothing. Always call `autoName` after composing a model.

3. **No separate backward/step**: the epoch function (`epochNative`, etc.) fuses forward, backward, and optimizer step. You don't manually call backward or step.

4. **Shapes are types, not values**: `Vector 3 Double` is a different type from `Vector 4 Double`. You can't write code that accidentally mixes them — it won't compile.

5. **IO for initialization, pure for computation**: layer constructors return `IO` because they use random number generators. Once built, `forward` is pure.
