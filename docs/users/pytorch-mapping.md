# PyTorch to idris-ml

A concept mapping for PyTorch users. This covers the mental model — for API details, use
`:t`, `:doc`, and `:browse` in the REPL or Jupyter notebooks. For the *why*, start at
[Why idris-ml](why-idris-ml.md).

## The big difference

PyTorch mutates models in-place. In idris-ml a **model is a single-owner linear
resource**: `forward` *consumes* the model handle and threads back a fresh one (plus the
output, riding a `(!*)` bang), all inside `Control.Linear.LIO.L IO`. Parameters live in a
C-side registry; the optimizer updates them via a fused train step.

```python
# PyTorch
output = model(input)           # model mutated in-place
loss = criterion(output, target)
optimizer.zero_grad()
loss.backward()                 # gradients accumulated in-place
optimizer.step()                # weights updated in-place
```

```idris
-- idris-ml: thread the linear model; fit owns zero_grad + backward + clip + step
(MkBang (epochs, loss) # trained) <-
  fitSupervised opt lossFn (batched stream) (simpleConfig 1000) model
```

No mutation, no manual `zero_grad` / backward / step. The `fit` driver runs the full train
step; if you need a custom loop, you consume-and-thread the model yourself through
`forward` and `trainStep`.

## Tensors

| PyTorch | idris-ml | Notes |
|---------|----------|-------|
| `torch.tensor([1,2,3])` | `tensor {dims=[3]} (FromVect [1,2,3])` | Shape in the type: `Tensor [3] ex dt g` |
| `torch.zeros(3,4)` | `tensor {dims=[3,4]} Zeros` | `InitSpec`: `Zeros`/`Const x`/`Normal μ σ`/`Uniform`/`FromVect` |
| a learnable parameter | `param "w" (Normal 0.0 0.02)` | registers in the optimizer registry |
| `x.shape` | no runtime query — shape is in the type | `Tensor [3] …` *is* shape `[3]`, always |
| `x + y` | `!(x +. y)` (or `tadd x y`) | elementwise; ops are `IO`-typed, use bang notation |
| `x * y` | `!(x *. y)` (or `tmul x y`) | **elementwise** — not matmul |
| `W @ x` | `tmv w x` / `tlinear2d w x b` | matrix–vector / fused linear, dimension-checked at compile time |
| `x.reshape(...)` | shape lives in the type | `Array.splitAt` for structural splits; `TVec`/`TMat` aliases for multiplicative shapes |
| Runtime `RuntimeError: shape mismatch` | Compile error: `Mismatch between: 8 and 5` | the point of the library |

Smart constructors are `IO`-typed (`tadd`, `tmul`, `ttanh`, …); elementwise infix aliases
`(+.)`, `(-.)`, `(*.)` and scalar-left `(*:)` work on already-evaluated tensors with bang
notation.

## Models

| PyTorch | idris-ml | Notes |
|---------|----------|-------|
| `nn.Module` | `Nn.Module` interface | batched-first linear `forward` |
| `nn.Linear(4, 8)` | `linear {i=4} {o=8}` | returns `Init (Linear 4 8 ex dt g)` |
| `nn.Sequential(l1, l2, l3)` | `l1 ~~> l2 ~~> l3 ~~> Nil` | a `Seq`; checks output dims match input dims |
| `model.parameters()` | the C param registry (`Nn.Init`) | `Nn.Group.groupOf submodel` for per-network scoping |
| `model(x)` | `forward model x` / `forwardSeq model x` | linear: consumes model, returns `(!* out) `LPair` model` |
| `model.eval()` | `eval model` | linear; retypes the model `WithGrad → NoGrad` |
| `model.train()` | `trainable model` | inverse of `eval` |

### Layer mapping

Every layer with learnable parameters is built in the `Init` monad (it allocates +
registers params); `runInit` / `runInitL` populates the registry, deriving names from the
scope path. Stateless activations are plain values.

| PyTorch | idris-ml | Builds to |
|---------|----------|-----------|
| `nn.Linear` | `linear {i} {o}` | `Init (Linear i o ex dt g)` |
| `nn.RNN` / `nn.LSTM` / `nn.GRU` | `rnn` / `lstm` / `gru` | recurrent (`Nn.Recurrent`) |
| `nn.Conv1d` / `nn.Conv2d` | `conv1d` / `conv2d` | output dim via `ConvOutDim` (type-level) |
| `nn.MaxPool1d` / `nn.MaxPool2d` | `maxPool1d` / `maxPool2d` | output dim via `PoolOutDim` |
| `nn.Dropout` | `dropout p` | `Init (Dropout n ex dt g)` |
| `nn.BatchNorm1d` | `batchNorm` | |
| `nn.LayerNorm` | `layerNorm` | |
| `nn.Embedding` | `embedding` | |
| `nn.ReLU` / `nn.Tanh` / `nn.Sigmoid` / `nn.GELU` | `reluA` / `tanhA` / `sigmoidA` / `geluA` | stateless |
| Custom NTM (Graves 2014) | `ntm` | LSTM controller + external memory (`Nn.Recurrent`) |
| Custom DNC (Graves 2016) | `dnc` | temporal links, allocation, multi-head read |

> Don't put a `softmax` layer in the chain. Apply `tlogSoftmax1d` to raw logits and feed
> `tnllLoss` — a softmax layer creates `1/p` intermediates that blow up. (See gotchas.)

### Model composition

```python
# PyTorch
model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 3))
```

```idris
-- idris-ml
Model : Type
Model = Seq 2 3 Ex F WithGrad

mkModel : Init Model
mkModel = do
  l1 <- linear {i=2} {o=8}
  l2 <- linear {i=8} {o=3}
  pure (l1 ~~> reluA ~~> l2 ~~> Nil)

trained <- runInitL mkModel    -- populates the C param registry
```

The compiler checks each layer's output dimension matches the next layer's input —
changing `o=8` to `o=10` without updating `i=8` is a compile error. There is **no
`autoName`**: `runInit` / `runInitL` derive parameter names from the scope path (the
PyTorch `state_dict` convention), and parameters reach the optimizer only through that
registry.

## Training data

PyTorch's three orthogonal joints map directly: `Dataset` (indexed access), `ShuffleSpec`
(order), `DataStream` (batching + collation).

| PyTorch | idris-ml | Notes |
|---------|----------|-------|
| `Dataset.__getitem__` | `Dataset { size : Nat; item : Fin size -> IO sample }` | `Fin` ⇒ out-of-bounds unrepresentable |
| in-memory dataset | `fromVect` / `fromVectIO` | hold host values, materialise fresh tensors per access |
| file / IO dataset | `fromIndexed size cb` | MNIST-family via `idxDataset` |
| `DataLoader` | `stream spec ds` + `batched` | shuffle via Fisher-Yates C engine; collation C-side |
| a sample | `(Tensor [i] …, Tensor [o] …)` | `batched` collates into `([b,i], [b,o])` |

## Loss functions

| PyTorch | idris-ml | Notes |
|---------|----------|-------|
| `nn.CrossEntropyLoss()` | `tlogSoftmax1d` then `tnllLoss` | apply to raw logits |
| `nn.NLLLoss()` | `tnllLoss` | expects log-probabilities |
| `nn.MSELoss()` | `tmseLoss` | **sum** reduction — scale by `1/n` for PyTorch's mean |
| `nn.BCELoss()` | `tbceLoss` | expects probabilities |

## Optimizers

Four IO constructors over `OptimOpts` (`defaultOpts` = PyTorch defaults; record-update to
override `beta1`/`beta2`/`eps`/`clip`):

| PyTorch | idris-ml |
|---------|----------|
| `optim.SGD(lr)` | `sgd lr defaultOpts` |
| `optim.Adam(lr, betas, eps)` | `adam lr defaultOpts` |
| `optim.AdamW(lr, …, wd)` | `adamW lr weightDecay defaultOpts` |
| `optim.RMSprop(lr, alpha, momentum)` | `rmsprop lr {alpha} {momentum} defaultOpts` |

Per-network scoping (`actor`/`critic`) is done after construction via `Train.Freeze`:
`restrictTo opt (groupOf net)` limits an optimizer's step to one network's exact param
set, and `freezeGroup opt =<< namesMatching (isPrefixOf "bert.")` freezes a name group
(LR override 0). Schedules: `withSchedule sched opt` + `tick opt epoch`.

## Training loop

```python
# PyTorch — manual loop
for epoch in range(1000):
    output = model(input)
    loss = criterion(output, target)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
```

```idris
-- idris-ml — one driver for everything
(MkBang (epochs, finalLoss) # trained) <-
  fitSupervised opt lossFn (batched stream) (simpleConfig 1000) model
```

`fit` owns the epoch loop, schedule `tick`, early stopping, checkpointing, NaN handling.
For RL / custom control flow, pass your own `EpochStep` to `fit`, or compose the engine
pieces (`runEpochLoop`, `withEpoch`, `postEpoch`, `earlyStopMachine`) — and inside the
loss body consume-and-thread the model through `forward` + a single `trainStep opt loss`.

### Early stopping

| PyTorch | idris-ml |
|---------|----------|
| no early stopping | `simpleConfig totalEpochs` |
| patience counter | `patienceConfig totalEpochs patience` |

## Saving and loading

```python
# PyTorch
torch.save(model.state_dict(), "model.pt")
model.load_state_dict(torch.load("model.pt"))
```

```idris
-- idris-ml — backend-agnostic SafeTensors
saveAll "model.safetensors"
res <- load "model.safetensors" defaultLoadOpts   -- Either LoadError ()
```

`.safetensors` is the only on-disk format; `allowCast = False` (the default) rejects any
dtype mismatch, `only := Just pfx` does a prefix-filtered warm-start. Python interop:
`safetensors.torch.load_file(...)` / MLX `mx.load(...)`. Loading real HuggingFace
checkpoints is `fromPretrained` in [`idris-transformers`](idris-transformers.md).

## Evaluation

```python
# PyTorch
model.eval()
with torch.no_grad():
    output = model(test_input)
```

```idris
-- idris-ml: eval retypes the model to NoGrad (linear); withNoGrad is the perf knob
infer <- eval trained                            -- infer : Model … NoGrad
(MkBang out # infer') <- forward infer testInput -- out : Tensor [b,o] … NoGrad
discard infer'
```

`eval` flips every param's `requires_grad` off and retypes `WithGrad → NoGrad`, so the
output can't be fed to `trainStep` (compile error). Wrap inference in `withNoGrad` for the
tape-free perf path. On mlx, push `withNoGrad` *inside* long eval loops (per-sequence).

## Key differences to internalize

1. **`forward` consumes the model**: it's a linear resource. Use the returned handle for
   the next call — the old name is consumed (reusing it is a compile error). This kills
   the "freeze/eval then accidentally train via the stale handle (silent no-op)" footgun.

2. **No `autoName`**: parameters are named by `runInit` from the scope path and reach the
   optimizer through the C registry. There's no separate paramId to pass.

3. **No separate backward/step**: `fit` (or a single `trainStep`) fuses forward, backward,
   clip, and optimizer step.

4. **Shapes (and device, dtype, grad-mode) are types, not values**: `Tensor [3] …` is a
   different type from `Tensor [4] …`; mixing them won't compile.

5. **`Init` for construction, `L IO` for the model lifecycle**: layer builders run in
   `Init` (they allocate + register); model forward/eval/freeze run in the linear `L IO`.
