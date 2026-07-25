# idris-ml

[![tests](https://github.com/willclarktech/idris-ml/actions/workflows/test.yml/badge.svg)](https://github.com/willclarktech/idris-ml/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/willclarktech/idris-ml/branch/main/graph/badge.svg)](https://codecov.io/gh/willclarktech/idris-ml)

A dependently-typed deep-learning framework in Idris 2: dynamic-graph ergonomics (define-by-run
autograd, ordinary `if`/`for`/`while`, normal debugging) with the constraints (shapes, devices,
dtypes, grad-mode) checked at compile time and erased at runtime. This is a **monorepo** of a
core library plus RL environments, an HF-aligned model library, supporting tools, and the
PyTorch reference implementations it's validated against.

## Why?

Here's a common bug class in PyTorch:

```python
fc1 = nn.Linear(784, 256)  # this hidden layer size got increased from 128 to 256
fc2 = nn.Linear(128, 10)   # bug: this value didn't get updated
some_inputs = torch.randn(64, 784)
fc2(fc1(some_inputs))
```
```text
RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x256 and 128x10)
```

The error arrives at runtime, possibly hours into a run, and only if that code path executes.
Here is the same model in idris-ml, with the same bug:

```idris
Batch : Nat
Batch = 64

Model : Type
Model = Seq 784 10 Ex F WithGrad

mkModel : Init Model
mkModel = do
  l1 <- linear {i = 784} {o = 256}
  l2 <- linear {i = 128} {o = 10}   -- same bug: 128 should be 256
  pure (l1 ~~> reluA ~~> l2 ~~> Nil)
```
```text
Error: While processing right hand side of mkModel. Can't find an implementation for ChainFits 256 128.

ShapeBug:21:16--21:36
 17 | mkModel : Init Model
 18 | mkModel = do
 19 |   l1 <- linear {i = 784} {o = 256}
 20 |   l2 <- linear {i = 128} {o = 10}
 21 |   pure (l1 ~~> reluA ~~> l2 ~~> Nil)
                     ^^^^^^^^^^^^^^^^^^^^
```

The program is rejected by the compiler, because shapes are part of the tensor's type. Nothing
had to run, and no data was needed. Fix the `128` to `256` and the rest is ordinary code — a
model is a record of layers, training is a function call:

```idris
loss : (1 _ : Model) ->
       (Tensor [Batch, 784] Ex F NoGrad, Tensor [Batch, 10] Ex F NoGrad) ->
       L IO {use = 1} (LPair (!* (Tensor [] Ex F WithGrad)) Model)
loss model (x, tgt) = do
  (MkBang out # model') <- forwardSeq {b = Batch} model (retypeGrad x)
  l <- tnllLossMeanL {b = Batch} {n = 10} out (retypeGrad tgt)
  pure1 (MkBang l # model')

train : DataStream (Tensor [Batch, 784] Ex F NoGrad, Tensor [Batch, 10] Ex F NoGrad) -> IO ()
train batches = run $ do
  opt   <- liftIO1 (adam 0.001 defaultOpts)
  model <- runInitL mkModel
  (MkBang (epochs, finalLoss) # trained) <-
    fitSupervised opt loss batches (simpleConfig 20) model
  discard trained
  liftIO1 (putStrLn ("epochs=" ++ show epochs ++ " loss=" ++ show finalLoss))
```

The `1` and `!*` annotations are the second language feature at work: a model is a linear
resource with a single owner, so reusing a stale handle after `eval` or `freeze` is a compile
error rather than a silent no-op. Between them, dependent types and linear types cover the rest:

| Bug class | PyTorch (dynamic) | TF 1.x (static) | hasktorch (Torch.Typed) | idris-ml |
|---|:---:|:---:|:---:|:---:|
| Shape mismatch | run time | graph build | **compile time** | **compile time** |
| Device mismatch | run time | run time | **compile time** | **compile time** |
| Grad-mode misuse | run time | n/a | not caught | **compile time** |
| Stale model handle after freeze | not caught | n/a | not caught | **compile time** |
| Lossy dtype cast | not caught | not caught | not caught | **compile time** (explicit opt-out) |
| Mixing multiple backends | unsupported | unsupported | unsupported | **compile time** |

→ [**Why idris-ml**](docs/users/why-idris-ml.md) makes the full case, side by side against PyTorch,
TensorFlow 1.x, and hasktorch (Torch.Typed), with the **literal error each one
produces**. It also runs real models: [`idris-transformers`](packages/idris-transformers/)
loads HuggingFace **BERT / GPT-2 / Llama-3.2-1B / BitNet** checkpoints by name and matches
PyTorch's forward pass to **4e-4**.

idris-ml is young: compile times are longer than you're used to, and performance today trails
PyTorch on many workloads (every example trains against a PyTorch reference implementation,
which doubles as the benchmark).

## Getting started

Runs on macOS (Apple Silicon and Intel) and Linux. The default backend is a self-contained C
tape requiring only a C compiler; the optional libtorch backend adds CUDA and Metal, and the
optional MLX backend is Apple Silicon only.

The quickest way to see the compile-time guarantees is the notebooks, which run against a real
kernel rather than a transcript:

```bash
make backend                               # build the C tape backend
make install                               # install core lib + gym
make jupyter-install && make jupyter-lab   # interactive notebooks
```

The tutorial sequence begins at
[`tutorials/01_tensors_and_types.ipynb`](packages/jupyter/notebooks/tutorials/01_tensors_and_types.ipynb),
and every notebook in [`packages/jupyter/notebooks/`](packages/jupyter/notebooks/) is executed
in CI. For a Jupyter-independent path, [Getting Started](docs/users/getting-started.md) is the
same walkthrough in text, and `make example-supervised` trains the simplest example end to end.

Toolchain requirements, the per-backend build matrix, and the test gates are in
[CONTRIBUTING.md](CONTRIBUTING.md).

## Packages

| Package | What it is |
| --- | --- |
| [`idris-ml`](packages/idris-ml/) | **Core library** — autograd `Tensor`, `Nn` models, optimizers, `fit`, data, checkpoints, pluggable backends |
| [`idris-transformers`](packages/idris-transformers/) | HF-aligned model library — load BERT / GPT-2 / Llama / BitNet via `fromPretrained`; LoRA + fine-tuning |
| [`idris-gym`](packages/idris-gym/) | Pure-Idris RL environments with a Gymnasium-parity API (CartPole, FrozenLake, Taxi, …) |
| [`idris-ml-examples`](packages/idris-ml-examples/) | Runnable example programs (supervised, recurrent, transformers, RL) + microbenchmarks |
| [`idris-args`](packages/idris-args/) | Typed CLI flag parsing (zero deps beyond base) |
| [`jupyter`](packages/jupyter/) | Jupyter kernel (Python) wrapping the Idris 2 REPL with FFI support |

The backends, the PyTorch oracle, the formatter, and the test harnesses are
[listed in CONTRIBUTING.md](CONTRIBUTING.md#internal-packages).

## Documentation

- [**Why idris-ml**](docs/users/why-idris-ml.md) — the five-guarantee case vs PyTorch / TF1 / hasktorch, with literal errors.
- [Getting Started](docs/users/getting-started.md) — first tensor to trained model, in text.
- [PyTorch Mapping](docs/users/pytorch-mapping.md) — concept translation for PyTorch users.
- [idris-transformers](docs/users/idris-transformers.md) — HuggingFace checkpoints, fine-tuning, LoRA.
- [Benchmarks](docs/users/benchmarks.md) — performance vs PyTorch across the tape, MLX, and torch backends.
- [docs/users/](docs/users/README.md) — the full user index, including the deep dives.
- [CHANGELOG.md](CHANGELOG.md) — completed work, most recent first.

## Contributing

[CONTRIBUTING.md](CONTRIBUTING.md) covers the build environment, the test gates, and the
conventions. Architecture and design rationale live in [docs/develop/](docs/develop/README.md).
