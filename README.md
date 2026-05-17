# idris-ml

Deep learning in Idris 2 with compile-time tensor shape checking and automatic differentiation.

## Why?

Dynamic graph frameworks like PyTorch catch shape errors at runtime:

```python
class NTM(nn.Module):
    def __init__(self, n=128, m=20, h=100):
        self.lstm = nn.LSTM(m + 9, h)        # input = memory_width + data_width
        self.read_fc = nn.Linear(h, m + 6)    # should be m + ShiftKernelSize + 3
        self.output_fc = nn.Linear(h + m, 8)  # hidden + memory_width -> output
```

Change the memory width `m` and five layer dimensions must update in concert. A typo in any one crashes mid-training -- or worse, silently broadcasts wrong shapes into plausible-looking garbage.

**idris-ml makes these compile errors.** Shape, device, dtype, and grad-mode are all part of the autograd-aware tensor type:

```idris
record Tensor (dims : Vect rank Nat) (0 d : Device) (0 dt : DType) (0 g : GradMode) where
  constructor MkTensor
  tensorPtr : AnyPtr        -- backend handle (carries autograd graph)
  paramId   : Maybe String  -- registry key for the optimizer
```

The network type chains layers with compile-time dimension threading:

```idris
(~~>) : AnyLayer i h d -> Network h hs o d -> Network i (h :: hs) o d

-- Compiles: output 10 matches input 10
ll <- linearLayerAny {i=2} {o=10} "ll0"
let model = ll ~~> OutputLayer reluLayerAny

-- Compile error: output 10 doesn't match input 5
ll2 <- linearLayerAny {i=5} {o=3} "ll1"
let bad = ll ~~> OutputLayer ll2  -- Error: Can't unify 10 with 5
```

NTM dimension relationships are type-level functions -- change one and the compiler tells you everywhere else that needs updating:

```idris
ReadParamWidth : Nat -> Nat
ReadParamWidth m = (m + ShiftKernelSize) + 3

WriteParamWidth : Nat -> Nat
WriteParamWidth m = ReadParamWidth m + m
```

The same compile-time discipline catches **device × dtype** mismatches. Metal GPU dropped float64 support in mlx 0.31; PyTorch users find out at runtime with a deep-in-C++ `RuntimeError`. idris-ml's `Compatible` capability interface lifts the check to the type system:

```idris
-- Compiles: MlxCpu supports both F32 and F64; MlxGpu supports F32.
gpuF32 : Tensor [4] (MlxDev MGpu) F32 WithGrad
cpuF64 : Tensor [4] (MlxDev MCpu) F64 WithGrad

-- Compile error: Can't find an implementation for Compatible (MlxDev MGpu) F64
gpuF64 : Tensor [4] (MlxDev MGpu) F64 WithGrad
```

`Tensor`'s dtype slot also carries a derived lossless-upcast partial order: `UpcastableTo F32 F64` resolves automatically (lossless), `UpcastableTo F64 F32` doesn't (narrowing — would need an explicit `tcast`). The same machinery applies to integer ladders (`Int 16 → Int 32`) and the brain-float family. Cross-family conversions (UInt 8 → F16, BF16 → F32) deliberately have no instance — even when the bit pattern fits, the semantic interpretation might not be what the user wants. See [`docs/develop/dtype-parameter.md`](docs/develop/dtype-parameter.md) for the design.

You get dynamic graph ergonomics (standard `if`/`for`/`while`, normal debugging, define-by-run autograd) with static graph safety (shape errors, illegal device-dtype combinations, and silently lossy casts are all impossible at runtime). See [docs/static-vs-dynamic-graphs.md](docs/static-vs-dynamic-graphs.md) for the full discussion.

## What works today

| Example | Description | Command |
|---------|-------------|---------|
| Supervised | 3-class classification with softmax | `make example-supervised` |
| RNN | Sequence prediction (repeating pattern) | `make example-rnn` |
| LSTM | Same task, LSTM controller | `make example-lstm` |
| NTM Copy | Neural Turing Machine binary vector copy | `make example-ntm-copy` |
| NTM Recall | NTM associative recall (content-based memory) | `make example-ntm-associative-recall` |
| Transformer | Autoregressive next-token prediction (causal self-attention) | `make example-transformer` |
| GPT | Character-level language model on Shakespeare | `make example-gpt` |
| MNIST | CNN digit classification (Conv2D + MaxPool2D) | `make example-mnist` |
| SeqClassify | 1D waveform classification (Conv1D + MaxPool1D) | `make example-seq-classify` |
| REINFORCE | Policy gradient on CartPole (pure Idris env) | `make example-reinforce` |

All examples accept `--epochs`, `--lr`, `--seed` and task-specific flags.

## Getting started

**Interactive notebooks** — progressive tutorial from tensors to training:

1. [Tensors and Types](packages/jupyter/notebooks/tutorials/01_tensors_and_types.ipynb) — shape-indexed types, the core value proposition
2. [Building Models](packages/jupyter/notebooks/tutorials/02_building_models.ipynb) — layer composition, dimension checking
3. [Data and Loss](packages/jupyter/notebooks/tutorials/03_data_and_loss.ipynb) — typed training data, loss functions
4. [Training](packages/jupyter/notebooks/tutorials/04_training.ipynb) — end-to-end classifier with evaluation
5. [Sequences](packages/jupyter/notebooks/tutorials/05_sequences.ipynb) — RNN/LSTM for time-series
6. [Device Safety](packages/jupyter/notebooks/tutorials/06_device_safety.ipynb) — phantom Device parameter, type-safe CPU/GPU placement
7. [Hyperparameter Optimization](packages/jupyter/notebooks/tutorials/07_hpo.ipynb) — `lr_find` and tuning workflows
8. [Precision and Devices](packages/jupyter/notebooks/tutorials/08_precision_devices.ipynb) — `Compatible (device, dtype)` admissibility, parametric dtype families, `UpcastableTo` derivation, build-mode targeting

**Quick start** — requires [Idris 2](https://github.com/idris-lang/Idris2) (0.8.0+) and a C compiler:

```bash
make backend                # build the C tape backend (no external dependencies)
make example-supervised     # run the simplest example
make example-ntm-copy       # train NTM on binary copy task
make test                   # run test suite
make jupyter-install && make jupyter-lab  # interactive notebooks
```

For the optional libtorch backend: `make BACKEND=torch backend`.

For the optional Apple MLX backend on Apple Silicon: `make BACKEND=mlx backend`. The nixpkgs `python3Packages.mlx` is CPU-only (Metal compute is hardcoded off — see `docs/develop/gotchas.md`); to get a Metal-capable build use a project-local pip install:

```bash
uv venv .venv-mlx && source .venv-mlx/bin/activate && uv pip install mlx
make BACKEND=mlx MLX_SITE=$VIRTUAL_ENV/lib/python3.13/site-packages/mlx backend
```

`MLX_DEVICE=gpu` enables Metal, but at the current example scales (RNN-cell, NTM/DNC, batch-32 MNIST) per-op kernel-launch overhead makes GPU 3-12× slower than the CPU stream. Default (`MLX_DEVICE=cpu`) is the right choice for the examples shipped here; GPU becomes interesting only with bigger batches/models or after `mx::compile`-style fusion lands.

## Performance

NTM-copy runs at ~110ms/epoch on the C tape backend (Apple M-series), comparable to the PyTorch reference (~130ms/epoch). See [docs/benchmarks.md](docs/benchmarks.md) for comparisons across all backends.

## Architecture

```
Array (Vect-of-Vect)  ->  Tensor (autograd)  ->  Layer (composable)  ->  Train (runner)
  [3,4] Double            shape + Device on value   LayerLike interface     runTraining
  pure-Idris ops          backend C handle          Network chains layers  early stopping
                          native optimizers         LSTM, Linear, NTM      CLI arg parsing
```

See [CLAUDE.md](CLAUDE.md) for the full module dependency order and development guide.

## References

- [Neural Turing Machines](https://arxiv.org/abs/1410.5401) (Graves, Wayne, Danihelka 2014)
- [Implementing Neural Turing Machines](https://arxiv.org/abs/1807.08518) (Collier & Beel 2018)
- [Idris 2: Quantitative Type Theory in Practice](https://arxiv.org/abs/2104.00480) (Brady 2021)
