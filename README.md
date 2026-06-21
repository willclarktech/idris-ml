# idris-ml

Deep learning in Idris 2 with compile-time tensor shape checking and automatic differentiation.

**New here?** [**Why idris-ml**](docs/why-idris-ml.md) makes the full case: dynamic-graph
ergonomics with safety guarantees stronger than any static graph, compared side-by-side
against PyTorch, TensorFlow 1.x / JAX, and Haskell (Grenade / hasktorch).

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
record Tensor (dims : Vect rank Nat) (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkTensor
  tensorPtr : AnyPtr        -- backend handle (carries autograd graph)
  paramId   : Maybe String  -- registry key for the optimizer
```

Models are records of `Nn` layers; a `Seq` chain threads dimensions at compile time — its
type pins only the endpoints, hidden dims are existential:

```idris
Model : Type
Model = Seq 2 3 Ex F WithGrad

mkModel : Init Model
mkModel = do
  l1 <- linear {i=2} {o=10}
  l2 <- linear {i=10} {o=3}
  pure (l1 ~~> reluA ~~> l2 ~~> Nil)   -- compiles: l1's out 10 unifies with l2's in 10

-- Swap l2 for `linear {i=5} {o=3}` and the chain won't elaborate:
--   Mismatch between: 10 and 5
```

**Five guarantees, one type mechanism.** Each of these is a compile error in idris-ml — and a runtime error, a silent bug, or simply impossible elsewhere:

1. **Shape** mismatches — type-level `Nat` arithmetic (the NTM dimensions above).
2. **Device** mismatches — including "CUDA on a Mac" (unspellable in a non-CUDA build) and Metal's F32-only limit (`Compatible (MlxExecutor MGpu) F64` deliberately doesn't exist).
3. **Multi-backend in one program** — `tape`, `torch`, and `mlx` tensors coexist in one type-checked program with explicit, checked transfers. No mainstream framework offers this.
4. **Grad-mode / model ownership** — models are single-owner linear resources; "freeze then train via the stale handle" (a silent no-op in PyTorch) is a linearity error.
5. **Lossy dtype casts** — narrowing must be code-visible; `F64 → F32` won't resolve without an explicit cast.

All with dynamic-graph ergonomics intact: ordinary `if`/`for`/`while`, define-by-run autograd, normal debugging.

→ [**Why idris-ml**](docs/why-idris-ml.md) makes the full case — every guarantee shown side by side in PyTorch, TensorFlow 1.x / JAX, Haskell (Grenade / hasktorch), and idris-ml, with the **literal error each one produces**.

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

**Not just toy tasks.** [`idris-transformers`](docs/users/idris-transformers.md) loads
real HuggingFace checkpoints by name — **BERT** (`google/bert_uncased_L-2_H-128_A-2`),
**GPT-2** (`distilgpt2`), **Llama-3.2-1B**, **BitNet** — via `fromPretrained` (parse
`config.json`, fill from `model.safetensors`, no remap machinery). CI gates regenerate
the PyTorch oracle and compare per-element: BERT matches HF's forward pass to **4e-4**
(`make test-e2e-bert-roundtrip`). LoRA + prefix-freeze fine-tuning are supported.

## Getting started

The fastest path is the text walkthrough — [**docs/getting-started.md**](docs/getting-started.md) — which takes you from a first tensor to a trained model against the current `Nn` / `fit` API. The same path, interactively, is the 8-part Jupyter tutorial (tensors → models → data → training → sequences → device safety → HPO → precision); see [packages/jupyter/README.md](packages/jupyter/README.md).

### Development environment

The toolchain — Idris 2 (via [pack](https://github.com/stefan-hoeck/idris2-pack)), Chez Scheme, a C compiler, Criterion, cppcheck, clang-tools, and uv — is pinned in [`flake.nix`](flake.nix). This is the **same shell CI runs in**, so local builds match CI exactly. You need [Nix](https://nixos.org/download) with flakes enabled.

**Recommended — [direnv](https://direnv.net) + [nix-direnv](https://github.com/nix-community/nix-direnv):** the repo ships an `.envrc` (`use flake`), so `cd` into the tree auto-loads the dev shell (cached, instant after the first eval) and `cd` out unloads it. After installing direnv + nix-direnv, one-time per checkout:

```bash
direnv allow                # in the repo root
```

**Or explicitly**, without direnv:

```bash
nix develop                                 # enter the dev shell, then run make targets
nix develop .#default --command make test   # run a single target in the shell
```

All `make` targets expect to run inside this shell. In particular the C unit tests (`make test`, `make test-unit-c`) compile against Criterion — like any C library, it's only on the compiler's search path inside the dev shell, never from a global install (Nix exposes libraries to compilation through the build environment, not the user profile).

**Quick start** — inside the dev shell above (or with a system Idris 2 0.8.0+ and a C compiler):

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
Array (Vect-of-Vect)  ->  Tensor (autograd)     ->  Nn (models-as-records)  ->  fit (driver)
  [3,4] Double            shape+executor+dtype       Module / Seq (~~>)          fitSupervised
  pure-Idris ops          +grad-mode on the type     linear single-owner models  early stopping
                          backend C handle, opts     Linear, Conv, LSTM, NTM…    checkpointing
```

See [CLAUDE.md](CLAUDE.md) for the full module dependency order and development guide.

## Documentation

- [Why idris-ml](docs/why-idris-ml.md) — the five-guarantee case vs PyTorch / TF1+JAX / Haskell, with literal errors
- [Getting Started](docs/getting-started.md) — first tensor → first trained model (text walkthrough)
- [PyTorch Mapping](docs/pytorch-mapping.md) — concept-by-concept translation for PyTorch users
- [idris-transformers](docs/users/idris-transformers.md) — load real HuggingFace BERT / GPT-2 / Llama / BitNet; fine-tuning + LoRA
- Deep dives: [Static vs Dynamic Graphs](docs/static-vs-dynamic-graphs.md) · [Grad-Mode & Device Typing](docs/grad-mode-and-device-typing.md) · [Benchmarks](docs/benchmarks.md)
- [docs/](docs/README.md) — full index · [CLAUDE.md](CLAUDE.md) — architecture + contributor guide

## References

- [Neural Turing Machines](https://arxiv.org/abs/1410.5401) (Graves, Wayne, Danihelka 2014)
- [Implementing Neural Turing Machines](https://arxiv.org/abs/1807.08518) (Collier & Beel 2018)
- [Idris 2: Quantitative Type Theory in Practice](https://arxiv.org/abs/2104.00480) (Brady 2021)
