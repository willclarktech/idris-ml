# idris-ml

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
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (64x256 and 128x10)
```

The error arrives at runtime, possibly hours into a run, and only if that code path executes.
In idris-ml the same program is rejected by the compiler, because shapes are part of the
tensor's type. The same two language features (dependent types, plus linear types for model
ownership) cover the rest:

| Bug class | PyTorch (dynamic) | TF 1.x (static) | hasktorch (Torch.Typed) | idris-ml |
|---|:---:|:---:|:---:|:---:|
| Shape mismatch | run time | graph build | **compile time** | **compile time** |
| Device mismatch | run time | run time | **compile time** | **compile time** |
| Grad-mode misuse | run time | n/a | not caught | **compile time** |
| Stale model handle after freeze | not caught | n/a | not caught | **compile time** |
| Lossy dtype cast | not caught | not caught | not caught | **compile time** (explicit opt-out) |
| Mixing multiple backends | unsupported | unsupported | unsupported | **compile time** |

→ [**Why idris-ml**](docs/why-idris-ml.md) makes the full case, side by side against PyTorch,
TensorFlow 1.x, and hasktorch (Torch.Typed), with the **literal error each one
produces**. It also runs real models: [`idris-transformers`](packages/idris-transformers/)
loads HuggingFace **BERT / GPT-2 / Llama-3.2-1B / BitNet** checkpoints by name and matches
PyTorch's forward pass to **4e-4**.

idris-ml is young: compile times are longer than you're used to, and performance today trails
PyTorch on many workloads (every example trains against a PyTorch reference implementation,
which doubles as the benchmark).

## Packages

| Package | What it is |
| --- | --- |
| [`idris-ml`](packages/idris-ml/) | **Core library** — autograd `Tensor`, `Nn` models, optimizers, `fit`, data, checkpoints, pluggable backends |
| [`idris-transformers`](packages/idris-transformers/) | HF-aligned model library — load BERT / GPT-2 / Llama / BitNet via `fromPretrained`; LoRA + fine-tuning |
| [`idris-gym`](packages/idris-gym/) | Pure-Idris RL environments with a Gymnasium-parity API (CartPole, FrozenLake, Taxi, …) |
| [`jupyter`](packages/jupyter/) | Jupyter kernel (Python) wrapping the Idris 2 REPL with FFI support |
| [`idris-ml-notebook`](packages/idris-ml-notebook/) | `Notebook.Prelude` re-export shim auto-loaded by the Jupyter kernel |
| [`idris-ml-examples`](packages/idris-ml-examples/) | Runnable example programs (supervised, recurrent, transformers, RL) + microbenchmarks |
| [`idris-args`](packages/idris-args/) | Typed CLI flag parsing (zero deps beyond base) |
| [`idris-fmt`](packages/idris-fmt/) | Compiler-native Idris formatter, gated by a round-trip safety oracle |
| [`backends`](packages/backends/) | C/C++ backends (tape, libtorch, MLX) + the shared training port |
| [`idris-test`](packages/idris-test/) | Shared Idris test harness (assertions, suites, property testing) |
| [`idris-test-c`](packages/idris-test-c/) | Cross-cutting C test infrastructure for the backend layer |
| [`pytorch`](packages/pytorch/) | PyTorch reference implementations (used as a correctness oracle) |

## Getting started

The recommended path is **[Nix](https://nixos.org/download) (with flakes) + [direnv](https://direnv.net)** —
it's how CI runs, so local builds match CI byte-for-byte. But **Nix is a convenience, not a
requirement**: any system with the toolchain below works.

### With Nix + direnv (recommended)

The repo ships an `.envrc` (`use flake`), so `cd` into the tree auto-loads the dev shell pinned in
[`flake.nix`](flake.nix) (the single source of truth for the toolchain):

```bash
direnv allow                # one-time, in the repo root
```

Or enter the shell explicitly:

```bash
nix develop                                 # enter the dev shell, then run make targets
nix develop .#default --command make test   # run a single target in the shell
```

### Without Nix — toolchain requirements

Install these yourself (matching the versions in `flake.nix` avoids skew). Only the **Core** row is
needed to build the default backend and run examples; the rest are per-feature:

| For | Needs |
| --- | --- |
| **Core** — build the tape backend, run examples, `make test` | Idris 2 0.8.0 (via [pack](https://github.com/stefan-hoeck/idris2-pack)), Chez Scheme, a C compiler, `make` |
| C unit tests (`make test-unit-c-*`) | [Criterion](https://github.com/Snaipe/Criterion) + dev headers, pkg-config |
| C lint (`make lint-c`) | cppcheck, clang-tools (clang-format / clang-tidy) |
| Python surfaces — PyTorch oracle, Jupyter | Python 3 + [uv](https://docs.astral.sh/uv/) |
| Linux only | OpenBLAS (`cblas.h`); macOS uses the Accelerate framework |
| Optional `torch` backend | libtorch |
| Optional `mlx` backend (Apple Silicon) | MLX |

The default tape backend has **no external dependencies** beyond the Core row — `make backend`
builds it with just a C compiler.

### Quick start

```bash
make backend                # build the C tape backend (no external dependencies)
make install                # install core lib + gym (needed for examples/tests)
make example-supervised     # run the simplest example
make test                   # run the Idris test suite
make jupyter-install && make jupyter-lab   # interactive notebooks
```

To try the compile-time guarantees live, start with the notebooks: the tutorial sequence begins
at [`tutorials/01_tensors_and_types.ipynb`](packages/jupyter/notebooks/tutorials/01_tensors_and_types.ipynb),
and every notebook in [`packages/jupyter/notebooks/`](packages/jupyter/notebooks/) is executed
in CI.

The optional libtorch / MLX backends and the full per-backend build matrix are documented in
[`packages/idris-ml/README.md`](packages/idris-ml/README.md#backends).

## Documentation

- [**Why idris-ml**](docs/why-idris-ml.md) — the five-guarantee case vs PyTorch / TF1 / hasktorch, with literal errors.
- [docs/](docs/README.md) — full user documentation index (getting-started, PyTorch mapping, deep dives, benchmarks).
- [CLAUDE.md](CLAUDE.md) — architecture, module dependency order, and the contributor guide.

## References

- [Neural Turing Machines](https://arxiv.org/abs/1410.5401) (Graves, Wayne, Danihelka 2014)
- [Implementing Neural Turing Machines](https://arxiv.org/abs/1807.08518) (Collier & Beel 2018)
- [Idris 2: Quantitative Type Theory in Practice](https://arxiv.org/abs/2104.00480) (Brady 2021)
