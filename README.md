# idris-ml

A dependently-typed deep-learning ecosystem in Idris 2: dynamic-graph ergonomics (define-by-run
autograd, ordinary `if`/`for`/`while`, normal debugging) with safety guarantees stronger than any
static graph ever offered — shapes, devices, dtypes, and grad-mode are checked at compile time
and erased at runtime. This is a **monorepo** of a core library plus RL environments, an
HF-aligned model library, supporting tools, and the PyTorch oracle it's validated against.

## Why?

Dynamic frameworks like PyTorch catch shape errors at runtime, devices at runtime, lossy casts
never. idris-ml makes them compile errors — and one mechanism (dependent + linear types) covers
all of it. Shape, executor (backend), dtype, and grad-mode all ride on the autograd tensor type:

```idris
record Tensor (dims : Vect rank Nat) (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkTensor
  tensorPtr : AnyPtr        -- backend handle (carries the autograd graph)
  paramId   : Maybe String  -- registry key for the optimizer
```

**Five guarantees, one type mechanism** — each a compile error here, a runtime error / silent
bug / outright impossibility elsewhere:

1. **Shape** mismatches — type-level `Nat` arithmetic threads dimensions through a whole model.
2. **Device** mismatches — including "CUDA on a Mac" (unspellable in a non-CUDA build) and
   Metal's F32-only limit (`Compatible (MlxExecutor MGpu) F64` deliberately doesn't exist).
3. **Grad-mode / model ownership** — models are single-owner linear resources; "freeze then
   train via the stale handle" (a silent no-op in PyTorch) is a linearity error.
4. **Lossy dtype casts** — narrowing must be code-visible; `F32 → BF16` won't resolve without an
   explicit cast.
5. **Multi-backend in one program** — `tape`, `torch`, and `mlx` tensors coexist in one
   type-checked program with explicit, checked transfers. No mainstream framework offers this.

→ [**Why idris-ml**](docs/why-idris-ml.md) makes the full case, side by side against PyTorch,
TensorFlow 1.x / JAX, and Haskell (Grenade / hasktorch), with the **literal error each one
produces**. It also shows this isn't a toy: [`idris-transformers`](packages/idris-transformers/)
loads real HuggingFace **BERT / GPT-2 / Llama-3.2-1B / BitNet** checkpoints by name and matches
PyTorch's forward pass to **4e-4**.

## Packages

| Package | What it is |
| --- | --- |
| [`idris-ml`](packages/idris-ml/) | **Core library** — autograd `Tensor`, `Nn` models, optimizers, `fit`, data, checkpoints, pluggable backends |
| [`idris-ml-examples`](packages/idris-ml-examples/) | Runnable example programs (supervised, recurrent, transformers, RL) + microbenchmarks |
| [`idris-transformers`](packages/idris-transformers/) | HF-aligned model library — load BERT / GPT-2 / Llama / BitNet via `fromPretrained`; LoRA + fine-tuning |
| [`idris-gym`](packages/idris-gym/) | Pure-Idris RL environments with a Gymnasium-parity API (CartPole, FrozenLake, Taxi, …) |
| [`idris-args`](packages/idris-args/) | Typed CLI flag parsing (zero deps beyond base) |
| [`idris-fmt`](packages/idris-fmt/) | Compiler-native Idris formatter, gated by a round-trip safety oracle |
| [`idris-ml-notebook`](packages/idris-ml-notebook/) | `Notebook.Prelude` re-export shim auto-loaded by the Jupyter kernel |
| [`jupyter`](packages/jupyter/) | Jupyter kernel (Python) wrapping the Idris 2 REPL with FFI support |
| [`backends`](packages/backends/) | C/C++ backends (tape, libtorch, MLX) + the shared training port |
| [`idris-test`](packages/idris-test/) | Shared Idris test harness (assertions, suites, property testing) |
| [`idris-test-c`](packages/idris-test-c/) | Cross-cutting C test infrastructure for the backend layer |
| [`pytorch`](packages/pytorch/) | PyTorch reference implementations — the correctness oracle (not shipped code) |

## Getting started

The toolchain (Idris 2 via [pack](https://github.com/stefan-hoeck/idris2-pack), Chez Scheme, a C
compiler, Criterion, clang-tools, uv) is pinned in [`flake.nix`](flake.nix) — the **same shell CI
runs in**, so local builds match CI. You need [Nix](https://nixos.org/download) with flakes.

**Recommended — [direnv](https://direnv.net) + [nix-direnv](https://github.com/nix-community/nix-direnv):**
the repo ships an `.envrc` (`use flake`), so `cd` into the tree auto-loads the dev shell:

```bash
direnv allow                # one-time, in the repo root
```

**Or explicitly:**

```bash
nix develop                                 # enter the dev shell, then run make targets
nix develop .#default --command make test   # run a single target in the shell
```

All `make` targets expect to run inside this shell. Quick start:

```bash
make backend                # build the C tape backend (no external dependencies)
make install                # install core lib + gym (needed for examples/tests)
make example-supervised     # run the simplest example
make test                   # run the Idris test suite
make jupyter-install && make jupyter-lab   # interactive notebooks
```

The optional libtorch / MLX backends and the full per-backend build matrix are documented in
[`packages/idris-ml/README.md`](packages/idris-ml/README.md#backends).

## Documentation

- [**Why idris-ml**](docs/why-idris-ml.md) — the five-guarantee case vs PyTorch / TF1+JAX / Haskell, with literal errors.
- [docs/](docs/README.md) — full user documentation index (getting-started, PyTorch mapping, deep dives, benchmarks).
- [CLAUDE.md](CLAUDE.md) — architecture, module dependency order, and the contributor guide.

## References

- [Neural Turing Machines](https://arxiv.org/abs/1410.5401) (Graves, Wayne, Danihelka 2014)
- [Implementing Neural Turing Machines](https://arxiv.org/abs/1807.08518) (Collier & Beel 2018)
- [Idris 2: Quantitative Type Theory in Practice](https://arxiv.org/abs/2104.00480) (Brady 2021)
