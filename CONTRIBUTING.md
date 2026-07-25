# Contributing

This is a monorepo: a core Idris 2 library, RL environments, an HF-aligned
model library, supporting tools, and the PyTorch reference implementations
everything is validated against. Everything builds from the root `Makefile`.

## Build environment

The recommended path is **[Nix](https://nixos.org/download) (with flakes) +
[direnv](https://direnv.net)** — it is how CI runs, so local builds match CI
byte-for-byte. Nix is a convenience, not a requirement: any system with the
toolchain below works.

### With Nix + direnv

The repo ships an `.envrc` (`use flake`), so `cd` into the tree auto-loads the
dev shell pinned in [`flake.nix`](flake.nix), the single source of truth for the
toolchain:

```bash
direnv allow                # one-time, in the repo root
```

Or enter the shell explicitly:

```bash
nix develop                                 # enter the dev shell, then run make targets
nix develop .#default --command make test   # run a single target in the shell
```

### Without Nix — toolchain requirements

Install these yourself; matching the versions in `flake.nix` avoids skew. Only
the **Core** row is needed to build the default backend and run examples, the
rest are per-feature:

| For | Needs |
| --- | --- |
| **Core** — build the tape backend, run examples, `make test` | Idris 2 0.8.0 (via [pack](https://github.com/stefan-hoeck/idris2-pack)), Chez Scheme, a C compiler, `make` |
| C unit tests (`make test-unit-c-*`) | [Criterion](https://github.com/Snaipe/Criterion) + dev headers, pkg-config |
| C lint (`make lint-c`) | cppcheck, clang-tools (clang-format / clang-tidy) |
| Python surfaces — PyTorch oracle, Jupyter | Python 3 + [uv](https://docs.astral.sh/uv/) |
| Linux only | OpenBLAS (`cblas.h`); macOS uses the Accelerate framework |
| Optional `torch` backend | libtorch |
| Optional `mlx` backend (Apple Silicon) | MLX |

The default tape backend has **no external dependencies** beyond the Core
row — `make backend` builds it with just a C compiler.

## Build and run

```bash
make backend                # build the C tape backend (no external dependencies)
make install                # install core lib + gym (needed for examples/tests)
make example-supervised     # run the simplest example
make test                   # run the Idris test suite
make jupyter-install && make jupyter-lab   # interactive notebooks
```

The optional libtorch / MLX backends and the full per-backend build matrix are
documented in
[`packages/idris-ml/README.md`](packages/idris-ml/README.md#backends). Build
artifacts are keyed per `(BACKEND, MLX_DEVICE, TORCH_DEVICE)` tuple, so
switching backends does not invalidate the previous one's tree.

## Internal packages

These exist to build and validate the library rather than to be used directly:

| Package | What it is |
| --- | --- |
| [`backends`](packages/backends/) | C/C++ backends (tape, libtorch, MLX) + the shared training port |
| [`pytorch`](packages/pytorch/) | PyTorch reference implementations, used as the correctness oracle |
| [`idris-fmt`](packages/idris-fmt/) | Compiler-native Idris formatter, gated by a round-trip safety oracle |
| [`idris-test`](packages/idris-test/) | Shared Idris test harness (assertions, suites, property testing) |
| [`idris-test-c`](packages/idris-test-c/) | Cross-cutting C test infrastructure for the backend layer |
| [`idris-ml-notebook`](packages/idris-ml-notebook/) | `Notebook.Prelude` re-export shim auto-loaded by the Jupyter kernel |

## Before you submit

```bash
make check-fmt              # every language's formatter, in check mode
make test                   # Idris unit tests
make lint-py typecheck-py   # ruff + pyright strict, if you touched Python
make lint-c                 # cppcheck + clang-tidy, if you touched C/C++
```

`make fmt` rewrites every source file in place. Formatting and linting are
separate gates; see [docs/develop/testing-taxonomy.md](docs/develop/testing-taxonomy.md)
for the full target-naming contract and
[docs/develop/testing.md](docs/develop/testing.md) for the test-layer overview.

Heavier gates — the five-lane example smoke matrix
(`make test-e2e-examples EXAMPLE_TIMEOUT=900`), the PyTorch reference suite
(`make test-e2e-pytorch-ref`), and the convergence campaign — take from minutes
to hours; CI runs them.

## Conventions

- **Commits** follow [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `refactor:`, `docs:`, `chore:`), imperative present tense,
  one logical change per commit.
- **Indentation** is governed by `.editorconfig` per extension — no formatter
  enforces it for you on new files.
- **Examples are paired**: an Idris example and its PyTorch reference must share
  every hyperparameter default. A change to one lands in the same commit as the
  matching change to the other. See
  [docs/develop/reference-alignment.md](docs/develop/reference-alignment.md).
- **Behaviour-bearing changes are test-driven**: write the test first and
  observe it fail for the right reason (a wrong value, gradient, dtype tag, or
  a crash — a compile or link error does not count).

[CLAUDE.md](CLAUDE.md) carries the architecture overview, the module dependency
order, and the working conventions in full.
[docs/develop/](docs/develop/README.md) has the design rationale and deep dives.
