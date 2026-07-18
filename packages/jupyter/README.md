# Jupyter Kernel for idris-ml

Interactive notebook experience wrapping the Idris 2 REPL with full C FFI support.

## Prerequisites

- **Idris 2** (0.8.0+) — installed and on `PATH`
- **Python 3.9+**
- **Built backend** — `make backend` (tape, MLX, or torch)

## Setup

From the project root:

```bash
make jupyter-install   # creates venv, installs kernel, registers with Jupyter
make jupyter-lab       # launches JupyterLab with example notebooks
```

Or manually:

```bash
make backend
make install-notebook       # installs idris-ml + the Notebook.Prelude shim (+ transformers)
cd packages/jupyter
python3 -m venv .venv
.venv/bin/pip install -e .
.venv/bin/python3 -m idris_ml_kernel.install
.venv/bin/jupyter lab --notebook-dir=notebooks
```

## Usage

The kernel wraps a persistent `idris2` REPL. All idris-ml modules are pre-loaded
via `Notebook.Prelude` — no manual imports needed.

### REPL commands (same as `idris2` REPL)

```
:t reluA                  -- type query
:doc linear               -- documentation
:browse Ml.Tensor         -- list module exports
:module Ml.Nn             -- import a module (persists across cells)
:exec putStrLn "hello"    -- execute an IO expression
```

### Definitions

Type signatures and function clauses are auto-prefixed with `:let`:

```
myDouble : Double -> Double
myDouble x = x * 2.0
```

Then use them in later cells:

```
:exec putStrLn (show (myDouble 21.0))
```

### Tensor operations

Construct and operate on tensors through the typed surface. Pin the executor + dtype at
the use site (`TapeExecutor` here; name your build's executor on a torch/mlx build):

```
:exec do { a <- tconstScalar {ex=TapeExecutor} {dt=F64} 2.0;
  b <- tconstScalar {ex=TapeExecutor} {dt=F64} 3.0;
  c <- tmul a b;
  putStrLn ("2 * 3 = " ++ show (tensorItem c)) }
```

### Multi-line `do` blocks

Idris 2's braced `do { }` syntax does not support bare `let` — use `<- pure` instead.
Multi-line cells are automatically joined — write naturally across lines.

### Shift-Tab inspection

Place cursor on a name and press Shift-Tab to see its type and documentation.

## How it works

1. The kernel spawns `idris2 -p contrib -p linear -p idris-ml -p idris-ml-notebook
   -p elab-util -p idris-transformers` and loads `:module Notebook.Prelude`
2. `Notebook.Prelude` re-exports all library modules via `import public`
3. `libidrisml.dylib` is copied to `build/exec/_tmpchez_app/` so `:exec` can load it
4. Each cell is parsed and sent to the REPL; output is captured via pexpect
5. Session state (`:module` imports, `:let` definitions) is tracked and replayed on crash recovery

## Limitations

- **No `let` in braced `do`** — `do { let x = y; ... }` fails with "Expected in". Use `x <- pure y` instead. Multi-line `let...in` chains work fine outside `do {}`
- **No incremental output** — long-running `:exec` cells buffer until complete. For training, use compiled executables (`make example-*`)
- **`:let` scope** — simple functions and values work. Complex types, interfaces, and implementations need `.idr` files + `:load`
- **macOS/Linux only** — pexpect requires PTY support (Windows users can use WSL)
- **First run slow** — cold TTC cache means ~10s startup while modules compile. Subsequent starts ~0.5s

## Notebooks

Two categories in `notebooks/`:

- **`tutorials/`** — Concept-oriented (01-09): tensors & types, building models, data & loss, training, model ownership, sequences, device safety, hyperparameter optimization, precision & devices
- **`models/`** — Architecture-oriented (10 notebooks): supervised, rnn_lstm, transformer, gpt, ntm, dnc, cnn, reinforce, seq_classify, bert. Each walks through model construction, type queries, and training or inference (interactive where feasible, CLI instructions for heavy models)

Executed outputs are committed in the `.ipynb` files so the notebooks display
fully rendered on GitHub, including the captured compile errors in the
expected-failure cells. After editing any notebook, run `make notebooks-refresh`
(re-executes every notebook in place) and commit the result; outputs are
deterministic per build, so a refresh with no notebook edits is a git no-op.
(The refresh runs at `IDRISML_LOG_LEVEL=warn` to keep `fit`'s wall-clock
timing epilogue out of the recorded outputs; a live kernel at the default
level prints it.)

## Tests

```bash
make test-e2e-jupyter    # kernel suite: parser + REPL integration + FFI + recovery
make test-e2e-notebooks  # run all notebooks headless (catches API breakage)
```

## Switching backends

The kernel uses whichever backend is active (`build/libidrisml.dylib` symlink). To switch:

```bash
make BACKEND=mlx backend    # rebuild with MLX
make jupyter-install        # re-copies dylib
```
