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
idris2 --build idris-ml.ipkg
cd jupyter
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
:t Var                    -- type query
:doc linearLayer          -- documentation
:browse Variable          -- list module exports
:module Layer.Core        -- import a module (persists across cells)
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

### FFI tensor operations

```
:exec (let t = prim__createScalar 3.14 0 in putStrLn (show (prim__item t)))
```

```
:exec (let a = prim__createScalar 2.0 1 in
  let b = prim__createScalar 3.0 0 in
  let c = prim__mul a b in
  putStrLn ("2*3=" ++ show (prim__item c)))
```

### Multi-line `do` blocks

Idris 2's braced `do { }` syntax does not support bare `let` — use `<- pure` instead:

```
:exec do { ll <- linearLayer {i=2, o=3};
  model <- pure (autoName (OutputLayer ll));
  buf <- pure (prim__setDouble (prim__setDouble (prim__allocDoubles 2) 0 1.0) 1 2.0);
  inT <- pure (prim__createState1d 2 buf);
  pair <- pure (forwardVarTensor model inT);
  putStrLn ("output sum = " ++ show (prim__item (prim__sum (snd pair)))) }
```

Multi-line cells are automatically joined — write naturally across lines.

### Shift-Tab inspection

Place cursor on a name and press Shift-Tab to see its type and documentation.

## How it works

1. The kernel spawns `idris2 --source-dir src -p contrib src/Notebook/Prelude.idr`
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

## Tests

```bash
make test-jupyter-unit   # cell parser only (no backend needed)
make test-jupyter        # full suite: parser + REPL integration + FFI + recovery
```

## Switching backends

The kernel uses whichever backend is active (`build/libidrisml.dylib` symlink). To switch:

```bash
make BACKEND=mlx backend    # rebuild with MLX
make jupyter-install        # re-copies dylib
```
