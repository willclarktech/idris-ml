# idris-ml

The core deep-learning library: an autograd `Tensor`, a models-as-records neural-network
surface (`Nn`), optimizers, a training driver (`fit`), data plumbing, and SafeTensors
checkpoints — all with compile-time shape / device / dtype / grad-mode safety and a pluggable
C/C++ backend (tape, libtorch, MLX).

For the *why* — the five compile-time guarantees compared side by side against PyTorch,
TensorFlow 1.x / JAX, and Haskell — see [**docs/users/why-idris-ml.md**](../../docs/users/why-idris-ml.md).

## Single-import surface

`import ML` brings the daily toolkit in one line — the autograd `Tensor` (+ operator aliases,
losses), the `Nn` model library, optimizers + the typed-scope surface, `Dataset`/`DataStream`,
the `fit` driver, checkpoints, and the `Backend` constraint bundle:

```idris
import ML            -- everything; you pin {ex=} / {dt=} at the leaf
import ML.Simple     -- ML plus the build's default (executor, dtype) as `Ex` / `F`
```

`ML.Simple` additionally pins the build's default `(executor, dtype)` cell as `Ex` / `F`, so
tutorial and example code writes `Tensor dims Ex F g` and never spells `{ex=}`. Granular imports
(`import Tensor`, `import Nn`, `import Optimizer`, `import Fit`, …) work too.

## The tensor type

Shape, executor (backend), dtype, and grad-mode all live on the type and are erased at runtime:

```idris
record Tensor (dims : Vect rank Nat) (0 ex : Executor) (0 dt : DType) (0 g : GradMode) where
  constructor MkTensor
  tensorPtr : AnyPtr        -- backend handle (carries the autograd graph)
  paramId   : Maybe String  -- registry key for the optimizer (Nothing = intermediate)
```

Construct with one surface over `InitSpec` — `tensor {dims=[2,3]} (Const 0.5)`,
`param "w" (Normal 0.0 0.02)`. Elementwise/matmul/loss ops are shape-checked: a `tadd` of
`[4]` and `[5]` is a compile error.

## Models as records

A model is a record of `Nn` layers or a `Seq` chain (`~~>`, endpoints-only index, hidden dims
existential), built in the `Init` monad and realised with `runInitL`:

```idris
Model : Type
Model = Seq 2 3 Ex F WithGrad

mkModel : Init Model
mkModel = do
  l1 <- linear {i=2} {o=10}
  l2 <- linear {i=10} {o=3}
  pure (l1 ~~> reluA ~~> l2 ~~> Nil)   -- l1's out 10 unifies with l2's in 10

-- Swap l2 for `linear {i=5} {o=3}` and the chain won't elaborate: Mismatch between: 10 and 5
```

A model is a **single-owner linear resource** threaded through `Control.Linear.LIO.L IO`:
`forward` / `eval` / `freeze` consume the handle and thread back a fresh one, so "freeze then
train via the stale handle" (a silent no-op against the shared C params) is a compile-time
linearity error. Tensors stay unrestricted. Train with the `fit` driver:

```idris
(trained, epochs, loss) <- fitSupervised opt lossFn (batched stream) (simpleConfig 1000) model
```

19 layers are available (Linear, Conv1D/2D, MaxPool, LayerNorm/BatchNorm/RmsNorm, Embedding,
Dropout, Residual, RNN/LSTM/GRU, NTM, DNC, Attention, TransformerBlock, RoPE, PosEncoding) plus
the four IO optimizers (`sgd` / `rmsprop` / `adam` / `adamW`).

## Architecture

```
Array (Vect-of-Vect)  ->  Tensor (autograd)     ->  Nn (models-as-records)  ->  fit (driver)
  [3,4] Double            shape+executor+dtype       Module / Seq (~~>)          fitSupervised
  pure-Idris ops          +grad-mode on the type     linear single-owner models  early stopping
                          backend C handle, opts     Linear, Conv, LSTM, NTM…    checkpointing
```

Module dependency order and the full development guide live in
[CLAUDE.md](../../CLAUDE.md).

## Backends

`Executor` is an *open* kind. `BACKEND` is a comma-separated list linked into one
`libidrisml.{so,dylib}`; the first item is primary. The C/C++ backend internals are documented
in [`packages/backends/README.md`](../backends/README.md).

```bash
make backend                                  # default: BACKEND=tape (lean, no C++ deps)
make BACKEND=tape,torch,mlx backend           # multi-link: all three in one dylib, tape primary
make BACKEND=torch backend                    # libtorch only
make BACKEND=mlx MLX_SITE=... backend         # MLX (Apple Metal)
make install                                  # install core lib + gym (needed for examples/tests)
```

| Build | Executor / dtype cell |
| --- | --- |
| `BACKEND=tape` | `TapeExecutor`, `F64` |
| `BACKEND=torch TORCH_DEVICE=cpu` | `TorchExecutor TCpu`, `F64` |
| `BACKEND=torch TORCH_DEVICE=mps` | `TorchExecutor TMps`, `F32` (Metal is F32-only) |
| `BACKEND=torch TORCH_DEVICE=cuda` | `TorchExecutor (TCuda 0)`, `F64` |
| `BACKEND=mlx MLX_DEVICE=cpu` | `MlxExecutor MCpu`, `F64` |
| `BACKEND=mlx MLX_DEVICE=gpu` | `MlxExecutor MGpu`, `F32` |

The nixpkgs `python3Packages.mlx` is CPU-only (Metal compute hardcoded off); for a
Metal-capable build use a project-local pip install:

```bash
uv venv .venv-mlx && source .venv-mlx/bin/activate && uv pip install mlx
make BACKEND=mlx MLX_SITE=$VIRTUAL_ENV/lib/python3.13/site-packages/mlx backend
```

At the current example scales (RNN-cell, NTM/DNC, batch-32 MNIST) per-op kernel-launch overhead
makes `MLX_DEVICE=gpu` 3–12× slower than the CPU stream — default `MLX_DEVICE=cpu` is the right
choice for the shipped examples.

## Performance

NTM-copy runs at ~110 ms/epoch on the C tape backend (Apple M-series), comparable to the PyTorch
reference (~130 ms/epoch). Full cross-backend comparisons: [docs/users/benchmarks.md](../../docs/users/benchmarks.md).

## See also

- [docs/users/why-idris-ml.md](../../docs/users/why-idris-ml.md) — the five-guarantee case.
- [docs/users/getting-started.md](../../docs/users/getting-started.md) — first tensor → first trained model.
- [docs/users/pytorch-mapping.md](../../docs/users/pytorch-mapping.md) — concept-by-concept for PyTorch users.
- [idris-ml-examples](../idris-ml-examples/) — runnable examples; [idris-transformers](../idris-transformers/) — real HF models.
