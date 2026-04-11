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

**idris-ml makes these compile errors.** Tensor shapes are part of the type:

```idris
data Tensor : Vect rank Nat -> Type -> Type where
  STensor : ty -> Tensor [] ty                                    -- scalar
  VTensor : Vect dim (Tensor dims ty) -> Tensor (dim :: dims) ty  -- n-dimensional
```

The network type chains layers with compile-time dimension threading:

```idris
(~>) : AnyLayer i h ty -> Network h hs o ty -> Network i (h :: hs) o ty

-- Compiles: output 10 matches input 10
ll <- linearLayer {i=2, o=10}
let model = ll ~> OutputLayer softmaxLayer

-- Compile error: output 10 doesn't match input 5
ll2 <- linearLayer {i=5, o=3}
let bad = ll ~> OutputLayer ll2  -- Error: Can't unify 10 with 5
```

NTM dimension relationships are type-level functions -- change one and the compiler tells you everywhere else that needs updating:

```idris
ReadParamWidth : Nat -> Nat
ReadParamWidth m = (m + ShiftKernelSize) + 3

WriteParamWidth : Nat -> Nat
WriteParamWidth m = ReadParamWidth m + m
```

You get dynamic graph ergonomics (standard `if`/`for`/`while`, normal debugging, define-by-run autograd) with static graph safety (shape errors are impossible at runtime). See [docs/static-vs-dynamic-graphs.md](docs/static-vs-dynamic-graphs.md) for the full discussion.

## What works today

| Example | Description | Command |
|---------|-------------|---------|
| Supervised | 3-class classification with softmax | `make supervised` |
| RNN | Sequence prediction (repeating pattern) | `make rnn` |
| LSTM | Same task, LSTM controller | `make lstm` |
| NTM Copy | Neural Turing Machine binary vector copy | `make ntm-copy` |
| NTM Recall | NTM associative recall (content-based memory) | `make ntm-associative-recall` |

All examples accept `--epochs`, `--lr`, `--seed` and task-specific flags.

## Quick start

Requires [Idris 2](https://github.com/idris-lang/Idris2) (0.8.0+) and a C compiler.

```bash
make backend       # build the C tape backend (no external dependencies)
make supervised    # run an example
make ntm-copy      # train NTM on binary copy task
make test          # run test suite
```

For the optional libtorch backend: `make BACKEND=torch backend`.

## Performance

NTM-copy runs at ~110ms/epoch on the C tape backend (Apple M-series), comparable to the PyTorch reference (~130ms/epoch). See [docs/performance-analysis.md](docs/performance-analysis.md).

## Architecture

```
Tensor (shape-indexed)  ->  Variable (autograd)  ->  Layer (composable)  ->  Train (runner)
  [3,4] Double              wraps C tensor          LayerLike interface     runTraining
  compile-time shapes       tape-based backward      Network chains layers  early stopping
                            native optimizers        LSTM, Linear, NTM      CLI arg parsing
```

See [CLAUDE.md](CLAUDE.md) for the full module dependency order and development guide.

## References

- [Neural Turing Machines](https://arxiv.org/abs/1410.5401) (Graves, Wayne, Danihelka 2014)
- [Implementing Neural Turing Machines](https://arxiv.org/abs/1807.08518) (Collier & Beel 2018)
- [Idris 2: Quantitative Type Theory in Practice](https://arxiv.org/abs/2104.00480) (Brady 2021)
