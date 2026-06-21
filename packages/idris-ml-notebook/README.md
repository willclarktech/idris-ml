# idris-ml-notebook

A one-module re-export shim: `Notebook.Prelude` imports the [idris-ml](../idris-ml/) daily toolkit
(Tensor, Nn, Optimizer, Fit, Dataset, DataStream, Checkpoint, Init, Schedule, RL helpers, …) plus
the linear-types prelude (`Control.Linear.LIO`, `Data.Linear.Notation`) and common stdlib modules,
so a Jupyter cell can use everything without manual `:module` directives.

The [Jupyter kernel](../jupyter/) auto-loads `Notebook.Prelude` at session start — see
[packages/jupyter/README.md](../jupyter/README.md) for the notebook experience, tutorials, and
setup. There's nothing to call here directly; it exists purely to give notebooks a batteries-included
import surface.
