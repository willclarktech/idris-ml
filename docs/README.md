# Documentation

## For Users

- [PyTorch Mapping](pytorch-mapping.md) -- concept translation for PyTorch users (tensors, models, optimizers, training loops)
- [Static vs Dynamic Graphs](static-vs-dynamic-graphs.md) -- how dependent types give you static shape safety with dynamic graph ergonomics
- [Benchmarks](benchmarks.md) -- performance comparison vs PyTorch across tape, MLX, and torch backends
- [Jupyter Notebooks](../jupyter/README.md) -- interactive notebook setup, tutorials, and per-model walkthroughs

## For Contributors

Architecture, design rationale, and implementation deep-dives live in [develop/](develop/):

- [Design Decisions](develop/design-decisions.md) -- autograd, optimizers, tensor ops, infrastructure choices
- [Performance Analysis](develop/performance-analysis.md) -- profiling methodology, optimization history, bottleneck analysis
- [Gotchas](develop/gotchas.md) -- Idris 2, Chez Scheme, and FFI pitfalls with workarounds
- [NTM Architecture](develop/ntm.md) -- Neural Turing Machine design and implementation
- [NTM Convergence](develop/ntm-convergence-results.md) -- ablation studies and experimental results
- [DNC Convergence](develop/dnc-convergence-results.md) -- Differentiable Neural Computer results
- [CUDA Testing](develop/cuda-testing.md) -- GPU testing on Google Colab
- [RefC Investigation](develop/refc-investigation.md) -- RefC backend compatibility and findings
- [RefC Upstream Bug](develop/refc-upstream-bug.md) -- draft bug report for idris-lang/Idris2
- [Reference Alignment](develop/reference-alignment.md) -- Idris/PyTorch example alignment policy and change log
