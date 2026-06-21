# Documentation

## For Users

**Start here:**

- [**Why idris-ml**](why-idris-ml.md) -- the case for the library: dynamic-graph ergonomics with safety stronger than any static graph, compared against PyTorch, TF1/JAX, and Haskell across five guarantees (shape, device, multi-backend, grad-mode, dtype)
- [Getting Started](getting-started.md) -- text walkthrough from a first tensor to a trained model (Jupyter-independent)
- [idris-transformers](users/idris-transformers.md) -- **load real HuggingFace BERT / GPT-2 / Llama / BitNet checkpoints** with `fromPretrained`; fine-tuning, LoRA, and the HF-roundtrip correctness gates (BERT matches PyTorch to 4e-4)

**Reference + deep dives:**

- [PyTorch Mapping](pytorch-mapping.md) -- concept translation for PyTorch users (tensors, models, optimizers, training loops)
- [Static vs Dynamic Graphs](static-vs-dynamic-graphs.md) -- deep dive: how dependent types give you static shape safety with dynamic graph ergonomics (the NTM dimension-threading example)
- [Grad-Mode and Device Typing](grad-mode-and-device-typing.md) -- deep dive: phantom enums vs dependent types vs linear types — what each guarantee actually requires
- [Benchmarks](benchmarks.md) -- performance comparison vs PyTorch across tape, MLX, and torch backends
- [Jupyter Notebooks](../packages/jupyter/README.md) -- interactive notebook setup, tutorials, and per-model walkthroughs

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
