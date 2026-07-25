# User documentation

**Start here:**

- [**Why idris-ml**](why-idris-ml.md) -- the case for the library: dynamic-graph ergonomics with safety stronger than any static graph, compared against PyTorch, TF1/JAX, and Haskell across five guarantees (shape, device, multi-backend, grad-mode, dtype)
- [Getting Started](getting-started.md) -- text walkthrough from a first tensor to a trained model (Jupyter-independent)
- [idris-transformers](idris-transformers.md) -- **load real HuggingFace BERT / GPT-2 / Llama / BitNet checkpoints** with `fromPretrained`; fine-tuning, LoRA, and the HF-roundtrip correctness gates (BERT matches PyTorch to 4e-4)

**Reference + deep dives:**

- [PyTorch Mapping](pytorch-mapping.md) -- concept translation for PyTorch users (tensors, models, optimizers, training loops)
- [Static vs Dynamic Graphs](static-vs-dynamic-graphs.md) -- deep dive: how dependent types give you static shape safety with dynamic graph ergonomics (the NTM dimension-threading example)
- [Grad-Mode and Device Typing](grad-mode-and-device-typing.md) -- deep dive: phantom enums vs dependent types vs linear types — what each guarantee actually requires
- [Benchmarks](benchmarks.md) -- performance comparison vs PyTorch across tape, MLX, and torch backends
- [Jupyter Notebooks](../../packages/jupyter/README.md) -- interactive notebook setup, tutorials, and per-model walkthroughs

Building the repo, the test layers, and the architecture rationale live in
[CONTRIBUTING.md](../../CONTRIBUTING.md) and [develop/](../develop/README.md).
