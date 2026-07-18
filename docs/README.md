# Documentation

## For Users

**Start here:**

- [**Why idris-ml**](users/why-idris-ml.md) -- the case for the library: dynamic-graph ergonomics with safety stronger than any static graph, compared against PyTorch, TF1/JAX, and Haskell across five guarantees (shape, device, multi-backend, grad-mode, dtype)
- [Getting Started](users/getting-started.md) -- text walkthrough from a first tensor to a trained model (Jupyter-independent)
- [idris-transformers](users/idris-transformers.md) -- **load real HuggingFace BERT / GPT-2 / Llama / BitNet checkpoints** with `fromPretrained`; fine-tuning, LoRA, and the HF-roundtrip correctness gates (BERT matches PyTorch to 4e-4)

**Reference + deep dives:**

- [PyTorch Mapping](users/pytorch-mapping.md) -- concept translation for PyTorch users (tensors, models, optimizers, training loops)
- [Static vs Dynamic Graphs](users/static-vs-dynamic-graphs.md) -- deep dive: how dependent types give you static shape safety with dynamic graph ergonomics (the NTM dimension-threading example)
- [Grad-Mode and Device Typing](users/grad-mode-and-device-typing.md) -- deep dive: phantom enums vs dependent types vs linear types — what each guarantee actually requires
- [Benchmarks](users/benchmarks.md) -- performance comparison vs PyTorch across tape, MLX, and torch backends
- [Jupyter Notebooks](../packages/jupyter/README.md) -- interactive notebook setup, tutorials, and per-model walkthroughs

## For Contributors

Architecture, design rationale, and implementation deep-dives live in [develop/](develop/).

**Architecture & design (living reference):**

- [Design Decisions](develop/design-decisions.md) -- the master rationale ledger: autograd, open executor/dtype kinds, the `fit` driver, checkpointing, infrastructure choices
- [Gotchas](develop/gotchas.md) -- Idris 2, Chez Scheme, numerics, and FFI pitfalls with workarounds
- [Linear Types and Effects](develop/linear-types-and-effects.md) -- the models-as-linear-resources design (`L IO`, stale-handle compile errors)
- [Tensor Lifecycle](develop/tensor-lifecycle.md) -- the wrapped-handle FFI ABI and refcount model
- [Device Availability Gating](develop/device-availability-gating.md) -- the two-gate system: compile-time `Linked` linkage + runtime EAFP hardware presence
- [NTM Architecture](develop/ntm.md) -- Neural Turing Machine design and implementation
- [LLM Inference](develop/llm-inference.md) -- the Llama end-to-end inference walkthrough (KV cache, RoPE, sampling)
- [Roadmap](develop/roadmap.md) -- the publishable-v1 workstream sequence

**Testing & quality policies:**

- [Testing](develop/testing.md) -- test layer overview; [Testing Taxonomy](develop/testing-taxonomy.md) -- the target-naming contract and C test layout
- [Coverage Policy](develop/coverage-policy.md) -- what "covered" means per backend; [Coverage Remaining](develop/coverage-remaining.md) -- the gap snapshot it tracks
- [Reachability Policy](develop/reachability-policy.md) -- what the Idris reachability gap-finder measures
- [Reference Alignment](develop/reference-alignment.md) -- Idris/PyTorch example alignment policy and change log
- [Example Coverage](develop/example-coverage.md) -- which library features each example exercises
- [CUDA Testing](develop/cuda-testing.md) -- the torch-backend CUDA smoke on Google Colab

**Performance regime** (four-file discipline; see CLAUDE.md "Performance documentation regime"):

- [perf-log.md](develop/perf-log.md) -- schema + jq cookbook for the append-only `perf-log.jsonl`; [perf-log-ref.md](develop/perf-log-ref.md) -- the third-party baseline log's schema
- [Perf Baseline](develop/perf-baseline.md) -- current-state Idris-vs-PyTorch ratio table
- [Perf Changes](develop/perf-changes.md) -- append-only log of every perf change, including reverted attempts
- [Performance Analysis](develop/performance-analysis.md) -- dated profiling/optimization analyses
- [Chez Profiling](develop/chez-profiling.md) -- source-level profiling of Idris-generated Scheme

**Historical records** (banner-marked; identifiers are era-accurate, see [path-c-migration.md](develop/path-c-migration.md) for the name decoder):

- [API Critique](develop/api-critique.md) (2026-06-11 audit) · [Dtype Parameter](develop/dtype-parameter.md) (design memo) · [Path C Migration](develop/path-c-migration.md)
- [NTM Convergence](develop/ntm-convergence-results.md) · [DNC Convergence](develop/dnc-convergence-results.md) · [NTM/DNC Perf Attribution](develop/ntm-dnc-perf-attribution.md) · [DNC Perf Baseline](develop/dnc-perf-baseline.md) · [Hyperparameter Tuning 2026](develop/hyperparameter-tuning-2026.md)
- Surveys: [mlx](develop/mlx-survey.md) · [PyTorch internals](develop/pytorch-survey.md) · [Glaive/TensorType](develop/glaive-survey.md) · [Idris JIT / JAX](develop/idris-jit-jax-investigation.md)
- [RefC Investigation](develop/refc-investigation.md) · [RefC Upstream Bug](develop/refc-upstream-bug.md) -- draft bug report for idris-lang/Idris2
