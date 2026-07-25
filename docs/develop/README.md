# Contributor documentation

Architecture, design rationale, and implementation deep-dives. Build
instructions and the test-gate overview live in
[CONTRIBUTING.md](../../CONTRIBUTING.md); [CLAUDE.md](../../CLAUDE.md) carries
the module dependency order and the working conventions.

**Architecture & design (living reference):**

- [Design Decisions](design-decisions.md) -- the master rationale ledger: autograd, open executor/dtype kinds, the `fit` driver, checkpointing, infrastructure choices
- [Gotchas](gotchas.md) -- Idris 2, Chez Scheme, numerics, and FFI pitfalls with workarounds
- [Linear Types and Effects](linear-types-and-effects.md) -- the models-as-linear-resources design (`L IO`, stale-handle compile errors)
- [Tensor Lifecycle](tensor-lifecycle.md) -- the wrapped-handle FFI ABI and refcount model
- [Device Availability Gating](device-availability-gating.md) -- the two-gate system: compile-time `Linked` linkage + runtime EAFP hardware presence
- [NTM Architecture](ntm.md) -- Neural Turing Machine design and implementation
- [LLM Inference](llm-inference.md) -- the Llama end-to-end inference walkthrough (KV cache, RoPE, sampling)
- [Roadmap](roadmap.md) -- the workstream sequence

**Testing & quality policies:**

- [Testing](testing.md) -- test layer overview; [Testing Taxonomy](testing-taxonomy.md) -- the target-naming contract and C test layout
- [Coverage Policy](coverage-policy.md) -- what "covered" means per backend; [Coverage Remaining](coverage-remaining.md) -- the gap snapshot it tracks
- [Reachability Policy](reachability-policy.md) -- what the Idris reachability gap-finder measures
- [Reference Alignment](reference-alignment.md) -- Idris/PyTorch example alignment policy and change log
- [Example Coverage](example-coverage.md) -- which library features each example exercises
- [CUDA Testing](cuda-testing.md) -- the torch-backend CUDA smoke on Google Colab

**Performance regime** (four-file discipline; see CLAUDE.md "Performance documentation regime"):

- [perf-log.md](perf-log.md) -- schema + jq cookbook for the append-only `perf-log.jsonl`; [perf-log-ref.md](perf-log-ref.md) -- the third-party baseline log's schema
- [Perf Baseline](perf-baseline.md) -- current-state Idris-vs-PyTorch ratio table
- [Perf Changes](perf-changes.md) -- append-only log of every perf change, including reverted attempts
- [Performance Analysis](performance-analysis.md) -- dated profiling/optimization analyses
- [Chez Profiling](chez-profiling.md) -- source-level profiling of Idris-generated Scheme

**Historical records** (banner-marked; identifiers are era-accurate, see [path-c-migration.md](path-c-migration.md) for the name decoder):

- [API Critique](api-critique.md) (2026-06-11 audit) · [Dtype Parameter](dtype-parameter.md) (design memo) · [Path C Migration](path-c-migration.md)
- [NTM Convergence](ntm-convergence-results.md) · [DNC Convergence](dnc-convergence-results.md) · [NTM/DNC Perf Attribution](ntm-dnc-perf-attribution.md) · [DNC Perf Baseline](dnc-perf-baseline.md) · [Hyperparameter Tuning 2026](hyperparameter-tuning-2026.md)
- Surveys: [mlx](mlx-survey.md) · [PyTorch internals](pytorch-survey.md) · [Glaive/TensorType](glaive-survey.md) · [Idris JIT / JAX](idris-jit-jax-investigation.md)
- [RefC Investigation](refc-investigation.md) · [RefC Upstream Bug](refc-upstream-bug.md) -- draft bug report for idris-lang/Idris2
