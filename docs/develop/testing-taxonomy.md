# Testing taxonomy

This is the **contract** for what every test target name means and
where it sits. New gates land in one of these layers; the
contributor-facing aggregator hides the leaves. The old organic
naming (`test` partial-aggregator, `test-all` everything-and-the-
kitchen-sink, `bench-*` / `check-*` / `coverage-*` siblings with no
shared rules) is replaced.

## The five test layers + one preflight + one release tier

A contributor working pre-commit runs **one** aggregator per layer:

| Tier | Aggregator | Wall (tape) | What lives here |
|---|---|---|---|
| **preflight** | `check-all` | ~30s | Pure type-checks of every package. Not a test layer; upstream of testing. ("Does it compile?") |
| **unit** | `test-unit` | ~2 min | Idris pure-logic + C-side ops + safetensors round-trip. No example training. |
| **integration** | `test-integration` | ~5 min | Negative-type gates, linters, multi-module integration probes that don't run a full training loop. |
| **e2e** | `test-e2e` | ~15 min | Every example × every available backend at 3–10 epochs; HF roundtrips; jupyter execution. The "does the user-facing program work end-to-end" layer. |
| **perf (Tier 1)** | `test-perf-fast` | ≤5 min | Op-kernel (Axis A) + single-layer fwd+bwd (Axis B), tape only. Auto-regenerates `BENCHMARKS.md`. |
| **perf (Tier 2)** | `test-perf-nightly` | ≤20 min | Tier 1 + e2e training (Axis C) + HF inference (Axis D), tape only. Scheduled nightly. |
| **perf (Tier 3)** | `test-perf-full` | hours | The full 80-cell sweep (every example × every backend). Manual / pre-tag. Wraps `scripts/perf-sweep.sh`. |
| **coverage** | `test-coverage` | ~10 min | Three-axis target (symbol + OP_* backward + F32 oracle). Advisory only — see [coverage-policy.md](coverage-policy.md). |
| **convergence** | `test-convergence` | hours | Every example to full default epochs, single seed=42, tape only. Release validation, not run on PRs. |

`make test-unit` is the local pre-commit default. `make
test-integration` adds another ~5 min if you touched a type-level
guarantee. CI runs unit + integration + e2e (+ perf-fast at Tier 1).

## Per-layer leaf naming convention

Every leaf is named `test-{layer}-{topic}` (or
`test-{layer}-{topic}-{backend}` when the same topic ships per
backend). Examples:

- `test-unit-idris-ml` — Idris unit tests for the core package.
- `test-unit-backend-tape` — Criterion C tests for the tape backend.
- `test-integration-typegate-gradmode` — negative-type-check gate
  for the GradMode aliasing rule.
- `test-integration-lint-paired-defaults` — drift detector for
  Idris ↔ PyTorch default hyperparameters.
- `test-e2e-hf-llama-roundtrip` — cross-language HfLlama correctness
  gate.
- `test-perf-rank3-broadcast` — single op-microbench under perf.
- `test-coverage-backend-mlx` — LLVM coverage report for the mlx
  backend.

The pattern is enforced *by convention* — the linter
`test-integration-lint-make-naming` (filed in Phase 6) is the
mechanical enforcement path.

## Perf layer — coverage axes

The perf layer is structured around four axes that decompose
"is the framework competitive?" along the same lines MLPerf /
TorchBench / DeepBench use at scale:

| Axis | Measures | Cadence | What it answers |
|---|---|---|---|
| **A. Op kernel** | Per-kernel wall-clock vs PyTorch on the same hardware. Pure forward, no autograd. | Tier 1 (CI) | "Is our matmul / SDPA / conv kernel competitive at the primitive level?" |
| **B. Layer composition** | Forward + backward through one layer type at representative sizes. | Tier 1 (CI) | "Does our FFI + tape wrap cost dominate the kernel?" |
| **C. End-to-end training** | Full training loop (data → forward → backward → optimizer) for one workload per training mode. | Tier 2 (nightly) | "How fast does a user-facing example train vs a PyTorch reference?" |
| **D. End-to-end inference** | Pretrained HF model forward / generate. | Tier 2 (nightly) | "How fast does a real production model run on this stack vs HF transformers?" |

**Selection rule per axis**: a workload earns a slot iff it
exercises a *distinct* compute pattern not covered by an existing
slot. Selection is by pattern, not count. If two end-to-end
workloads exercise the same compute pattern (`rnn`/`gru` both ride
the RNN cell path), only one earns a slot.

Tier 1 + Tier 2 auto-regenerate `BENCHMARKS.md` at the repo root —
the external-facing artifact answering "how does idris-ml compare
to PyTorch / JAX?" Tier 3 (`test-perf-full`) is the existing full
sweep; it does *not* regenerate the doc (it covers the same axes
but across all backends, producing the deeper apples-to-apples
table elsewhere in `docs/develop/perf-baseline.md`).

## Current → new target rename table

Every existing target gets renamed in one wave (Phase 3 + Phase 4
land together). **No deprecation aliases** — per "no users → no
backcompat", old names disappear in the same commit they're
renamed. In-repo callsites (`scripts/*.sh`, in-Makefile invocations,
`.github/workflows/`) are updated in the same wave.

### Unit layer (16 entries)

| Old | New |
|---|---|
| `test` (partial agg) | *deleted* — use `test-unit` |
| `test-idris` | `test-unit-idris-ml` |
| `test-multi` | `test-unit-multi-backend` |
| `test-gym` | `test-unit-gym` |
| `test-transformers` | `test-unit-idris-transformers` |
| `test-examples-unit` | `test-unit-examples` |
| `test-backend-criterion` | `test-unit-backend` |
| `test-backend-criterion-tape` | `test-unit-backend-tape` |
| `test-backend-criterion-mlx` | `test-unit-backend-mlx` |
| `test-backend-criterion-torch` | `test-unit-backend-torch` |
| `test-safetensors` | `test-unit-safetensors` |
| `test-ntm-grad` | `test-unit-ntm-grad` |
| `test-ntm-timestep` | `test-unit-ntm-timestep` |
| `test-mlx-compile` | `test-unit-mlx-compile` |
| *(new)* | `test-unit` — aggregator |

### Integration layer (10 entries)

| Old | New |
|---|---|
| `check-gradmode-gate` | `test-integration-typegate-gradmode` |
| `check-gradmode-aliasing` | `test-integration-typegate-gradmode-aliasing` |
| `check-lossy-cast-gate` | `test-integration-typegate-lossy-cast` |
| `check-int-overflow-cast-gate` | `test-integration-typegate-int-overflow-cast` |
| `check-rename-headers` | `test-integration-lint-rename-headers` |
| `check-ffi-wrap-template` | `test-integration-lint-ffi-wrap-template` |
| `check-non-io-side-effects` | `test-integration-lint-non-io-side-effects` |
| `check-paired-defaults` | `test-integration-lint-paired-defaults` |
| `check-example-hf-llama-inference` | `test-integration-lint-hf-llama-inference` |
| `test-checkpoint-resume` | `test-integration-checkpoint-resume` |
| `test-jupyter-unit` | `test-integration-jupyter-cellparser` |
| *(new)* | `test-integration` — aggregator |

(The `check-*` linters move under `test-integration-*` because they
*test* a property of the source code — drift, FFI conformance,
type-level negative gates — and so semantically belong with the
other gates. Pure type-checks like `check`, `check-gym`,
`check-notebook`, `check-examples`, `check-transformers`,
`check-all` stay under `check-*` because they're upstream
preflight, not behaviour gates.)

### E2E layer (14 entries)

| Old | New |
|---|---|
| `test-hf-bert-roundtrip` | `test-e2e-hf-bert-roundtrip` |
| `test-hf-bitnet-roundtrip` | `test-e2e-hf-bitnet-roundtrip` |
| `test-hf-gpt2-roundtrip` | `test-e2e-hf-gpt2-roundtrip` |
| `test-hf-llama-roundtrip` | `test-e2e-hf-llama-roundtrip` |
| `test-hf-llama-generate-roundtrip` | `test-e2e-hf-llama-generate-roundtrip` |
| `test-transformers-oracle` | `test-e2e-transformers-oracle-bert` |
| `test-transformers-oracle-gpt2` | `test-e2e-transformers-oracle-gpt2` |
| `test-transformers-oracle-llama` | `test-e2e-transformers-oracle-llama` |
| `test-transformers-oracle-llama-generate` | `test-e2e-transformers-oracle-llama-generate` |
| `test-rope-oracle` | `test-e2e-rope-oracle` |
| `test-examples` | `test-e2e-examples` |
| `test-jupyter` | `test-e2e-jupyter` |
| `test-notebooks` | `test-e2e-notebooks` |
| `test-cuda` | `test-e2e-cuda` |
| `test-ref` / `ref-test` | `test-e2e-pytorch-ref` (drop `ref-test` alias) |
| *(new)* | `test-e2e` — aggregator |

### Perf layer (replaces all `bench-*` targets)

Phase 5 designs the perf benchmark suite from scratch around the
four axes (Axis A op kernel, Axis B layer composition, Axis C e2e
training, Axis D e2e inference). The existing `bench-*` targets
fold into the new structure rather than being renamed 1:1.

| Old | New |
|---|---|
| `bench-ops` | `test-perf-ops` (Axis A driver; per-op breakdown) |
| `bench-ops-py` | *deleted* — merged into `test-perf-ops` (the Python ref runs in-process) |
| `bench-ops-compare` | *deleted* — merged into `test-perf-ops` |
| `bench-rank3-broadcast` | `test-perf-microbench-rank3-broadcast` |
| `bench-rank3-broadcast-wrapped` | `test-perf-microbench-rank3-broadcast-wrapped` |
| `bench-py` | *deleted* — folded into `test-perf-nightly` Axis C driver |
| `bench-compare` | *deleted* — folded into `test-perf-nightly` |
| `bench-gym` | `test-perf-gym` |
| *(new, Phase 5)* | `test-perf-fast` — Axis A + B, ≤5 min |
| *(new, Phase 5)* | `test-perf-nightly` — A + B + C + D, ≤20 min |
| *(new, Phase 5)* | `test-perf-full` — wraps `scripts/perf-sweep.sh` |

### Coverage layer (5 entries)

| Old | New |
|---|---|
| `coverage-backend` | `test-coverage-backend` |
| `coverage-backend-tape` | `test-coverage-backend-tape` |
| `coverage-backend-mlx` | `test-coverage-backend-mlx` |
| `coverage-backend-torch` | `test-coverage-backend-torch` |
| `coverage-gap-probe` | `test-coverage-gap-probe` |
| *(new)* | `test-coverage` — aggregator |

### Convergence layer (1 entry)

| Old | New |
|---|---|
| `test-examples-convergence` | `test-convergence` |

### Removed aggregators

| Old | Why removed |
|---|---|
| `test` (partial agg of unit + criterion + safetensors) | Replaced by `test-unit` (true unit aggregator). The old `test` did *not* aggregate everything fast; the new `test-unit` does. |
| `test-all` | Replaced by `test-unit && test-integration && test-e2e`. The "everything" target is not a coherent layer. |
| `all` | Replaced by `check-all && test-unit && test-integration && test-e2e`. |
| `all-backends` | Replaced by `test-e2e` (which already iterates available backends). |

### Preserved as-is (preflight type-checks)

These are *upstream* of testing and aren't renamed:

- `check` — type-check the core library on the active backend.
- `check-gym` / `check-notebook` / `check-examples` /
  `check-transformers` — per-package type-checks.
- `check-all` — aggregate type-check.

## CI workflow consumption

`.github/workflows/test.yml` is generated from a spec
(`.github/workflows/test.yml.spec.yaml`) by `scripts/gen-ci-workflow.sh`
(Phase 4). The spec lists which aggregators run on which platform
with which env. Adding a new gate to `test-unit` auto-includes it in
the CI's unit job — no separate workflow edit. The preflight
`make check-ci-workflow` fails CI if a hand-edit of `test.yml`
diverges from the regenerated output.

## When to run what

| Situation | Command |
|---|---|
| Before opening a PR | `make check-all && make test-unit` (~3 min) |
| Touched a type-level guarantee | also `make test-integration` |
| Touched an example or training-loop module | also `make test-e2e` (~15 min) |
| Investigating a perf regression | `make test-perf-fast` (Tier 1, fast) → `make test-perf-nightly` (Tier 2, deeper) |
| Pre-release validation | `make test-convergence` (hours) + `make test-perf-full` (hours) |
| Iterating on one example | `make example-<name>` |

## Why this is the contract

Every contributor friction point from TODO row 10 collapses to a
single rule above:

- "What's the right gate for this change?" — pick the layer.
  Aggregator name reveals it.
- "Add a new gate" — drop it in the right `test-{layer}-{topic}`
  slot; the aggregator picks it up automatically; the CI workflow
  picks it up automatically (Phase 4 generation).
- "`make test` is a partial aggregator" — replaced by `test-unit`
  (true unit aggregator) + per-layer aggregators above it.
- "Per-package harnesses are not uniform" — Phase 2 collapses the
  four duplicate `Harness.idr` copies to `packages/idris-test`.
- "Coverage + perf are siblings to tests but live in their own
  naming worlds" — now they sit *in* the test layer cake under
  `test-coverage-*` and `test-perf-*`.

The taxonomy is the contract; if a change wants to add a target
that doesn't fit, the right move is to update *this doc* first to
say where it belongs, then land the Makefile change.
