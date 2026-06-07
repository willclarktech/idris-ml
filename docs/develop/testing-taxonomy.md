# Testing taxonomy

This is the **contract** for what every test target name means and
where it sits. New gates land in one of these layers; the
contributor-facing aggregator hides the leaves. The old organic
naming (`test` partial-aggregator, `test-all` everything-and-the-
kitchen-sink, `bench-*` / `check-*` / `coverage-*` siblings with no
shared rules) is replaced.

## Three verbs, three tiers, a few aggregators

Three top-level verbs map to genuinely distinct questions:

| Verb | Question | Reads code? | Runs code? | Writes perf-log.jsonl? |
|---|---|---|---|---|
| `check` | does it compile? | yes | no | no |
| `test` | does it behave? | yes | yes | no |
| `bench` | how fast? | yes | yes | **yes** |

Each verb's bare form (`check`, `test`, `bench`) is the daily-driver
default — the fast, sensible tier. Suffixes scale up (`-all`,
`-full`) or down (`-fast`). The bare verb is always an alias for the
default tier of that verb; explicit tiered names exist alongside.

### `check` family — type-check only, no code execution

| Target | Scope | Wall (warm) |
|---|---|---|
| `check` (= `check-idris`) | all Idris-side library packages — core + gym + transformers + notebook | minutes |
| `check-idris-ml` | just the core library | shortest |
| `check-{gym,transformers,notebook}` | per-package | shortest |
| `check-examples` | build every example as an executable | tens of minutes (cold) |
| `check-all` | `check` + `check-examples` | tens of minutes (cold) |

C type-checking isn't separately exposed — the C compile step IS the
type-check, and `make backend` already runs it incrementally with
cached object files. Most cold rebuilds of `check-all` take 20-60 min
because of `check-examples`; warm rebuilds (TTC + object cache present)
are much faster.

### `test` family — execute test code, no perf measurement

| Target | Scope | Wall (warm) |
|---|---|---|
| `test` (= `test-unit`) | all unit suites — Idris + C | a few minutes |
| `test-unit-idris` | Idris-side unit suites across packages (core, gym, args, transformers, examples) | a few minutes |
| `test-unit-c` | Criterion C-side suite | a minute |
| `test-unit-{idris-ml,gym,args,idris-transformers,examples}` | per-package | shortest |
| `test-unit-c-{tape,mlx,torch}` | C suite with that backend forced | a minute each |
| `test-unit-multi-backend` | cross-backend Idris suite (requires BACKEND=tape,torch,mlx) | a few minutes |
| `test-integration` | negative-type gates, linters, multi-module probes | ~5 min |
| `test-e2e` | every example × every available backend at 3–10 epochs; HF roundtrips; jupyter | tens of minutes |
| `test-convergence` | every example to full default epochs, single seed=42 | hours |
| `test-coverage` | three-axis coverage report (symbol + OP_* backward + F32 oracle) | ~10 min |
| `test-all` | unit + integration + e2e (NOT convergence — too long) | tens of minutes |

`make test` is the local pre-commit default. `make test-integration`
adds another few minutes if you touched a type-level guarantee. CI
runs unit + integration + e2e per-backend.

### `bench` family — perf measurement, writes `perf-log.jsonl`

| Target | Scope | Wall (warm) |
|---|---|---|
| `bench` (= `bench-fast`) | Tier 1: Axis A (op kernels) + Axis B (single-layer fwd+bwd), tape only | ≤5 min |
| `bench-deep` | Tier 2: Tier 1 + Axis C (e2e training) + Axis D (HF inference), tape only | ≤20 min |
| `bench-full` | Tier 3: the cross-backend sweep (every example × every backend) | hours |

All three append to `docs/develop/perf-log.jsonl` and regenerate
`BENCHMARKS.md` via `scripts/render-benchmarks.py`. The verb axis is
load-bearing: `bench` writes a perf-log entry, `test` doesn't.

> **Wall-clock note**: numbers above are rough hardware-aware guidance,
> not guarantees. This is a dependently-typed project; Idris-2
> elaboration dominates and the cost is non-linear in module count +
> implicit-arg width. On a slower box or with `nice -n 19`, multiply
> by 2-3×. The only honest measurement is `time make <target>` on the
> box you care about.

## Per-layer leaf naming convention

Every leaf is named `test-{layer}-{topic}` (or
`test-{layer}-{topic}-{backend}` when the same topic ships per
backend). Examples:

- `test-unit-idris-ml` — Idris unit tests for the core package.
- `test-unit-c-tape` — Criterion C tests for the tape backend.
- `test-integration-typegate-gradmode` — negative-type-check gate
  for the GradMode aliasing rule.
- `test-integration-lint-paired-defaults` — drift detector for
  Idris ↔ PyTorch default hyperparameters.
- `test-e2e-hf-llama-roundtrip` — cross-language HfLlama correctness
  gate.
- `bench-rank3-broadcast` — single op-microbench under perf.
- `test-coverage-backend-mlx` — LLVM coverage report for the mlx
  backend.

The pattern is enforced *by convention* — the linter
`test-integration-lint-make-naming` (filed in Phase 6) is the
mechanical enforcement path.

## Test file layout — hybrid colocation

Test files live as close to the source they exercise as Criterion and
Idris-2's module-to-file constraint allow. **The `packages/<pkg>/test/`
tree is gone.** Two rules cover where a new test belongs:

### C side (Criterion suites + standalone main()s)

| Test kind | Lives at |
|---|---|
| Per-op unit test (cross-backend FFI) | `packages/backends/backend_tape/<subsystem>/test_<op>.c` (next to the tape source — `find` glob picks it up regardless of which backend is primary) |
| Backend-specific test (touches internals) | `packages/backends/backend_<b>/<subsystem>/test_<topic>.c` with `#ifdef BACKEND_<NAME>` gating the body |
| Cross-cutting integration / oracle / framework infra | `packages/idris-test-c/src/` (peer package; cross-cutting "shared infra is a package, not just a directory") |
| 1:1 with a source file (`safetensors.c` ↔ `test_safetensors.c`) | next to the source — `packages/backends/test_safetensors.c` |

Discovery is the Makefile's `find` glob (`CRITERION_BACKEND_TEST_SRCS`)
which walks `backend_{tape,torch,mlx}/` for `test_*.c`. The dylib
build's source glob excludes `test_*.c` via `! -name 'test_*.c'` so
test files don't get compiled into `libidrisml.dylib`.

Include paths: tests use bare `#include "backend.h"` and `#include
"test_helpers.h"`; the Makefile passes
`-Ipackages/backends -Ipackages/idris-test-c/include` to resolve.

### Idris side (per-package Test.* subtree under same sourcedir)

Each Idris package has TWO `.ipkg` files at the package root pointing
at the same `src/` sourcedir:

| File | Modules | Purpose |
|---|---|---|
| `packages/<pkg>/<pkg>.ipkg` | `<library modules>` (excludes `Test.*`) | What `pack install` publishes; what library consumers depend on |
| `packages/<pkg>/<pkg>-tests.ipkg` | `Test.Main`, `Test.<Topic>`, …      | What `make test-unit-<pkg>` builds |

Test files live at `packages/<pkg>/src/Test/<Topic>.idr` declaring
`module Test.<Topic>`. The entry point is `Test.Main` at
`src/Test/Main.idr`. idris-ml's generated `Test.Config` (build-time
device/dtype pinning) lives at `src/Test/Config.idr`; the template
lives next to it as `Config.idr.in`.

**Why this layout** (vs the more common pack-db convention of a
separate `test/` directory): puts each test next to the module it
tests — one directory hop, no parallel tree to navigate. Idris-2's
module-path-mirrors-file-path rule blocks literal adjacency (you
can't have `Tensor.idr` and `Test.Tensor.idr` in the same directory
since `Test.Tensor` *must* live at `Test/Tensor.idr` per the
dot-namespace convention), so this is the closest pattern Idris-2
permits. Documented separately in
[`testing.md`](testing.md#dual-ipkg-pattern).

## Perf layer — coverage axes

The perf layer is structured around four axes that decompose
"is the framework competitive?" along the same lines MLPerf /
TorchBench / DeepBench use at scale:

| Axis | Measures | Cadence | What it answers |
|---|---|---|---|
| **A. Op kernel** | Per-kernel wall-clock vs PyTorch on the same hardware. Pure forward, no autograd. | Tier 1 (CI) | "Is our matmul / SDPA / conv kernel competitive at the primitive level?" |
| **B. Layer composition** | Forward + backward through one layer type at representative sizes. | Tier 1 (CI) | "Does our FFI + tape wrap cost dominate the kernel?" |
| **C. End-to-end training** | Full training loop (data → forward → backward → optimizer) for one workload per training mode. | Tier 2 (per publication push) | "How fast does a user-facing example train vs a PyTorch reference?" |
| **D. End-to-end inference** | Pretrained HF model forward / generate. | Tier 2 (per publication push) | "How fast does a real production model run on this stack vs HF transformers?" |

**Selection rule per axis**: a workload earns a slot iff it
exercises a *distinct* compute pattern not covered by an existing
slot. Selection is by pattern, not count. If two end-to-end
workloads exercise the same compute pattern (`rnn`/`gru` both ride
the RNN cell path), only one earns a slot.

Tier 1 + Tier 2 auto-regenerate `BENCHMARKS.md` at the repo root —
the external-facing artifact answering "how does idris-ml compare
to PyTorch / JAX?" Tier 3 (`bench-full`) is the existing full
sweep; it does *not* regenerate the doc (it covers the same axes
but across all backends, producing the deeper apples-to-apples
table elsewhere in `docs/develop/perf-baseline.md`).

**Current implementation status**: Axis A is landed
(`scripts/perf-fast.sh` drives `make bench-ops` + `make bench-ops-py`,
parses output, appends `kind: "op_bench"` entries to
`perf-log.jsonl`, regenerates `BENCHMARKS.md` via
`scripts/render-benchmarks.py`). Today's Axis A panel covers
matmul, matvec, elementwise, softmax, conv2d (PyTorch-only —
disabled on tape pending fix), train-step, SDPA (GQA),
embedding gather, RMSNorm. Axes B/C/D are placeholders; the
renderer emits an "No entries yet" stub per missing axis. The
selection rule + the planned B/C/D inventories are tracked in
the relevant `TODO.md` rows.

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
| `test-backend-criterion` | `test-unit-c` |
| `test-backend-criterion-tape` | `test-unit-c-tape` |
| `test-backend-criterion-mlx` | `test-unit-c-mlx` |
| `test-backend-criterion-torch` | `test-unit-c-torch` |
| `test-safetensors` | `test-unit-safetensors` |
| `test-ntm-grad` | `test-unit-ntm-grad` |
| `test-ntm-timestep` | `test-unit-ntm-timestep` |
| `test-mlx-compile` | `test-unit-mlx-compile` |
| *(new)* | `test-unit` — aggregator |

(Historical table — the four standalone-main() targets in the last
rows were subsequently converted to Criterion suites on 2026-06-05
and their dedicated recipes deleted; their assertions now ride
`test-unit-c` via the glob discovery.)

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

### Perf layer (`bench-*` tier aggregators + axis drivers)

The perf benchmark suite is structured around the four axes (Axis A
op kernel, Axis B layer composition, Axis C e2e training, Axis D e2e
inference). Three cadence tiers + a handful of axis drivers:

| Target | What |
|---|---|
| `bench` (= `bench-fast`) | Tier 1: Axes A + B, tape only, ≤5 min |
| `bench-deep` | Tier 2: Axes A + B + C + D, tape only, ≤20 min |
| `bench-full` | Tier 3: cross-backend sweep, hours, wraps `scripts/perf-sweep.sh` |
| `bench-ops` | Axis A driver (Idris-side op-level wall-clock) |
| `bench-ops-py` | Axis A driver (PyTorch reference) |
| `bench-ops-compare` | Idris-vs-PyTorch op comparison |
| `bench-layers` / `bench-layers-py` | Axis B driver pair |
| `bench-rank3-broadcast` / `-wrapped` | Microbench microbench for the rank-3 broadcast op |
| `bench-gym` | Gym package microbench |
| `bench-compare` / `bench-py` | Legacy convenience targets — prefer the Axis-specific drivers above |

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

## `.expect` vs `Test.Property.Golden` — when to use which

Both harnesses live side-by-side. They have different semantics; pick by what
the test is asserting, not by which framework is newer.

| Question | Pick |
|---|---|
| Numerical metric with a tolerance? (`loss < 5.0`, `accuracy >= 0.05`) | **`.expect`** — threshold-based RESULT-line check via `scripts/check-result.sh`. NaN-safe. |
| Byte-deterministic output across runs + machines? (boolean state, schema dump, CLI help) | **`Test.Property.Golden`** — verbatim equality. `GOLDEN_UPDATE=1` to re-baseline. |
| Wall-clock or RSS in the output? | **`.expect`** presence-only (no threshold), or skip from RESULT entirely. **Never `Golden`** — flakes on machine drift. |
| Boolean or categorical state? (`overall=ok`, `status=converged`) | **`Test.Property.Golden`** if integration cost is low. **`.expect`** with `key == value` operator if the test target is already running under the example harness. |
| Pure-Idris invariant (sum-to-one, round-trip)? | Neither — use **`Test.Property`** (Hedgehog) for property-based generation. |

### Why we didn't migrate `example-precision-demo` to Golden

`test-examples.expect` includes `example-precision-demo overall == ok` —
the cleanest "byte-deterministic categorical state" fixture in the codebase.
Migration to `Test.Property.Golden` looked attractive but doesn't pay off
in practice:

1. `Example.PrecisionDemo.main` requires the multi-backend build
   (`BACKEND=tape,torch,mlx`) — Part 3 hops Tape → Torch → Mlx → Tape. The
   per-backend `test-unit-examples` CI lane can't load the example's
   `main` since two of three backend symbol sets aren't linked. Wiring
   Golden into `test-unit-multi-backend` works but breaks the "one
   adjacent test per example" colocation expectation.
2. The example writes to stdout directly (via `putStrLn`); `Test.Property.Golden`'s
   `checkGolden : String -> String -> IO String -> IO Bool` wants an `IO String`
   action. Either refactor the example to return its rendered output (invasive
   for one fixture) or capture stdout via `popen` (heavier than the test's value).
3. The current `.expect` line catches the failure mode the row cares about
   ("did all 3 parts pass"). Golden would additionally pin the exact F32-cast
   readback numbers — strictly more information, but the current row gives no
   evidence that line was the wrong layer.

So the migration was filed against the wrong fixture. The decision tree above
is what stays; the row closes without code change.

## CI workflow consumption

CI jobs mirror the taxonomy one-to-one (the 2026-06-11 detector
restructure): `.github/workflows/test.yml` has jobs named `lint`,
`lint-full`, `build`, `test-unit`, `test-integration`,
`test-e2e-examples`, `test-e2e-hf`, and `coverage`;
`.github/workflows/perf.yml` runs `bench-deep`. Each test job's
make-invocation block is generated from
`.github/workflows/test.yml.spec.json` by
`scripts/gen-ci-workflow.py` (one marker-bounded region per job; the
generator also injects `!cancelled()` into every step so one red
gate never hides its siblings). Two lints keep the three layers in
sync: `make test-integration-lint-ci-workflow` fails CI if a
hand-edit of `test.yml` diverges from the spec, and
`make test-integration-lint-ci-coverage`
(`scripts/check-ci-gate-coverage.py`) fails if a workflow invokes a
nonexistent make target or if an aggregator leaf runs in no workflow
without being a named exception in the spec. So adding a new gate to
an aggregator means adding it to the spec too — the coverage lint
reminds you.

## When to run what

| Situation | Command |
|---|---|
| Inner-loop edit to one package | `make check-<package>` (shortest) |
| Before opening a PR | `make check && make test` |
| Touched a type-level guarantee | also `make test-integration` |
| Touched an example or training-loop module | also `make test-e2e` |
| Touched any C kernel | also `make test-unit-c-{tape,mlx,torch}` for the backends you touched |
| Touched an example you want to compile-test | `make check-examples` (or just `make check-all`) |
| Investigating a perf regression | `make bench` (Tier 1) → `make bench-deep` (Tier 2) |
| Pre-release validation | `make test-convergence` + `make bench-full` |
| Iterating on one example | `make example-<name>` |

## Why this is the contract

Every contributor friction point from TODO row 10 collapses to a
single rule above:

- "What's the right gate for this change?" — pick the layer.
  Aggregator name reveals it.
- "Add a new gate" — drop it in the right `test-{layer}-{topic}`
  slot; the aggregator picks it up automatically; the CI coverage
  lint then demands a matching spec entry (see "CI workflow
  consumption").
- "`make test` is a partial aggregator" — replaced by `test-unit`
  (true unit aggregator) + per-layer aggregators above it.
- "Per-package harnesses are not uniform" — Phase 2 collapses the
  four duplicate `Harness.idr` copies to `packages/idris-test`.
- "Coverage + perf are siblings to tests but live in their own
  naming worlds" — now they sit *in* the test layer cake under
  `test-coverage-*` and `bench-*`.

The taxonomy is the contract; if a change wants to add a target
that doesn't fit, the right move is to update *this doc* first to
say where it belongs, then land the Makefile change.
