# Testing layers

The canonical contract for test target names and layers lives in
[testing-taxonomy.md](testing-taxonomy.md). This doc covers the
*operational* details — thresholds, expected-failure matrix,
when-to-run guidance — that the contract doesn't.

## TL;DR — the three commands you'll use

| Command | What it does | Wall time | When |
|---|---|---|---|
| `make test-unit` | Idris unit + C-side ops + safetensors round-trip. The pre-commit default. | ~2 min | Every PR |
| `make test-e2e` | Smoke gate: every example × every available backend, 3–10 epochs each, safety-net thresholds. Catches crashes / NaN / divergence / missing RESULT keys. | ~15 min | PR touching an example or training-loop module |
| `make test-convergence` | Convergence: every example at full default epochs, single seed=42, tape only, tight thresholds. Catches "model trains in the wrong direction" and similar correctness regressions. | hours | Release validation |
| `make example-<name>` | Run one example with full default config. The standard dev-iteration command. | varies | Dev iteration |

To run a *subset* of examples to convergence: invoke the
per-example targets you care about (`make example-supervised &&
make example-rnn && …`). There's intentionally no
`convergence-supervised` / `convergence-memory` / sub-targets —
keep the command surface small.

## Per-target reference (post-taxonomy names)

Renames follow the rename table in
[testing-taxonomy.md](testing-taxonomy.md). What each layer
catches:

| Aggregator | Wall (warm) | What it catches | What it doesn't |
|---|---|---|---|
| `check` | minutes (Idris elaboration of 4 lib packages) | Type errors, missing imports, syntax — Idris library packages only | C-side errors (those come out at `make backend` link time); example-level type errors (those need `check-examples`) |
| `check-all` | 20-60 min cold | `check` + every example executable builds | Runtime bugs |
| `test` (`= test-unit`) | a few minutes | Idris-side correctness across packages + C-level op correctness + safetensors round-trip | Anything that requires training a model |
| `test-integration` | ~5 min | Negative-type gates (`test-integration-typegate-*`), lint drift (`test-integration-lint-*`), checkpoint resume, jupyter cell parser, NTM grad/timestep | Full example training |
| `test-e2e` | tens of minutes | Example smoke matrix × 5 backend lanes, HF-roundtrip gates, transformer / oracle gates, jupyter notebook execution | Multi-seed sensitivity (single seed=42 only); strict convergence quality |
| `bench` (`= bench-fast`) | ≤5 min | Op-kernel + single-layer fwd+bwd regressions vs PyTorch | E2E training perf, HF inference perf |
| `bench-deep` | ≤20 min | Tier 1 + e2e training perf + HF inference perf, tape only | Cross-backend perf |
| `bench-full` | hours | Cross-backend perf — every example × every backend | Correctness (perf signal only) |
| `test-coverage` | ~10 min | Three-axis target (symbol + OP_* backward + F32 oracle) | Convergence; advisory only |
| `test-convergence` | hours | "Model trains in the wrong direction"; optimizer-step bugs that drop convergence rate | Multi-seed sensitivity (single seed=42 only); cross-backend (tape only) |

> Wall-clock numbers above are rough hardware-aware guidance, not
> guarantees. Idris-2 elaboration cost dominates and is non-linear in
> module count + implicit-arg width; on a slower box or under
> `nice -n 19`, multiply by 2-3×. `time make <target>` on your box is
> the only honest answer.

## Two thresholds, two files

- **`test-examples.expect`** — safety-net thresholds. Used by
  `test-e2e-examples`. Examples: `loss < 5.0`, `avg_return >= -2500`,
  `acc >= 0.3`. Loose enough to tolerate normal noise at smoke epoch
  counts; tight enough to catch NaN, full divergence, or a regression
  where the metric somehow becomes 0.
- **`test-examples-convergence.expect`** — convergence-level
  thresholds. Used by `test-convergence`. Examples: `loss < 0.5`,
  `acc_short >= 0.9`, `avg_return >= 150`. These are the values a
  healthy run at full default epochs actually achieves.

`scripts/check-result.sh <target> <result_line> [<expect_file>]`
parses both files. Default expect file is `test-examples.expect`.

## What we explicitly don't test automatically

- **MLX or torch convergence** — `test-convergence` is tape-only.
  Backend correctness is covered by `test-unit-c-{mlx,torch}`
  and the smoke gate; we don't try to verify each example converges
  identically on each backend. (Different backends have different
  numerical precision and op timing, so single-seed convergence
  comparisons are noisy anyway.)
- **Multi-seed for any example** — single seed=42 is the
  convergence baseline. The CLAUDE.md ≥5-seed policy is for
  *making convergence claims in PR descriptions / docs*, not as a
  CI gate. If a multi-seed regression shows up in the wild, run
  the per-example target manually with several seeds.
- **PyTorch parity per example** — `bench-deep` Axis C
  exists for the representative panel; we don't enforce parity on
  every example automatically. Drift between Idris and PyTorch is
  captured in `docs/develop/reference-alignment.md`.
- **GPU / CUDA convergence** — `scripts/test_cuda_colab.sh` exists
  for opportunistic CUDA testing on Colab; not part of `make` flow.

## Expected-failure matrix (example × backend lane)

`test-e2e-examples` walks five lanes — `tape`, `mlx` (CPU stream),
`mlx-gpu` (Metal), `torch` (CPU), `torch-mps` (Metal, F32) —
building each backend and skipping the lane if its build fails.
**The default expectation is that every (example × lane) cell
passes.** This matrix records the *known exceptions*, classified
so a genuine regression isn't waved off as a known issue. The rule
(per CLAUDE.md "all backends first-class"): a real bug is **fixed**,
not added here; only environment limitations and not-yet-reproducible
flakes live in this table.

| Example | Lane | Status | Classification / cause |
|---|---|---|---|
| `transformer` | `mlx` (CPU) | ⚠ intermittent SIGTRAP | **Flake — P-investigate, do not mask.** Rare crash in the epoch-*transition* mlx lifecycle (per-epoch generation free / drain / sweep). Epoch-0 output is bit-identical to passing runs, so it clears epoch 0 and trips only sometimes at the boundary. Not reproducible on demand (0/38 local: seeds 99/1/2/7/42/123/5/13/21 × 5–8 epochs). Suspected mlx-on-paravirtualized-VM allocator/lifecycle instability — the same class the generation-free machinery (`docs/develop/tensor-lifecycle.md`) exists to mitigate. Observed once in a full `test-e2e-examples` run (2026-05-22). Fix when a deterministic repro appears. |

Notes on lane-specific failure *classes* (so a new failure can be
triaged fast):

- **F32-only lanes (`torch-mps`, `mlx-gpu`)** surface dtype-honesty
  bugs that the F64 lanes hide: any tensor created as F64 and fed
  to F32 params crashes with a "double vs float" / dtype-mismatch
  on these lanes only. torch-mps is strict (aborts); mlx-gpu often
  silently up/down-casts and passes — so torch-mps is the canary.
  (Example: the `mnist_get_image` F64-image regression, fixed
  2026-05-22 — see CHANGELOG.)
- **`mlx` (CPU) on long workloads** has documented Metal-allocator
  ceilings under the paravirt VM; mitigated by the
  generation-scoped free but the source of intermittent lifecycle
  crashes like the `transformer` row above.
- **Impossible-by-design cells are not failures**: F64 on Metal
  (`torch-mps` / `mlx-gpu`) is `Compatible`-gated, so those builds
  pin F32 and never attempt F64 — there is no cell to fail.

## When to run what

| Situation | Command |
|---|---|
| Before opening a PR | `make check && make test` |
| PR touched a type-level guarantee | also `make test-integration` |
| PR touched an example or training-loop module | also `make test-e2e` |
| PR touched any example you want compile-tested | also `make check-examples` (long) |
| Before merging a PR that touches a backend | `make test-unit-c-{tape,mlx,torch}` for the backends you touched |
| When you suspect a math/primitive regression | `make test` |
| Iterating on one example | `make example-<name>` |
| Pre-release validation | `make test-convergence` + `make bench-full` (both hours) |
| Comparing Idris to PyTorch | `make bench-deep` (then read `BENCHMARKS.md`) |
| Investigating a flaky training run | re-run the example with several `--seed` values manually |

## Threshold philosophy in one paragraph

`test-e2e-examples` thresholds are calibrated to NOT FAIL on a
healthy run, even at low epoch counts and across 3 seeds of
backend variance. They exist to catch *regressions that produce
nonsense* (NaN, RESULT line missing, return = -infinity).
`test-convergence` thresholds are calibrated to PASS at full
default epochs on a healthy run with margin, but FAIL if the loss
curve has plateaued earlier than it should. The two layers cover
different failure classes.
