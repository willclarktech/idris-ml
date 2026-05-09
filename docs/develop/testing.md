# Testing layers

This is the single reference for what each test target does, what it catches, and when to run it.

## TL;DR — the three commands you'll use

| Command | What it does | Wall time | When |
|---|---|---|---|
| `make test-examples` | Smoke gate: every example × 5 lanes (tape/mlx/mlx-gpu/torch/torch-mps, where the backend builds), 3-10 epochs each, safety-net thresholds. Catches crashes / NaN / divergence / missing RESULT keys. See the expected-failure matrix below. | ~13 min+ | per-PR |
| `make test-examples-convergence` | Convergence: every example at full default epochs, single seed=42, tape only, tight thresholds. Catches "model trains in the wrong direction" and similar correctness regressions. | hours | release validation |
| `make example-<name>` | Run one example with full default config. The standard dev-iteration command. | varies | dev iteration |

To run a *subset* of examples to convergence: invoke the per-example targets you care about (`make example-supervised && make example-rnn && …`). There's intentionally no `convergence-supervised` / `convergence-memory` / sub-targets — keep the command surface small.

## Testing pyramid

From cheapest/fastest/narrowest at the bottom to broadest/slowest at the top:

```
                      ┌───────────────────────────┐
                      │ test-examples-convergence │  hours · single seed · tape
                      │  (correctness regressions)│
                      ├───────────────────────────┤
                      │      test-examples        │  ~13 min · 5 lanes · smoke
                      │     (crash-only gate)     │
                      ├───────────────────────────┤
                      │    test-ref / bench-*     │  PyTorch parity / perf
                      ├───────────────────────────┤
                      │  test-backend-{tape,mlx,  │  C-level FFI op correctness
                      │   torch} + safetensors +  │
                      │   ntm-grad + ntm-timestep │
                      ├───────────────────────────┤
                      │  test + test-gym +        │  Idris pure-logic unit tests
                      │  test-examples-unit       │
                      ├───────────────────────────┤
                      │         check-all         │  type-checks every package
                      └───────────────────────────┘
```

`test-all` is an aggregate: type-check → unit → C-FFI → smoke + reference + jupyter, in that order. It does *not* run the convergence target — that's separate.

## Per-target reference

| Target | Wall time | What it catches | What it doesn't |
|---|---|---|---|
| `check-all` | ~30s | Type errors, missing imports, syntax | Runtime bugs |
| `test` | ~1 min | Idris-side math/tensor/layer/NTM/RL primitive correctness | Anything backend-specific |
| `test-gym` | ~30s | Gym envs (step/reset/space semantics) | Training |
| `test-examples-unit` | ~10s | Synthetic data-generator correctness | Training |
| `test-backend-{tape,mlx,torch}` | ~30s each | C-level tensor ops + autograd correctness | Idris-side; only the unique paths per backend |
| `test-safetensors` | ~5s | Save/load round-trip | Cross-backend portability beyond what test-examples-transfer-demo checks |
| `test-ntm-grad` | ~5s | NTM gradient NaN regression | NTM convergence |
| `test-ntm-timestep` | ~5s | NTM single-timestep integration | Multi-timestep state |
| `test-examples` | ~13 min | Crashes, NaN, divergence, missing RESULT keys, multi-epoch state-lifecycle bugs (≥3 epochs) | Whether models actually learn |
| `test-examples-convergence` | hours | "Model trains in the wrong direction", optimizer-step bugs that drop convergence rate | Multi-seed sensitivity (single seed=42 only); cross-backend (tape only) |
| `test-ref` | ~1 min | PyTorch reference correctness | Idris correctness; only runs if `uv` and the pytorch venv are set up |
| `bench-compare` | minutes | End-to-end Idris vs PyTorch perf parity | Correctness |
| `bench-ops-compare` | minutes | Op-level perf parity | Correctness |
| `test-jupyter` / `test-notebooks` | minutes | Jupyter kernel + notebook execution | Training correctness in notebooks |
| `test-all` | ~30 min | All of the above except `test-examples-convergence` | Convergence regressions |

## Two thresholds, two files

- **`test-examples.expect`** — safety-net thresholds. Used by `test-examples`. Examples: `loss < 5.0`, `avg_return >= -2500`, `acc >= 0.3`. Loose enough to tolerate normal noise at smoke epoch counts; tight enough to catch NaN, full divergence, or a regression where the metric somehow becomes 0.
- **`test-examples-convergence.expect`** — convergence-level thresholds. Used by `test-examples-convergence`. Examples: `loss < 0.5`, `acc_short >= 0.9`, `avg_return >= 150`. These are the values a healthy run at full default epochs actually achieves.

`scripts/check-result.sh <target> <result_line> [<expect_file>]` parses both files. Default expect file is `test-examples.expect`.

## What we explicitly don't test automatically

- **MLX or torch convergence** — `test-examples-convergence` is tape-only. Backend correctness is covered by `test-backend-{mlx,torch}` and the smoke gate; we don't try to verify each example converges identically on each backend. (Different backends have different numerical precision and op timing, so single-seed convergence comparisons are noisy anyway.)
- **Multi-seed for any example** — single seed=42 is the convergence baseline. The CLAUDE.md ≥5-seed policy is for *making convergence claims in PR descriptions / docs*, not as a CI gate. If a multi-seed regression shows up in the wild, run the per-example target manually with several seeds.
- **PyTorch parity per example** — `bench-compare` and `ref-convergence-{copy,recall}` exist for specific examples; we don't enforce parity on every example automatically. Drift between Idris and PyTorch is captured in `docs/develop/reference-alignment.md`.
- **GPU / CUDA convergence** — `scripts/test_cuda_colab.sh` exists for opportunistic CUDA testing on Colab; not part of `make` flow.

## Expected-failure matrix (example × backend lane)

`test-examples` walks five lanes — `tape`, `mlx` (CPU stream), `mlx-gpu`
(Metal), `torch` (CPU), `torch-mps` (Metal, F32) — building each backend
and skipping the lane if its build fails. **The default expectation is
that every (example × lane) cell passes.** This matrix records the *known
exceptions*, classified so a genuine regression isn't waved off as a known
issue. The rule (per CLAUDE.md "all backends first-class"): a real bug is
**fixed**, not added here; only environment limitations and
not-yet-reproducible flakes live in this table.

| Example | Lane | Status | Classification / cause |
|---|---|---|---|
| `transformer` | `mlx` (CPU) | ⚠ intermittent SIGTRAP | **Flake — P-investigate, do not mask.** Rare crash in the epoch-*transition* mlx lifecycle (per-epoch generation free / drain / sweep). Epoch-0 output is bit-identical to passing runs, so it clears epoch 0 and trips only sometimes at the boundary. Not reproducible on demand (0/38 local: seeds 99/1/2/7/42/123/5/13/21 × 5–8 epochs). Suspected mlx-on-paravirtualized-VM allocator/lifecycle instability — the same class the generation-free machinery (`docs/develop/tensor-lifecycle.md`) exists to mitigate. Observed once in a full `test-examples` run (2026-05-22). Fix when a deterministic repro appears. |

Notes on lane-specific failure *classes* (so a new failure can be triaged
fast):

- **F32-only lanes (`torch-mps`, `mlx-gpu`)** surface dtype-honesty bugs
  that the F64 lanes hide: any tensor created as F64 and fed to F32 params
  crashes with a "double vs float" / dtype-mismatch on these lanes only.
  torch-mps is strict (aborts); mlx-gpu often silently up/down-casts and
  passes — so torch-mps is the canary. (Example: the `mnist_get_image`
  F64-image regression, fixed 2026-05-22 — see CHANGELOG.)
- **`mlx` (CPU) on long workloads** has documented Metal-allocator ceilings
  under the paravirt VM; mitigated by the generation-scoped free but the
  source of intermittent lifecycle crashes like the `transformer` row above.
- **Impossible-by-design cells are not failures**: F64 on Metal
  (`torch-mps` / `mlx-gpu`) is `Compatible`-gated, so those builds pin F32
  and never attempt F64 — there is no cell to fail.

## When to run what

| Situation | Command |
|---|---|
| Before opening a PR | `make test-examples` |
| Before merging a PR that touches a backend | `make test-backend-{tape,mlx,torch}` for the backends you touched |
| When you suspect a math/primitive regression | `make test test-gym` |
| Iterating on one example | `make example-<name>` |
| Pre-release validation | `make test-examples-convergence` (hours) |
| Comparing Idris to PyTorch | `make bench-compare` (or `ref-convergence-*`) |
| Investigating a flaky training run | re-run the example with several `--seed` values manually |

## Threshold philosophy in one paragraph

`test-examples` thresholds are calibrated to NOT FAIL on a healthy run, even at low epoch counts and across 3 seeds of backend variance. They exist to catch *regressions that produce nonsense* (NaN, RESULT line missing, return = -infinity). `test-examples-convergence` thresholds are calibrated to PASS at full default epochs on a healthy run with margin, but FAIL if the loss curve has plateaued earlier than it should. The two layers cover different failure classes.
