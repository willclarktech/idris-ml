# Perf baseline — all examples × all backends vs PyTorch ref

> **Goal**: every (example, backend) cell ≤ 1.10× PyTorch ref ms/epoch
> AND tape full-convergence ≤ 30 min/example. See plan in
> `docs/develop/reference-alignment.md` for the alignment policy that
> bounds problem-shrink decisions.

## How this table is filled in

Run `scripts/perf-baseline.sh <example-key> <backend>`. The script does
two-point timing (`(T_long − T_short) / (N_long − N_short)`) to
subtract idris2 build / startup overhead. One CSV row per call. See
the script header for the per-example budgets.

Each cell's *idris ms* and *pytorch ms* are wall-clock per epoch at the
*currently aligned* configuration recorded in
`docs/develop/reference-alignment.md`. If a config later changes, the
matching cells need to be re-measured paired-side.

**Marker conventions**:
- `unmeasured` — script not yet run for this cell. Default state.
- `noisy` — example runs sub-millisecond per epoch; two-point timing
  resolution insufficient. Both sides are firmly in Bucket A.
- `task-bound` — wall-clock dominated by the env-step / data-gen
  side, not the backend forward/backward. Ratio is not a meaningful
  perf signal; gated only on convergence-time budget.
- `no-ref` — example has no 1:1 PyTorch script (e.g. SAC ↔ MountainCarCont
  shares a model but the Idris example uses a different env). Ratio
  N/A; gate on convergence time alone.

## Convergence-expected examples (12)

These have target accuracy thresholds in
`test-examples-convergence.expect`; full convergence runtime matters.

| Example | tape ms | mlx ms | torch ms | pytorch ms | tape ratio | mlx ratio | torch ratio | conv epochs | tape conv | budget | bucket |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| supervised | **0.27** (2026-05-07) | unmeasured | unmeasured | **0.49** (2026-05-07) | **0.55×** (faster) | unmeasured | unmeasured | 1000 | <1 min | 30 min | A |
| rnn | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | 2000 | <1 min (est.) | 30 min | A (est.) |
| lstm | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | 2000 | <1 min (est.) | 30 min | A (est.) |
| gru | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | 2000 | <1 min (est.) | 30 min | A (est.) |
| transformer | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | 1000 | unmeasured | 30 min | unmeasured |
| seq-classify | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | 30 min | unmeasured |
| mnist | ~120000 (5 full passes ~10 min) | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | 5 epochs | ~10 min | 30 min | A |
| gpt (embedded) | ~1000 (per epoch ≈1 s) | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | 30 epochs | ~30 s | 30 min | A |
| ntm-copy | see [NTM/DNC table below](#ntmdnc-currentstate-postphase15) | | | | | | | 50000 (max) | see below | 30 min | see below |
| ntm-recall | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | 50000 | unmeasured | 30 min | unmeasured |
| dnc-copy | see [NTM/DNC table below](#ntmdnc-currentstate-postphase15) | | | | | | | 50000 (max) | see below | 30 min | see below |
| dnc-recall | ~4360 (pre-rewrite) | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | ≥50000 | **>10 h** | 30 min | **D** |

### NTM/DNC current-state (post-Phase-1.5)

Detailed per-backend convergence after Phase 1.5 model alignment.
Full-history measurements live in [`perf-log.md`](perf-log.md).

| Example | Backend | ms/ep | Convergence (seed=42) | acc_short | acc_full | Status |
|---|---|---:|---|---:|---:|---|
| ntm-copy | torch | ~33-60 | 5,000 ep / 2:48-5:05 | 100% | 100% | ✅ matches PyTorch ref |
| ntm-copy | tape | ~19 | 35,500 ep / 11:40 | 96% | 80% | ⚠️ backward-rule drift |
| ntm-copy | mlx | ~100 | killed at 17K (no learning) | ~50% | — | ❌ not training |
| ntm-copy | pytorch ref | ~40 | 4,600 ep / 3:02 | 99.6% | 100% | (oracle) |
| dnc-copy | torch | ~133 | 3,500 ep / 7:46 | 100% | 84% | ✅ alignment correct (RNG variance — see seed=1 below) |
| dnc-copy | tape | (untested post-fix) | known-broken `(n,1)·(1,n)→(n,n)` broadcast in link-matrix update | — | — | ❌ broadcast-bug |
| dnc-copy | mlx | unmeasured | — | — | — | unmeasured |
| dnc-copy | pytorch ref | ~9 | 3,400 ep / 0:31 | 100% | 99% | (oracle) |

**dnc-copy seed-variance check** (Phase 1.5c follow-up):

| seed | Idris-on-torch acc_full | PyTorch ref acc_full |
|---:|---:|---:|
| 42 | 84% | 99% |
| 1 | 96% | 64% |

Wide seed-variance on length-generalization (35-pt PyTorch spread,
12-pt Idris spread). Multi-seed mean is comparable. Confirmed not
an algorithmic gap. See [`perf-log.md`](perf-log.md) for raw entries.

## RL examples (13)

These are largely env-step-bound or short-convergence. Ratio is less
meaningful (env step time on the PyTorch side is part of the loop).
Gate primarily on convergence time.

| Example | tape ms | mlx ms | torch ms | pytorch ms | conv epochs | tape conv | budget | bucket |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| reinforce | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | 30 min | unmeasured |
| dqn | unmeasured | unmeasured | unmeasured | unmeasured | 300 | unmeasured | 30 min | unmeasured |
| q-learning | task-bound | task-bound | task-bound | task-bound | unmeasured | unmeasured | 30 min | unmeasured |
| sarsa | task-bound | task-bound | task-bound | task-bound | unmeasured | unmeasured | 30 min | unmeasured |
| monte-carlo | task-bound | task-bound | task-bound | task-bound | unmeasured | unmeasured | 30 min | unmeasured |
| frozen-lake | task-bound | task-bound | task-bound | task-bound | unmeasured | unmeasured | 30 min | unmeasured |
| taxi | task-bound | task-bound | task-bound | task-bound | unmeasured | unmeasured | 30 min | unmeasured |
| mountain-car | unmeasured | unmeasured | unmeasured | unmeasured | 500 | ~17 min (5/5) | 30 min | A (est.) |
| mountain-car-cont | unmeasured | unmeasured | unmeasured | unmeasured | 30000 | ~11 min | 30 min | A |
| a2c | 29 | unmeasured | unmeasured | unmeasured | 5000 | ~2.5 min | 30 min | A |
| ppo | ~6000 | unmeasured | unmeasured | unmeasured | 100 | ~10 min | 30 min | A (env-bound) |
| sac | no-ref | no-ref | no-ref | no-ref | 24000 | ~36 min | 30 min | **slightly over** |
| transfer | n/a (composite demo) | n/a | n/a | unmeasured | 500+500 | unmeasured | 30 min | unmeasured |

## Other (3)

| Example | Notes |
|---|---|
| bench | Internal driver only — covered by `bench-compare`. |
| profile | Internal profiler — `make example-profile` runs NTM-copy with per-op profiling enabled. |
| profile-micro | Internal microbench — see `docs/develop/ntm-dnc-perf-attribution.md`. |

## Important: total convergence runtime ≠ max-epochs × ms/epoch

The example configs set `--epochs N` as a **maximum**; early stopping
(`esThreshold`/`esWindow`/`esPatience`) terminates when loss
converges. The "tape conv" column above multiplies max-epochs ×
ms/epoch which **overestimates** wall-clock for any example that
early-stops. Real convergence numbers come from running to
early-stop with `--seed 42` and recording total wall-clock + epochs
returned in the RESULT line.

Documented historical convergence runs:

| Example | Config | Epochs to converge | Reference |
|---|---|---:|---|
| ntm-copy | seed=42 batch=1 | **9300** | `docs/develop/design-decisions.md:516` |
| dnc-copy | seed=42 batch=16 N=128 | 4100 | `docs/develop/dnc-convergence-results.md` (PyTorch ref, since reverted) |
| dnc-copy | seed=42 batch=1 N=32 | ~46000 (estimated) | Idris current config, ongoing |

So with current ms/epoch:
- ntm-copy at batch=1: 9300 × 559 ≈ **86 min** (over the 30-min budget,
  but much closer than the 7.7 h max-epochs estimate)
- ntm-copy at current batch=16 default: **unknown** (no recorded run);
  measurement in progress (`/tmp/ntm-copy-converge.log`).

The user's "NTM-copy in about half an hour" recollection lines up
with **pre-Path-C 228 ms × 9300 ≈ 35 min**. Path C doubled ms/epoch;
recovering that is what Phase 2 perf work targets.

## Phase 1 attack list (provisional)

Based on the partial baseline above, ordered by expected wall-clock
gain. Bucket assignments will firm up once full-convergence runs
land for the over-budget cells.

1. **`dnc-recall` tape** (Bucket D): >10 h convergence, well over the
   30-min budget. Two-pronged: Phase 2b (DNC layer perf) + Phase 3
   (paired-side shrink — `R` reads or `maxLen` smaller).
2. **`ntm-copy` tape** (Bucket C/D candidate): 559 ms/epoch × ~9300
   epochs = ~86 min on current code. Phase 2c (per-op profiler re-run
   post-Phase-1) targets the ms/epoch side. Phase 3 (`maxLen` 20 → 10
   or epoch shrink) compensates if we can't hit ~228 ms/epoch.
3. **`dnc-copy` tape** (Bucket C/D): 113 ms/epoch, 12.5× pytorch
   ratio. Phase 2b (zeroDiag mask, batched FCs, buildMatrixRows)
   before Phase 3.
4. **`ntm-recall` tape** (unmeasured but expected Bucket C/D by
   architectural similarity to NTM-copy). Run measurement first.
5. **`sac` tape** (slightly over 30-min budget): borderline; may be
   in budget after Phase 4 (mlx/torch tuning).
6. **mlx + torch baselines** (Phase 4): unmeasured for ~all examples.
   Run after tape-side Phase 2/3 lands so the baseline reflects the
   shared layer-side improvements.

## Source-of-truth notes

- `docs/develop/dnc-convergence-results.md`: DNC config history,
  N=32 vs N=128, ms/epoch trajectory.
- `docs/develop/dnc-perf-baseline.md`: Phase 1 (DNC tensor-tape
  rewrite, 8× speedup, 1040 → 130 ms/epoch).
- `docs/develop/ntm-dnc-perf-attribution.md`: per-op forward
  profiling, microbench results, prim-floor structural finding.
- `docs/develop/reference-alignment.md`: alignment policy +
  paired-side history.
- `docs/develop/example-coverage.md`: smoke-gate args + thresholds.

## Next actions

1. Fill in unmeasured tape cells: run
   `scripts/perf-baseline.sh <key> tape` for each row marked
   `unmeasured` in the convergence-expected table. Skip
   `task-bound` rows for now.
2. Phase 1 → Phase 2 / 3: act on the attack list above. Update the
   table after each shrink or perf change.
3. Phase 4: re-measure all cells on mlx and torch backends.
4. Phase 5: re-validate ≤ 1.10× across all (example, backend) cells
   that aren't `task-bound` / `no-ref`.
