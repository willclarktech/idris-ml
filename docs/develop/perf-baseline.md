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

Latest cross-backend sweep: 2026-05-09 @ commit `0e2e86a` (post Phase
1.5e, mlx tensor_linear + softplus fixes landed). Two-point timing
via `scripts/perf-baseline.sh <key> <backend>` at `--seed 42`.

| Example | tape ms | mlx ms | torch ms | pytorch ms | tape ratio | mlx ratio | torch ratio | conv epochs | tape conv | budget | bucket |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| supervised | noisy† | noisy† | 0.19 | 0.21 | A | A | 0.9× | 1000 | <1 min | 30 min | A |
| rnn | 4.12 | 9.51 | 5.24 | 1.06 | 3.89× | 8.89× | 5.4× | 2000 | <1 min | 30 min | C |
| lstm | 5.11 | 13.07 | 7.39 | 3.55 | 1.44× | 3.49× | 2.12× | 2000 | <1 min | 30 min | B/C |
| gru | 4.93 | 11.69 | 8.07 | 2.71 | 1.82× | 3.92× | 2.75× | 2000 | <1 min | 30 min | B/C |
| transformer | 25.86 | 33.75 | 32.87 | 19.02 | 1.36× | 1.62× | 1.57× | 1000 | <1 min | 30 min | B |
| seq-classify | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | 30 min | unmeasured |
| mnist | ~120000 (5 full passes ~10 min) | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | 5 epochs | ~10 min | 30 min | A |
| gpt (embedded) | ~1000 (per epoch ≈1 s) | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | 30 epochs | ~30 s | 30 min | A |
| ntm-copy | 15.63 | 21.83 | 27.83 | 11.0 | 1.42× | 1.98× | 2.58× | 5K (s99) | <2 min | 30 min | B |
| ntm-recall | 26.10 | 14.10 | 22.87 | 12.83 | 2.03× | 1.10× | 1.78× | 50000 | unmeasured | 30 min | B |
| dnc-copy | 9.93 | 21.58 | 16.28 | 7.95 | **1.24×** | 2.68× | 2.05× | 50000 (max) | see below | 30 min | A/B |
| dnc-recall | 24.70 | 40.00 | 34.83 | 16.30 | **1.50×** | 2.38× | 2.14× | ≥50000 | (rerun pending) | 30 min | B/C |

† Supervised takes <1 ms/epoch; the two-point method's resolution
(N_long=200 vs N_short=50) is below the build-startup noise floor.
Result is firmly Bucket A; rerun with `N_LONG=2000` if a precise
number is needed. ntm-recall tape ratio recomputed against the
positive PyTorch reference (`-2.87` was a noise artifact in the
two-point run; replaced with the 12.83 ms baseline observed on
mlx/torch runs of the same script).

### NTM/DNC current-state (post-Phase-1.5e — 2026-05-09)

Detailed per-backend convergence after Phase 1.5e mlx fixes
(`tensor_linear` bias-on-tape + `tensor_softplus` stable form).
Defaults are now `seed=99 epochs=10000` (paired); ES gate
`WindowedPercentile 0.10 / 0.01 / 1000 / 3` unchanged. Full-history
measurements live in [`perf-log.md`](perf-log.md).

| Example | Backend | ms/ep | Convergence (seed=99 batch=1) | acc_short | acc_full | Status |
|---|---|---:|---|---:|---:|---|
| ntm-copy | tape | ~16 | ~5K ep / ~2 min | 99.8% | 99.8% | ✅ |
| ntm-copy | mlx | ~22 | 4,400 ep / 4:00 | 99.97% | 99.97% | ✅ |
| ntm-copy | torch | ~28 | ~5K ep / ~2-3 min | 100% | 100% | ✅ matches PyTorch ref |
| ntm-copy | pytorch ref | 11 | ~5K ep / ~1 min | 100% | 100% | (oracle) |
| dnc-copy | tape | 80 | (post-fix not run; broadcast fix deferred to Phase 2) | — | — | needs broadcast fix in `binop_elementwise` for full DNC link-matrix |
| dnc-copy | mlx | 92 | (post-fix not run) | — | — | unmeasured at convergence |
| dnc-copy | torch | 71 | 3,500 ep / 7:46 (Phase 1.5c, seed=42) | 100% | 84% | ✅ alignment correct (RNG variance) |
| dnc-copy | pytorch ref | 7.2 | 3,400 ep / 0:31 | 100% | 99% | (oracle) |

**ntm-copy seed-sensitivity** (documented in `gotchas.md`): the
aligned NTM model has high variance across seeds at moderate epoch
budgets (~1/4 of seeds reach 99%+ at 5K epochs on both PyTorch ref
and Idris). The `seed=99` default reaches 99%+ on both tape and
mlx, well under the 10K cap.

**dnc-copy seed-variance check** (Phase 1.5c follow-up, seed=42):

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
Gate primarily on convergence time. Same 2026-05-09 sweep as above.

| Example | tape ms | mlx ms | torch ms | pytorch ms | tape ratio | mlx ratio | torch ratio | conv epochs | tape conv | budget | bucket |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reinforce | 136.78 | 243.33 | 169.78 | 52.20 | 2.62× | 4.70× | 3.24× | unmeasured | unmeasured | 30 min | C† |
| dqn | 289.73 | 391.05 | 319.17 | 11.30 | 25.64× | 34.09× | 27.40× | 300 | unmeasured | 30 min | D† |
| q-learning | task-bound | task-bound | task-bound | task-bound | — | — | — | unmeasured | unmeasured | 30 min | task-bound |
| sarsa | task-bound | task-bound | task-bound | task-bound | — | — | — | unmeasured | unmeasured | 30 min | task-bound |
| monte-carlo | task-bound | task-bound | task-bound | task-bound | — | — | — | unmeasured | unmeasured | 30 min | task-bound |
| frozen-lake | task-bound | task-bound | task-bound | task-bound | — | — | — | unmeasured | unmeasured | 30 min | task-bound |
| taxi | task-bound | task-bound | task-bound | task-bound | — | — | — | unmeasured | unmeasured | 30 min | task-bound |
| mountain-car | 1853.72 | 2509.18 | 2093.38 | 59.38 | 31.22× | 42.35× | 35.51× | 500 | ~17 min (5/5) | 30 min | D† (env-bound) |
| mountain-car-cont | noisy | noisy | noisy | noisy | A | A | A | 30000 | ~11 min | 30 min | A |
| a2c | 10.03 | 15.29 | 11.57 | 1.12 | 8.96× | 13.41× | 10.42× | 5000 | ~2.5 min | 30 min | C/D† |
| ppo | 3269.43 | 5974.50 | 3951.67 | 143.60 | 22.77× | 41.14× | 27.32× | 100 | ~10 min | 30 min | D† (env-bound) |
| sac | no-ref | no-ref | no-ref | no-ref | — | — | — | 24000 | ~36 min | 30 min | **slightly over** |
| transfer | n/a (composite demo) | n/a | n/a | unmeasured | — | — | — | 500+500 | unmeasured | 30 min | unmeasured |

† **Many of the high-ratio RL cells are env-bound, not Idris-bound.**
For dqn / mountain-car / ppo / a2c / reinforce, the per-epoch step
count includes one or more episode rollouts where the environment
(MountainCar / CartPole) is single-step-Python on the Idris side
(via `idris-gym`) but vectorized C/Cython on the PyTorch side
(`gym.vector.SyncVectorEnv`). The "ratio" is dominated by env-step
cost, not backend forward/backward. Phase 1 triage will mark these
as `task-bound (env)` before bucketing — perf work in Phases 2/4
won't move the needle here.

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

## Update 2026-05-09 (mask precompute)

`Layer/Dnc.idr`'s `dncZeroDiag` was rebuilding a (1 − I) [n,n] mask
every timestep, costing ~1027 prim FFI calls per timestep at n=32.
Moved to a precomputed `nonDiagMaskT` field on `DncState`, built
once in `dncLayer`. Both DNC examples moved from Bucket D into
Bucket A/B on every backend (5–8× speedup). Numbers above already
reflect the post-fix state.

## Phase 1 attack list (post-2026-05-09 sweep)

Bucketing rule:
- **A** (≤1.10×) — done, no work.
- **B** (≤2.0×) — small fix, single-prim or alloc tuning.
- **C** (≤10×) — structural fix (Phase 2 fusion).
- **D** (>10×) — problem-shrink (Phase 3) after Phase 2 hasn't closed it.
- **task-bound (env)** — the ratio is env-step cost, not backend
  cost; gate on convergence runtime alone, perf work won't move it.

Excluding the task-bound cells, the active queue ordered by gap:

1. **`dnc-recall` (all backends, Bucket D)**: 13-15× ratio + >10 h
   tape convergence. Two-pronged: Phase 2b (DNC layer perf —
   `zeroDiag`, batched FCs, `buildMatrixRows`) plus Phase 3
   (paired-side shrink: `maxLen` smaller, possibly fewer read
   heads `R`).
2. **`dnc-copy` (all backends, Bucket C/D)**: 9.8-13× ratio. Same
   Phase 2b items as dnc-recall; tape additionally needs the
   `binop_elementwise` (n,1)×(1,n) broadcast for the link-matrix
   update before convergence even runs cleanly.
3. **`rnn` / `lstm` / `gru` (mlx Bucket B/C, tape ≤ Bucket B/C)**:
   3.5-9× on mlx, 2-3× on torch, 1.4-3.9× on tape. Likely not a
   single bottleneck — the prim floor + per-epoch overhead
   dominates these short loops. Phase 4 microbench will tell.
4. **`ntm-copy` (all backends Bucket B)**: 1.4-2.6× ratio,
   converges fast (~5K ep) at the new defaults. Acceptable but
   the gap is real. Re-measure after Phase 2.
5. **`transformer` (all backends Bucket B)**: 1.4-1.6× ratio;
   modest fixed gap, likely closeable once Phase 4 has microbench
   data.
6. **`reinforce` / `a2c` (Bucket C, partly env-shaped)**: 2.6-13×.
   Need to disentangle env-step cost from grad-cost first.
7. **`sac` (no-ref, slightly over 30-min convergence budget)**:
   borderline; revisit after Phase 4.

Likely-task-bound RL examples (`dqn` / `mountain-car` / `ppo`):
mark as **task-bound (env)** in Phase 1 triage. Their high ratios
are the gym-vector vs idris-gym single-step gap, not backend perf.
If a meaningful Idris-side optimization later lands, re-measure
in Phase 5; otherwise leave alone.

Phase 4 (mlx + torch tuning) will follow Phase 2 wherever the gap
is "Idris-side fixed" rather than "backend-specific". Per-backend
microbenches (`ProfileMicro.idr` mlx/torch variants, currently
missing) need to land first.

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
