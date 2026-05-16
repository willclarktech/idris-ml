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

> **Latest full sweep 2026-05-25 @ `54c8dba`** — post-Phase-6 closeout
> + 4 `bug | S` fixes + mlx backward per-op split + transformer F64
> leak fix. Six examples × five cells (tape / torch-cpu / torch-mps /
> mlx-cpu / mlx-gpu). Run on an idle Tart VM (the morning sweep at
> `6578b81` was contaminated by concurrent editing + a perf-sweep
> harness `|| true` that ran stale binaries). Methodology: in-script
> `PERF_MS_PER_EP` marker. See `perf-changes.md` 2026-05-25 entry for
> the delta-vs-`461ad12` table and the per-cell crash discussion.
>
> | Example     | tape ms | torch-cpu | torch-mps | mlx-cpu | mlx-gpu | pytorch ms | tape | torch-cpu | torch-mps | mlx-cpu | mlx-gpu |
> |-------------|--------:|----------:|----------:|--------:|--------:|-----------:|-----:|----------:|----------:|--------:|--------:|
> | rnn         |  0.38   |  1.78     |  1.72     |  74.48  | 111.79  |  1.66      | 0.23×| 1.07×     | 1.04×     | 44.87×  | 67.34×  |
> | lstm        |  0.43   |  2.77     |  2.77     | 125.21  | 177.75  |  4.02      | 0.11×| 0.69×     | 0.69×     | 31.15×  | 44.22×  |
> | gru         |  0.35   |  3.28     |  3.28     |  89.72  | 147.07  |  4.11      | 0.09×| 0.80×     | 0.80×     | 21.83×  | 35.78×  |
> | transformer |  1.15   |  7.52     |  6.74     |  39.44  | **crashed** | 29.83  | 0.04×| 0.25×     | 0.23×     | 1.32×   | N/A     |
> | ntm-copy    |  3.67   | 15.31     | 13.78     | 231.16  | 325.10  | 13.14      | 0.28×| 1.17×     | 1.05×     | 17.59×  | 24.74×  |
> | ntm-recall  |  4.46   | 17.81     | 15.96     | 259.60  | 384.40  | 14.42      | 0.31×| 1.24×     | 1.11×     | 18.00×  | 26.66×  |
>
> **Tape**: dominates every cell — 0.04-0.31× PyTorch. **Torch**:
> competitive on every cell; faster on transformer (4× faster) and
> rnn/lstm/gru (matched). **mlx-cpu**: 17-44× PyTorch on small-net
> training (per-FFI overhead + per-op Metal Performance Shaders
> dispatch cost dominating sub-ms kernels). **mlx-gpu**: 25-67×
> PyTorch — kernel-launch wall persists, and transformer crashes at
> 200 epochs with `Exception: invalid memory reference` (filed as a
> TODO row; separate from the F64-on-Metal crash fixed in `f7354bd`).
> mlx remains the right backend for the matmul-bench compute-bound
> regime (4.3 TFLOPS at N=4096); these training-loop cells live in
> the opposite regime where its overhead dominates.

Latest cross-backend sweep: 2026-05-09 @ commit `6f7792a` (post DNC
mask + retention + linkTrans fixes, torch free_intermediates
simplification). Two-point timing via `scripts/perf-baseline.sh <key>
<backend>` at `--seed 42`.

> **New example added 2026-05-15** — `Example/MatmulBench` (pure forward
> matmul at N=2048/4096) is the canonical "mlx GPU > CPU" demo for
> idris-ml. Measured on M-series in a Tart VM via the typed `Tensor`
> API:
>
> | N | mlx CPU ms/call | mlx GPU ms/call | GPU speedup | CPU GFLOPS | GPU GFLOPS |
> |---|---:|---:|---:|---:|---:|
> | 2048 | 13.76 | 7.81 | **1.76×** | 1248 | 2197 |
> | 4096 | 120.96 | 33.97 | **3.56×** | 1136 | 4045 |
>
> Above N≈1024, mlx GPU dominates; below, CPU wins on per-op kernel-launch
> wall. See `perf-changes.md` 2026-05-15 "New `Example/MatmulBench`" entry.
> `Example/GptLarge` retired same day; was never structurally able to
> show GPU > CPU at dModel=256.

> **Full sweep 2026-05-17 @ commit `b894fbb`** (post IO refactor +
> per-sequence withNoGrad, see `perf-changes.md` 2026-05-17 entry).
> Six examples × four cells (tape / torch / mlx-cpu / mlx-gpu) via
> `scripts/perf-sweep.sh`. mlx-gpu added as a first-class cell.
>
> | Example | tape ms | torch ms | mlx-cpu ms | mlx-gpu ms | pytorch ms | tape | torch | mlx-cpu | mlx-gpu |
> |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
> | rnn         |  0.34 |  1.36 |  76.0 | 123.3 |  1.75 | 0.19× | 0.78× | 43× | 70× |
> | lstm        |  0.29 |  3.48 | 140.6 | 183.1 |  3.81 | 0.08× | 0.91× | 37× | 48× |
> | gru         |    ~0 |  3.97 |  95.2 | 157.6 |  3.78 | noise | 1.05× | 25× | 42× |
> | transformer |  1.08 |  8.28 |  40.6 |  74.9 | 29.39 | 0.04× | 0.28× | 1.4× | 2.6× |
> | ntm-copy    |    ~0 | 25.10 | 281.0 | 335.9 | 12.30 | noise | 2.04× | 23× | 27× |
> | ntm-recall  |  3.13 | 23.53 | 285.5 | 360.9 | 13.13 | 0.24× | 1.79× | 22× | 27× |
>
> **Tape**: beats PyTorch on every measurable cell. **Torch**:
> competitive on small ops, 4× faster on transformer. **mlx-cpu**:
> 22-43× PyTorch on small-net training — the IO refactor's per-FFI
> overhead compounding on mlx's already-high per-op cost. **mlx-gpu**:
> 1.4-1.7× slower than mlx-cpu in this regime — kernel-launch wall.
> Trade-off accepted: 5× small-op mlx training regression buys
> correctness (eval truly skips autograd, no_grad bracket actually
> brackets) on examples that previously crashed.
>
> Matmul-bench (the compute-bound regime mlx exists for) is intact:
>
> | N | tape GFLOPS | torch GFLOPS | mlx-cpu GFLOPS | mlx-gpu GFLOPS |
> |---:|---:|---:|---:|---:|
> | 1024 | 305 |  365 | 1054 |  682 |
> | 2048 | 339 |  329 | 1319 | **2993** |
> | 4096 | 317 |  334 | 1215 | **4290** |
>
> mlx-gpu hits **4.3 TFLOPS at N=4096, 13.5× the CPU backends**. The
> per-FFI wrapping is invisible at this scale.

> **Partial re-measurement 2026-05-15 @ commit `db20f12`** (post
> Transformer PE-caching fix + `prim__tile2d` rewrite, see
> `perf-changes.md`). The transformer row is now stale across all
> backends. Fresh numbers from `scripts/perf-baseline.sh`:
>
> | Example | tape ms | mlx ms | torch ms | pytorch ms | tape ratio | mlx ratio | torch ratio |
> |---|---:|---:|---:|---:|---:|---:|---:|
> | transformer (post-fix, tile2d) | **6.4** | 37.09 | **9.95** | ~20 | **0.31× (3.2× faster than ref)** | 1.89× | **0.5× (2× faster than ref)** |
>
> tape and torch are now **faster than the PyTorch reference**. mlx
> hovers around 1.89-1.95× — investigated and traced to fundamental
> overhead of carrying the cached PE tensor on `TransformerState`,
> NOT to the reshape ops it initially seemed (replacing
> `reshape3d → add → reshape2d` with `tile2d → add` left the ratio
> unchanged). The 12-20% mlx regression on this small-model config
> (dModel=16, blocks=2) is the cost of the architectural fix that
> gave a **22× wall reduction on `gpt-large`** (dModel=256) — net
> across the project this trades a small absolute slowdown on the
> tiny demo for a massive speedup on a real model. The
> pre-2026-05-14 transformer row below is retained for historical
> context but is stale.

| Example | tape ms | mlx ms | torch ms | pytorch ms | tape ratio | mlx ratio | torch ratio | conv epochs | tape conv | budget | bucket |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| supervised | noisy† | noisy† | ~1 | ~0.2 | A | A | A | 1000 | <1 min | 30 min | A |
| rnn | 5.11 | 11.60 | 7.32 | 1.66 | 3.08× | 6.59× | 4.38× | 2000 | <1 min | 30 min | C |
| lstm | 5.49 | 16.24 | 8.68 | 4.10 | 1.32× | 4.22× | 2.12× | 2000 | <1 min | 30 min | B/C |
| gru | 4.81 | 16.67 | 8.16 | 4.10 | 1.19× | 4.05× | 1.96× | 2000 | <1 min | 30 min | A/B |
| transformer | 25.64 | 33.21 | 31.78 | 19.54 | 1.26× | 1.73× | 1.63× | 1000 | <1 min | 30 min | B |
| seq-classify | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | 30 min | unmeasured |
| mnist | ~120000 (5 full passes ~10 min) | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | 5 epochs | ~10 min | 30 min | A |
| gpt (embedded) | ~1000 (per epoch ≈1 s) | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | unmeasured | 30 epochs | ~30 s | 30 min | A |
| ntm-copy | 10.83 | 38.10 | 27.37 | 10.73 | **1.01×** | 3.53× | 2.57× | 5K (s99) | <2 min | 30 min | A/B |
| ntm-recall | 15.63 | 8.70 | 26.60 | 12.27 | 1.35× | **0.71×** | 2.07× | 50000 | unmeasured | 30 min | A/B |
| dnc-copy | 8.22 | 17.98 | 13.42 | 7.18 | **1.14×** | 2.50× | 1.91× | 50000 (max) | see below | 30 min | A/B |
| dnc-recall | 18.30 | 53.80 | 30.07 | 14.20 | 1.27× | 3.96× | 2.12× | ≥50000 | (rerun pending) | 30 min | B/C |

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
Defaults are now per-backend after the 2026-05-10 broadcast
adoption in `Layer/Ntm.idr`: `Makefile` picks `--seed 42` for
tape/torch and `--seed 99` for mlx, `epochs=10000`, ES gate
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
and Idris). The Makefile picks the seed that converges with
broadcast per backend: seed=42 for tape (~4400 ep) and torch
(~5300 ep); seed=99 for mlx (~4400 ep, matches pre-broadcast
4400 ep / 99.97% baseline).

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
| reinforce | 136.69 | 264.51 | 167.41 | 53.31 | 2.56× | 5.02× | 3.08× | unmeasured | unmeasured | 30 min | C† |
| dqn | 283.47 | 374.33 | 318.25 | 11.37 | 24.93× | 32.75× | 26.37× | 300 | unmeasured | 30 min | D† |
| q-learning | task-bound | task-bound | task-bound | task-bound | — | — | — | unmeasured | unmeasured | 30 min | task-bound |
| sarsa | task-bound | task-bound | task-bound | task-bound | — | — | — | unmeasured | unmeasured | 30 min | task-bound |
| monte-carlo | task-bound | task-bound | task-bound | task-bound | — | — | — | unmeasured | unmeasured | 30 min | task-bound |
| frozen-lake | task-bound | task-bound | task-bound | task-bound | — | — | — | unmeasured | unmeasured | 30 min | task-bound |
| taxi | task-bound | task-bound | task-bound | task-bound | — | — | — | unmeasured | unmeasured | 30 min | task-bound |
| mountain-car | 1850.73 | 2703.57 | 2079.03 | 59.48 | 31.12× | 36.39× | 35.47× | 500 | ~17 min (5/5) | 30 min | D† (env-bound) |
| mountain-car-cont | noisy | noisy | noisy | noisy | A | A | A | 30000 | ~11 min | 30 min | A |
| a2c | 9.93 | 18.27 | 11.46 | 1.73 | **5.74×** | 15.22× | 10.81× | 5000 | ~2.5 min | 30 min | C† |
| ppo | 3582.37 | 5314.90 | 4155.47 | 151.63 | 23.63× | 32.28× | 28.91× | 100 | ~10 min | 30 min | D† (env-bound) |
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


## 2026-05-19: 5-cell Apple-Silicon sweep (post device-taxonomy refactor)

First full sweep after Phases 1-9 land the parametric TorchDev,
TORCH_DEVICE=mps build cell, and UserDeviceTransfer machinery. Cells:
`tape,torch-cpu,torch-mps,mlx-cpu,mlx-gpu`. Examples: the
perf-sweep default set (rnn, lstm, gru, transformer, ntm-copy,
ntm-recall) plus reinforce as a one-off. CUDA cell intentionally
skipped — no CUDA hardware on the Apple-Silicon CI lane.

Numbers are `idris_ms_per_epoch / pytorch_ref_ms_per_epoch`; lower
is better (Idris-ml beating PyTorch CPU baseline). PyTorch ref runs
once per example regardless of cell (one shared reference).

| example     | tape  | torch-cpu | torch-mps | mlx-cpu | mlx-gpu |
|-------------|-------|-----------|-----------|---------|---------|
| reinforce   | 0.18  | 0.68      | 0.16      | (skipped) | (skipped) |
| rnn         | 0.20  | 1.01      | 1.07      | 45.42   | 64.94   |
| lstm        | 0.10  | 0.78      | 0.79      | 37.63   | 59.72   |
| gru         | 0.06  | 0.64      | 0.54      | 17.13   | 25.57   |
| transformer | 0.04  | 0.30      | 0.28      | 1.51    | 3.47    |
| ntm-copy    | 0.31  | 1.41      | **crash** | 34.52   | 34.07   |
| ntm-recall  | 0.27  | 1.55      | **crash** | 18.34   | 27.94   |

Raw measurements in `perf-log.jsonl` (filter `methodology ==
"in_script_marker"` AND `date == "2026-05-19"`).

### Key observations

1. **tape dominates on every measured workload** — confirms tape's
   "fastest on small-and-medium" character. Even after the device-
   taxonomy refactor, every cell on every example sees tape ≤ 0.31×
   PyTorch CPU. The "Idris-ml is faster than PyTorch on these
   workloads" claim survives the refactor.
2. **torch-cpu and torch-mps tie at this scale** — across rnn /
   lstm / gru / transformer, torch-mps comes in within 5-15% of
   torch-cpu (sometimes faster, sometimes slower). Kernel-launch
   overhead dominates these small tensors; MPS's compute wins don't
   materialise until something matmul-heavy at larger sizes lands
   (LLM-class workload remains the canonical such case).
3. **mlx is consistently 10-65× slower at this scale** — mirrors
   the existing finding documented in
   `project_mlx_gpu_environment.md`: mlx's kernel-launch wall
   crushes the tiny workloads idris-ml's examples exercise.
   transformer is the outlier where mlx is "only" 1.5×/3.5× —
   probably because its larger model + batched attention amortise
   the launch cost.
4. **reinforce on torch-mps comes in at 0.16×, beating tape's 0.18×**
   — the 128-hidden policy network is *just* large enough for MPS
   to win on the matmul. First case in the codebase where torch-mps
   visibly beats tape on a real example.
5. **NTM-copy and NTM-recall crash on torch-mps with abort trap 6**
   — exit code 134, no Python-side error message; the SIGABRT fires
   below the Idris-Chez layer (libtorch internals). Likely related
   to NTM's cosine-similarity / softmax loops hitting an MPS kernel
   coverage gap. Filed as TODO row "Investigate NTM crash on
   torch-mps". The torch-cpu lane on these examples runs fine.

### Why torch-cpu is sometimes slower than tape

torch-cpu lands at 1.0-1.55× on rnn / lstm / ntm — slower than PyTorch
CPU. That's expected: torch-cpu goes through libtorch's autograd
graph + per-op kernel dispatch, picking up Python-equivalent overhead
per op. Tape's hand-rolled tape allocator + per-op fused kernels avoid
the dispatch tax. The "free" Idris-ml win on small recurrent loops
comes from the tape backend, not from libtorch.

### Why MPS doesn't help here

These workloads are small enough that MPS kernel launch (~50-100µs
per op) dominates the actual matmul time (~5-10µs for a 128×4 matmul
at F32). The MPS win comes when:
- Matmul size grows past the kernel-launch crossover (~1024×1024 N
  at F32 per `MatmulBench.idr` data).
- Batch dim grows (amortises launch cost across more arithmetic).
- The model fits in MPS unified memory and avoids host transfers.

idris-ml's example bank is calibrated for fast convergence + tape-
compatible scales, so MPS's headroom isn't visible yet. The LLM-class
example (parked TODO) is the canonical workload to exercise it.
4. Phase 5: re-validate ≤ 1.10× across all (example, backend) cells
   that aren't `task-bound` / `no-ref`.
