# Performance log

**Append-only measurement log.** Each entry records one performance
measurement with the commit hash where it was produced. **Never edit
or delete prior entries** — historical numbers are valuable as
regression evidence and to avoid re-running expensive measurements.
If a measurement is later determined to be invalid (e.g. wrong
config), add a follow-up entry that says so; don't remove the
original.

When recording a new measurement, append to the bottom of the
relevant `## <example>` section. New entries near the top of the
file (under `## Convention reminders`) are non-data — see the
section above the data for those.

---

## Convention reminders

- **Commit hash**: the short hash (`git rev-parse --short HEAD`) at
  the time the measurement was produced. If uncommitted changes
  were in the tree, append `+dirty`.
- **Date**: ISO-8601 (YYYY-MM-DD). Use the date the measurement
  finished, not the date you wrote it up.
- **Config**: the full CLI args used (or the default if implicit).
  At minimum: backend, seed, batch.
- **Result line**: copy the example's `RESULT` line verbatim if
  one exists. Plus wall-clock from the runtime.
- **Multi-seed runs**: one entry per (commit × seed × backend).
- **Caveats**: if the measurement was on a system under load, with
  unusual cache state, or any other reason the wall-clock might
  not be representative, note it in the entry.

---

## NTM-Copy

### 2026-05-08 — commit `9477d58` (pre-Phase-1.5)

config: `--seed 42 --batch 16` (pre-alignment default), backend=tape
result: 488 ms/epoch (post-Path-C regression baseline)
notes: Path-C migration broke per-epoch perf vs main's 228 ms/epoch.
       Forward regression; backward unchanged. Documented in
       `docs/develop/ntm-dnc-perf-attribution.md`.

### 2026-05-07 — commit `185bb1d` (post-batch=1, post-percentile-ES)

config: `--seed 42 --batch 1` (after batch=1 paired-side change)
backend: tape
result:  `RESULT epochs=27700 acc_short=0.9867 acc_full=0.8186 seed=42`
         17:42 wall-clock, 38 ms/epoch
notes:   First full-convergence run at the new aligned config.
         Pre-model-fix — convergence quality limited by additive-write
         and other algorithmic gaps, NOT a tape backend issue.

### 2026-05-08 — commit `dbd8ebf` (Phase 1.5b NTM model alignment)

config: `--seed 42 --batch 1`
backend: torch
result:  `RESULT epochs=5000 acc_short=1.0 acc_full=1.0 seed=42`
         5:05 wall-clock, 61 ms/epoch
notes:   First convergence run with the fully-aligned NTM model.
         Matches PyTorch ref's 4,600 epochs / 99.6% / 100% within
         9% epoch budget. Algorithmic alignment confirmed correct.

### 2026-05-08 — commit `dbd8ebf` (Phase 1.5b NTM model alignment)

config: `--seed 42 --batch 1`
backend: torch
result:  `RESULT epochs=5000 acc_short=1.0 acc_full=1.0 seed=42`
         2:48 wall-clock, 33 ms/epoch
notes:   Re-run on same commit, faster wall-clock — warm caches /
         JIT amortized from prior run. Per-epoch rate is the real
         steady-state metric.

### 2026-05-08 — commit `8b…` (Phase 1.5d tape verification)

config: `--seed 42 --batch 1`
backend: tape
result:  `RESULT epochs=35500 acc_short=0.9583 acc_full=0.8051 seed=42`
         11:40 wall-clock, 19 ms/epoch
notes:   Tape doesn't track Idris-on-torch on the aligned model.
         7× more epochs and 20-percentage-point lower acc_full.
         Forward parity at epoch 0 within 36 ULPs of torch
         (`0.7018285801505979` vs `0.7018285801506005`); divergence
         from backward-rule drift compounding over many epochs.
         Filed as backend-side follow-up.

### 2026-05-08 — commit `8b…` (Phase 1.5d mlx verification)

config: `--seed 42 --batch 1`
backend: mlx
result:  killed at epoch 17,000 / 29 min — loss stuck at ~0.69
         (random level), no learning observed
notes:   mlx fails to train the aligned NTM model entirely.
         Forward at epoch 0 differs from torch starting at digit 7
         (`0.70182865858078` vs torch `0.7018285801506005`) — float32
         internally, drifts faster. Filed as backend-side follow-up.

### Reference: PyTorch ref NTM-Copy at seed=42

config: `--seed 42 --batch 1`
result:  `RESULT epochs=4600 acc_short=0.9956 acc_full=0.9999 seed=42`
         3:02 wall-clock, 39 ms/epoch
notes:   Algorithmic oracle. All Idris-on-torch convergence comparisons
         use this as the reference point.

### 2026-05-08 — commit `6068d5c` (Phase 1.5d, tape seed=1 variance check)

example: ntm-copy
backend: tape
args:    --seed 1 --batch 1
wall:    2m 13s
converged at epoch 5700 (p10_loss=0.007001773762984365)
result:  `RESULT epochs=5700 acc_short=0.8421 acc_full=0.7091 seed=1`
notes:   Tape ES fires earlier than seed=42 (5,700 vs 35,500) but to
         WORSE accuracy (84/71 vs 96/80). Suggests p10 dipped below
         threshold on noise — premature ES, not real convergence.

### 2026-05-08 — commit `6068d5c` (PyTorch ref NTM-Copy seed=1)

example: ntm-copy
backend: pytorch_ref
args:    --seed 1 --batch 1
wall:    1m 58s
converged at epoch 6600 (p10_loss=0.003023173427209258)
result:  `RESULT epochs=6600 acc_short=1.0 acc_full=1.0 seed=1`
notes:   PyTorch ref converges to 100/100 at seed=1 too (matching
         seed=42's 100/100). For NTM, PyTorch is robust to seed.

### 2026-05-08 — commit `6068d5c` (Idris-on-torch seed=1, **killed**)

example: ntm-copy
backend: torch
args:    --seed 1 --batch 1
wall:    >10 min (killed at epoch ~17,500, no convergence)
notes:   Idris-on-torch at seed=1 doesn't converge in the same time
         budget that PyTorch needs (1:58). Loss volatile around
         0.3-0.6 at epoch 17K. Suggests Idris-on-torch may also have
         seed-sensitivity that PyTorch doesn't share — possibly RNG
         init differences (Idris C `rand()` vs torch PCG produce
         different actual init values for "seed=1") or numeric
         differences in the autograd path. Re-run with longer budget
         and a more permissive ES might surface convergence.

---

## DNC-Copy

### 2026-04-29 — commit `51e97d5` (pre-Phase-1.5)

config: `--seed 42 --batch 1 --max-len 10` (post-revert default)
backend: tape
result: 1033 ms/epoch (1733 ms/epoch under contention)
notes:  Pre-Phase-1 tensor rewrite baseline. Documented in
        `docs/develop/dnc-convergence-results.md`.

### 2026-05-02 — commit `2368bc7` (DNC Phase-1 tensor rewrite)

config: `--seed 42 --batch 1 --max-len 10`
backend: tape
result:  130 ms/epoch (8× speedup vs pre-Phase-1)
notes:   `docs/develop/dnc-perf-baseline.md` Phase-1 result.

### 2026-05-08 — commit `8b…` (Phase 1.5c DNC model alignment, seed=42)

config: `--seed 42 --batch 1` (default)
backend: torch
result:  `RESULT epochs=3500 acc_short=1.0 acc_full=0.8391 seed=42`
         7:46 wall-clock, 133 ms/epoch
notes:   Aligned-DNC convergence at seed=42. Matches PyTorch ref's
         3,400-epoch convergence within 3%. acc_full diverges from
         PyTorch's 99.4% by 15 points — **RNG seed variance, not
         algorithmic** (see seed=1 entries below for confirmation).

### 2026-05-08 — commit `8b…` (Phase 1.5c, seed=1 variance check)

config: `--seed 1 --batch 1`
backend: torch
result:  `RESULT epochs=4600 acc_short=1.0 acc_full=0.9627 seed=1`
         9:26 wall-clock, 123 ms/epoch
notes:   At seed=1 Idris hits 96.3% acc_full vs PyTorch's 64.3% —
         the seed=42 ranking inverts. Multi-seed mean ~90% (Idris)
         vs ~82% (PyTorch). Confirms seed-variance, not bug.

### Reference: PyTorch ref DNC-Copy

| seed | epochs | acc_short | acc_full | wall-clock |
|---:|---:|---:|---:|---:|
| 42 | 3,400 | 100.0% | 99.4% | 0:31 (8 ms/ep) |
| 1 | 3,800 | 100.0% | 64.3% | 0:31 (8 ms/ep) |

Wide seed-variance on length-generalization (35-point spread between
two seeds) is a known property of the PyTorch reference itself.

---

## DNC-Recall

(no Phase-1.5 measurements yet)

### 2026-05-04 (pre-Phase-1.5)

backend: tape, default config
result:  ~4360 ms/epoch (pre-DNC-tensor-rewrite estimate)
notes:   Documented in early survey. Way over the 30-min/example
         convergence budget. Awaits Phase 2 perf work.

---

## NTM-Recall

(no Phase-1.5 measurements — still using inherited V1 baseline)

---

## Supervised / RNN / LSTM / GRU / Transformer / GPT / MNIST

(no Phase-1.5 measurements — short examples below the 30-min budget,
deferred until Phase 0/1 baseline sweep.)

---

## RL examples (Reinforce / Dqn / MountainCar / etc.)

(deferred — task-bound or short-convergence; ratio is less meaningful
than wall-clock budget. See `docs/develop/perf-baseline.md` for
inherited estimates.)

### 2026-05-08 — commit `6068d5c+dirty`

example: ntm-copy
backend: tape
args:    --seed 42 --batch 1 --epochs 5
exit:    2
wall:    6.635s (6635 ms)

### 2026-05-08 — `ntm-copy` [tape] @ `6068d5c+dirty` — `--seed 42 --batch 1 --epochs 5`

exit:    0
wall:    11.010s (11010 ms)
stats:   Completed in 0s (5 epochs, 0ms/epoch)
result:  `RESULT	epochs=5	acc_short=0.4875416666666667	acc_full=0.49835609787099727	seed=42`
