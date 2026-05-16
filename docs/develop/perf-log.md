# Performance log — schema and conventions

**The canonical data store is `docs/develop/perf-log.jsonl`.** This
markdown file documents the schema and querying conventions; it
also keeps the pre-2026-05-09 markdown-format entries below as
historical record (in an `## Archive` section).

Both `scripts/perf-run.sh` (full-convergence runs) and
`scripts/perf-baseline.sh` (two-point ms/epoch ratios) append a
single JSON object on its own line to `perf-log.jsonl` per
invocation. **Append-only**; never edit or delete prior entries —
historical numbers are regression evidence and avoid re-running
expensive measurements. If a measurement is later determined to be
invalid (e.g. wrong config), append a follow-up entry that says so
rather than removing the original.

## JSONL schema

Every entry has these fields:

| field | type | notes |
|---|---|---|
| `ts` | string | ISO-8601 UTC timestamp (`2026-05-09T15:14:00Z`) |
| `date` | string | ISO date (`2026-05-09`) |
| `kind` | string | `"run"` (perf-run.sh) or `"baseline"` (perf-baseline.sh) |
| `example` | string | `ntm-copy`, `dnc-recall`, `a2c`, etc. |
| `backend` | string | `tape`, `mlx`, or `torch` |
| `device` | string | `cpu` or `gpu`. For `mlx` reflects `MLX_DEVICE`; for `tape`/`torch` always `cpu` (we don't pin tensors to MPS/CUDA at the wrapper). Added 2026-05-11 — entries before that date can be assumed `cpu`. |
| `commit` | string | abbreviated git hash (`+dirty` if uncommitted changes) |

**`kind: "run"`** entries also have:

| field | type | notes |
|---|---|---|
| `args` | string | full CLI args passed to the example |
| `exit` | int | process exit code (0 = success) |
| `wall_ms` | int | total wall-clock in ms |
| `wall_human` | string | human-readable wall (`1m 11s`, `643658 ms`) |
| `converged_at_epoch` | int? | present when the run hit early-stop |
| `diverged_at_epoch` | int? | present when training NaN'd |
| `stats` | object? | `{total_epochs, ms_per_epoch, wall}` from "Completed in …" line |
| `result` | object? | parsed `RESULT` line, e.g. `{epochs, acc_short, acc_full, seed}` |

**`kind: "baseline"`** entries also have:

| field | type | notes |
|---|---|---|
| `idris_ms_per_epoch` | float | two-point ms/epoch (Idris side) |
| `pytorch_ms_per_epoch` | float | two-point ms/epoch (PyTorch ref) |
| `ratio` | float | `idris_ms_per_epoch / pytorch_ms_per_epoch` |
| `n_long` | int | the long-side epoch count used in the two-point timing |
| `seed` | int | seed used (always 42 for the baseline script) |

## Conventions

- **Commit hash**: short hash at run time, `+dirty` if uncommitted.
- **Multi-seed runs**: one entry per (commit × seed × backend × example).
- **Caveats**: jot in the args string or as a follow-up entry; don't
  silently drop an outlier.

## Querying (jq cookbook)

```sh
# All runs of one cell
jq 'select(.example == "dnc-copy" and .backend == "tape")' \
  docs/develop/perf-log.jsonl

# Latest baseline ratio per (example, backend)
jq -s '
  map(select(.kind == "baseline"))
  | group_by([.example, .backend])
  | map(sort_by(.ts) | last)
' docs/develop/perf-log.jsonl

# Wall-clock leaderboard for a single example
jq -s '
  map(select(.kind == "run" and .example == "dnc-recall"))
  | sort_by(.wall_ms) | .[:5]
' docs/develop/perf-log.jsonl

# Convergence epochs across backends, latest run per backend
jq -s '
  map(select(.example == "ntm-copy" and (.converged_at_epoch | not | not)))
  | group_by(.backend) | map(sort_by(.ts) | last)
  | map({backend, commit, converged_at_epoch, wall_human})
' docs/develop/perf-log.jsonl
```

## Archive (pre-2026-05-09 entries)

Entries below are in the original markdown format and predate the
JSONL switchover. They remain as historical record. New entries
go to `perf-log.jsonl`; **don't add new entries here**.

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

### 2026-05-08 — INVALIDATION note for the `commit \`6068d5c+dirty\`` exit-2 entry above

The 2026-05-08 entry that says `exit: 2 / wall: 6.635s` (no result, no
completed line) is **NOT a valid measurement**. It was the first run of
`scripts/perf-run.sh` while debugging the script; make returned 2 due
to leftover dylib state from a prior interrupted run, not a real model
result. After `make clean && make install` the same args produced the
adjacent successful entry.

Per the convention, leaving the original entry in place rather than
deleting; this note marks it invalid.

### 2026-05-08 — `ntm-copy` [tape] @ `f6e628b` — `--seed 42 --batch 1 --epochs 5000 --es-threshold 0.0`

exit:    0
wall:    3m 15s (195101 ms)
stats:   Completed in 2m 56s (5000 epochs, 35ms/epoch)
result:  `RESULT	epochs=5000	acc_short=0.7544166666666663	acc_full=0.594306809660283	seed=42`

### 2026-05-08 — `ntm-copy` [tape] @ `f6e628b+dirty` — `--seed 42 --batch 1 --epochs 5000 --es-threshold 0.0`

exit:    0
wall:    2m 38s (158960 ms)
stats:   Completed in 2m 30s (5000 epochs, 30ms/epoch)
result:  `RESULT	epochs=5000	acc_short=0.757645833333333	acc_full=0.610204074820948	seed=42`

### 2026-05-08 — `ntm-copy` [tape] @ `f6e628b+dirty` — `--seed 42 --batch 1 --epochs 5000 --es-threshold 0.0`

exit:    0
wall:    1m 43s (103658 ms)
stats:   Completed in 1m 36s (5000 epochs, 19ms/epoch)
result:  `RESULT	epochs=5000	acc_short=0.9985625	acc_full=0.9916108120680491	seed=42`

### 2026-05-08 — `ntm-copy` [tape] @ `f6e628b+dirty` — `--seed 42 --batch 1 --epochs 5000 --es-threshold 0.0`

exit:    0
wall:    3m 3s (183227 ms)
stats:   Completed in 2m 44s (5000 epochs, 32ms/epoch)
result:  `RESULT	epochs=5000	acc_short=0.7520416666666668	acc_full=0.5949439477898263	seed=42`

### 2026-05-08 — `ntm-copy` [tape] @ `f6e628b+dirty` — `--seed 42 --batch 1 --epochs 5000 --es-threshold 0.0`

exit:    0
wall:    1m 49s (109156 ms)
stats:   Completed in 1m 42s (5000 epochs, 20ms/epoch)
result:  `RESULT	epochs=5000	acc_short=0.7530416666666665	acc_full=0.6126344269228362	seed=42`

### 2026-05-08 — `ntm-copy` [tape] @ `f6e628b+dirty` — `--seed 42 --batch 1 --epochs 5000 --es-threshold 0.0`

exit:    0
wall:    1m 39s (99429 ms)
stats:   Completed in 1m 34s (5000 epochs, 18ms/epoch)
result:  `RESULT	epochs=5000	acc_short=0.9985625	acc_full=0.9916108120680491	seed=42`

### 2026-05-08 — `ntm-copy` [tape] @ `f96c798` — `--seed 7 --batch 1 --epochs 5000 --es-threshold 0.0`

exit:    0
wall:    1m 44s (104458 ms)
stats:   Completed in 1m 37s (5000 epochs, 19ms/epoch)
result:  `RESULT	epochs=5000	acc_short=0.8222916666666669	acc_full=0.7384260855404914	seed=7`

### 2026-05-08 — `ntm-copy` [tape] @ `f96c798+dirty` — `--seed 99 --batch 1 --epochs 5000 --es-threshold 0.0`

exit:    0
wall:    2m 39s (159069 ms)
stats:   Completed in 2m 33s (5000 epochs, 30ms/epoch)
result:  `RESULT	epochs=5000	acc_short=0.9981874999999999	acc_full=0.997890777243912	seed=99`

### 2026-05-08 — `ntm-copy` [tape] @ `f96c798+dirty` — `--seed 123 --batch 1 --epochs 5000 --es-threshold 0.0`

exit:    0
wall:    1m 47s (107732 ms)
stats:   Completed in 1m 41s (5000 epochs, 20ms/epoch)
result:  `RESULT	epochs=5000	acc_short=0.7492708333333331	acc_full=0.6188554234024918	seed=123`

### 2026-05-08 — `ntm-copy` [mlx] @ `f96c798+dirty` — `--seed 99 --batch 1 --epochs 1000 --es-threshold 0.0`

exit:    0
wall:    59.746s (59746 ms)
stats:   Completed in 50s (1000 epochs, 50ms/epoch)
result:  `RESULT	epochs=1000	acc_short=0.4742083333333334	acc_full=0.48806552435567524	seed=99`

### 2026-05-08 — `ntm-copy` [mlx] @ `f96c798+dirty` — `--seed 99 --batch 1 --epochs 5000 --es-threshold 0.0`

exit:    0
wall:    5m 46s (346512 ms)
stats:   Completed in 5m 39s (5000 epochs, 67ms/epoch)
result:  `RESULT	epochs=5000	acc_short=0.4967916666666666	acc_full=0.49314633203913405	seed=99`

### 2026-05-08 — `ntm-copy` [mlx] @ `f96c798+dirty` — `--seed 42 --batch 1 --epochs 5000 --es-threshold 0.0`

exit:    0
wall:    6m 35s (395821 ms)
stats:   Completed in 6m 27s (5000 epochs, 77ms/epoch)
result:  `RESULT	epochs=5000	acc_short=0.8058749999999998	acc_full=0.707252760907608	seed=42`

### 2026-05-08 — `ntm-copy` [mlx] @ `7be9acf` — `--seed 42 --batch 1 --epochs 30000 --es-threshold 0.01`

exit:    0
wall:    18m 44s (1124112 ms)
converged: Converged at epoch 8200 (p10_loss=0.0071136243641376495)
stats:   Completed in 18m 34s (8200 epochs, 135ms/epoch)
result:  `RESULT	epochs=8200	acc_short=0.9935416666666665	acc_full=0.9987700892857143	seed=42`

### 2026-05-08 — `ntm-copy` [mlx] @ `7be9acf+dirty` — `--seed 99 --batch 1 --epochs 30000 --es-threshold 0.01`

exit:    0
wall:    9m 30s (570490 ms)
diverged: Diverged (NaN) at epoch 6448
stats:   Completed in 9m 23s (6448 epochs, 87ms/epoch)
result:  `RESULT	epochs=6448	acc_short=0.48483333333333334	acc_full=0.5045799697593426	seed=99`

### 2026-05-08 — `ntm-copy` [mlx] @ `7be9acf+dirty` — `--seed 7 --batch 1 --epochs 30000 --es-threshold 0.01`

exit:    0
wall:    13m 51s (831718 ms)
converged: Converged at epoch 9300 (p10_loss=0.008000670932233334)
stats:   Completed in 13m 44s (9300 epochs, 88ms/epoch)
result:  `RESULT	epochs=9300	acc_short=0.979375	acc_full=0.9939276960784313	seed=7`

### 2026-05-08 — `ntm-copy` [mlx] @ `11e38f0+dirty` — `--seed 99 --batch 1 --epochs 30000 --es-threshold 0.01`

exit:    0
wall:    4m 34s (274546 ms)
converged: Converged at epoch 4400 (p10_loss=0.005980806890875101)
stats:   Completed in 4m 28s (4400 epochs, 60ms/epoch)
result:  `RESULT	epochs=4400	acc_short=0.9997499999999999	acc_full=0.9996747344771242	seed=99`

### 2026-05-08 — `ntm-copy` [mlx] @ `ac2e370` — `--seed 42 --batch 1 --epochs 30000 --es-threshold 0.01`

exit:    0
wall:    21m 10s (1270833 ms)
converged: Converged at epoch 13200 (p10_loss=0.005320791155099869)
stats:   Completed in 21m 4s (13200 epochs, 95ms/epoch)
result:  `RESULT	epochs=13200	acc_short=0.9997499999999999	acc_full=0.9964816176470589	seed=42`

### 2026-05-08 — `ntm-copy` [mlx] @ `0e2e86a+dirty` — `--seed 1 --batch 1 --epochs 30000 --es-threshold 0.01`

exit:    0
wall:    6m 51s (411736 ms)
converged: Converged at epoch 6200 (p10_loss=0.00479522068053484)
stats:   Completed in 6m 45s (6200 epochs, 65ms/epoch)
result:  `RESULT	epochs=6200	acc_short=0.9221666666666666	acc_full=0.6840611154035595	seed=1`

### 2026-05-08 — `ntm-copy` [mlx] @ `0e2e86a+dirty` — `--seed 7 --batch 1 --epochs 30000 --es-threshold 0.01`

exit:    0
wall:    31m 9s (1869313 ms)
converged: Converged at epoch 17100 (p10_loss=0.009659020230174065)
stats:   Completed in 31m 2s (17100 epochs, 108ms/epoch)
result:  `RESULT	epochs=17100	acc_short=0.9734375	acc_full=0.9368180315234301	seed=7`

### 2026-05-08 — `ntm-copy` [mlx] @ `0e2e86a+dirty` — `--seed 99 --batch 1 --epochs 30000 --es-threshold 0.01`

exit:    0
wall:    4m 6s (246432 ms)
converged: Converged at epoch 4400 (p10_loss=0.005980806890875101)
stats:   Completed in 4m 0s (4400 epochs, 54ms/epoch)
result:  `RESULT	epochs=4400	acc_short=0.9997499999999999	acc_full=0.9996747344771242	seed=99`

### 2026-05-08 — `ntm-copy` [mlx] @ `0e2e86a+dirty` — `--seed 123 --batch 1 --epochs 30000 --es-threshold 0.01`

exit:    0
wall:    1h 12m 49s (4369694 ms)
converged: Converged at epoch 25500 (p10_loss=0.0031838188879191875)
stats:   Completed in 1h 12m (25500 epochs, 171ms/epoch)
result:  `RESULT	epochs=25500	acc_short=1.0	acc_full=1.0	seed=123`

### 2026-05-09 — `ntm-copy` [mlx] @ `0e2e86a+dirty` — `--seed 42 --batch 1 --epochs 30000 --es-threshold 0.01`

exit:    2
wall:    3.621s (3621 ms)

### 2026-05-09 — `ntm-copy` [mlx] @ `0e2e86a+dirty` — `--seed 99 --batch 1 --epochs 30000 --es-threshold 0.01`

exit:    2
wall:    3.510s (3510 ms)

### 2026-05-09 — `ntm-copy` [mlx] @ `0e2e86a+dirty` — `--seed 42 --batch 1 --epochs 30000 --es-threshold 0.01`

exit:    2
wall:    3.338s (3338 ms)

### 2026-05-09 — `ntm-copy` [mlx] @ `0e2e86a+dirty` — `--seed 99 --batch 1 --epochs 30000 --es-threshold 0.01`

exit:    2
wall:    3.310s (3310 ms)

### 2026-05-09 — `ntm-copy` [tape] @ `0e2e86a+dirty` — `--seed 42 --batch 1 --epochs 30000 --es-threshold 0.01`

exit:    0
wall:    9m 11s (551011 ms)
stats:   Completed in 9m 5s (30000 epochs, 18ms/epoch)
result:  `RESULT	epochs=30000	acc_short=0.9997499999999999	acc_full=0.9625012271493987	seed=42`

### 2026-05-09 — `ntm-copy` [tape] @ `0e2e86a+dirty` — `--seed 99 --batch 1 --epochs 30000 --es-threshold 0.01`

exit:    0
wall:    42m 19s (2539456 ms)
converged: Converged at epoch 6300 (p50_loss=0.00889734677538643)
stats:   Completed in 2m 14s (6300 epochs, 21ms/epoch)
result:  `RESULT	epochs=6300	acc_short=0.9665833333333333	acc_full=0.9554162412879517	seed=99`

### 2026-05-09 — `dnc-copy` [tape] @ `6dde57a` — `--seed 42 --batch 1 --epochs 50000 --es-threshold 0.01`

exit:    0
wall:    1m 11s (71562 ms)
converged: Converged at epoch 5500 (p10_loss=0.007755757543974216)
stats:   Completed in 1m 7s (5500 epochs, 12ms/epoch)
result:  `RESULT	epochs=5500	acc_short=0.9956250000000001	acc_full=0.932104930270323	seed=42`

### 2026-05-09 — `dnc-copy` [tape] @ `6dde57a+dirty` — `--seed 42 --batch 1 --epochs 30000 --es-threshold 0.01`

exit:    0
wall:    1m 10s (70518 ms)
converged: Converged at epoch 5500 (p10_loss=0.007755757543974216)
stats:   Completed in 1m 7s (5500 epochs, 12ms/epoch)
result:  `RESULT	epochs=5500	acc_short=0.9956250000000001	acc_full=0.932104930270323	seed=42`

### 2026-05-09 — `dnc-copy` [torch] @ `6dde57a+dirty` — `--seed 42 --batch 1 --epochs 30000 --es-threshold 0.01`

exit:    0
wall:    59.780s (59780 ms)
converged: Converged at epoch 3100 (p10_loss=0.0050969075849020065)
stats:   Completed in 56s (3100 epochs, 18ms/epoch)
result:  `RESULT	epochs=3100	acc_short=1.0	acc_full=0.9100805852644088	seed=42`

### 2026-05-09 — `dnc-recall` [tape] @ `6dde57a+dirty` — `--seed 42 --batch 1 --epochs 30000 --es-threshold 0.01`

exit:    0
wall:    8m 2s (482700 ms)
converged: Converged at epoch 20100 (p10_loss=0.009474297106862779)
stats:   Completed in 7m 58s (20100 epochs, 23ms/epoch)
result:  `RESULT	epochs=20100	acc_k2=1.0	acc_k4=0.9222222222222222	acc_k6=0.9083333333333332	seed=42`

### 2026-05-09 — `dnc-recall` [torch] @ `26a336c+dirty` — `--seed 42 --batch 1 --epochs 30000 --es-threshold 0.01`

exit:    0
wall:    8m 22s (502152 ms)
converged: Converged at epoch 14800 (p10_loss=0.008118813353628134)
stats:   Completed in 8m 17s (14800 epochs, 33ms/epoch)
result:  `RESULT	epochs=14800	acc_k2=1.0	acc_k4=0.7388888888888888	acc_k6=0.6555555555555556	seed=42`

### 2026-05-09 — `ntm-copy` [tape] @ `26a336c+dirty` — `--seed 42 --batch 1 --epochs 30000 --es-threshold 0.01`

exit:    0
wall:    2m 46s (166690 ms)
converged: Converged at epoch 9600 (p10_loss=0.005619597162917928)
stats:   Completed in 2m 42s (9600 epochs, 16ms/epoch)
result:  `RESULT	epochs=9600	acc_short=1.0	acc_full=0.9989820261437907	seed=42`

### 2026-05-09 — `ntm-copy` [torch] @ `26a336c+dirty` — `--seed 42 --batch 1 --epochs 30000 --es-threshold 0.01`

exit:    0
wall:    2m 31s (151318 ms)
converged: Converged at epoch 5300 (p10_loss=0.0034115108035240255)
stats:   Completed in 2m 26s (5300 epochs, 27ms/epoch)
result:  `RESULT	epochs=5300	acc_short=1.0	acc_full=0.99375	seed=42`

### 2026-05-09 — `ntm-recall` [tape] @ `26a336c+dirty` — `--seed 42 --batch 1 --epochs 30000 --es-threshold 0.01`

exit:    0
wall:    9m 40s (580580 ms)
stats:   Completed in 9m 33s (30000 epochs, 19ms/epoch)
result:  `RESULT	epochs=30000	acc_k2=0.9916666666666667	acc_k4=0.9622222222222221	acc_k6=0.8577777777777779	seed=42`

### 2026-05-09 — `ntm-recall` [torch] @ `f6d8378+dirty` — `--seed 42 --batch 1 --epochs 30000 --es-threshold 0.01`

exit:    0
wall:    10m 43s (643658 ms)
converged: Converged at epoch 20000 (p10_loss=0.009346269686353154)
stats:   Completed in 10m 34s (20000 epochs, 31ms/epoch)
result:  `RESULT	epochs=20000	acc_k2=0.9955555555555555	acc_k4=0.9561111111111111	acc_k6=0.8894444444444445	seed=42`
