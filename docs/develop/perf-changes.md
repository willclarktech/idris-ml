# Perf changes log

Append-only log of every performance change we make: motivation,
change, impact (before/after ms-per-epoch and ratio), commit hash,
date. The companion files are:

- `docs/develop/perf-log.md` — raw `scripts/perf-run.sh` /
  `scripts/perf-baseline.sh` measurement entries (one per run).
  Records the *measurements*, not the code changes.
- `docs/develop/perf-baseline.md` — current-state table of every
  example × backend ratio. Reflects the latest measurement.

This file records the *changes themselves*: what was tried, what
landed, what didn't. New entries go at the bottom.

## Convention

```
### YYYY-MM-DD — <short title> — <commit-hash>

**Plan job**: Job 1 / Job 2a / Job 2b / cross-cutting (specify which)

**Motivation**: 1–3 sentences on why this change. What did the
profile show? What was the hypothesis?

**Change**: 1–3 sentences on what landed (or was tried). Reference
the file(s) and key function(s).

**Impact**: a small table of before/after ms-per-epoch and ratios
on the affected (example, backend) cells. Note "noise" if the
delta is below the measurement floor; "regression" if it got
worse on some cells.

**Outcome**: landed / reverted / partial. If reverted, why.

**Cross-references**: relevant `perf-log.md` entries (date + commit),
relevant `perf-baseline.md` rows updated.
```

When measurement noise is in play, take 3+ samples and report
mean + range. Don't claim a win from a single run inside the noise
floor.

When a change is reverted, leave the entry — the negative result
is still useful (saves someone trying it again).

----

## Entries

### 2026-05-09 — DNC `dncZeroDiag` mask precompute — `20f4dab`

**Plan job**: cross-cutting (helps Job 1 + Job 2a + Job 2b
together; the mask is a per-step constant rebuild, so reducing
it benefits every backend equally).

**Motivation**: DNC-copy on torch was 9.8× PyTorch ref ms-per-epoch.
Profiling showed `prim_forward_ms` ≈ 114 ms/epoch, dominating the
120 ms/epoch total. `Layer/Dnc.idr`'s `dncZeroDiag` was rebuilding
a (1 − Iₙ) [n, n] mask every timestep — for n = 32 that's
1 + n² + 1 prim FFI calls (`allocDoubles` + 1024 × `setDouble` +
`create2d` + `mul`) on a constant.

**Change**: moved the mask construction into the `DncState`
constructor (new `nonDiagMaskT : AnyPtr` field). `dncZeroDiag` is
now a single `prim__mul` against the precomputed mask.
Mathematically identical (mask is a constant); training trajectory
is bit-identical.

**Impact** (`scripts/perf-baseline.sh <ex> <be>`, seed=42):

| Example    | Backend | Before  | After   | Speedup |
|------------|---------|--------:|--------:|--------:|
| dnc-copy   | tape    | 11.11×  |  1.24×  |  ~9×    |
| dnc-copy   | mlx     | 13.01×  |  2.68×  |  ~5×    |
| dnc-copy   | torch   |  9.80×  |  2.05×  |  ~5×    |
| dnc-recall | tape    | 13.20×  |  1.50×  |  ~9×    |
| dnc-recall | mlx     | 15.25×  |  2.38×  |  ~6×    |
| dnc-recall | torch   | 14.24×  |  2.14×  |  ~7×    |

Both DNC examples moved from Bucket D (>10×) into Bucket A/B.
`dnc-copy` on tape is now within Bucket A (≤1.10×) territory.

**Outcome**: landed.

**Cross-references**: `perf-log.md` 2026-05-09 entries for
dnc-copy/dnc-recall on each backend; `perf-baseline.md`
"NTM/DNC current-state" subtable updated with new ms/epoch.

### 2026-05-09 — DNC `dncRetention` scalar 1.0 reuse — `b209ab1`

**Plan job**: cross-cutting (small; same family as the mask fix).

**Motivation**: After the mask fix, DNC torch was still ~2×.
`Layer/Dnc.idr`'s `dncRetention` recursed once per read head, calling
`prim__createScalar 1.0` inside each recursion to build the
`(1 − fg·rw)` factor. For R = 4 read heads that's 4 redundant FFI
calls per timestep on a constant.

**Change**: pass the precomputed `onesScalar` (already built once
at the call site in `applyDnc`) into `dncRetention` as the leading
argument; reuse it inline.

**Impact** (3 samples each on `scripts/perf-baseline.sh dnc-copy
<be>`):

| Example  | Backend | Before  | After   | Note      |
|----------|---------|--------:|--------:|-----------|
| dnc-copy | tape    | 1.24×   | ~1.14×  | small win |
| dnc-copy | torch   | 2.05×   | ~1.75×  | small win |

Within measurement noise, but consistently slightly better.

**Outcome**: landed.

### 2026-05-09 — NTM `onesM` precompute — *(reverted)*

**Plan job**: Job 1 (mostly torch) + Job 2a (tape).

**Motivation**: `Layer/Ntm.idr`'s `ntmInterpWriteIdris` was building
a length-`m` all-ones tensor per timestep via
`prim__addScalar (zeroState1d m) 1.0`. Same pattern as the DNC mask
fix on a smaller constant. Estimated 3 prim FFI calls × ~50
timesteps/epoch saved.

**Change**: added `onesMPtr : AnyPtr` field to `NtmState`, built
once in `ntmLayer`. Threaded through `applyNtm` to
`ntmInterpWriteIdris`.

**Impact** (3 samples each):

| Example  | Backend | Before  | After   | Note         |
|----------|---------|--------:|--------:|--------------|
| ntm-copy | tape    | 1.42×   | ~1.20×  | small win    |
| ntm-copy | torch   | 2.58×   | ~1.88×  | win          |
| ntm-copy | mlx     | 1.98×   | ~3.6 ×  | **regression** |

mlx regressed ~2×. Likely related to the existing `gotchas.md`
note: "MLX requires non-grad tensors to be non-persistent —
`prim__createState1d` marks them persistent and the lazy graphs
that reference them survive `tape_reset` and dangle after the next
epoch starts." A precomputed persistent ones-tensor used inside
`mx::outer` apparently triggers a slow path on mlx (or builds an
ever-growing lazy graph).

**Outcome**: reverted. Net negative across backends. Could be
re-attempted with a Maybe-cached lazy-init approach (build on
first call of each epoch, reset to Nothing on `resetNtmState`),
which would behave like the existing memT pattern and probably
avoid the mlx interaction. Filed as future-todo, not active.

**Re-attempt 2026-05-09 under priority-torch-and-tape framing** (after
plan was updated to make tape + torch primary and mlx Job 3): hoped
the small torch wins would justify the mlx regression as an
acceptable tradeoff. 3 samples each on tape and torch show no
measurable change vs pre-fix on either: ntm-copy tape ~1.42×
(unchanged), ntm-copy torch ~2.36× (unchanged within noise), ntm-recall
tape ~1.25× (unchanged), ntm-recall torch ~2.64× (worse but high
variance). The "torch wins" I thought I saw on the first attempt were
sampling noise. mlx still regresses 2× as expected. Reverted again.

**Hypothesis for why no win even on tape/torch**: Idris's compiler is
very likely CSE'ing the per-timestep `prim__addScalar (zeroState1d m)
1.0` chain across timesteps — they have identical inputs every call,
so the result handle is shared and the FFI calls only fire once per
sequence anyway. If true, the precompute-into-state plan is just
moving CSE'd work into an explicit field with no perf delta.

This optimization shape (precompute a constant in the state record)
isn't worth pursuing unless we find a constant where Idris's CSE
doesn't fire. The DNC mask precompute (above) DID work because it
saved hundreds of `prim__setDouble` calls in a loop, which CSE can't
fold across the loop.

### 2026-05-09 — `withNoGrad` + a2c rollout — `452eb7e`

**Plan job**: Job 1 (torch wrapper overhead) + Job 2a (tape).

**Motivation**: PyTorch's `torch.no_grad()` suppresses autograd graph
construction for forward passes whose results aren't backprop'd —
the standard pattern for RL rollouts, evaluation, anything that just
wants the forward result. Our a2c rollout was running 480+ prim ops
per epoch under autograd tracking only to extract Doubles for sampling
+ bootstrap; the gradient came from `buildLoss`'s own batched forward.
All 480 ops were wasted graph construction.

**Change**: wired up the existing-but-stubbed `tensor_no_grad_begin/end`
in `backend_tape.c`, `backend_mlx.cpp`, `backend_torch.cpp` with a
nesting counter (matches PyTorch's nestable `torch.no_grad()`).

- tape: `tape_append` becomes a no-op when `no_grad_depth > 0`;
  results marked `requires_grad=0` so downstream doesn't propagate.
  Returns a writable static dummy entry so callers that do
  `e->op_meta = ...` don't null-deref.
- torch: nests a `torch::NoGradGuard` while depth > 0.
- mlx: `tape_append` skipped + result `requires_grad=false` so the
  VJP-replay closure doesn't track these ops.

Idris API: `withNoGrad : IO a -> IO a` in `Tensor.idr`. Uses primIO
sequencing on begin/end (same pattern as `prim__backwardC`).

Wired up in `Example.A2c.a2cEpoch`: rollout phase wrapped in
`withNoGrad`.

**Impact** (3 samples each at default config):

| Backend | Pre   | Post  | Note |
|---------|------:|------:|------|
| tape    | 9.93× | 8.66× | tape entries 600+ → 12, backward 5ms → 0.2ms |
| torch   | 10.81× | 9.26× | autograd graph saved on rollout |
| mlx     | 15.22× | 13.04× | fewer VJP constants |

Wins are smaller than hoped because the per-call Chez FFI floor
(~9 µs/call) dominates per-prim cost on these examples; no_grad
saves a portion of that, not the dispatch itself.

**Outcome**: landed. Useful as a library feature even where the perf
delta is small — anyone writing RL or eval code in Idris-ml would
expect `withNoGrad` to exist, just like PyTorch users expect
`torch.no_grad()`. Future opportunity: also wrap `bootstrapV` and
the eval phase, plus other examples' eval paths.

### 2026-05-09 — DNC `dncReadHeads` link-transpose hoist — `eaab884`

**Plan job**: cross-cutting (mostly tape/torch).

**Motivation**: `dncReadHeads` recursed once per read head, calling
`prim__transpose2d linkT` inside each recursion. `linkT` is shared
across heads — the transpose is head-invariant.

**Change**: compute `linkTransT = prim__transpose2d newLinkT` once
in `applyDnc` and thread it into `dncReadHeads` as an extra
argument. Removes R-1 redundant FFI calls per timestep on a
head-invariant value.

**Impact** (3 samples each, post `b209ab1`):

| Example    | Backend | Before  | After   | Note         |
|------------|---------|--------:|--------:|--------------|
| dnc-copy   | tape    | ~1.25×  | ~1.25×  | noise        |
| dnc-copy   | torch   | ~2.05×  | ~2.14×  | noise        |
| dnc-recall | tape    | ~1.50×  | ~1.32×  | small win    |
| dnc-recall | torch   | ~2.14×  | ~1.91×  | small win    |

Theoretical savings: R-1 = 3 FFI calls × 40 timesteps × 9 µs ≈
1 ms/epoch. Within measurement noise on dnc-copy (small absolute
ms/epoch) but visible on dnc-recall.

**Outcome**: landed.
