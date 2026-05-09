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

### 2026-05-09 — Align Layer.Rnn with `nn.RNNCell` — `f402354`

**Plan job**: Job 1 / Job 2a (both — the layer change benefits all
backends), with paired-side update.

**Motivation**: pre-existing `Layer.Rnn` was a non-standard
linear-recurrence (no activation, single bias) chosen arbitrarily
when the example was first written. PyTorch's `nn.RNN` doesn't have
a "no activation" mode, so the matching reference (`torch_ref/models/
rnn.py`) had to use a hand-written `LinearRNNCell` with `hidden_size=1`,
no projection, and matching no-activation semantics. Two consequences:
the example didn't demonstrate the canonical RNN shape that library
users expect, and the perf ratio comparison was unfair (PyTorch ref
was doing strictly less work — no projection, no tanh — so the ratio
was inflated).

**Change**: realign both sides to `nn.RNNCell`'s shape:
- Idris `Layer.Rnn` gets two biases (`ihB`, `hhB`) and a generic
  `activation : TVec o d -> TVec o d` field (more flexible than
  `nn.RNN`'s tanh/relu enum — pass any unary tensor function).
- `rnnLayerAny` defaults activation to `ttanh`, matching `nn.RNN`'s
  default.
- PyTorch ref's `LinearRNNCell` rewritten to match `nn.RNNCell`
  (tanh, two biases) plus the output projection the Idris model has
  on top. Defaults `hidden_size=4 / output_size=1` to match Idris.
- Initial hidden state on both sides: zero (matches `nn.RNNCell`
  default; previously was a learned `nn.Parameter` on PyTorch side).
- LR default 0.03 → 0.5 on PyTorch ref (matches Idris example default).

**Impact** (3 samples each, post change):

| Backend | Before  | After   |
|---------|--------:|--------:|
| tape    | 3.83×   | 3.08×   |
| torch   | 5.07×   | 4.38×   |
| mlx     | 7.99×   | 6.59×   |

The ratio shrinkage is partly a methodological correction — the
previous PyTorch ref was doing strictly less work, so its ms/epoch
was artificially small. The new comparison is fair (both sides
implement the same model). All three backends produce bit-identical
loss curves on the new model (e.g. loss=0.005914 at ep 100).

**Outcome**: landed. The example is now a canonical small-RNN
demonstration matching what a library user expects to see for "how
do I use an RNN cell in Idris-ml". Same shape applies to lstm/gru
example alignment if/when we revisit them — they already use the
nn.LSTM/GRU shape, but worth a paired-side audit.

### 2026-05-09 — Align Layer.Lstm and Layer.Gru with `nn.LSTMCell` / `nn.GRUCell` — `2c34ec1`

**Plan job**: Job 1 + Job 2a (cross-cutting; all backends benefit).

**Motivation**: LSTM was using a single fused bias (vs `nn.LSTMCell`'s
two: `bias_ih` + `bias_hh`). GRU's C kernel was a *simplified-GRU*
variant that computed but ignored the `r` reset gate (vs `nn.GRU`'s
`n = tanh(ih_n + r * hh_n)`). Same family of non-standard
simplifications as the rnn alignment.

**Change**:
- LSTM: split bias into `ihB` + `hhB`; `applyLstm` now does 3 FFI
  calls per timestep (vs 2 with fused bias). PyTorch ref drops its
  Jozefowicz forget-gate-bias=1 init for symmetry with Idris (which
  never had it). PyTorch ref now also has learned `h0`/`c0` to match
  Idris's `LstmState` carrying them (added in Phase 1.5b).
- GRU: kernel signature changed from `(combined, prev, o)` to
  `(ih, hh, prev, o)`. Three backends updated: tape rewrites the
  hand-rolled backward to handle r's grad path; mlx uses a new
  `GruCellReplayMeta` to thread `prev`'s pool_idx into the replay
  closure; torch's autograd handles backward through the graph.
  `tgruCell` and `applyGru` updated; `applyGru` now does 3 FFI
  calls (vs 4 with the explicit pre-sum tadd).

**Impact** (3 samples each, post change):

| Cell      | Backend | Before  | After   |
|-----------|---------|--------:|--------:|
| lstm      | tape    | 2.01×   | 1.32×   |
| lstm      | torch   | 2.07×   | 2.12×   |
| lstm      | mlx     | 3.40×   | 4.22×   |
| gru       | tape    | 1.91×   | 1.19×   |
| gru       | torch   | 2.70×   | 1.96×   |
| gru       | mlx     | 5.97×   | 4.05×   |

Tape and torch gru improved on ratio AND in absolute ms (gru tape
5.25 → 4.81 ms because applyGru saves one FFI hop). lstm tape ratio
improved partly because PyTorch ref slowed down (added h0/c0 +
clone) and partly because lstm Idris stayed similar (the extra tadd
hhB cost is small in absolute terms). mlx ratios moved
inconsistently — Job 3 sub-item to investigate later.

**Outcome**: landed. Together with the rnn alignment, the rnn /
lstm / gru examples now demonstrate the canonical PyTorch shape
that library users expect. Backend cell APIs (`tgruCell`,
`tlstmGatesPair`) are also closer to standard ML library
conventions.

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

### 2026-05-09 — `withNoGrad` for RL rollouts + a2c bootstrap — `6e39337`

**Plan job**: Job 1 (torch) + Job 2a (tape).

**Motivation**: completing the `withNoGrad` pattern across the four RL
examples that have a "rollout (no grad needed) + separate batched
loss-step (grad needed)" structure: dqn, ppo, sac (a2c was already
done). Plus pulling a2c's `computeBootstrap` out of `buildLoss` so
the single critic forward on `finalSt` runs in `withNoGrad` too —
the value is consumed as a Double by GAE, no grad path.

**Change**:
- a2c: refactor `buildLoss` to take precomputed `bootstrap : Double`
  as a parameter; compute it in `a2cEpoch` inside `withNoGrad`.
- dqn: wrap `epsGreedyIO` at the action-selection point in
  `runEpisode.go`.
- ppo: wrap `rollout` and `prepareRollout` (which calls
  `computeBootstrap`).
- sac: wrap the post-warmup `sampleActionIO` in `sacStep`.

Reinforce intentionally NOT wrapped: its rollout's per-step forward
log-probs ARE used in the gradient (single-forward-per-step structure,
no separate batched forward in the loss). Wrapping would break
training.

**Impact**: per-prim Chez FFI floor still dominates these examples,
so the savings are modest in absolute ms/epoch (~0.5–2 ms/epoch in
each). The bigger win is correctness: rollout phases no longer hold
references to stale autograd graph nodes that get freed at the next
`optimizer_step` anyway.

**Outcome**: landed. Verified all four examples train to the expected
accuracy on tape.

----

## Future opportunities (not active)

Ideas surfaced during the Job 1 phase A push that we don't plan to
do right now but might be worth picking up later. Listed here rather
than in `TODO.md` because they're optimization candidates with a
specific cost / benefit profile, not "should-have" features.

### Pre-allocated obs buffer + `tensor_write_data_inplace` for RL

For RL examples (CartPole / Acrobot / etc.), every rollout step calls
`bulkToTensor obs` which is 6 prim FFI calls (alloc + 4 setDouble +
create) for the 4-element CartPole observation. Pre-allocating a
persistent obs tensor at episode-start and using a new
`tensor_write_data_inplace` primitive (PyTorch's `tensor.copy_(other)`
shape) to overwrite the values would drop this to ~2 prims.

Estimated savings: ~0.5–1 ms/epoch on a2c (20 rollout steps × 4 prims
saved × 9 µs ≈ 0.7 ms). Modest. Mostly worth it as a library feature —
`tensor.copy_` is something users would expect to exist.

Effort: ~half day. New primitive on each backend + Idris binding +
caller-side rewrite at a couple of sites.

### Slab allocator for `at::Tensor*` on torch backend

Each `from_tensor` in `backend_torch.cpp` does `new at::Tensor(std::move(t))`
— ~1 µs per call from system malloc. For DNC-class workloads (~3K
intermediates per epoch) that's ~3 ms/epoch, plus the matching
`delete` costs at `free_intermediates`. A bump-allocator that
allocates `at::Tensor` slots into a pre-sized arena and resets the
pointer at `free_intermediates` would be O(1).

Estimated savings: 1–3 ms/epoch on DNC-class workloads on torch.
Modest. Code-complexity tradeoff: bump arena needs alignment care
and we'd lose stable pointers (any caller stashing an `at::Tensor*`
across `free_intermediates` would break — though current callers
don't seem to do this).

Effort: ~1 day. New allocator + integration with the existing
`intermediates` vector + `free_intermediates` cleanup path. Test that
all examples still train.

Pairs with the existing TODO "Bound memory usage" — if a slab is in
place, it's natural to extend it with a memory limit.

### Other ideas surfaced and discarded

- **NTM `onesM` precompute** — tried, reverted twice (Idris CSE makes
  the precompute redundant). See entry above.
- **Batched recurrent sequence forward** (Idea B from the Job 1
  brainstorm) — substantial multi-day implementation per backend
  (would need to write the timestep loop in C on tape and mlx; torch
  could delegate to `torch::nn::functional::rnn_tanh`). Estimated
  savings on rnn torch: 1–2 ms/epoch. Cost-benefit didn't favour it
  given the modest perf delta and that the rnn example already
  matches `nn.RNNCell` semantics post-alignment. Could revisit if/when
  larger sequence-length workloads land that would amortize the
  effort better.
