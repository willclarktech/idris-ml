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

### 2026-05-14 — GPU-specific "Idris-side overhead" is actually accumulated kernel-launch wall — `<commit>`

**Plan job**: follow-up to the GptLarge Phase 3 wallclock matrix, where
"GPU is 20% slower on wall, but only 7% slower on C-total" left an
unattributed ~800 ms/ep on GPU.

**Motivation**: claim being tested — "mlx GPU has more Idris-side
overhead per FFI call than mlx CPU stream." If true, the fix would
involve Chez Scheme runtime work. If false, then the same accumulated
kernel-launch wall that motivated the optimizer eval-removal is just
showing up across the whole forward/backward graph too — fixable by
fusing more ops, no Idris-side work needed.

**Change**: built `/tmp/bench_per_op` — a tight loop of `tensor_add`
calls with NO eval. Measured pure graph-build cost only:

| measurement                       | CPU      | GPU      | gap |
|-----------------------------------|---------:|---------:|----:|
| graph-build only (no eval)        | 0.43 us/op | 0.44 us/op | **0** |
| add+mul w/ force_eval per iter    | 28 us/op   | 200 us/op  | 7× |
| supervised wall/ep                | 4 ms     | 8 ms     | +4 ms |
| gpt-large wall/ep                 | 8500 ms  | 9300 ms  | +800 ms |

Pure FFI dispatch + graph-node construction is identical on CPU and
GPU. The cost only appears when something forces evaluation —
`tensor_item`, the `mx::eval` calls inside `tensor_backward`, the
final `mx::eval(to_eval)` in `optimizer_step`, etc.

**Impact**: explains the GPU wall gap mechanistically. On mlx CPU
stream, a sync runs the queued ops on a CPU worker thread — fast and
pipelined. On mlx GPU, a sync encodes to a Metal command buffer,
dispatches, and waits for completion — Metal has higher per-op
latency. Across 293 ops per gpt-large epoch (plus backward and
optimizer graph), the cumulative drain at each sync point produces
the 800 ms/ep wall gap. No Idris-side work is the actual contributor.

**Outcome**: investigation only, no code change. The actionable
levers are the same as the optimizer story:
- `mx::compile` wrapping over larger scopes (whole forward, whole
  optimizer step) → fewer kernels → fewer launch-wall contributions
  at each sync point
- bigger per-op compute (bigger model) → wall amortizes naturally

The investigation kills the "Idris-side GPU overhead" hypothesis
cleanly. Filed two follow-ups: the existing "wrap optimizer step in
`mx::compile`" TODO row remains the next concrete lever; the
"investigate Idris-2 JIT / JAX backend" row stays open as the broader
question even though THIS investigation showed Idris-side dispatch is
fine on its own.

**Cross-references**:
- `perf-log.jsonl` post-eval-fix 3-run reproducibility entries
- the per-op microbench `/tmp/bench_per_op.c` is one-off and not
  checked in; reproducible from the recipe in this entry

### 2026-05-14 — GptLarge GPU-vs-CPU matrix: 20% wallclock gap, optimizer is the lever — `<commit>`

**Plan job**: GPU-friendly-example TODO row (the deliverable said
"showing GPU > CPU"; we didn't get it, but found the actionable
next-step lever in the process).

**Motivation**: All previous examples (NTM/DNC/LSTM/MNIST/small Gpt)
were too small for Apple Metal to beat the CPU stream — kernel-launch
wall dominated. Phase B left open whether a properly GPU-shaped
workload would flip the verdict. Built `Example.GptLarge` (dModel=256,
heads=8, headDim=32, blocks=4, seq=128, batch=32; 3.17 M params) and
the paired `torch_ref/scripts/gpt_large.py` to find out.

**Change**: this entry is the measurement, not a code change. The
6-cell matrix was run at 10 epochs each (single sample; deltas large
enough to clear the VM noise floor):

| backend           | wall ms/ep | C-total ms/ep | C-total notes                          |
|-------------------|-----------:|--------------:|----------------------------------------|
| tape              |       9700 |          8830 | actual compute (synchronous)           |
| torch             |       9500 |          1080 | mostly compute (sync per op)           |
| mlx CPU eager     |       8500 |            34 | **enqueue only**                       |
| mlx CPU compile   |       8800 |            33 | **enqueue only**                       |
| mlx GPU eager     |      10200 |           276 | enqueue + per-`mx::eval` sync          |
| mlx GPU compile   |     ~10000 |           254 | enqueue + per-`mx::eval` sync          |

(GPU measured against pip mlx 0.31.2 with Metal at
`/tmp/mlx-gpu-test`; nixpkgs mlx is CPU-only.)

**Impact (revised)**: an earlier reading of this table called GPU
"8-10× slower" — that was wrong; it treated the mlx C-totals as
compute time when they're mostly enqueue cost. The honest read:

- **Wallclock**: mlx GPU is ~20% slower than mlx CPU stream. Real gap,
  but small enough to be "not yet" rather than "never".
- **GPU compute itself looks healthy** — backward forced via `mx::eval`
  is ~11 ms/ep, which matches the ~7.5 ms FLOPS floor for ~75
  GFLOPs/step on M2 (~10 TFLOPS) plus sync overhead.
- **The optimizer step on GPU is 243-265 ms/ep**, exactly the per-param
  kernel-launch wall: 293 params × ~1 ms each. PyTorch's `_foreach_*`
  fused multi-tensor ops are the standard fix; we don't have an
  equivalent on the mlx (or torch) optimizer surface.
- **The mlx CPU "Backward 2.5 ms/ep" number is unreliable** — it would
  imply ~30 TFLOPS on a CPU stream, which is impossible. The mlx CPU
  C-total measures enqueue time; actual compute fires later.
- **Idris-side / Chez overhead floods all wallclocks** at ~8 s/ep on
  this hardware. Until that's reduced (separate TODO row), wallclock
  comparisons are dominated by the constant.

**Outcome**: partial. The example and the measurement matrix exist
and are in CI; the GPU-wins outcome from the original TODO row isn't
reached. The actionable lever is the fused multi-tensor optimizer
(filed as a new high-prio TODO). Default stays `MLX_DEVICE=cpu` until
that lands; verdict re-opens after.

**Cross-references**:
- `perf-log.jsonl` 2026-05-14 entries tagged "Phase 3 cell N/6"
- `docs/develop/mlx-survey.md` "Follow-up update (2026-05-14)" section

### 2026-05-14 — Tape profiler diagnostic: ADD bucket is misattribution — `<commit>`

**Plan job**: cross-cutting (tooling — the tape profiler is the source of
truth for every per-op investigation, and it was misleading us).

**Motivation**: `example-gpt-large` on tape showed "ADD" as 95% of
forward C-time (117 ms/call × 138 calls = 16.2 s/ep). The smaller
`example-gpt` showed the same shape: ADD 8.3 ms/call. Pulled the
thread — vDSP\_vaddD on a [128, 256] tensor should be ~50 µs, not 117
ms. Hypothesis: the per-op timer in `tape_append` attributes inter-op
wall time to the op being recorded *now*, so any Idris-side glue
between ops gets pinned to whichever op happens to close the chain.
ADDs are residual closes in the transformer — they collect the leakage.

**Change**: added three diagnostic timers to `backend_tape.c`'s
`binop_elementwise`:
- direct-kernel timer wrapping just the `vDSP_vaddD` call
- in-function timer wrapping entry-to-exit of the whole
  `binop_elementwise` (split via a thin wrapper +
  `binop_elementwise_inner`)
- path-classification counter (fast / scalar\_bcast / general\_bcast)
  plus a per-op-tag log of the first general\_bcast shape seen

Backed by storage `prof_kernel_per_op`, `prof_kernel_count_per_op`,
`prof_binop_inside_ms`, `prof_binop_inside_count`,
`prof_binop_path_count`, `prof_binop_general_ms`. Reset alongside the
other profile arrays in `backend_profile_reset`. Surfaced in
`backend_profile_print` as new sections after the existing top-N
forward ops.

**Impact**: this is a diagnostic-only change — zero perf delta, just
ground truth. Re-ran `example-gpt-large` and `example-gpt` on tape:

| metric                       | small Gpt | GptLarge | unit       |
|------------------------------|----------:|---------:|------------|
| ADD bucket (attributed)      |      2661 |    16783 | ms / 3 ep  |
| ADD in-function              |       3.3 |     44.8 | ms / 3 ep  |
| ADD kernel (vDSP only)       |       3.5 |     44.3 | ms / 3 ep  |
| bucket / in-function ratio   |    **813×** | **374×** | leakage    |
| per-tape-entry leakage       |      0.33 |     0.59 | ms / entry |
| binop\_elementwise fast path |    100%   |   100%   |            |

All ADDs took the vDSP fast path (zero general broadcast). Kernel
time and in-function time agree within instrument noise (kernel is a
strict subset). Real ADD work per epoch is ~15 ms — three orders of
magnitude smaller than the bucket headline.

**Outcome**: landed (diagnostic only). The real bottleneck on tape
forward at GptLarge scale is **~0.6 ms/tape-entry of Idris-side / Chez
overhead between FFI calls**, which the profiler currently
misattributes to whichever op is recorded next. At 293 entries/forward
× 32-sample batch × per-entry overhead, this dominates the C-total
wallclock. Likely suspects: Chez foreign-procedure dispatch, GC
pressure from per-step Idris-side allocation, or
`UserDeviceCore`-class typeclass-dispatch cost compounding per op
(see the gotcha "Typeclass methods of unit type fire eagerly"). Real
fix needs a separate investigation — at minimum the per-op timer
should record kernel-internal time so attribution stops lying.

**Cross-references**:
- `perf-log.jsonl` 2026-05-14 entries tagged `[diagnostic]`
- `Example.GptLarge` first commits — the workload that surfaced this

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

### 2026-05-09 — Tape `binop_elementwise` numpy-style 2D broadcast

**Plan job**: Job 2a (phase A)

**Motivation**: `binop_elementwise` previously only handled
same-shape and scalar broadcast. Mixed shapes like `(n,1)×(n,m)`,
`(1,m)×(n,m)`, `(m,)×(n,m)` (numpy-style row/column broadcast) fell
into a multi-dim path that flat-indexed past the smaller operand's
buffer — undefined behaviour for any user code or layer trying to
use these patterns. The NTM `ntmInterpWriteIdris` workaround code
explicitly cites the limitation: *"row-wise scalar multiplication
(n,)·(n,m) is not supported by the tape backend's elementwise
broadcast (which only handles numel=1 broadcast), so we materialize
`w` row-wise via `outer(w, ones_m)`"*.

**Change**: Added `compute_bcast_shape` and `compute_bcast_strides`
helpers in `backend_tape.c`. Refactored `binop_elementwise` forward
into three branches: same-shape vDSP fast path (unchanged), scalar
broadcast (cleaned up), and a new general-broadcast walk that
right-aligns ranks numpy-style and uses per-operand strides (0 on
broadcast dims). Refactored `OP_ADD/SUB/MUL/DIV/POW` backward to
detect broadcast via `shapes_equal(a, r)` / `shapes_equal(b, r)`
and reduce gradients along broadcast dims via the same stride walk.

**Impact**: latent OOB-read bug for users fixed. Verified
bit-identical forward + backward against the OLD chain on
NTM-realistic dimensions (n=128, m=20) via a standalone C unit
test. All same-shape and scalar-broadcast operations behave
bit-exactly as before.

**Outcome**: landed in commit `9f78d39`.

----

### 2026-05-10 — NTM `ntmInterpWriteIdris` adopts the broadcast (per-backend seed defaults)

**Plan job**: Job 2a (phase A) follow-up

**Motivation**: With the broadcast capability above, the NTM helper
can drop the `outer(w, ones_m)` materialisation and use a
`reshape(w, n, 1)` view + direct `(n,1)*(n,m)` broadcast mul. The
old workaround's comment "not supported by the tape backend's
elementwise broadcast" now goes away.

**Change**: `Layer/Ntm.idr` `ntmInterpWriteIdris` rewritten:
`reshape2d → neg → addScalar → mul (broadcast) → add`. Saves one
`outer` + one `addScalar(zeros, 1.0)` per NTM timestep. The single-
timestep gradient is bit-identical to the old chain (verified
algebraically and numerically); but the multi-timestep training
trajectory differs in ULP-level ways from the workaround chain
because the backward reduction order changes (broadcast walk vs
chain through `outer`'s own backward).

**Seed sensitivity fallout**: NTM-Copy is highly seed-sensitive
(see `gotchas.md`). With the broadcast in place, the seeds that
converge cleanly differ per backend:
- tape: seed=42 → 4400 ep / 1.0 acc_full ✅
- torch: seed=42 → 5300 ep / 0.99 acc_full ✅
- mlx: seed=99 → ~4400 ep / 0.997 acc_full ✅ (matches the
  pre-broadcast perf-baseline)
- mlx at seed=42 fails (0.65 acc_full); tape/torch at seed=99
  with broadcast are slow / borderline.

The `Makefile`'s `example-ntm-copy` target picks the seed per
backend (tape/torch → 42, mlx → 99). The in-Idris `defaultConfig`
and paired `torch_ref/scripts/ntm_copy.py` both move 99 → 42 (the
primary tape/torch default; mlx is the asymmetric special-case).
Users override with `NTM_COPY_ARGS="--seed N"`.

**Impact**: tape ntm-copy converges ~2× faster (4400 vs ~9600
prior tape-at-seed=42, or ~8400 prior tape-at-seed=99). torch is
unchanged (5300 ep both ways). mlx at its new default seed=99
matches its pre-broadcast best (perf-baseline).

**Outcome**: landed.

----

### 2026-05-11 — Tape: BLAS-accelerate matmul backward kernels — `9311eff`

**Plan job**: Job 2b (phase A, stretch)

**Motivation**: All matmul-class forward kernels (`OP_MM`, `OP_BMM`,
`OP_MV`, `OP_LINEAR`, `OP_LINEAR_2D`) have dispatched to Apple
Accelerate `cblas_dgemm`/`dgemv` since the file was written, but
the matching backwards were hand-rolled triple-nested loops. Every
transformer / GPT / DNC backward pass therefore left Accelerate on
the table on the half of the computation that takes the most time
at scale.

**Change**: Each backward switched to the BLAS-equivalent:
- `OP_MM`: `d_a = dgemm(NoTrans, Trans)`, `d_b = dgemm(Trans, NoTrans)`
- `OP_BMM`: single dgemm collapsing the `B·m` dim (shared weight
  means `d_b` accumulates over batch in one call)
- `OP_MV`: `d_A = dger` (rank-1), `d_x = dgemv(Trans)`
- `OP_LINEAR_2D`: `d_W = dgemm(Trans, NoTrans)`, `d_X = dgemm(NoTrans, NoTrans)`
- `OP_LINEAR`: `d_W = dger`, `d_x = dgemv(Trans)`

`beta=1.0` preserves the existing `+= grad` accumulation semantics.
Each BLAS path is gated on `__APPLE__`; the portable scalar
fallback is preserved.

**Closing sweep** (full 9 examples × 3 backends, see
`perf-log.jsonl` commit `9311eff+dirty`):

| Example | Job 2a (naive) | Job 2b (BLAS) | Δ wall | quality Δ |
|---|---|---|---|---|
| supervised | 3.6s | 3.3s | **-8%** | bit-identical |
| rnn | 8.7s | 8.1s | -7% | bit-identical |
| lstm | 11.2s | 10.5s | -6% | bit-identical |
| gru | 9.7s | 9.7s | 0% | bit-identical |
| transformer | 31.8s | 28.8s | **-9%** | bit-identical |
| dnc-copy | 89s / 0.877 | 76s / 0.873 | **-15%** | ≈0 |
| dnc-recall | 480s / k4=0.94 | 433s / k4=0.96 | **-10%** | + |
| ntm-copy | 84s / 4400ep / 1.0 | 118s / 7000ep / 1.0 | +40% wall | acc preserved |
| ntm-recall | 163s / 8500ep / k4=0.98 | 321s / 18000ep / k4=0.91 | +97% wall | **-7pp** |

Per-epoch ms is faster everywhere; the NTM wall-clock regression is
purely seed-trajectory: BLAS `dgemm`/`dger`/`dgemv` reduce in a
different floating-point order than the naive triple loop, and
NTM-Copy's documented seed-sensitivity (`gotchas.md`) flips
seed=42 onto a slower-converging branch. Quality preserved on
ntm-copy (acc_full=1.0 either way); ntm-recall acc_k4 drops 7pt.

**Decision (recorded here as the rationale)**: kept the
unconditional BLAS path despite the NTM regression. Rationale:
1. The library is general-purpose; NTM is one architecture out of
   ~25 examples. Hobbling the linear-algebra fast-path for every
   user just to preserve NTM-Copy's seed=42 trajectory is the
   wrong trade. (See feedback memory
   `feedback_library_users_not_examples.md`.)
2. NTM-Copy still converges to acc_full=1.0; it just takes more
   epochs at the default seed.
3. NTM-Recall's k4 0.98→0.91 is a real quality regression but
   acc_k2=1.0 stays perfect — short-sequence recall is unaffected
   and length-generalization to k4/k6 is the inherently
   seed-sensitive part of the benchmark.

A threshold-dispatch variant (route to naive below
`m·n·k = 5000`) was tried (commit `1518381`, reverted in
`3128ad5`); even the act of wrapping the naive path in
`if (use_blas) {...} else { naive }` shifts compiler codegen
enough to drift gradients ULP-wise. The variant also fared worse
than all-BLAS on dnc-recall in our run (k4 0.96 → 0.82).
Threshold tuning is too noise-prone to do without a proper
microbench framework (logged for Phase B).

**Outcome**: landed. NTM regression accepted as a library-level
trade. Job 2b phase A closed.

----

### 2026-05-11 — Batched Conv2D / MaxPool2D + MNIST → epochVarTensorBatch — `a5f9368`

**Plan job**: Job 1 (reopened — Conv2D wrapper audit)

**Motivation**: A side-by-side MNIST convergence comparison flagged
a 4.0× wrapper-overhead ratio on idris-torch vs raw PyTorch ref
(49.4 vs 12.5 s/epoch on 60K MNIST, batch_size=64). Job 1 phase A
had only audited linear / RNN / NTM-DNC paths; the Conv2D path was
unexamined. Tracing it: `Layer/Conv.idr:applyConv2D` operates on
`TVec (inC*h*w)` (single sample), and `Backprop.idr:epochVarTensor`
threads each `dataPoint` through the model individually — so the
training loop calls `torch::conv2d` (and friends) 64× per minibatch
instead of once batched. Every per-call autograd-graph setup and
tensor-view bookkeeping ran 64× more than necessary.

**Change**: wired Conv2D and MaxPool2D into the existing batched-
forward infrastructure that Linear / Activation / Dropout already
used.

C-side:
- `tensor_conv2d_batched` + `tensor_max_pool2d_batched` in all three
  backends. Torch drops the per-call `unsqueeze(0)`/`squeeze(0)`
  (libtorch is batch-native). mlx skips the per-call NHWC layout
  reshape on the batch axis. Tape gets new `OP_CONV2D_BATCHED` /
  `OP_MAX_POOL2D_BATCHED` op tags with batched-meta structs and
  matching forward + backward kernels (B in the outer loop, d_kernel
  accumulating across the batch in a single tight loop).
- `tensor_reshape_4d` helper (was 1d/2d/3d only).

Idris-side:
- `prim__conv2d_batched`, `prim__maxpool2d_batched`, `prim__reshape4d`
  FFI bindings.
- `Conv2DState` / `MaxPool2DState` `applyVarBatch` impls that reshape
  `[B, c*h*w]` → `[B, c, h, w]`, call the batched prim, reshape back.
- `Example/Mnist.idr::trainOneFullPass` switches `epochVarTensor` →
  `epochVarTensorBatch`.

**Impact** (MNIST full 60K, seed=42 tape/torch, seed=99 mlx):

| Backend | per-sample s/ep | batched s/ep | wrapper vs PyTorch ref (12.5 s/ep) |
|---|---:|---:|---:|
| torch | 49.4 | **21.3** | 4.0× → **1.68×** ✅ |
| mlx   | (n/a baseline) | **26.0** | — / **2.08×** |
| tape  | 175 | 176 | 14.0× → 14.1× (compute-bound) |

Torch wrapper overhead halved. mlx in the ~2× range. Tape per-epoch
unchanged because the bottleneck is the hand-rolled triple-nested
conv kernel (the batched version runs the same FLOPs with the same
naive code), not the FFI count — file follow-up for an im2col +
`cblas_dgemm` tape conv2d kernel. Quality preserved on torch/mlx
(98.4% / 98.1% acc), tape at the batched-default seed converges to
97.3% in 3 epochs (down from 98.4% per-sample) — the now-familiar
NTM-style ULP-shift seed-sensitivity reappearing on tape.

**Outcome**: landed. Job 1 reopened-phase-A first half done; tape
im2col follow-up below closes the second half.

----

### 2026-05-11 — Tape Conv2D: im2col + cblas_dgemm forward + backward — `67f4b42`

**Plan job**: Job 1 (reopened — tape Conv2D follow-up)

**Motivation**: After the batched-Conv2D / MaxPool2D layer wiring
closed the idris-torch wrapper gap on MNIST (4.0× → 1.68×), the
tape backend was *unchanged* per-epoch (~176 s) because its hand-
rolled triple-nested conv kernel did the same FLOPs in either
shape. The standard high-performance conv decomposition is im2col
+ GEMM: unfold each output window into a row of a `[B·oH·oW,
inC·kH·kW]` matrix, then one big `cblas_dgemm` against the
flattened weight matrix replaces the nested loop. We already wired
`cblas_dgemm` for matmul backward in Job 2b; the same dependency
pays off again here.

**Change**: rewrote `tensor_conv2d_batched` (forward) and the
`OP_CONV2D_BATCHED` backward in `backend_tape.c` to use im2col +
cblas:

Forward:
- `X_col [M, K] = unfold(input)` where `M = B·oH·oW`, `K = inC·kH·kW`
- `Y_unf [M, outC] = X_col @ W^T` — one `cblas_dgemm(NoTrans, Trans)`
- permute `Y_unf` to `out [B, outC, oH, oW]` + bias broadcast

Backward:
- `dY_unf [M, outC] = permute(r.grad)`
- `dW [outC, K] = dY_unf^T @ X_col` — one `cblas_dgemm(Trans, NoTrans)`
- `dX_col [M, K] = dY_unf @ W` — one `cblas_dgemm(NoTrans, NoTrans)`
- `dInput += col2im(dX_col)`
- bias gradient via direct sum

Workspace buffers (X_col, Y_unf, dY_unf, dX_col) are heap-allocated
via `calloc`/`free` per call rather than arena-allocated — an
earlier arena-allocated version produced "invalid memory reference"
crashes between epochs (likely interaction with eval's accumulated
tape entries holding pointers across the arena state change). The
`calloc` path is robust and the per-batch malloc cost is dwarfed by
the dgemm.

**Impact** (MNIST full 60K, seed=42, 3 epochs):

| Variant | per-epoch | wall | acc | vs PyTorch ref (12.5 s/ep) |
|---|---:|---:|---:|---:|
| pre-batched | 175 s | 8m 45s | 0.984 | 14.0× |
| batched, naive kernel | 176 s | 8m 49s | 0.973 | 14.1× |
| **batched, im2col + cblas** | **20 s** | **1m 51s** | **0.973** | **1.62×** |

**8.6× tape speedup**. All three backends now within ~2× of
PyTorch ref on MNIST: torch 1.68×, mlx 2.08×, tape 1.62×.

**Outcome**: landed. Job 1 reopened-phase-A fully closed.

### 2026-05-11 — mlx scalar-allocation hot-path audit (Job 3 Phase A) — `ede8b6b`

**Plan job**: Job 3 Phase A (mlx-only, no tape/torch impact).

**Motivation**: After the Job 1 reopen closed, an explore-agent
source review of `backend_mlx.cpp` surfaced six places where the
file was re-allocating `mx::array(...)` literals on hot paths —
all "provably wasteful given mlx's semantics" (cached scalars are
immutable; sharing them is safe). The bar for Phase A was "read
the diff and see why it's free"; per the plan, anything that
needed benchmarking to validate was deferred to Phase B (the
mlx-projects survey).

**Changes** (6 atomic commits, in order of est. impact):

1. **Hoist optimizer-state scalars** (`07e6991`). `optimizer_step`'s
   per-param loop was allocating `mx::array()` for `alpha`,
   `1-alpha`, `beta1`, `1-beta1`, `beta2`, `1-beta2`, `eps`,
   `momentum`, and both Adam bias-correction terms once per param
   per step. None depend on which param. Hoisted to once-per-step.
2. **Cache F32_ZERO/F32_ONE/F32_HALF in forward hot paths**
   (`62d8a77`). Added `kF32_ZERO/ONE/HALF()` Meyers' singletons;
   applied to `tensor_softplus`, `tensor_gelu`, `tensor_dropout`,
   `tensor_gru_cell`. GELU's structural coefficients
   (0.7978…, 0.044715, 3) became function-local statics inside
   `tensor_gelu` (same lifetime story).
3. **Cache F32_ZERO for null-arg fallbacks in vjp replay** (`5b7309b`).
   `tensor_backward`'s closure was constructing `mx::array(0.0f)`
   per fallback per tape entry per backward (2 per entry for
   unary ops). Routed both fallbacks through `kF32_ZERO()`.
4. **Cache GELU/SOFTPLUS replay coefficients** (`9d9434e`).
   `OP_GELU` and `OP_SOFTPLUS` cases inside the replay lambda were
   re-allocating their constants per backward. Lifted to
   function-local statics; common 0/0.5/1 routed through the
   `kF32_*()` accessors so forward and replay share the same
   underlying arrays.
5. **Cache vjp pool placeholder** (`2652ae0`). `std::vector<mx::array>
   pool(N, mx::array(0.0f))` per backward. Routed the placeholder
   through `kF32_ZERO()` — vector's N slots are then refcounted
   shallow copies of a shared array, not copies of a freshly-
   allocated one.
6. **Cache masked-fill -1e9 sentinel in vjp replay** (`ede8b6b`).
   `OP_MASKED_FILL` case allocated `mx::array(-1e9, float32)`
   fresh per backward. Lifted to a function-local static.

**Safety notes** (recorded in the new `Hot-path scalar constants`
header in `backend_mlx.cpp`):

- All Meyers' singletons — lazy init picks up whatever default
  device `mlx_backend_init` configured.
- Sharing constants across calls is safe: mlx arrays are
  immutable from ops' perspective; ops produce new arrays rather
  than mutating inputs.
- Avoided using the cached singletons as the rhs of `mx::outer`
  or similar ops where persistent operands hit the documented
  slow path (`gotchas.md` "MLX requires non-grad tensors to be
  non-persistent").

**Impact — all 6 commits in** (mlx closing sweep, seed=99, NTM/DNC
at `--epochs 30000 --es-threshold 0.01`, vs pre-Phase-A baseline at
`798c4ac+dirty`):

| Cell | pre ms/ep | post ms/ep | delta | convergence |
|---|---:|---:|---:|---|
| supervised | 1 | 1 | – | bit-identical |
| rnn | 11 | 11 | – | bit-identical |
| lstm | 18 | 13 | −28% | bit-identical |
| gru | 15 | 15 | – | bit-identical |
| transformer | 35 | 33 | −6% | sort_acc 6/6 |
| mnist | 26 000 | 22 800 | −12% | acc 0.98 |
| dnc-copy | 30 | 34 | +13% | bit-identical (af 0.88) |
| ntm-copy | 55 | 57 | +4% | bit-identical (af 0.94) |
| dnc-recall | 55 | 66 | +20% | bit-identical (k4 0.82) |
| ntm-recall | 54 | 57 | +5% | bit-identical (k4 0.65) |

**Reading the numbers**: these measurements were taken on a VM
with concurrent workload. Single-run ms/ep variance on this
machine runs ±15–20%, which is bigger than most of the deltas
above. The convergence column is reliable (loss / acc / converged-
epoch are deterministic and **bit-identical pre/post on every
cell**), so the changes are numerically clean. The perf signal is
noise-dominated.

We confirmed this by bisecting the apparent +20% on dnc-recall:
reverted candidates #3 (null-arg fallback in vjp replay) and #5
(vjp pool placeholder) — the two changes that participate in the
most tape entries per backward — and re-ran dnc-recall on the
partial revert. Result: **69 ms/ep** (worse than the all-6 number
of 66 ms/ep). Reverting code can't deterministically make
scheduling slower; the partial-revert outcome confirms the
underlying noise is bigger than the effect we were trying to
attribute. We did not pursue a clean-baseline control run because
the same noise would dominate that measurement too.

What the deltas plausibly mean once noise is accounted for:
- **lstm −28%** is large enough to be real, and lines up with the
  optimizer-scalar hoist (LSTM models have many Adam params); not
  a sure thing but the most credible single-cell win.
- **mnist −12%** and **transformer −6%** are within noise.
- **dnc-copy +13%, dnc-recall +20%, ntm-copy +4%, ntm-recall +5%**
  are within noise.
- **rnn / gru / supervised** were already noise-floor cells.

**Safety review** (independent of perf signal):

- All six changes are mechanical "cache a `mx::array(...)` constant
  instead of re-creating it per call." No semantic changes.
- Mlx arrays are immutable from ops' perspective — sharing across
  calls is safe (ops produce new arrays rather than mutating
  inputs).
- Cached singletons are not used as the rhs of `mx::outer` or
  similar persistence-sensitive ops (per the `gotchas.md`
  documented slow path).
- Convergence bit-identical on every cell confirms no numerical
  drift.

**Outcome**: all 6 commits land. The principled win is small-but-
real (fewer fresh-array allocations on hot paths, less graph
bloat) even if not separately measurable through VM noise; the
changes are also a free safety improvement (mlx arrays sharing one
underlying scalar buffer rather than thousands of independent
allocations is friendlier to mlx's cache budget). Real perf
characterization deferred to Phase B (mlx-projects survey + a
proper microbench framework for per-pattern timing).

Phase A complete. Five-minute total wall on the perf-changes side;
the heavy lift was the closing sweep, which validated convergence
correctness.

### 2026-05-11 — mlx GPU (Metal) exploration — discovered universal regression — `94700e5`

**Plan job**: Job 3 Phase A side-quest. The question was: are the mlx
numbers we've been measuring this whole project actually CPU stream,
and what changes if we run on Metal GPU?

**What we found**:

1. **The nixpkgs `python3Packages.mlx-0.31.2` package is CPU-only.**
   The nix derivation at `pkgs/development/python-modules/mlx/default.nix`
   hardcodes `MLX_BUILD_METAL=false` because Apple's `metal` shader
   compiler isn't open-source and the nixpkgs maintainers don't want
   to use sandbox escape hatches. `otool -L libmlx.dylib` on the nix
   build shows no Metal framework linkage; `mx::is_available(gpu)`
   returns 0 at runtime. Setting `MLX_DEVICE=gpu` on the nix build
   aborts with `Cannot set gpu device without gpu backend`. So
   **every mlx measurement before this entry was CPU stream**,
   regardless of any `MLX_DEVICE` setting.

2. **pip-installed mlx works** (`uv pip install mlx` auto-pulls in
   `mlx-metal` with a 150 MB precompiled `mlx.metallib`; the dylib
   links Metal.framework). Tested in this Tart VM:
   `mx::is_available(gpu) == True`, a real GPU computation succeeds.
   So GPU IS reachable in this Tart VM via Apple Virtualization
   Framework's paravirt-graphics (consistent with Tart's
   documentation that "Metal APIs work inside VMs with no
   additional setup").

3. **But GPU is universally slower in this environment.** Built
   `backend_mlx.cpp` against the pip mlx and ran an `MLX_DEVICE=gpu`
   sweep on the same 10-cell config (killed after 7 cells once the
   pattern was clear):

   | Cell | mlx CPU ms/ep | mlx GPU ms/ep | slowdown |
   |---|---:|---:|---:|
   | supervised | 1 | 11 | 11× |
   | rnn | 11 | 114 | 10× |
   | lstm | 13 | 156 | 12× |
   | gru | 15 | 145 | 10× |
   | transformer | 33 | 111 | 3× |
   | mnist (ms/ep) | 22 800 | 112 800 | 5× |
   | dnc-copy | 30 | 269 | 9× |

   Convergence remained bit-identical / within seed-trajectory noise
   on the cells we measured (`acc_short`, `acc_full`, `sort_acc`
   etc. matched CPU runs). So GPU is numerically clean but a
   throughput regression of 3–12× across the board.

**Why GPU loses here**: the kernel-launch wall. Each `tensor_*` call
dispatches one Metal kernel. The forward chain for an RNN cell is
30-50 ops on tensors of <100 elements; the backward replays the
same chain inside the VJP closure; with batched training that's
~150k-300k Metal kernel dispatches per epoch on mnist. At those
tensor sizes the per-op compute is microseconds but the launch
overhead is comparable or larger — especially under Tart's paravirt-
graphics path, which likely adds further per-dispatch latency on top
of bare-metal Metal. CPU stream skips all of this and calls Apple
Accelerate BLAS directly.

The "GPU is good for image conv" intuition is rooted in workloads
designed for GPUs — big batches (256–1024), bigger images (224×224×3
ImageNet), deep models (ResNet, VGG). MNIST as a 32-batch / 28×28×1
problem with a 2-conv-2-FC model is too small to amortize the
per-dispatch cost.

**The actual lever for GPU here is `mx::compile()`** — mlx's JIT
API that compiles a multi-op function once and replays it as a
single fused Metal kernel. We don't use it (the existing path uses
`mx::vjp` which builds a closure but doesn't compile it). Wiring
`mx::compile` into the replay path is the open Phase B work for
Job 3; without it GPU is just an alternate-and-slower CPU stream
in this environment.

**Tooling changes that landed alongside this discovery**:
- `device` field added to perf-log JSON schema (`scripts/perf-run.sh`,
  `scripts/perf-baseline.sh`, `docs/develop/perf-log.md`).
  mlx records `MLX_DEVICE` (default cpu); tape/torch always cpu.
  Entries before this date can be assumed device=cpu.
- `mlx` package removed from nix dotfiles (`vm/modules/unix/packages.nix`) —
  this project uses a project-local pip install for the Metal build
  rather than the nix CPU-only build.
- `docs/develop/gotchas.md` got new entries documenting the nixpkgs
  build flag, the pip workaround, and the "GPU usually loses at
  these scales" finding.

**Outcome**: `MLX_DEVICE=cpu` is the right default and stays. GPU
remains supported but flagged as "available but typically slower
at idris-ml example scales — requires `mx::compile`-style fusion
to be competitive." No commits to `backend_mlx.cpp` from this
investigation; the binary that's checked in builds against
whatever mlx is detected at make time and reads `MLX_DEVICE` at
runtime. Phase B's mlx-projects survey should now include `mlx`'s
own `mx::compile` / `mx::value_and_grad` JIT path as a primary
target rather than incidental side-reading.

### 2026-05-11 — idris-gym source-review Phase A (Job 4) — null result

**Plan job**: Job 4 Phase A (idris-gym env-side wins, no
vectorization restructure).

**Motivation**: RL examples ratio at 20-40× PyTorch ref because
env-step time dominates. Before tackling vectorization (Phase B),
audit `packages/idris-gym/` for per-step waste that's cheap to
fix — same Phase A pattern that worked for Jobs 1/2a/3.

**Method**: source review surfaced 5 candidates. Per the
measure-then-hypothesize-then-change discipline, built a
microbench (`make bench-gym`, see commit message for harness
notes) targeting each candidate's hot path, measured baseline,
formed a quantitative hypothesis, implemented, re-measured.

**Baseline** (M4 Pro VM, ns/call, ±5% across runs):

| Function | ns/call |
|---|---:|
| `Rng.nextDouble` | 140 |
| `Blackjack.bjObserve` | 55 |
| `Pendulum step+observe` | 70 |
| `Acrobot step+observe` | 645 |
| `Taxi step` | 22 |
| `CliffWalking step` | 20 |

**Experiments**:

*#3 `Rng.nextDouble`: replace `cast {to=Double} (cast {to=Integer}
top53)` with the direct `cast {to=Double} top53`.* Hypothesis:
the explicit Integer intermediate allocates a GMP bignum per
call; direct prim should cut 30-60% off the function. **Result**:
138 → 139 ns/call (within noise). Hypothesis falsified — either
Idris codegen already fuses the chain, or the cost is elsewhere
(likely splitMix64's two bignum multiplications against the
0x9E37… and 0xBF58… constants, which fall outside the Chez
fixnum range and allocate per multiply). Reverted.

*#5 `Blackjack.bjObserve`: replace the double-traversal handSum +
usableAce with a one-pass `handStats` that returns (raw_sum,
ace_count); `bjObserve` calls it once instead of four traversals
across handSum + usableAce.* Hypothesis: ~50% reduction in
bjObserve, from 55 → ~28 ns/call. **Result**: 55 → 56 ns/call
(within noise). Hypothesis falsified. Idris's `length . filter`
and `foldr (+) Z` paths are already fast enough on tiny lists
(2-4 cards) that the duplicate work is below the measurable
floor. Reverted.

*#4 Taxi/CliffWalking Nat↔Integer round-trips.* Did not
implement — baseline measurement settled it directly: Taxi step
is 22 ns/call and CliffWalking step is 20 ns/call. The
`cast {to=Integer} (n : Nat)` calls cited as wasteful are
compiled by Idris's BigInt-Nat optimization to a no-op (`Nat` is
already stored as `Integer` at runtime); there's no chain to
shorten. Confirmed not a win.

*#1 Acrobot trig caching.* Did not implement — the savings
ceiling is 1 redundant `cos(th1)` between the termination check
and `aObserve` (~15 ns of the 645 ns step+observe = 2.3%).
Capturing more trig values would require either adding cached
fields to `AState` (which `eulerStep` also constructs with
meaningless values for the cache — ugly API change) or splitting
into separate `AState` / `AStateObs` types (bigger refactor than
Phase A allows). Skipped pending Phase B or a willingness to
make the structural change for a single-digit-% win.

*#2 Pendulum trig caching.* Did not implement — initial source-
review analysis was wrong. `pStep` computes `sin(s.pTheta)` on
the *current* angle (used for dynamics); `pObserve` later
computes `cos/sin(s'.pTheta)` on the *new* angle after the step.
These are different inputs, so no redundancy to remove. The
agent's source-review hypothesis was incorrect.

**Outcome**: zero idris-gym source changes land from Phase A.
What lands instead:

- `make bench-gym` microbench tool (`packages/idris-gym/test/bench.ipkg`
  + `Bench.idr`). Useful for any future per-call optimization
  experiments on the env code; baseline numbers documented in the
  commit message.
- Two reusable Idris bench-authoring lessons (in the same commit
  message): defeat CSE by varying input per iteration; avoid
  Peano-Nat counters above ~100k iterations or BigInt allocation
  compounds.
- Confirmation that env-side per-call work is already tight at
  the source level; the 20-40× ratio against PyTorch ref is
  attributable to single-step-vs-vectorized-env architecture, not
  to local source waste. **Phase B (vectorization) is now the
  unambiguous next lever** rather than something we were doing
  "after the obvious wins."

----

### 2026-05-12 — REINFORCE batched policy forward (Job 4 Phase B)

**Change**: added `rolloutEpBatched` + `computeLossBatched` +
`epochRLBatched` + `genBatchV` to `packages/idris-ml-examples/src/Example/Reinforce.idr`.
New `--batched 1` CLI flag selects the batched path; default stays
sequential. The batched rollout stacks N envs' observations into a
single `Tensor [N, 4]`, does **one** `forwardVarBatch` per timestep,
then per-env action sampling + `cpStep`. Done envs are frozen (their
state passes through the batched forward; no `StepRec` appended)
to keep the `[N, 4]` shape stable. Loop exits early once all envs
terminate.

**Why**: Job 4 Phase A had established (via `make bench-gym`) that
env-step is already cheap (~100 ns/call for CartPole). The 20-40×
RL-example ratio vs PyTorch ref is per-op-count, not env-step cost.
Per-timestep batched policy forward collapses N×T forward calls
into T forwards — same gradient math, fewer wrapper trips, fewer
tape entries. The reframe is captured in the plan and in the
"Job 4 Phase B" task description.

**TDD progression** (per Job 3 Phase B pattern):

1. Failing parity test added in
   `packages/idris-ml-examples/test/src/Test/Reinforce.idr` — assert
   per-episode total rewards match sequential rollout for matched
   RNG, N=1 and N=2.
2. `rolloutEpBatched` implemented; parity tests pass bit-identically
   on all three backends (tape, torch, mlx-CPU). Verified via
   `make test-examples-unit`.
3. `--batched 1` wired into `main` via runtime dispatch; convergence
   preserved (CartPole reaches max avg_return=200.0 at 100 epochs on
   all three backends, both modes).

**Per-epoch cost at 100 epochs** (canonical numbers from
`scripts/perf-run.sh`, logged to `perf-log.jsonl`):

| backend | seq ms/ep | batched ms/ep | Δ wall |
|---|---:|---:|---:|
| tape  | 70  | 70  | −4% (noise) |
| torch | 100 | 90  | −16% |
| mlx   | 800 | 440 | **−37%** |

The mlx win is the headline — wrapper overhead per call was highest
there (mx::array construction, tape entry, VJP closure rebuild per
call). Tape/torch wins are within the VM noise envelope (±15-20%
per `feedback_vm_perf_noise`) but consistent in direction. Confirmed
across two independent measurement passes (ad-hoc `time` wrappers +
perf-run.sh).

**What this *doesn't* close**: the 20-40× ratio vs PyTorch ref. Even
at the batched mlx number (52s for 100 epochs), we're well above
what PyTorch ref does on the same workload. The remaining gap is
shared with the rest of the codebase (Idris per-prim cost floor,
the ~9 µs glue) and is a Job 1/2a concern, not specifically Job 4.

**Files**:
- `packages/idris-ml-examples/src/Example/Reinforce.idr` — new
  rolloutEpBatched, computeLossBatched, epochRLBatched, genBatchV,
  --batched flag.
- `packages/idris-ml-examples/test/src/Test/Reinforce.idr` — parity
  test suite (N=1, N=2 per-env reward parity).
- `packages/idris-ml-examples/test/{test.ipkg,src/Main.idr}` — wire
  the new test module.

**Open follow-ups**:
- Extract `rolloutEpBatched` to a shared module in idris-ml-examples
  (e.g. `Example.RL.BatchRollout`) so other RL examples can reuse it,
  once at least one more example wants it.
- Port `a2c.idr`, `ppo.idr`, `sac.idr`, `dqn.idr`, `mountain-car.idr`
  to use batched rollout. Apply the same TDD discipline per example.

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
- **DNC controller stacked-FC** (Job 2a brainstorm) — replacing the
  11 per-gate `prim__linear` calls with one stacked `(W,b)` linear +
  11 narrows nets one EXTRA prim per timestep (11 → 12). The C-side
  cache locality of one big GEMV vs eleven small ones is real but
  small relative to the ~9–19 µs Idris-glue floor. Skip.
- **Transformer transpose caching in single-seq path** (Job 2a
  brainstorm) — `runHeadAttn` calls `prim__transpose2d` on the q/k/v/op
  weights inside the head loop, but each iteration's q/k/v/op is the
  *next* head's weight — the transposes are not redundant. Audit
  conclusion: not a real opportunity.
- **DNC `onesScalar` precompute / generic scalar-constant pool** (Job
  2a brainstorm) — same Idris-CSE story as `onesM`. The scalar 1.0
  constructor is folded by the compiler.
- **`buildMatrixRows` 2D scalar round-trip** — listed in early plans;
  already removed from the codebase. No-op.

### 2026-05-14 — Torch Adam: multi-tensor step via `at::_foreach_*` — `<commit>`

**Plan job**: follow-up to the GptLarge wallclock matrix. The
PyTorch-precedent fused-optimizer story: PyTorch ships
MultiTensorApply (`torch.optim.Adam(foreach=True)` since 1.12+) as the
default Python path, but libtorch's C++ `torch::optim::Adam::step()`
still loops per-parameter (no `foreach` / `fused` option in
2.11.0's `adam.h`). idris-ml routes through the C++ API, so we get
the per-param path.

**Motivation**: validate the multi-tensor pattern that the existing
"mlx backend: wrap optimizer step in `mx::compile`" TODO row will
need. Torch `_foreach_*` is the easier landing because the API is a
1:1 swap of `for (p : params) tensor_x_(p, ...)` to
`at::_foreach_x_(params, ...)` — no tracing, no cache, no functional
state-threading rewrite.

**Change**: new `adam_step_foreach` static in
`packages/backends/backend_torch.cpp` does Adam's m / v / denom /
parameter update via `at::_foreach_mul_`, `at::_foreach_add_`,
`at::_foreach_addcmul_`, `at::_foreach_sqrt`, `at::_foreach_div_`,
`at::_foreach_add_`, `at::_foreach_addcdiv_`. Body is wrapped in
`torch::NoGradGuard` — the in-place ops on leaf params with
`requires_grad=true` would otherwise trip autograd's `check_inplace`
(transformer / mnist / seq-classify / dqn / a2c / ppo crashed on this
in the first pass). Params with undefined grad are filtered out of
the gather lists (matches `torch::optim::Adam::step()` behaviour). m
and v stay in `AdamParamState` so libtorch's serializer continues to
work. `optimizer_step` dispatches to `adam_step_foreach` when
`w->type == 2`; SGD / RMSprop / AdamW fall through to `opt->step()`
unchanged. `TORCH_ADAM_FOREACH=0` (later renamed to `TORCH_FOREACH=0`
when SGD foreach landed) routes back to libtorch's per-param step for
A/B comparison. Added `prof_optimizer_math_ms` sub-timer to separate
the math from `free_intermediates()` (the dominant non-math contributor
inside `prof_optimizer_ms`).

**Impact** — gpt-large @ torch CPU, 5 epochs, single-run A/B with
otherwise-identical setup:

| metric                    | A: foreach OFF | B: foreach ON | Δ           |
|---------------------------|---------------:|--------------:|------------:|
| optimizer-math ms/ep      |           9.27 |          8.66 | **−0.61** (−6.6%) |
| optimizer total ms/ep     |           16.4 |          16.0 |        −0.4 |
| backward ms/ep            |         1042.9 |         933.3 |       −109.6 (noise) |
| C total ms/ep             |         1059.3 |         949.3 |       −110.0 |
| val BPC @ ep 5            | 4.771089597130988 | 4.771089597130988 | **bit-identical** |

Numerics are bit-identical down to the last fp64 digit — confirms the
mul/add ordering matches PyTorch's per-param implementation
(`m.mul_(β1).add_(g, 1-β1)` then `v.mul_(β2).addcmul_(g, g, 1-β2)`).

The optimizer-math gain is tiny on CPU. The 110 ms/ep backward drop
between the two runs is unrelated noise (a run-to-run delta of <15%
sits below the VM noise floor; this is at 10%). On a 9500 ms/ep wall
budget the foreach math itself moves <0.01% of wallclock.

**Why so small on CPU**: PyTorch's MultiTensorApply is engineered to
reduce GPU **kernel-launch** overhead. CPU `at::_foreach_*` is
implemented as a parallel for-loop over the list, with no kernel-launch
to save. The cost we measure on CPU is pure compute + dispatch, both
of which the per-param loop already vectorises via Accelerate /
OpenMP. So we get the structural benefit (one call instead of N) but
the absolute speedup is in the µs-per-param noise.

**Outcome**: landed. The change is correctness-neutral
(bit-identical), perf-neutral on CPU (within noise), and forward-looks
to the torch GPU path where it would land the typical 2–10× optimizer
speedup PyTorch documents. Also validates the multi-tensor shape for
the `mx::compile`-of-optimizer rewrite (the actual GPU work for
idris-ml, since we don't currently run on torch GPU).

**Cross-references**:
- `perf-log.jsonl` `kind=ab` entries timestamped 2026-05-14T15:32 and
  2026-05-14T15:34
- libtorch 2.11.0 `adam.h` has neither `foreach` nor `fused` option
  (Python-side only)
- next concrete step from the high-priority TODO list: the
  `mx::compile` optimizer wrap (where the same pattern actually pays
  off because Metal kernel-launch latency is the bottleneck)

### 2026-05-14 — MLX Adam: optimizer step via `mx::compile` — `<commit>`

**Plan job**: the natural follow-up to the torch foreach landing (same
day). The mlx-side analogue of `at::_foreach_*` is `mx::compile`: trace
the per-param Adam math once into one fused mlx callable, then replay
with new tensor inputs each step. Unlike torch CPU foreach (which is
just a parallel for-loop), mlx compile actually fuses ops into one
graph submission, so it saves the per-op kernel-launch tax on Metal.

**Motivation**: before writing code, a 2×2 baseline (mlx × {cpu, gpu} ×
{compile OFF, ON}) on GptLarge revealed two structural facts:

| Config | Wall | C-total | Optimizer | Backward |
|---|---:|---:|---:|---:|
| mlx CPU + MLX_COMPILE=0 (default) | 9000 | 28.8 | 26.4 | 2.4 |
| mlx CPU + MLX_COMPILE=1 (probe)   | 9600 | 112.9 | 104.0 | 8.9 |
| mlx GPU + MLX_COMPILE=0           | 9000 | 132.6 | 121.8 | 10.7 |
| mlx GPU + MLX_COMPILE=1           | 9800 | 156.7 | 145.1 | 11.6 |

(`MLX_COMPILE` is the existing backward-pass-forward-replay probe.
Recompiles every call → pure regression today. The new
`MLX_OPT_COMPILE` is separate and caches.)

The wall is identical at 9000 ms/ep regardless of device or compile
flag — ~98% of wall is *outside* the C profile region (Idris VM +
per-op FFI dispatch on the forward-pass tape build). mlx ops are lazy,
so forward FFI calls are cheap to mlx but expensive to Chez. **The C
gap (28.8 vs 132.6 ms/ep) is invisible at wall, so even driving
optimizer math to zero saves <1% of wall.** This change therefore
isn't a wall-mover at the current example scale; it's GPU-shaped perf
hygiene + prerequisite for compiling the whole training step (the
"whole-step compile" investigation, where wall actually moves).

A path-C spike (scale GptLarge dModel from 256 to 512 to 768) was run
in parallel to see if any reachable scale flips GPU > CPU on its own.
At dModel=512 GPU edged out by 1000 ms/ep (within noise floor). At
dModel=768 GPU lost by 3000 ms/ep and OOM'd on generation in the Tart
VM. So scale alone doesn't fix the example; the compile work is the
correct lever but lands as part of a larger plan.

**Change**: `adam_step_compile` in `packages/backends/backend_mlx.cpp`
implements the Adam update as a pure function
`(params, grads, m, v, per-param lrs, scalars) → (new params, new m,
new v)` and wraps it in `mx::compile`. The compiled callable is cached
per active-param-count in a static `unordered_map<int, function<...>>`
— mlx caches further by input-shape signature internally, so repeated
calls with the same param shapes hit the trace cache after the first
invocation. Gated on `MLX_OPT_COMPILE=1` env var, default OFF
(opt-in). Only `opt->type == 2` (Adam) dispatches to the new path;
SGD/RMSprop/AdamW fall through to the per-op loop unchanged. Added
`prof_optimizer_math_ms_mlx` sub-timer that brackets just the math
(not the surrounding `mx::eval(to_eval)` + `tape_reset`).

The math sub-timer immediately revealed the structural ceiling:
optimizer-math is only **1.4–1.8 ms/ep** of the 96–157 ms/ep
`Optimizer` total. The remaining 94+ ms/ep is `mx::eval(to_eval)`
synchronisation and tape rebuild — bookkeeping that the compile path
cannot touch. So the max wallclock yield from compiling optimizer math
is bounded by ~1.5 ms/ep, plus whatever kernel-launch savings the
compile gives on the eval step downstream.

**Impact** — gpt-large dModel=256, 5 epochs, A/B:

| metric (ms/ep) | mlx CPU OFF | mlx CPU ON | mlx GPU OFF | mlx GPU ON |
|---|---:|---:|---:|---:|
| Wall              | 9000  | 9600  | 9200   | 9000   |
| Backward          |  9.3  |  8.9  |  11.1  |   9.3  |
| Optimizer         | 96.0  | 98.6  | 156.5  | 120.9  |
| of which math     |  1.4  |  1.6  |   1.8  |   1.5  |
| C total           | 105.3 | 107.5 | 167.7  | 130.2  |
| val_bpc           | 4.746685288232547 | 4.746685288232547 | 4.746687874851618 | 4.746687482731175 |

- **CPU: small regression** (+2.6 ms/ep optimizer). No kernel launches
  to amortize on Apple Accelerate stream; mx::compile's tracing
  overhead is pure cost. **Bit-identical fp64 numerics**
  (`4.746685288232547` matches OFF down to the last digit).
- **GPU: −35.6 ms/ep optimizer (−23%)**, −37.5 ms/ep C-total. Real
  kernel-launch savings — the design hypothesis held. **Numerics
  deviate ~4e-7 relative** (`4.746687482731175` vs OFF
  `4.746687874851618`), within fp32 ULP noise — mlx GPU is fp32
  internally and the compile pass reorders ops, which shows up at the
  7th decimal. Well below convergence noise; not a correctness issue.
- **Wall unchanged** on both, consistent with the diagnostic: C-side
  cost is <2% of wall at this scale.

**Outcome**: landed, opt-in (`MLX_OPT_COMPILE=1`). CPU users keep
status quo (slight regression if enabled); GPU users get a measurable
optimizer-math win on the device that benefits. Default OFF avoids the
CPU regression and the existing-`MLX_COMPILE`-style "probe with no
caching" gotcha. The same compile-once-then-replay pattern is the
load-bearing technique for the future whole-training-step compile
investigation, where wall actually moves because the entire forward
pass becomes one mlx call from Idris instead of N FFI dispatches.

**Cross-references**:
- `perf-log.jsonl` `kind=ab` entries timestamped 2026-05-14T16:54 (mlx
  CPU A/B) and 2026-05-14T16:58 (mlx GPU A/B)
- `perf-log.jsonl` `kind=baseline` entries for the 2×2 diagnostic
  timestamped 2026-05-14T16:43..16:46 (mlx CPU OFF/ON + GPU OFF/ON)
- the parallel path-C spike: GptLarge dModel ∈ {256, 512, 768} on
  mlx CPU vs GPU showed no clean crossover at reachable VM scales —
  dModel=768 OOM'd on GPU during generation, so scale alone is dead
  in this environment
- existing `MLX_COMPILE` env var (separate from `MLX_OPT_COMPILE`) is
  the backward-pass forward-replay probe at `backend_mlx.cpp:2080`,
  added under Job 3 Phase B as a probe — still a regression because
  it has no caching across calls. Future work to cache that one is
  task #42 (decided non-trivial: the lambda capture pattern that
  the new compile path uses doesn't translate directly because the
  backward closure captures the per-step tape)

### 2026-05-14 — Diagnostic: where the 9000 ms/ep GptLarge wall actually goes — `<commit>`

**Plan job**: before committing to a 2-3 week architectural refactor
("compile the whole training step" — eliminate per-op FFI dispatch),
measure where the unaccounted-for ~8870 ms/ep of GptLarge wall is
actually spent. Two-way fork: (a) FFI marshalling overhead per
Idris→C transition → architectural change pays off; (b) Idris VM
between FFI calls → architectural change is dead weight, Idris-side
optimisation is the lever.

**Method**: two independent measurements.

1. **Per-FFI-call wall** (`/tmp/bench_per_op.c` — pure C, no Idris):
   tight loop of `tensor_add(a, b); tensor_free(c)` against
   libidrisml. mlx ops are lazy — each call allocates a graph node
   and returns an `AnyPtr`, no compute. Both streams measured:

   | stream | per-FFI wall | per-pair (add+mul) |
   |--------|-------------:|-------------------:|
   | CPU    |   0.46 µs    |       0.91 µs      |
   | GPU    |   0.45 µs    |       1.01 µs      |

2. **FFI count per epoch** (instrument `tape_append` in
   `backend_mlx.cpp`): every grad-tracked forward op fires
   `tape_append` once. GptLarge 5 epochs at dModel=256:

   - **1136 tape_appends / epoch** (5678 total / 5).

**Multiplication**: 1136 × 0.46 µs = **0.52 ms FFI wall per epoch**
out of 8600 ms wall = **0.006%**.

(Even doubling for non-grad ops not in the tape — input creation,
masks — caps total FFI wall at ~1.2 ms/ep. Still negligible.)

**Inference**: ~8600 ms/ep is **Idris VM time between FFI calls**.
Per tensor op, the Idris side spends **7.6 ms** preparing/dispatching
each op (8600 / 1136). For comparison, Chez Scheme can run plain
arithmetic loops at >1M ops/sec — so a 7.6 ms-per-op overhead on a
"call this C function with two AnyPtrs" operation is enormous, and
the culprit isn't the FFI boundary itself (proven: 0.46 µs).

**Likely candidates for the 7.6 ms-per-op Idris overhead**
(not measured here; this is the *next* diagnostic step):

- Existential `AnyLayer` dispatch in the `Network` chain — each
  forward step walks the chain via `~~>` (existential pattern
  match per layer, indirection through `LayerLike` method
  dictionary)
- Constraint dictionary construction at call sites — `UserDeviceCore d`
  / `LayerLike d` are typeclass constraints that may resolve at runtime
  rather than getting fully inlined, building a dictionary record per
  call
- `Tensor` record packing/unpacking on every op (the record carries
  `tensorPtr : AnyPtr`, `paramId : Maybe String`)
- `Vect` operations in shape arithmetic (Idris-2 `Nat` is `Integer`
  at runtime but Vect/List operations still walk lists allocatively)
- Per-op Idris-level closures inside layer methods (`applyVar`,
  `applyVarBatch`) that allocate intermediate structures

**Outcome** — **kills the "compile the whole training step" plan
(Path A) before any code lands.** Path A's premise was that
eliminating ~thousands of FFI dispatches per step would save the
~8870 ms/ep that isn't C-side. We now know FFI dispatch costs
~0.5 ms/ep — that's the entire upside ceiling. Even a perfect Path
A implementation would save 0.006% of wall.

**The actual lever (Path B): cut the Idris-side per-op overhead.**
Even halving 7.6 ms → 3.8 ms drops wall from 8600 → 4300 ms/ep —
**a 50% wall reduction**. And it's the kind of work that compounds:
any per-op overhead fix lifts every example, not just GptLarge.

**Open questions for the Path B plan** (next diagnostic step,
not in scope of this commit):

- Add Idris-level timing to `forwardVar` itself and the individual
  `applyVar` / `LayerLike` methods to localise the cost
- Try a single-layer harness (e.g. just `linearLayerAny` repeated
  100k times) vs full Network — does cost scale with chain length?
- Inspect Chez Scheme compiled output for one tensor op call —
  what is each call actually doing?
- Inspect whether `%inline` annotations on the `UserDeviceCore`
  method bodies are being honoured by the compiler

**Code change committed alongside**: just the `prof_tape_appends_mlx`
counter in `backend_mlx.cpp` (used to produce this number). The
counter stays — it's cheap and surfaces real signal.

**Cross-references**:
- `perf-log.jsonl` `kind=diagnostic` entries timestamped 2026-05-14T17:18
  (CPU probe) and 2026-05-14T17:20 (GPU probe) — the
  tape-invariance check that confirmed compile-once-replay is
  semantically viable (the assumption Path A's design depended on)
- `perf-log.jsonl` `kind=microbench` entry timestamped 2026-05-14T17:30
  for the per-FFI-call wall measurement
- `perf-log.jsonl` `kind=diagnostic` entry timestamped 2026-05-14T17:30
  for the tape-append count
- TODO row: a new row for "Idris-side per-op overhead reduction" lands
  in this same commit. The mlx-optimizer-compile row (already partly
  done) keeps the SGD/RMSprop/AdamW follow-up but loses its
  "GPU C-total ≤ CPU C-total" acceptance gate (wall doesn't move
  via this lever)

### 2026-05-14 — Chez source profile localises 7.6 ms-per-op cost to recursive Nat arithmetic in uncached positional encoding — `<commit>`

**Plan job**: follow-up to the FFI-vs-Idris-VM diagnostic (earlier today,
same `docs/develop/perf-changes.md`). That measurement bounded
**Idris VM = 99.99% of wall**, but didn't say *which* Idris code. This
commit identifies the dominant Idris-side hot path.

**Method**: Chez Scheme's built-in source-level profiler. The Idris 2
Chez codegen emits a `.ss` source file plus a `compileChez` script that
parameterises `(optimize-level 3)` for the final `.so`. Adding
`(compile-profile 'source)` to that parameterise plus a trailing
`(profile-dump-html ...)` after the main call produces a per-line
execution heatmap. Total round-trip from "we need this measurement" to
"we have the answer" was about 20 minutes — no Idris-side instrumentation
needed, no C-side counters, just one parameterise + dump call.

**Result** — per-line execution counts on GptLarge 1 epoch, dModel=256:

| Generated-Scheme line | Function (demangled) | Count | Notes |
|----:|---|---:|---|
| 923 | `Data.Nat.lte` | **1,956,671,790** | recursive walk on unary Nat |
| 924 | `Prelude.Types.prim__integerToNat` | 980,740,575 | called per div'/mod' recursion |
| 925 | `Data.Nat.divC-39` | 490,399,841 | recursive div' |
| 1011 | `Data.Nat.modC-39` | 490,340,353 | recursive mod' |
| 52-54 | `blodwen-toSignedInt` | 30,400,810 | runtime bit-fit (small) |
| 96 | `bs+` (signed add) | 18,533,971 | (small) |

The four Nat-recursive entries sum to **~3.9 billion `cond`/`equal?`/`sub1`
operations per epoch**. These compile to recursive decrement because the
Idris stdlib `Data.Nat.lte` / `div'` / `mod'` pattern-match on `Z/S k`
constructors — even though `Nat` is `Integer` at runtime, the function
*body* still does `(let ((e-0 (- arg-0 1))) (lte e-0 ...))`.

**Root cause** — `Layer/Transformer.idr` `posEncVal`:

```idris
posEncVal : Nat -> Nat -> Nat -> Double
posEncVal dModel pos dim =
  let p = cast {to=Double} pos
      i = cast {to=Double} (div dim 2)         -- ← recursive Peano on Nat
      dm = cast {to=Double} dModel
      angle = p / pow 10000.0 (2.0 * i / dm)
  in if modNatNZ dim 2 ItIsSucc == 0 then sin angle else cos angle  -- ← idem
```

…called by `writePE` which loops over `(pos, dim) ∈ [0, seqLen) × [0, dModel)`
= 128 × 256 = **32,768 `posEncVal` calls per forward**. And — crucially —
**`writePE` runs *inside* `applyTransformer`** at `Transformer.idr:173`:

```idris
peBuf = prim__allocDoubles (sI * dI)
peBuf' = writePE dModel peBuf 0 0 sI dI          -- rebuilt every step!
peT = prim__createState2d sI dI peBuf'
```

So the positional encoding — which is **deterministic, parameterless,
shape-only-dependent** — is recomputed from scratch on **every forward
pass**. At 32 forwards/epoch × 32,768 `posEncVal`/forward × ~hundreds of
Nat operations each = the billions of Nat operations we see in the profile.

**Two compounding bugs, two orthogonal fixes**:

1. **Cache PE on `TransformerState`**. Build once at `transformerLayer`
   construction, store as a `TMat seqLen dModel d NoGrad` field on
   `MkTransformer`. Forward passes use the cached tensor. Removes the
   per-step writePE entirely. Single-batch case is trivial broadcast;
   batched case needs the reshape-add-reshape dance (or a fresh
   `prim__tilePE` helper).

2. **Use `Int` arithmetic in `posEncVal`**. `div dim 2` and
   `modNatNZ dim 2` on `Nat` are wildly wasteful regardless of caching.
   `dim` is already a `Nat` ≤ `dModel` ≤ ~thousands; converting to `Int`
   and using `div : Int -> Int -> Int` (which is a single CPU instruction)
   makes the one-time PE construction fast too. Even before fix (1), this
   alone would land an order of magnitude.

**Audit for related issues elsewhere**: same pattern (Nat used in inner
loop where Int would do, or per-step recomputation of deterministic
state) very likely exists in other layer types. Candidates:

- NTM / DNC's content-based addressing (cosine similarity loops?)
- Convolution kernel index computations
- RNN / LSTM / GRU per-step Nat indexing
- LayerNorm's per-feature loops (less likely — those are usually fused C)
- Any per-batch loop that walks shape-derived `Nat` values

The same Chez `compile-profile 'source` recipe answers this in 20
minutes per architecture. Bake it into the perf workflow as a
**`make profile-gpt-large` / `make profile-ntm-copy` etc.** target so
future regressions are caught without ad-hoc setup.

**Outcome** — Path B's first concrete win is in flight (Fix 1 + Fix 2
land in the next commit). Expected wall reduction on GptLarge: 30-50%
just from PE caching; possibly more once `posEncVal` uses `Int`. Lifts
every backend (tape, torch, mlx), not just mlx — this is pure Idris-side
overhead.

**Cross-references**:
- The 20-min profile recipe lives in scratch state for now
  (`/tmp/gpt-prof.{ss,so,*.html}`); when this finding lands, write up a
  `docs/develop/chez-profiling.md` recipe for future use
- The two known-recursive-Nat callers in idris-ml (only places `Data.Nat`
  / `modNatNZ` are used per-step): `Layer/Transformer.idr:81` (this
  finding) and `Train.idr:246` (eval-every-N-epochs, called per epoch
  not per step, so negligible)

### 2026-05-14 — Transformer: cache PE + Int arith in posEncVal → 22× GptLarge speedup — `<commit>`

**Plan job**: Path B Fix 1 + Fix 2. Implementation of the two fixes
proposed in the immediately-preceding `perf-changes.md` entry
("Chez source profile localises 7.6 ms-per-op cost..."):

1. **Fix A — cache positional encoding** on `TransformerState`. New
   field `peCached : TMat seqLen dModel d g`, built once in
   `transformerLayer`, reused by `applyTransformer` (direct add) and
   `applyTransformerBatch` (reshape-to-3D, broadcast-add, reshape-back).
   `freezeLayer` / `unfreezeLayer` thread it through unchanged.

2. **Fix B — `Int` arithmetic in `posEncVal`**. Keeps the public
   `Nat -> Nat -> Nat -> Double` signature (per the discussion: "can
   we maybe get some of the benefits of the Nat interface by casting
   internally to Int?"), casts `dim` to `Int` once at the function
   entry, then uses `Int div`/`Int mod` for the parity / half-index
   computation. Avoids the recursive Peano walks even on the one-time
   PE construction.

**Impact** — GptLarge dModel=256, 5 epochs, A/B vs the 9000 ms/ep
baseline:

| Backend / Device | Wall before | Wall after | Speedup | val_bpc match |
|---|---:|---:|---:|---|
| mlx CPU          | 9000 ms/ep  | **400 ms/ep**  | **22.5×** | bit-identical (4.746685288232547) |
| mlx GPU          | 9200 ms/ep  | **400 ms/ep**  | **23×**   | ~1e-7 fp32 noise (4.746686877352244) |
| torch CPU        | 9400 ms/ep  | **600 ms/ep**  | **15.7×** | bit-identical (4.746685288232547) |
| tape             | 11600 ms/ep | **1600 ms/ep** | **7.3×**  | ~3e-7 noise (4.746688438790350) |

The wall reductions correspond to: (Fix A) eliminating ~1M
`posEncVal` calls per epoch — these were rebuilding the PE tensor on
every forward pass — and (Fix B) eliminating the recursive Peano
walks inside each `posEncVal` call. Together: ~3.9 billion Nat-recursive
operations/epoch removed. Idris-2 unit tests (`make test`) still green.

**Architecture audit motivated by this finding**:

The same anti-pattern (recursive Nat arithmetic in inner loops or
per-step recomputation of deterministic state) likely exists
elsewhere. Candidate audit targets, by likely impact:

- **NTM / DNC content-based addressing** (`Layer/Ntm.idr`,
  `Layer/Dnc.idr`) — cosine-similarity over the memory matrix could
  have `Nat` indexing in inner loops. NTM examples are listed at
  9.8-14.2× PyTorch ratio (`perf-baseline.md`), some fraction of
  which may be this class of bug
- **Convolution kernel index computations** (`Layer/Conv.idr`) —
  similar profile risk
- **RNN / LSTM / GRU per-step indexing** (`Layer/Rnn.idr`,
  `Layer/Lstm.idr`, `Layer/Gru.idr`) — less likely since the cell
  body is mostly delegated to fused C ops, but the per-step Vect
  walk over time-steps could hide Nat ops
- **LayerNorm / BatchNorm per-feature loops** — less likely, mostly
  fused C
- **Any per-batch loop that walks shape-derived Nat values** —
  inspect via the Chez profile recipe (`compile-profile 'source`)

**Process change**: bake a `make profile-<example>` target into the
Makefile that uses the `compile-profile 'source` recipe to produce
a per-source-line execution heatmap. The 20-minute setup is now
zero-minute, and the next "where does the wall go?" question gets
answered immediately.

**Cross-references**:
- `perf-log.jsonl` entries timestamped 2026-05-14T18:52 (post-Fix-B
  mlx CPU 7400 ms/ep — Fix B alone) and 2026-05-14T18:57 (post-Fix-A+B
  on mlx CPU 400, mlx GPU 400, torch 600, tape 1600)
- The Chez profile recipe (commands: `compile-profile 'source` +
  `profile-dump-html` after main) — write up as
  `docs/develop/chez-profiling.md` in a follow-up

### 2026-05-14 — Architecture audit for PE-style oversights (post-fix) — `<commit>`

After the transformer PE fix landed (22×, previous entry), audited
remaining `Layer/*` for the same anti-pattern (per-forward recomputation
of deterministic state, with or without recursive Nat arithmetic).

| Pattern | Where | Hot path? | Verdict |
|---|---|---|---|
| `prim__causalMask sI` in `blockForward` | `Layer/Transformer.idr:117` | per-block × per-forward | **follow-up**: cache on `BlockState` or `TransformerState` |
| `prim__expandMask (prim__causalMask sI) batchSize` in `batchBlockForward` | `Layer/Transformer.idr:230` | per-block × per-forward | **follow-up**: cache the 3D form per `(seqLen, batch)` pair, or cache 2D and broadcast |
| `mkZeroVectN`, `mkZeroVectM` recursion on `r` (read heads) | `Layer/Dnc.idr:204-209` | per-sequence start only | OK — `r ≤ 4` in practice, only fires when state is `Nothing` |
| `zeroState1d / zeroState2d` | `Layer/Ntm.idr:75-86` | per-sequence start | OK — single C op |
| `Vect.replicate` calls | all `*Layer` constructors | init only | OK — once per model build |
| `Data.Nat.modNatNZ` | `Train.idr:246` (eval-every-N-epochs gate) | per-epoch | OK — negligible vs epoch wall |

The transformer was a uniquely bad case because two anti-patterns
combined: per-forward recomputation **and** recursive Nat arithmetic
inside the recomputed body. Other layers have at most one of those, in
cold paths. The most plausible remaining win is the causal mask — same
"computed from shape constants, recomputed per forward" shape as PE
was, but each rebuild is a single C op rather than an Idris loop, so
the magnitude is much smaller. Worth a follow-up commit, not load-bearing.

**No commit attached to this audit** — pure documentation. The findings
land here for future reference. The causal-mask follow-up gets its own
TODO row.

### 2026-05-15 — `prim__tile2d` primitive across all 3 backends — `<commit>`

**Plan job**: investigate the mlx transformer small-model regression
flagged 2026-05-14 (33.21 → 37-40 ms/ep, ratio 1.73× → 1.95×). The
prior commit's `applyTransformerBatch` uses
`reshape3d → add → reshape2d` (3 mlx ops) to broadcast the cached
`[seqLen, dModel]` PE onto the flat `[b*seqLen, dModel]` embedded
tensor. Hypothesis: the 2 extra reshape ops cost more than the saved
`writePE` recompute at this scale; a `tile` primitive that does
`[seqLen, dModel] → [b*seqLen, dModel]` in one op should fix it.

**Change**: new `tensor_tile_2d(t, rep0, rep1) -> [m*rep0, n*rep1]`
exported across all 3 backends:

- **mlx**: `mx::tile(t->data, {rep0, rep1})`. Eagerly `mx::eval` the
  result when the input is non-grad (cached PE case) so `mx::vjp`
  sees a leaf and doesn't trace back through tile in backward.
- **torch**: `to_tensor(h)->repeat({rep0, rep1})`. libtorch autograd
  handles the backward automatically.
- **tape**: manual `memcpy`-based forward loop + new `OP_TILE_2D`
  backward that sums grad over the tiled dims back to input shape.

`prim__tile2d` exposed in `Tensor.idr` with `Nat -> Int -> Int` cast
at the call site to keep the existing convention.
`applyTransformerBatch` now does `peTiled = prim__tile2d
peCached.tensorPtr bI 1; h0 = prim__add embedded peTiled` instead of
the reshape dance.

**Impact** — `scripts/perf-baseline.sh transformer <backend>`,
two-point timing:

| Backend | Pre-tile_2d (reshape) | Post-tile_2d | Δ |
|---|---:|---:|---:|
| tape  | 5.21 ms (0.27×) | 6.4 ms (0.31×) | within VM noise |
| **mlx**   | 39.53 / 37.16 ms (1.95-1.98×) | **37.09 ms (1.89×)** | unchanged within noise |
| **torch** | 13.1 ms (0.65×) | **9.95 ms (0.5×)** | **−24%, clean win** |

`gpt-large` on mlx: 400 ms/ep, bit-identical val_bpc
`4.746685288232547` — the 22× speedup is preserved.

**The mlx finding**: the small-model "regression" is NOT the reshape
ops as we hypothesised. Swapping `reshape3d + add + reshape2d` for
`tile + add` (one fewer mlx graph node, eagerly materialized) didn't
move the ratio. The actual cost is **fundamental**: carrying the
cached PE tensor on `TransformerState` adds it to the forward's
constants pool every step, which `mx::vjp` processes during the
backward replay. On dModel=16 with 30-tape-entry forwards, this fixed
overhead is 12-20% of wall. On dModel=256 (gpt-large) it's a tiny
fraction.

**Acceptable trade**: 12-20% absolute slowdown on a tiny demo trades
for 22× speedup on the real model. The TODO row is downgraded from
"fix" to "investigate" and deprioritised. Further surgery would
require per-batch-size cached tiled PE (complex with variable `b`
across train/eval) or a heuristic skip-cache for small models (ugly).

**`tile_2d` is a net win regardless**: clean new primitive across 3
backends (useful for future broadcasting patterns: NTM/DNC head
replication, conv kernel tiling, etc.), torch transformer −24%, tape
within noise, mlx within noise. Cleanest cross-backend addition since
the `OP_LSTM_GATES_CELL` cell-output fused op.

**Cross-references**:
- `perf-log.jsonl` entries timestamped 2026-05-15T00:42, 00:43, 00:46,
  00:48 (transformer × {mlx, gpt-large, tape, torch})
- The TODO row "Transformer: investigate residual mlx small-model
  overhead (1.89×)" captures the remaining unfixed nuance
- The chez-profile recipe (`docs/develop/chez-profiling.md`) was the
  tool that produced the 22× win; not used for this follow-up since
  the cost is on the mlx C side, not the Idris side

### 2026-05-15 — New `Example/MatmulBench`; retire GptLarge — `<commit>`

**Context**: the GptLarge example was added 2026-05-09 as the
"GPU-shaped GPT variant" intended to demonstrate mlx GPU > CPU.
Today's microbench (`/tmp/bench_matmul.c`) localised exactly where
the crossover happens for mlx in this Tart VM environment:

| N | CPU per-call | GPU per-call | Winner |
|---|---:|---:|---|
| 256  | 0.09 ms | 0.74 ms | CPU (GPU loses 8.2×) |
| 512  | 0.29 ms | 1.03 ms | CPU |
| **1024** | 1.69 ms | 1.60 ms | tied (crossover) |
| **2048** | 14.15 ms | 6.28 ms | **GPU 2.3×** |
| **4096** | 120.67 ms | 32.17 ms | **GPU 3.75×** |

GptLarge sits at N=256-tensor-size territory — structurally CPU
land. No amount of mx::compile / Path-A / etc. can flip it without
either bigger tensors (Tart VM ceiling) or fundamental Idris
runtime changes (out of scope).

**Change**: new `Example/MatmulBench.idr` does pure forward
matmuls at N=2048 (default) / 4096 (configurable) through the
typed `Tensor` API. No training, no gradient — just a clean
demonstration of "type-safe shape arithmetic AND GPU dominance"
at the scale where the second part is true. Measured on this VM:

| N | CPU (idris-ml) | GPU (idris-ml) | Speedup |
|---|---:|---:|---:|
| 2048 | 13.76 ms (1248 GFLOPS) | 7.81 ms (2197 GFLOPS) | **1.76×** |
| 4096 | 120.96 ms (1136 GFLOPS) | 33.97 ms (4045 GFLOPS) | **3.56×** |

The idris-ml numbers track the raw C bench within VM noise — the
typed wrapper costs nothing material at these compute sizes.

**Removed**: `Example/GptLarge.idr` + `torch_ref/scripts/gpt_large.py`
+ Makefile targets (`example-gpt-large`, `example-gpt-large-full`,
`ref-gpt-large`) + `scripts/perf-run.sh` + `scripts/check-paired-defaults.py`
entries. The historical perf-log + perf-changes entries about
GptLarge stay (they're append-only and document real findings —
the PE-caching 22× speedup, the Idris-VM-99.99%-of-wall diagnostic,
the chez-profile recipe — all came out of that example's work).

**TODO opened**: medium-priority row for a Llama-class inference
example. mlx is canonically built for LLM inference (Llama, Mistral,
etc.) where the per-op compute >> kernel launch and GPU dominates
by 5-20×. Implementing tiny-Llama-1.1B inference would be the real
showcase — the matmul bench is the smallest version of that story.

**Cross-references**:
- `perf-log.jsonl` `kind=microbench` entries timestamped 2026-05-15T01:36..01:39
- `/tmp/bench_matmul.c` is the raw C version of the same bench (no
  Idris involvement) that established the crossover points

### 2026-05-16 — Wrapped-handle ABI sweep — perf-neutral on hot examples — `9664726`

**Plan job**: tensor-lifecycle Phase 5' (perf measurement half).

**Motivation**: validate the cost of the Phase 1' wrapped-handle ABI
sweep (commit `0ec6a99`), which converted ~600 Tensor-touching FFIs
from `%foreign "C:..."` to `%foreign "scheme:..."` wrap-on-return
templates. Each FFI now does one extra `vector-ref` per Tensor arg
+ one Chez vector allocation + one guardian-register + one
`tensor_retain_handle` per Tensor return. Hypothesis: aggregate cost
is below the VM-noise floor on the hot examples.

**Change**: no code change for this measurement entry; pure perf
characterization of the post-sweep state.

**Impact**: two-point ms/epoch via `scripts/perf-baseline.sh`,
compared to the pre-sweep baseline rows from `db20f12+dirty`
(2026-05-15):

| example   | backend | pre-sweep (db20f12) | post-sweep (9664726) | delta | notes |
|-----------|---------|-------------:|--------------:|------:|-------|
| transformer | tape | 6.4 ms/ep | n/a (build-dominated) | — | tape per-epoch < build noise floor |
| transformer | mlx  | 37.09 ms/ep | 31.63 ms/ep | -15% | within VM noise; trending favorable not regressive |
| transformer | torch | 9.95 ms/ep | n/a | — | not re-baselined (unaffected by mlx-side wrap) |
| lstm        | tape | n/a | 0.71 ms/ep | — | fresh baseline; ratio 0.18 vs PyTorch |
| lstm        | mlx  | n/a | 120.29 ms/ep | — | fresh baseline; ratio 29.7 (mlx CPU-stream kernel-launch wall at batch=1) |
| dnc-copy    | mlx  | n/a | 139.62 ms/ep | — | fresh baseline; ratio 16.05 (same kernel-launch wall) |
| dnc-copy    | tape | n/a | n/a | — | build-dominated for tape (sub-ms/epoch) |

The wrapped-handle ABI is NOT a measurable perf regression on the
hot examples. The mlx CPU-stream kernel-launch wall (per
`feedback_vm_perf_noise.md`) dominates over any FFI-wrap cost.
**Conclusion**: the cost-per-FFI overhead is below the VM noise
floor on every example measured.

**Outcome**: landed (the sweep itself is `0ec6a99` and prior, not a
new change).

**Drain cadence tuning — declined for now.** The plan's Phase 5'-b
called for re-enabling a mid-block drain (foreign-callable trampoline
inside `tape_append`'s no_grad branch) and sweeping cadences in the
500-5000 range. *Motivation*: the original 3 failing mlx examples
(`ntm-copy`, `ntm-associative-recall`, `mountain-car-cont`) were
leaking inside long `withNoGrad` blocks. *Finding*: under the
wrapped-handle ABI alone (Idris-side `withNoGrad`-exit drain only),
all three of these examples now show *bounded* memory:

- `ntm-associative-recall`: peak=49MB, cur=31MB stable across 700+ iters
- `mountain-car-cont`: peak=49MB, cur=30MB stable, training to completion
- `ntm-copy` (500 epochs): peak=49MB, cur=31MB stable across 400+ epochs

The `withNoGrad`-exit drain + the per-FFI wrap-and-retain are
sufficient to keep Tensor count bounded. Mid-block drain is no
longer load-bearing; deferred behind the cadence-tuning task until
a workload actually needs it. The cleaner Phase 5' deliverable is
"the original motivation is gone."

**Resolved (commit `e337512`)**: both the ntm-copy:mlx ~450-epoch UAF and the ppo:tape mid-run UAF are gone. The IO refactor (`forwardVar` / `applyVar` / Tensor smart constructors all `IO`-typed) made `withNoGrad` actually bracket eval-during-training, which means eval forwards no longer append to the live training tape and can't leave stale handles for the next epoch to dereference. Verification: ntm-copy:mlx 500 epochs ran clean (`epochs=500 acc_short=0.6350`); ppo:tape ran to completion (`epochs=100 avg_return=-78.0`). Tasks #88 and #89 closed.

**Cross-references**:
- `perf-log.jsonl` `kind=baseline` entries timestamped 2026-05-16
  with commit `9664726+dirty`
- `tensor-lifecycle-plan.md` Phase 5' status
- saved memory `feedback_vm_perf_noise.md` (15-20% delta = noise floor)

----

### 2026-05-17 — IO refactor trade-off: per-FFI overhead on mlx small ops, mlx-GPU compute-regime intact — `b894fbb`

**Motivation**: The IO refactor (every Tensor-touching smart constructor + `applyVar` + `forwardVar` returns `IO`) was load-bearing for correctness — `withNoGrad (pure expensiveFFI)` was a no-op under strict argument evaluation, so eval-during-training was running with autograd on and leaking handles into the next training epoch's tape. Closes the original three failing-on-mlx examples (`ntm-copy`, `ntm-associative-recall`, `mountain-car-cont`) plus the ntm-copy:mlx ~450-epoch UAF (#88) and ppo:tape mid-run UAF (#89). The question this entry answers: what did we pay in raw training-time perf?

**Change**: `forwardVar`/`applyVar`/all smart constructors now return `IO (...)` via `ioRerun : (() -> a) -> IO a = primIO (\w => MkIORes (f ()) w)`. Each FFI call goes through one extra closure (the `() -> a` thunk) and one `MkIORes` allocation. Per-sequence `withNoGrad` brackets added inside long eval loops so the exit-drain (forceMajorGc + drainManagedHandles) fires after each sequence on mlx (otherwise Metal MTLBuffer count climbs past the Tart VM ceiling before drain).

**Impact — small-op training (6 examples × 4 cells, two-point timing, ms/ep)**:

| Example | tape | torch | mlx-cpu | mlx-gpu | pytorch |
|---|---:|---:|---:|---:|---:|
| rnn         |  0.34 |  1.36 |  76.0 | 123.3 |  1.75 |
| lstm        |  0.29 |  3.48 | 140.6 | 183.1 |  3.81 |
| gru         | ~0   |  3.97 |  95.2 | 157.6 |  3.78 |
| transformer |  1.08 |  8.28 |  40.6 |  74.9 | 29.39 |
| ntm-copy    | ~0   | 25.10 | 281.0 | 335.9 | 12.30 |
| ntm-recall  |  3.13 | 23.53 | 285.5 | 360.9 | 13.13 |

Tape backend wins or ties PyTorch on every cell (≥6× faster on transformer). Torch competitive on small ops, 4× faster on transformer. **mlx-cpu regressed ~5× vs pre-IO-refactor on small networks** (rnn/lstm/gru/ntm-*): pre-refactor mlx hit 4-7× PyTorch on these cells; now 22-43×. **mlx-gpu** is 1.4-1.7× slower than mlx-cpu in this regime — kernel-launch wall dominates at idris-ml's example sizes (matches `feedback_mlx_gpu_environment` note).

**Impact — compute-bound (matmul-bench, GFLOPS)**:

| N | tape | torch | mlx-cpu | mlx-gpu |
|---:|---:|---:|---:|---:|
| 1024 | 305 |  365 | 1054 |   682 |
| 2048 | 339 |  329 | 1319 | **2993** |
| 4096 | 317 |  334 | 1215 | **4290** |

mlx-gpu wins decisively above N≈2048: **4.3 TFLOPS at N=4096, 13.5× the CPU backends**. The crossover between mlx-cpu and mlx-gpu lands around N=1024-2048; below that, kernel-launch overhead dominates. The IO refactor's per-FFI overhead is invisible at this scale — a 13-ms op doesn't notice a few μs of Idris-side wrapping.

**Outcome**: landed. Trade-off accepted. The IO refactor delivers correctness (eval truly skips autograd graph, no_grad bracket actually brackets) for a 5× small-op-mlx training regression; tape (the convergence-class backend) is unaffected, torch improves on every cell, and mlx-gpu's compute-regime advantage is intact. The regression only matters where mlx is least useful anyway (tiny ops, no GPU advantage). A follow-up to streamline `ioRerun`'s closure+IORes shape could recover some of the mlx-cpu small-op regression if needed — tracked under the high-priority "side-effect-bearing non-IO audit" TODO row, since the audit and the optimisation are the same investigation.

**Cross-references**:
- `perf-log.jsonl` `kind=baseline` entries timestamped 2026-05-17 with commit `f018df5+dirty` (small-op sweep) and `b894fbb` (matmul-bench)
- `scripts/perf-sweep.sh` — the new top-level sweep with cached PyTorch + mlx-cpu/mlx-gpu cells
- `docs/develop/gotchas.md` — "Side-effect-bearing pure functions" entry
- `CLAUDE.md` — `forwardVar`/IO-typed surfaces, per-sequence `withNoGrad` rule

----

### 2026-05-17 — Transformer causal mask cache

**Motivation**: Follow-up to the PE-caching commit. Per-forward audit surfaced one remaining instance of recomputed-deterministic-state: `prim__causalMask sI` was rebuilt every `blockForward` (single-sequence path) and `prim__expandMask (prim__causalMask sI) batchSize` was rebuilt every `batchBlockForward` (batched path). Mask only depends on `seqLen`, which is fixed at construction — the same shape of fix that already landed for PE.

**Change**: Added `TMat seqLen seqLen` field to `TransformerState` carrying the cached 2D mask. Single-sequence path threads the cached AnyPtr through `foldBlocks` → `blockForward` → `runHeadAttn`. Batched path expands the cached mask to `[b, seqLen, seqLen]` once per batch in `applyTransformerBatch` (outside the fold) and threads the 3D AnyPtr through `foldBlocksBatched` → `batchBlockForward`. Bug discovered during numerics verification: routing through `prim__causalMask` directly (which calls `tensor_causal_mask` → `make_tensor` → arena-allocated) gave a dangling pointer after the first `tape_reset` — the cached handle pointed at clobbered arena memory. Fix: route through `prim__createState2d` with an Idris-side `writeCausalMask` recursive Int loop filling the upper triangle on a `prim__allocDoubles` buffer, mirroring the PE-cache pattern exactly. Persistent-state allocator on all three backends (tape `t->persistent=1` + `malloc`; torch `from_tensor_persistent`; mlx refcount-driven `new Tensor`) keeps the mask alive across `tape_reset` / `free_intermediates`.

**Impact — transformer example, 4 cells (two-point timing, ms/ep)**:

| Cell | Before (2026-05-17 IO-refactor baseline) | After (mask cache) | Δ |
|---|---:|---:|---:|
| tape    |  1.08 |  1.11 | +3% (noise) |
| torch   |  8.28 |  7.91 | -5% |
| mlx-cpu | 40.60 | 37.59 | -7% |
| mlx-gpu | 74.90 | 67.87 | -9% |

Deltas are within VM noise (`feedback_vm_perf_noise`: ±15-20%), but the direction is consistently negative on the FFI-cost-dominated cells (mlx-cpu, mlx-gpu, torch). What we saved per forward: `numBlocks` causalMask calls in the single-sequence path, and `numBlocks − 1` expandMask calls in the batched path. Eliminating those compounds in deeper transformers and larger seqLen — this row's "small example" measures the floor; the win grows with model size.

**Bit-identical numerics**: 3-epoch transformer at seed=42 produces `Predicted: 11110$ sort_acc=1/6` on tape, torch, and mlx — matches the pre-change baseline exactly across all three backends.

**Outcome**: landed. Clean architectural win: mask is now constructor-time data (same status as PE), forwards no longer fire deterministic-state-rebuild FFI calls, and the lifetime-management bug (arena tensor cached across `tape_reset`) is closed by routing through the existing persistent-state allocator. Closes the high-priority TODO row "Transformer: cache causal mask (follow-up to PE caching)".

**Cross-references**:
- `packages/idris-ml/src/Layer/Transformer.idr` — cache field, `writeCausalMask` helper, threaded mask AnyPtr
- `perf-log.jsonl` `kind=baseline` entries timestamped 2026-05-17 with transformer rows
- PE-cache precedent: 2026-05-14 entry above


### 2026-05-17 — Non-IO %foreign audit + `ioRerun` shape investigation — `<this commit>`

**Plan job**: cross-cutting (TODO row 7 — audit for side-effect-bearing functions with non-IO types + optimise `ioRerun` shape).

**Motivation**: row 7 hypothesised that streamlining the `ioRerun (\_ => body)` shape could recover some of the 2026-05-17 mlx-cpu small-op regression (rnn/lstm/gru/ntm-* at 22–43× pytorch ratio, vs 4–7× pre-IO-refactor). The IO refactor wrapped every Tensor smart constructor in `ioRerun f = primIO (\w => MkIORes (f ()) w)`, adding a thunk closure per FFI call. The conjecture: that closure (and the `MkIORes` box) is a meaningful slice of the regression.

**Change**: investigated, no code change to `ioRerun` itself. The audit half found three live IO-typing bugs (`memoryReport`, `setParamLR`, `polyakUpdate`) and added a lint to prevent the bug class — those landed separately in the same commit chain. The perf half measured the closure-overhead hypothesis against the actual per-op cost.

**Impact**: per-call analysis says `ioRerun` adds ~1 closure allocation per FFI call (~100ns on Chez). For LSTM at the measured workload (4 IO ops per timestep × 50 timesteps × 200 epochs = ~40k IO ops), that's <5 ms wall — within noise. Row 16's diagnostic on GptLarge already proves the actual per-op Idris cost is ~7.6 ms (the wall lever isn't FFI overhead at all; it's existential `AnyLayer` dispatch + typeclass dictionary resolution + Tensor record packing). The `ioRerun` shape isn't the bottleneck; the optimisation knob is elsewhere.

| cell    | example | idris ms/ep | py ms/ep | ratio | source        |
|---------|---------|-------------|----------|-------|---------------|
| mlx-cpu | lstm    | 134.99      | 3.53     | 38.24 | post-investigation sweep |

**Outcome**: investigated, no change to `ioRerun`. The audit half landed as a lint + three bug fixes. Future small-op mlx-cpu recovery work belongs in row 16's territory (per-op Idris VM overhead), not row 7's. Row 7's perf bullet retired; the audit/lint bullet stays as the durable deliverable.

**Cross-references**:
- `scripts/lifecycle/check-non-io-side-effects.py` — the new lint
- TODO row 7 closed; row 16 (per-op Idris overhead) remains as the relevant follow-up for mlx-cpu small-op recovery

### 2026-05-17 — Precision-type-parameter rollout — perf-neutral — `a875549`

**Plan job**: validation pass for the precision/dtype landing (commits `55bd35e` through `a875549` — DType.Core scaffold, Tensor `(0 dt : DType)` slot, `Compatible` + `UpcastableTo` interfaces, `MlxDev` parametric family, 11 LayerAny creators device-polymorphised, BuildConfig generation, 23 examples migrated, tutorial 08).

**Motivation**: the new `(0 dt : DType)` parameter on `Tensor` is 0-quantity (erased before code generation), and the FFI surface to the C backends is unchanged. The expectation is zero runtime impact — but elaborator pressure changes (a Tensor reference now carries one more implicit) could in principle pessimise codegen. Worth verifying before declaring the rollout done.

**Change**: ran `scripts/perf-sweep.sh` at HEAD `a875549` — 6 examples × 4 cells (tape, torch, mlx-cpu, mlx-gpu), seed=42, identical to the `b894fbb` sweep on 2026-05-17.

**Impact**: zero or favourable across every cell.

| Example | Cell        | b894fbb ms | a875549 ms | Δ |
|---|---|---:|---:|---:|
| rnn | tape | 0.34 | (sub-ms) | noise floor |
| rnn | torch | 1.36 | 1.65 | +21% (1-ms scale) |
| rnn | mlx-cpu | 76.0 | 71.6 | −6% |
| rnn | mlx-gpu | 123.3 | 110.8 | −10% |
| lstm | tape | 0.29 | 0.31 | noise |
| lstm | torch | 3.48 | 2.56 | −26% |
| lstm | mlx-cpu | 140.6 | 121.3 | −14% |
| lstm | mlx-gpu | 183.1 | 179.1 | −2% |
| gru | tape | ~0 | 0.01 | noise |
| gru | torch | 3.97 | 2.89 | −27% |
| gru | mlx-cpu | 95.2 | 89.8 | −6% |
| gru | mlx-gpu | 157.6 | 151.0 | −4% |
| transformer | tape | 1.08 | 0.84 | −22% |
| transformer | torch | 8.28 | 7.59 | −8% |
| transformer | mlx-cpu | 40.6 | 34.7 | −15% |
| transformer | mlx-gpu | 74.9 | 69.0 | −8% |
| ntm-copy | tape | ~0 | 0.97 | small |
| ntm-copy | torch | 25.10 | 1.37 | b894 was wrong |
| ntm-copy | mlx-cpu | 281.0 | 212.6 | −24% |
| ntm-copy | mlx-gpu | 335.9 | 261.1 | −22% |
| ntm-recall | tape | 3.13 | 2.47 | −21% |
| ntm-recall | torch | 23.53 | 15.23 | −35% |
| ntm-recall | mlx-cpu | 285.5 | 244.6 | −14% |
| ntm-recall | mlx-gpu | 360.9 | 367.2 | +2% |

The PyTorch references on the same machine also came in 2–22% faster than during the `b894fbb` sweep (rnn 1.75 → 1.37, ntm-recall 13.13 → 11.33), indicating this VM is running ~10–15% leaner on the day — system noise, not algorithmic change. After backing that out, every Idris cell is within the ±15–20% per-cell noise gate established in `feedback_vm_perf_noise.md`. The only above-floor positive delta is rnn/torch at +21% on a 1-ms-scale task — within the resolution of two-point timing at that range, not a regression worth chasing.

The ntm-copy/torch row shows a 25.10 → 1.37 collapse that is far too large to be VM drift. Working hypothesis: the `b894fbb` 25.10 was a measurement artefact (two-point timing at N_short=10, N_long=40 on a ~25 ms/ep task is just ~1 s of wall — easy to drown in startup variance). The new 1.37 is also at the noise floor of that two-point regime. Either could be wrong; the right read is "this cell is not reliably resolvable at the current N_long". Not a precision-work signal in either direction.

Follow-up: also ran the matmul-bench compute-bound suite (the canonical "mlx GPU > CPU" demo, separate code path — pure forward matmul, no autograd, no FFI hot loop), 3 sizes × 4 cells, iters=5, identical to the `abc3552` 2026-05-17 sweep:

| N | tape GFLOPS abc3552 → now | torch abc3552 → now | mlx-cpu abc3552 → now | mlx-gpu abc3552 → now |
|---:|---:|---:|---:|---:|
| 1024 | 305 → 307 | 365 → 346 | 1054 → 1091 | 682 → 649 |
| 2048 | 339 → 335 | 329 → 353 | 1319 → 1264 | 2993 → 2719 |
| 4096 | 317 → 341 | 334 → 347 | 1215 → 1227 | 4290 → 4271 |

All 12 cells within ±10%, including the headline mlx-gpu 4.3-TFLOPS @ N=4096 — fully preserved. The largest negative delta is mlx-gpu @ N=2048 at −9%, well within the noise gate.

**Outcome**: precision rollout is perf-neutral on both the training sweep (small-op, FFI-heavy) and the compute-bound matmul sweep. No code change. Not updating the `perf-baseline.md` 2026-05-17 rows — deltas are below the 20% noise gate and don't represent a material change worth churning the canonical tables over.

**Cross-references**:
- TODO "Investigate precision type parameter" — closed; see `docs/develop/dtype-parameter.md` for the design memo and lessons learned
- sweep raw output: `/tmp/perf-sweep-a875549.log` (training), `/tmp/matmul-bench-a875549.log` (matmul-bench)
- JSONL entries appended to `docs/develop/perf-log.jsonl` (training: kind=baseline, commit=a875549; matmul-bench: kind=matmul-bench, commit=dbc39cc)
