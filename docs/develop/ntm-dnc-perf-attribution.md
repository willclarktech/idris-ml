# NTM / DNC perf attribution (Phase 0)

Per-op forward profiling of NTM and DNC on the tape backend, captured
2026-05-07. Establishes where time is going so Phase 1 can target the
biggest lever.

The tape backend profiler was extended (`prof_forward_per_op[]` in
`packages/backends/backend_tape.c`) to attribute the wall-clock gap
between consecutive `tape_append` calls to the op being recorded —
the same pattern as the existing `prof_backward_per_op`. It surfaces
via the existing `backend_profile_report()` call, which `Example.Profile`
now invokes after a 5-epoch warmup and 10-epoch timed window.

## NTM-copy default (N=128 M=20 H=100 batch=16, seqLen 1-20)

Forward: **488 ms/epoch** baseline — confirms the documented
post-Path-C regression vs the pre-merge ~228 ms/epoch.

Top forward ops:

| Op | Total (ms / 10 ep) | Calls | µs/call |
|----|---:|---:|---:|
| **MV** | **2774** | **16560** | **167** |
| CAT | 281 | 6480 | 43 |
| ADD | 221 | 28224 | 7.8 |
| SELECT | 148 | 19872 | 7.4 |
| SOFTMAX | 104 | 13248 | 7.8 |
| MUL | 140 | 19872 | 7.0 |
| ADD_S | 137 | 19872 | 6.9 |
| NARROW | 134 | 16560 | 8.1 |
| SOFTPLUS | 100 | 13248 | 7.6 |
| VECMAT | 56 | 6624 | 8.5 |

**MV alone is 60 % of forward time** at 167 µs/call — far higher
than the working hypothesis of "softplus is the regression source"
that the prior memory file recorded.

### Where the 167 µs/MV-call goes

Inline timers inside `tensor_mv` reveal the body itself is < 1 µs:

| Stage | Total (ms / 27380 calls inc. warmup) | µs/call |
|---|---:|---:|
| `arena_alloc` output buffer | 0.5 | 0.02 |
| `cblas_dgemv` kernel | 19.3 | 0.7 |
| `make_tensor_arena` | 5.4 | 0.2 |
| backward-meta save | 3.2 | 0.1 |
| **C body total** | **~28** | **~1.0** |
| **"Glue" — gap from previous tape_append to entering tensor_mv** | **4078** | **149** |

99 % of MV's profiled cost is **Idris-side execution between two
consecutive C calls**, not the C kernel.

### Why MV's glue (149 µs) is wildly higher than ADD/MUL's (~6 µs)

ADD/MUL/SELECT calls happen immediately after another C call returns
— minimal Idris work between them.

MV calls in NTM happen at FC boundaries where there's substantial
Idris-side work between the previous op's `tape_append` and the
`prim__mv` invocation:

```idris
-- After LSTM forward (last tape_append is inside LSTM):
let cellPtr = case updLstm.cellT of
                Just c => c.tensorPtr
                Nothing => idris_crash "..."
    -- Sub-layer weight tensor handles
    rfcW = readFc.weightT.tensorPtr   -- 5 record-field accesses
    rfcB = readFc.biasT.tensorPtr
    wfcW = writeFc.weightT.tensorPtr
    ...
    mI = cast {to=Int} m              -- Nat → Int cast
    skI = cast {to=Int} ShiftKernelSize
    -- Read FC: cell -> [ReadParamWidth m]
    readResultT = prim__add (prim__mv rfcW cellPtr) rfcB
```

The gap from "LSTM's last tape_append" to "tensor_mv entry" is the
time for: `case updLstm.cellT`, several record-field accesses, two
`Nat → Int` casts, and a Chez Scheme foreign-call dispatch.

This matches the pre-Path-C-vs-post-Path-C story:

- Pre-Path-C scalar Variable path: less typed-record machinery,
  intermediate state passed as flat values. Per-FFI-call glue
  was small.
- Post-Path-C tensor path: typed `LstmState` / `NtmState` records
  with `Maybe (Tensor [n,m] ex)` fields. Each access path through
  the typed surface adds Idris-side work.

The 2× forward regression (228 → 488 ms/epoch) ≈ the increased
per-call glue overhead at every MV site.

## DNC-copy default (N=32 M=20 H=100 batch=1, seqLen 1-10)

Forward: **119 ms/epoch** at the small (reverted) config.

Top forward ops:

| Op | Total (ms / 10 ep) | Calls | µs/call |
|----|---:|---:|---:|
| **MUL** | **1021** | **2097** | **486** |
| MV | 70 | 1755 | 40 |
| ADD | 25 | 2394 | 10 |
| CAT | 14 | 342 | 41 |
| SUB | 9 | 927 | 9 |
| CLAMP | 5 | 585 | 8 |
| RESHAPE | 5 | 576 | 8 |
| SIGMOID | 4 | 468 | 8 |
| SELECT | 3 | 351 | 9 |
| SOFTPLUS | 3 | 234 | 12 |

**MUL is 86 % of forward time at 486 µs/call.** Same root cause
pattern as NTM's MV: in DNC's addressing logic, MUL calls happen
between record-heavy operations (link-matrix slicing, per-head
loops over `Vect r AnyPtr`, usage / precedence updates), so the
per-MUL Idris glue is enormous.

DNC also has 10 FC layers all consuming the same LSTM cell input
(`writeKey`, `writeBeta`, `eraseFc`, `addFc`, `freeGates`,
`allocGate`, `writeGate`, `readKeys`, `readBetas`, `readModes`).
Each is currently a separate `prim__add (prim__mv W x) b`. This
is the highest-leverage fusion target on the DNC side.

## NTM-copy small config (N=32 M=10 H=32 batch=1)

Per-call MV cost drops to **83 µs** at this config — 84 µs less than
default. The fixed per-call overhead is **~80 µs** independent of
matrix size; the kernel itself scales with size but is small.

This decomposition matters for Phase 4 fallback: shrinking N/M/H
helps but only until the per-call FFI floor dominates.

## Spike experiments (within Phase 0 to validate the lever)

| Spike | Forward ms/epoch | Δ vs baseline |
|---|---:|---:|
| Baseline (post-Path-C, no changes) | 488 | — |
| `tensor_mv` heap → arena | 466 | −22 (−5 %) |
| `tensor_mv` cblas → hand-rolled loop | 590 | +102 (worse) |
| Above + `tensor_linear` heap → arena + `prim__linear` exposed + 3 NTM FC sites fused | 462 | −26 (−5 %) |

`acc_short@100 = 0.7054` matches the documented post-Path-C baseline
bit-for-bit. The fusion is numerically equivalent.

The single-figure spike on three NTM FC sites doesn't move the needle
much because the dominant remaining cost is **the per-FFI-call glue
itself**, not the kernel work. Fusion helps where the glue cost is
the cost we're saving (one call vs two). Read/write addressing's
remaining `prim__matmul` calls still cost 380 µs/call — they're
called in NTM's read-head body which is full of record destructuring
and 2D ops in between.

## Top-3 perf levers (ranked by projected wall-clock impact)

### 1. Op fusion at every Linear-style site (recommended Phase 1)

Wherever the codebase has `prim__add (prim__mv W x) b` or
equivalent, replace with `prim__linear W x b`. One C call instead of
two; the Idris-side glue collapses from "execute mv → execute add"
to "execute linear" — and there's only one tape entry, so the
backward replay is also faster.

Apply systematically:
- `Layer/Linear.idr` — the canonical apply site for every FC.
- `Layer/Ntm.idr` — 3 FC sites (read FC, write FC, output FC).
- `Layer/Dnc.idr` — 11 FC sites in the controller.
- `Layer/Lstm.idr` — gate computation
  (`Wih @ x + Whh @ h + bias`) is a candidate for a slightly more
  complex fused op (`tensor_lstm_gates_fused` or two consecutive
  `tensor_linear`s that share a tape entry).

Estimated impact: 10–15 % on NTM-copy, 5–10 % on DNC, 1–3 % on
non-recurrent examples (their per-FC overhead is amortized across
batches).

Risk: each fused linear takes a 3-arg FFI signature. Per Chez foreign-call
dispatch, 3-arg may be slower than 2-arg by a small constant — need to
confirm via measurement after the change. The Phase-0 spike of 3 NTM
sites already showed net win (462 < 488), so the constant must be smaller
than the savings.

### 2. Batched FCs in DNC controller (filed as Phase 2b in the plan)

`Layer/Dnc.idr` registers 10 separate per-gate FCs all from the same
LSTM cell input. One fused linear (`h → sum of head dims`) +
`prim__narrow` splits is structurally cheaper:

- 10 `prim__mv` + 10 `prim__add` → 1 `prim__linear` + 10 `prim__narrow`
- ~10× FFI overhead reduction at the DNC controller block

Estimated impact: 25–40 % on DNC at default config.

Risk: each FC has its own bias and weights. The fused weight is the
vstack of individual weights; the fused bias is the concat. Need to
construct at `dncLayer` time. Gradient correctness is automatic via
`prim__narrow`'s existing backward rule.

### 3. DNC `zeroDiag` + 2D-pass-through (filed as Phase 2a / 2c)

Already-filed sub-items targeting DNC's per-element FFI cascades.
- `zeroDiag` allocates an N² mask via Scheme-recursive `prim__setDouble`
  per timestep. At N=128: ~5 M scalar FFI calls/epoch, each going
  through Chez foreign-call dispatch.
- `buildMatrixRows` round-trips through scalar Variables to assemble
  per-row tensors.

Estimated impact: 15–30 % on DNC at the PyTorch-aligned N=128
batch=16 config (currently can't run end-to-end on tape).

## Phase 1 decision

**Attack lever 1 first** (op fusion at every Linear-style site).
It's the lowest-risk change, exposes the right pattern for lever 2,
and has the broadest applicability across the codebase (every
non-NTM/DNC example with a Linear layer also benefits).

After lever 1 lands and is verified multi-seed across 3 backends,
proceed to lever 2 (batched DNC FCs). Lever 3 follows lever 2 as
filed in the plan.

## Phase 1 result (committed 2026-05-07)

Applied `prim__linear` (typed `tlinear` wrapper) systematically:

| Layer | Sites | Pattern |
|---|---:|---|
| `Layer/Linear.idr` | 1 | `tadd (tmv W x) bias` → `tlinear W x bias` |
| `Layer/Ntm.idr` | 3 (Phase 0) | read FC, write FC, output FC |
| `Layer/Dnc.idr` | 11 | 10 controller FCs + output FC |
| `Layer/Lstm.idr` | 1 | nested: `tlinear rw h (tlinear iw x bT)` (4 → 2 FFI) |
| `Layer/Rnn.idr` | 1 | same nested pattern (4 → 2 FFI) |
| `Layer/Gru.idr` | 2 | input-side and hidden-side gates (4 → 2 FFI) |

Cross-backend smoke gate green.
`example-ntm-copy seed=42 100 epochs acc_short=0.7053541666666666`
unchanged (bit-identical).

Wall-clock measurements (10-epoch profile, tape backend):

| Benchmark | Pre-Phase-1 | Post-Phase-1 | Δ |
|---|---:|---:|---:|
| NTM-copy default (N=128 batch=16) | 488 ms/epoch | **436 ms/epoch** | **−11 %** |
| DNC-copy small (N=32 batch=1) | 119 ms/epoch | **97 ms/epoch** | **−19 %** |

DNC's larger relative win confirms the prior — DNC has more FC
sites per timestep (11) than NTM (3) so the per-call overhead
reduction compounds further.

### What's still left

Top forward op on NTM-copy is now `LINEAR` at **163 µs/call**, ~the
same as MV's pre-Phase-1 167 µs/call — **the per-FFI-call overhead
floor hasn't moved**. Phase 1 reduced *call count*, not per-call
cost.

## Phase 2a (NTM read+write FC fusion) — abandoned

Tried fusing NTM's read FC + write FC into a single
`prim__linear` of combined width `ReadParamWidth m +
WriteParamWidth m` followed by 2 narrows. Math is identical;
seed=42 numerics bit-identical (preserved via per-half xavier
draws in original RNG order).

**Result: ~5 % regression** (448 → 473 ms/epoch tape, 3-run avg).

Hypothesis test failed. The Idris-side glue between two
*consecutive* `prim__linear` calls in the NTM forward (read FC →
narrows → write FC, both consuming `cellPtr`) turns out to be
small enough that:

- Saving = 1 fewer `prim__linear` (~30 µs of glue + 1 µs body)
- Cost  = 2 added `prim__narrow` (~16 µs total) + ~50 µs added
  per-call cost on the now-bigger fused matrix (`[72, 100]` vs the
  separate `[26, 100]` + `[46, 100]`).

Net loss. The fusion is reverted.

**Lesson**: per-FFI-call overhead reduction works for ops that
weren't *already* consecutive (Phase 1's `mv + add → linear` —
two different functions with two different Idris-side wrappers).
It does NOT work for ops that are already syntactically adjacent
in Idris (two consecutive `prim__linear` calls share the same
call-site overhead pattern; fusing them doesn't remove glue
that wasn't there).

## Phase 2b (DNC 10-FC fusion) — not attempted

The same dynamic likely applies to DNC's 10 consecutive controller
FCs. The per-call savings would be small (each FC's glue is
already the "second consecutive call" pattern), and the added
narrow cost would offset most of it. Risk of regression matches
NTM Phase 2a. Skipped until a more targeted measurement justifies
it.

## Microbench-based investigation (committed in this session)

Built `Example.ProfileMicro` (run via `make example-profile-micro`)
that calls primitives in tight Idris loops with minimal surrounding
work. **The 165 µs/LINEAR attribution from the C profiler is an
artifact, not a real per-FFI-call cost.**

Tight-loop per-call cost (NTM-copy default H=100, with grad-tracked
params so tape_append fires every call):

| Bench | Per-call | Notes |
|---|---:|---|
| `prim__add` (1d, W=26) | 7 µs | Floor for any 2-arg primitive |
| `prim__mv` (W=26, H=100) | 9 µs | + cblas_dgemv on a small matrix |
| `prim__linear` (mv+add fused) | 10 µs | Slightly larger 3-arg FFI |
| `tlinear` (typed wrapper) | 10 µs | No overhead from typed surface |
| `Layer.Linear` `applyVar` | 10 µs | LinearState record extract free |
| `applyLstm` (2 linear + 1 lstm_gates_pair) | 59 µs | ~20 µs per internal prim |
| `applyNtm` at default dims (n=128 m=20 h=100) | 574 µs | ~30 prims internally → ~19 µs/prim |

So **per-prim cost averaged across a real layer's internal
sequence is ~19 µs**, comprising:
- ~10 µs FFI dispatch + tape_append + arena alloc + LinearMeta
- ~9 µs of Idris-side glue per prim (let bindings, record
  destructuring, the MkTensor wrap/unwrap dance, etc.)

The C profiler's "163 µs/LINEAR" is an **attribution artifact** of
the time-between-tape_appends model: it sums up *all* Idris work
between two consecutive tape_appends and credits it to whichever
op's tape_append closes the interval. When LINEARs sit at major
code-path boundaries (after LSTM cell extraction, after read-head
logic, after concat-output-prep), they get credited with a huge
chunk of glue from many other ops that don't themselves fire a
tape_append.

This has a sharp implication for the Phase 1+2 fusion strategy:
**fusion only helps if it removes a FULL prim__ call's worth of
work** (~19 µs). It doesn't reduce per-call cost — it reduces call
count.

Phase 1 (mv + add → linear) saved 19 µs per fused pair (one fewer
prim) — real win. Phase 2a (two consecutive prim__linears → one
prim__linear + 2 narrows) saved one prim (19 µs) but added two
narrow prims (~40 µs combined) — net regression. Confirmed.

## What's left to try (next session)

1. **Reduce the number of prims per applyNtm**. The ~30 prims/
   timestep is the dominant cost. Each prim is ~19 µs total, so
   eliminating any one prim saves ~19 µs/timestep. Multi-FC
   fusion failed because the added narrow prims cost more than
   the saved linear. Better targets:

   - Fuse `cosine_similarity + mul + softmax` (the content
     addressing chain in `ntmReadHeadIdris`) into one C op.
     Saves 2 prims per call to ntmReadHeadIdris; called twice
     per applyNtm → 4 prims saved per timestep → ~76 µs/timestep
     ≈ 25 ms/epoch saved. PyTorch convert would not have an
     exact equivalent but `F.softmax(beta * F.cosine_similarity)`
     is a recognisable enough pattern.

   - Fuse `pow + sum + div` (sharpening + normalize at end of
     `ntmReadHeadIdris`) into a single sharpening op. Saves
     ~2 prims per call, ~38 µs/timestep ≈ 13 ms/epoch.

   - Both of these are NTM-specific. Per the
     architecture-specific-fused-ops principle, only worth doing
     if a PyTorch convert would expect them. They're sub-paper-
     specific (used in NTM, MANN, DNC, sparse-attention) so it's
     borderline. Decision deferred.

2. **Reduce the per-prim Idris-side glue cost** (~9 µs/prim).
   This is harder — it's the cost of the typed surface itself
   (record wrap/unwrap, let-bindings). Options:
   - Idris codegen improvements (upstream)
   - Replacing typed Tensor wrappers with raw AnyPtr in hot paths
     — explicitly trades back Path-C's type safety. Last resort.

3. **Accept the ~11 % Phase 0+1 win**. The pre-Path-C 228 ms/
   epoch baseline used the scalar-Variable path which we
   traded away for type safety. Recovering it without giving up
   Path C requires fusion of NTM-architecture-specific patterns
   (see option 1) — which is the same kind of architecture-
   specific fused C op we just removed.

   Conclusion this session: **the per-FFI-call overhead has a
   floor of ~19 µs that's hard to reduce further without
   architecture-specific fusion or codegen-level work**. This is
   a structural limit of the current typed-surface design, not a
   bug.

The microbench infrastructure (`Example.ProfileMicro`) is committed
so future sessions can re-test these floors quickly.

## Implementation note

The tape backend already had `tensor_linear` (1D fused linear) but it
wasn't exposed via Idris FFI — only `tensor_linear_2d` (batched, for
RL examples) had a `prim__linear2d` binding. Phase 0 added the 1D
`prim__linear` binding in `Tensor.idr`. The C `tensor_linear`
implementations exist in all three backends (`backend_tape.c:942`,
`backend_mlx.cpp:612`, `backend_torch.cpp:221`), so there's no
backend work needed beyond what already exists.

## What this measurement does NOT tell us

- **PyTorch-side per-op timing.** We compared totals against
  PyTorch's documented numbers but didn't run `torch.profiler` for
  op-level cross-reference. If the post-fusion ratio is still ≫ 1×
  vs PyTorch, we'll need that next.
- **mlx / torch tape timing.** mlx and torch have their own replay /
  native-autograd paths. The forward-glue regression may not exist
  there since their typed-record traversal is different. Multi-seed
  smoke after Phase 1 will reveal cross-backend ratios.
- **Whether reducing Idris glue (vs reducing call count) is also
  feasible.** The glue is part of the typed-record surface that Path
  C committed to. Refactoring it without giving up type safety is a
  larger investigation; the current plan addresses the symptom (call
  count), not the cause (per-call glue). If lever 1+2+3 don't close
  enough of the gap, the cause-side investigation becomes Phase 1B.
