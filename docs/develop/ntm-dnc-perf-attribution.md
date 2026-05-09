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
  with `Maybe (Tensor [n,m] d)` fields. Each access path through
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
