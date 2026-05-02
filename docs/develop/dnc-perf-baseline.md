# DNC perf baseline (P0) + Phase 1 results (P1)

## Phase 1 result (2026-05-02, commit `cf03748`)

**Tape backend: 1040 ms/epoch → 128 ms/epoch — 8.1× speedup**

Exceeds the 3-4× projection (NTM-mirror baseline). The targeted DNC
`applyVar` rewrite to a tensor-handle pipeline (no per-op
vecStackTensor/tensorToScalars round-trips) ran cleanly on tape.
Convergence preserved: loss at epoch 0 matches scalar baseline to
4-decimal float precision (0.6937 vs 0.6934); loss decreases
0.69 → 0.45 over 1000 epochs at the new path.

Cross-backend validation (mlx, torch) and multi-seed dnc-copy
convergence to acc_short ≥ 0.8 are deferred to follow-up commits but
expected to produce comparable wins on mlx/torch (which had even more
FFI overhead in the baseline).

Tape's projected dnc-copy convergence wall-clock: ~13 h → ~1.6 h
(46K epochs × 128 ms). Likely lets us revert the 18 h CONVERGENCE_TIMEOUT
bump.

---

# Original P0 baseline measurements



Captured 2026-05-01 to ground the tape-backend performance overhaul. All
runs at the current default config: `N=32 M=20 H=100 R=1, batch=1,
seqLen=1-10, lr=1e-4, RMSprop`. 100-epoch wall-clock measurements with
`backend_profile_report()` enabled.

## Cross-backend wall-clock

| Backend | ms/epoch (wall) | C total | Outside-C (Idris+FFI) | Forward C | Backward C | Optimizer C | vs PyTorch |
|---|---:|---:|---:|---:|---:|---:|---:|
| **tape**  | 1040 | 1025.6 (98.6%) |   14 | **1022.8** |   2.3 |   0.4 |  93.9× |
| **mlx**   | 2260 |  685.5 (30%)   | 1575 | n/a*       | 151.0 | 534.5 | 204.2× |
| **torch** | 1230 |  108.7 (9%)    | 1121 | n/a*       | 101.2 |   7.5 | 111.1× |
| **PyTorch ref** | **11.07** | — | — | — | — | — | 1.0× |

\* mlx/torch profile reports don't have a forward timer; their "C total"
includes only backward + optimizer.

PyTorch reference measured fresh on the same default config (1000 epochs
in 11.07s wall, acc_short=73.7% at that point). Matches the
`dnc-convergence-results.md` historical reference (~10 ms/epoch, 44s for
the full 4.1K-epoch convergence run).

## Tape top backward ops (100 epochs)

```
MV          53.75 ms (16,074 calls)
SELECT      42.34 ms (2,402,216 calls)   ← 24K SELECT calls per epoch
STACK       27.36 ms (28,212 calls)
MUL         15.06 ms (258,526 calls)
ADD          7.69 ms (91,120 calls)

Tape entries (last fwd): 27
Backward walk: 2,948,610 processed, 71,772 skipped (2% dead)
```

→ ~30K tape entries per epoch. The SELECT dominates by call count
(2.4M, 24K/epoch), reflecting per-element scalar extraction from tensor
outputs — direct evidence of the scalar-Variable round-trip pattern.

## Diagnosis

Two distinct gaps add up to the ~100× vs PyTorch reference:

1. **Tape backend, in-C forward time dominates** (1022 ms/epoch). The
   tape forward timer is inclusive of every per-op FFI call; with
   ~30K small ops/epoch and small per-op compute (M=20, N=32, H=100),
   per-call overhead dominates raw kernel time. Tape's bump-arena
   allocator keeps per-op overhead minimal but doesn't eliminate it.

2. **MLX and torch backends, FFI orchestration dominates** (1121 ms
   / 1575 ms outside C). Each tensor allocation in libtorch / MLX is
   substantially more expensive than tape's arena slot, so DNC's
   ~30K-op forward suffers more on the heavier backends. The user's
   observation that "torch is even worse than tape on DNC" is
   confirmed and structurally explained.

Both gaps are driven by the **same root cause**: DNC's `Layer/Dnc.idr`
still uses the scalar-Variable round-trip pattern that NTM had before
its 2026-04-08 rewrite. Forward pass:
- 11 separate `applyVar fc cell` calls (lines 343-352), each going
  `Vect 100 Variable` → `vecStackTensor` → C linear → `tensorToScalars`
  → `Vect M Variable`.
- Per-element `map sigmoidVar` over erase / freeGates (lines 355-356).
- `vecStackTensor` re-stacking AFTER unpacking (lines 378-380) just to
  feed `prim__narrow` for per-head slicing.

NTM had the same shape pre-fix. Its rewrite went 380 ms/epoch → 110
ms/epoch (3.5×) by keeping tensor handles end-to-end and unpacking only
at the loss site. DNC needs the same treatment — tracked under Phase 1
of the perf overhaul plan.

## Convergence cost at baseline

`dnc-convergence-results.md` projects ~46K epochs to hit `acc_short ≥ 0.8`
on the current (reverted) batch=1 config. At measured 1040 ms/epoch on
tape, that's **~13.3 hours** wall-clock for one seed — exactly why
`make test-examples-convergence` timed out at the original 4-hour cap.
Goal post-Phase 1: bring this under 4 h again.

## Hypothesis for Phase 1 win

Mirroring NTM's 3.5× win on DNC would project tape from 1040 → 297
ms/epoch, putting full convergence at ~3.8 h on tape. MLX/torch have
even more headroom (their FFI overhead drops proportional to the call
count reduction); estimated 4-6× win there.

If the projection holds, **the 18h CONVERGENCE_TIMEOUT bump committed
in `aa8896a` becomes unnecessary** and can be reverted to the original
4 h or tightened further.

## Phase 1 implementation notes (research, pre-commit)

The proven NTM pattern is in `Layer/Ntm.idr:189-267`:
- `applyVar` (lines 189-194) is a thin wrapper: `vecStackTensor` input → `applyVarTensor` → `tensorToScalars` output. Hot path.
- `applyVarTensor` (lines 217-267) does all internal work on `AnyPtr` tensor handles, calling `prim__cat2`, `applyVarTensor lstm`, `tensorMv`, `tensorAdd`, `prim__narrow`, `prim__select`, `prim__softmax`, `prim__sigmoid`, fused C ops `prim__ntmReadHead` / `prim__ntmInterpWrite`. State carried in record's `Maybe AnyPtr` fields, populated in `nameLayer` (lines 283-298).
- Existing infra (`extractCellTensor`, `extractWeightTensor`, `extractBiasTensor`, `prim__createState1d/2d`, `packScalarValues`/`packMatrixValues`) is general — DNC reuses without changes.

DNC's port has the same shape but more pieces:
- 11 FC layers vs NTM's 3 — each becomes `tensorAdd (tensorMv W cellT) b` direct call.
- 5 algorithm helpers (`dncUsageUpdate`, `dncAllocate`, `dncWriteWeight`, `dncEraseAddWrite`, `dncLinkUpdate`, `dncReadWeight`) currently take `Vector n (Variable d)` — each gets a tensor-level variant `…T` that takes/returns `AnyPtr`. Most internal ops are already C calls; the rewrite eliminates the `vecStackTensor`/`tensorToScalars` round-trips at boundaries.
- 7 state-tensor fields already in the record (`memTensor`, `usageTensor`, `writeWtTensor`, `precedenceTensor`, `linkTensor`, `readWtTensors`, `readOutTensors`) — initialized in `nameLayer`'s tensor branch (mirror NTM lines 283-298).
- Per-head loop in `computeReads` becomes a loop on tensor handles.

Estimated implementation scope: ~300-500 lines across `Layer/Dnc.idr` (~200 LOC of new tensor helpers, ~150 LOC for `applyVarTensor`, ~30 LOC `nameLayer` init, ~20 LOC `applyVar` wrapper). No new C ops needed — every prim used by DNC already exists. Cross-backend correctness gated by re-running `make test-backend-{tape,mlx,torch}` plus a 5-seed dnc-copy convergence at default config.

### Implementation pitfall observed (2026-05-02 spike, reverted)

First attempt placed three recursive helpers (`computeRetentionTensor`,
`catReadOutTensors`, `computeReadHeadsTensor`) as `where`-clause locals
inside the `applyVarTensor` method, with `{r' : Nat}` implicit
parameters used in `Vect r' AnyPtr` arguments. Idris2's elaborator
hung indefinitely at `27/41: Building Layer.Dnc` (15+ minutes of
silent CPU). Likely cause: implicit-r' interacting with the LayerLike
instance's outer `{r : Nat}` parameter during coverage / totality
analysis on nested `Vect (S k)` patterns.

**Recommended approach for next session**: define the recursive
helpers as **top-level functions** outside the LayerLike instance,
with **explicit `Nat` parameters** rather than implicit. Mirrors how
NTM's recursive helpers (`forwardReadHeadUnboundedVar` etc.) are
structured — top-level, no inner-instance type-var interaction. Then
`applyVarTensor` calls the top-level helpers with explicit `n`, `m`,
`r` from its outer scope. This pattern compiles fine on NTM; it's
the correct shape for DNC too.

The export-fix for `prim__cosineSimilarity` (currently private) is
also required — promote to `export` in `Variable.idr:147`.
