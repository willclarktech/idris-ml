# DNC Convergence Results

> **2026-04-29 update**: Both Idris and PyTorch reverted to N=32, batch=1, max-len 10 (copy) / batch=1 (recall). The previous N=128 batch=16 alignment proved untenable on the Idris tape backend (~5 min/epoch vs PyTorch's 276 ms/epoch). See `docs/develop/reference-alignment.md` for the alignment-policy rationale and `TODO.md` for the layer-perf work that would re-enable PyTorch's prior config.

## PyTorch Reference (Oracle)

Configuration: N=32, M=20, H=100, R=1, lr=1e-4, RMSprop (alpha=0.95, momentum=0.9), clip=10.0, **batch=1**, seed=42.

### Copy Task (seqLen 1-10)

Converged at **4,100 epochs** via early stopping (windowed avg < 0.01).

| Metric | Result |
|--------|--------|
| Short (len 1-5) | 100.0% |
| Full (len 1-20) | 81.8% |
| Time | 44s (10ms/epoch) |

Loss trajectory: 0.63 → 0.61 → 0.49 (epoch 1500, breakthrough) → 0.001 → converged.

Note: full-length eval is on len 1-20 sequences, but training caps at len 10 — the 81.8% full reflects how well the in-distribution-trained model generalizes; copy short (the in-distribution metric) is fully solved at 100%.

### Associative Recall (items 2-6, seqLen=3)

Ran **50,000 epochs** (no convergence trigger — sporadic loss spikes kept windowed avg above the 0.01 threshold even though pointwise loss was usually below).

| Metric | Result |
|--------|--------|
| K=2 items | 100.0% |
| K=4 items | 99.7% |
| K=6 items | 93.3% |
| Time | 22m 48s (27ms/epoch) |

Loss trajectory: 0.69 → 0.63 → 0.27 (epoch ~12500, breakthrough) → 0.0005 with periodic spikes → ran out 50K-epoch budget.

### Historical (N=128, batch=16, pre-2026-04-29)

The previous PyTorch oracle ran at the larger config:
- Copy: 2,900 epochs / 13m 21s / 276ms/epoch / 100% short / 100% full
- Recall: 7,600 epochs / 39m 49s / 314ms/epoch / 100% K=2 / 99.7% K=4 / 99.7% K=6

The convergence quality was higher (full-length copy at 100%, not 81.8%), but at a config the Idris tape backend can't run end-to-end. Re-aligning at N=128 batch=16 is blocked on DNC layer perf work — see `TODO.md`.

## Idris Implementation

Configuration: N=32, M=20, H=100, R=1, lr=1e-4, nativeRmsprop (alpha=0.95, momentum=0.9), clip=10.0, batch=1, seed=42.

### Copy Task (seqLen 1-10)

| Epochs | Backend | Short (1-5) | Full (1-20) | Time | ms/epoch |
|--------|---------|-------------|-------------|------|----------|
| 500 | tape | 69% | 59% | 9m 11s | 1102 |
| 2,000 | tape | 77% | 60% | 34m 26s | 1033 |
| 2,000 (2026-04-29 rerun) | tape | 76.5% | 60.7% | 57m 47s | 1733 |

The 2026-04-29 rerun (post-revert) reproduces the documented `acc_short ≈ 77%` / `acc_full ≈ 60%` baseline at 2K epochs, confirming the smaller config still works as documented. The slower ms/epoch number reflects parallel CPU contention from running PyTorch's DNC recall convergence on the same machine simultaneously; the 1033 ms/epoch baseline remains the clean per-epoch reference.

Convergence trajectory at batch=1 tracks PyTorch's early phase (PyTorch at epoch 500: 62%, Idris at epoch 500: 69%). Estimated full convergence: ~46K epochs (~13 hours on tape backend).

### Associative Recall (items 2-6, seqLen=3)

| Epochs | Backend | K=2 | K=4 | K=6 | Time |
|--------|---------|-----|-----|-----|------|
| 100 | tape | 49% | 49% | 55% | 3m 31s |

Early stage — recall task needs 5K+ epochs to show meaningful accuracy improvement (PyTorch needed ~5000 epochs before breakthrough).

## Comparison with NTM

| Metric | NTM (10K epochs) | DNC (2K epochs) | DNC PyTorch (4.1K/50K, N=32 batch=1) |
|--------|-------------------|-----------------|------------------------|
| Copy short | 100% | 77% | 100% |
| Copy full | 92% | 60% | 81.8% |
| Recall K=2 | 100% | 49%* | 100% |
| Recall K=6 | 98% | 55%* | 93.3% |

*DNC recall at only 100 epochs, not comparable.

NTM converges faster per-epoch because it has simpler addressing (shift+focus vs allocation+links+mode mixture). DNC's O(n^2) link matrix also makes each epoch slower.

## Key Differences: Idris vs PyTorch

| Factor | Idris | PyTorch | Impact |
|--------|-------|---------|--------|
| Batch size | 1 | 1 (was 16, reverted) | Aligned |
| Memory slots | 32 | 32 (was 128, reverted) | Aligned |
| Backend | tape/torch | libtorch+autograd | Idris tape ~100× slower per epoch (1100 vs 10 ms) |
| Autograd | Wengert tape (tape) / native (torch) | Native | Comparable gradient quality |

The remaining ~100× per-epoch cost on tape (vs PyTorch) is the gating perf gap. NTM exhibits a similar but smaller multiplier — DNC's per-timestep work is dominated by `Layer/Dnc.idr`'s `zeroDiag` scalar fill loop and `buildMatrixRows` per-row `prim__select` extraction. See the Medium-priority TODO entry "DNC layer perf — re-enable PyTorch-aligned config".

## Stability Fixes Applied

Six clamping points prevent forward-pass explosion (previously NaN at seqLen >= 4):

1. Link matrix decay clamped to [0, inf) — prevents sign flip when w_i + w_j > 1
2. Link matrix entries clamped non-negative after diagonal zeroing
3. Allocation sorted usage clamped to [1e-6, inf) before cumprod
4. Retention factors clamped to [1e-10, inf)
5. Read weights clamped and normalized after mode mixture
6. Weight projection in syncBuffers (clamp to [1e-8, inf) + renormalize)

Additional correctness fix: output FC uses current timestep's read outputs (was using previous timestep's).

## R=4 Multi-Head Testing

The DNC layer is parameterized at type level with `r : Nat`. R=4 matches the Graves et al. 2016 paper. All type-level dimensions stay under the ~1000 Peano Nat threshold (max dim = 180 for output FC input = h + r*m = 100 + 4*20).

### PyTorch Reference R=4

| Task | Epochs | Short/K=2 | Full/K=4/K=6 | ms/epoch |
|------|--------|-----------|--------------|----------|
| Copy | 3,000 | 99.2% | 92.1% | 580 |
| Recall | 500 | 69% K=2 | 59% K=4, 56% K=6 | 796 |

(R=4 numbers above are from the historical N=128 batch=16 config; not yet re-run at N=32 batch=1.)

Copy converges to near-100% by 3K epochs. Recall at 500 epochs is still early (R=1 needed 5K+ epochs for breakthrough).

### Idris R=4

| Task | Epochs | Backend | Result | ms/epoch |
|------|--------|---------|--------|----------|
| Copy | 100 | tape | 54% short | 1100 |
| Recall | 100 | tape | 56% K=2 | 4360 |

Both compile and run with no NaN. Recall is slower per-epoch (4.4s vs 1.1s) due to 4 read heads each doing O(n^2) link matrix operations. R=4 is a compile-time constant change (`R = 4` in example files), requiring no layer code modifications.
