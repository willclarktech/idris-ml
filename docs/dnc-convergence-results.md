# DNC Convergence Results

## PyTorch Reference (Oracle)

Configuration: N=128, M=20, H=100, R=1, lr=1e-4, RMSprop (alpha=0.95, momentum=0.9), clip=10.0, batch=16, seed=42.

### Copy Task (seqLen 1-20)

Converged at **2,900 epochs** via early stopping (windowed avg < 0.01).

| Metric | Result |
|--------|--------|
| Short (len 1-5) | 100.0% |
| Full (len 1-20) | 100.0% |
| Time | 13m 21s (276ms/epoch) |

Loss trajectory: 0.69 → 0.61 → 0.48 → 0.14 (epoch 1700, breakthrough) → 0.001 → converged.

### Associative Recall (items 2-6, seqLen=3)

Converged at **7,600 epochs** via early stopping.

| Metric | Result |
|--------|--------|
| K=2 items | 100.0% |
| K=4 items | 99.7% |
| K=6 items | 99.7% |
| Time | 39m 49s (314ms/epoch) |

Loss trajectory: 0.69 → 0.57 → 0.30 (epoch 5100, breakthrough) → 0.001 → converged.

## Idris Implementation

Configuration: N=32, M=20, H=100, R=1, lr=1e-4, nativeRmsprop (alpha=0.95, momentum=0.9), clip=10.0, batch=1, seed=42.

### Copy Task (seqLen 1-10)

| Epochs | Backend | Short (1-5) | Full (1-20) | Time | ms/epoch |
|--------|---------|-------------|-------------|------|----------|
| 500 | tape | 69% | 59% | 9m 11s | 1102 |
| 2,000 | tape | 77% | 60% | 34m 26s | 1033 |

Convergence trajectory at batch=1 tracks PyTorch's early phase (PyTorch at epoch 500: 62%, Idris at epoch 500: 69%). The batch=1 vs batch=16 difference means Idris needs ~16x more epochs for equivalent gradient signal. Estimated full convergence: ~46K epochs (~13 hours on tape backend).

### Associative Recall (items 2-6, seqLen=3)

| Epochs | Backend | K=2 | K=4 | K=6 | Time |
|--------|---------|-----|-----|-----|------|
| 100 | tape | 49% | 49% | 55% | 3m 31s |

Early stage — recall task needs 5K+ epochs to show meaningful accuracy improvement (PyTorch needed ~5000 epochs before breakthrough).

## Comparison with NTM

| Metric | NTM (10K epochs) | DNC (2K epochs) | DNC PyTorch (2.9K/7.6K) |
|--------|-------------------|-----------------|------------------------|
| Copy short | 100% | 77% | 100% |
| Copy full | 92% | 60% | 100% |
| Recall K=2 | 100% | 49%* | 100% |
| Recall K=6 | 98% | 55%* | 99.7% |

*DNC recall at only 100 epochs, not comparable.

NTM converges faster per-epoch because it has simpler addressing (shift+focus vs allocation+links+mode mixture). DNC's O(n^2) link matrix also makes each epoch slower.

## Key Differences: Idris vs PyTorch

| Factor | Idris | PyTorch | Impact |
|--------|-------|---------|--------|
| Batch size | 1 | 16 | ~16x more epochs needed |
| Memory slots | 32 | 128 | Less capacity, but faster epochs |
| Backend | tape/torch | libtorch+autograd | Similar speed per epoch |
| Autograd | Wengert tape (tape) / native (torch) | Native | Comparable gradient quality |

## Stability Fixes Applied

Six clamping points prevent forward-pass explosion (previously NaN at seqLen >= 4):

1. Link matrix decay clamped to [0, inf) — prevents sign flip when w_i + w_j > 1
2. Link matrix entries clamped non-negative after diagonal zeroing
3. Allocation sorted usage clamped to [1e-6, inf) before cumprod
4. Retention factors clamped to [1e-10, inf)
5. Read weights clamped and normalized after mode mixture
6. Weight projection in syncBuffers (clamp to [1e-8, inf) + renormalize)

Additional correctness fix: output FC uses current timestep's read outputs (was using previous timestep's).
