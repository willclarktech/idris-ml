# Backlog

## High Priority

| Item | Difficulty | Notes |
|------|-----------|-------|
| PyTorch benchmarks | M | Compare training speed/accuracy against PyTorch baseline |

## Medium Priority

| Item | Difficulty | Notes |
|------|-----------|-------|
| Minibatch sampling | S | Currently trains on full dataset every epoch |
| Unit tests | M | No test framework yet; verify via `--check` + examples |
| Automatically name parameters | S | |
| More Tensor functions (eg concatenation) | M | |
| Reshaping layers | M | |
| Noise other than uniform (eg Gaussian) | S | |

## Low Priority

| Item | Difficulty | Notes |
|------|-----------|-------|
| Convolutional layers | L | |
| Transformer | XL | |
| Regularisation/normalisation layers | M | |
| Heterogeneous context (CPU/GPU) | XL | |
| Write README.md | S | |

## Done

- NTM
- Optimize backward pass (SortedMap, single-pass)
- Node IDs + memoized graph traversal
- Momentum/Adam optimizer
- SGD optimizer
- Bounded NTM gamma (focus sharpening)
- Global gradient norm clipping (`adamGlobalClip`)
- O(n) topoSort (accumulator-based)
- Hyperparameter sweep script (`scripts/sweep.sh`)
- Tape-based autograd (Wengert list) with Chez FFI storage
- Buffer-backed tensor ops + C FFI (Phase 1: matmul/dot, 1.3-1.9x speedup)
- Persistent weight buffers + bulk tape registration (Phase 3: ~1.14x NTM speedup)
- Early stopping (patience-based + NaN detection in Backprop)
- Hyperparameters type (Config record in Ntm.idr)
- Learning rate schedules (one-cycle, cosine annealing in Schedule.idr)
- C-backed softmax/logSoftmax (Phase 2 of buffer-backed tensors)
- Xavier/He/LeCun weight initialization (Init.idr)
- NTM debug/diagnostics module (Debug.idr, `--diagnose` flag)
