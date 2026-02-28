# Backlog

## High Priority

| Item | Difficulty | Notes |
|------|-----------|-------|
| C-backed softmax/logSoftmax/memory ops | M | Phase 2 of buffer-backed tensors |

## Medium Priority

| Item | Difficulty | Notes |
|------|-----------|-------|
| Automatically name parameters | S | |
| More Tensor functions (eg concatenation) | M | |
| Reshaping layers | M | |
| Noise other than uniform (eg Gaussian) | S | |

## Low Priority

| Item | Difficulty | Notes |
|------|-----------|-------|
| Convolutional layers | L | |
| Transformer | XL | |
| Early stopping | S | |
| Hyperparameters type | S | |
| Regularisation/normalisation layers | M | |
| Heterogeneous context (CPU/GPU) | XL | |
| Write README.md | S | |

## Done

- NTM
- Optimize backward pass (SortedMap, single-pass)
- Node IDs + memoized graph traversal
- Momentum/Adam optimizer
- Minibatches/SGD
- Bounded NTM gamma (focus sharpening)
- Global gradient norm clipping (`adamGlobalClip`)
- O(n) topoSort (accumulator-based)
- Hyperparameter sweep script (`scripts/sweep.sh`)
- Tape-based autograd (Wengert list) with Chez FFI storage
- Buffer-backed tensor ops + C FFI (Phase 1: matmul/dot, 1.3-1.9x speedup)
- Persistent weight buffers + bulk tape registration (Phase 3: ~1.14x NTM speedup)
