# Backlog

## High Priority

| Item | Difficulty | Notes |
|------|-----------|-------|
| Try RefC backend | S | Blocked: `srand` FFI missing in RefC codegen |
| Tape-based autograd | L | Wengert list, standard ML approach |
| Buffer-backed tensors + C FFI | L | Contiguous memory for tensor data, 10-50x speedup |

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
