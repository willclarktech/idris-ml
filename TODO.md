# Backlog

## High Priority

| Item | Difficulty | Notes |
|------|-----------|-------|
| Try RefC backend | S | Blocked: `srand` FFI missing in RefC codegen |
| Tape-based autograd | L | Wengert list, standard ML approach |
| Buffer-backed tensors + C FFI | L | Contiguous memory for tensor data |

## Medium Priority

| Item | Difficulty | Notes |
|------|-----------|-------|
| Minibatches/SGD | M | |
| Momentum/Adam | M | |
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
| Hyperparameter optimization | L | |
| Heterogeneous context (CPU/GPU) | XL | |
| Write README.md | S | |

## Done

- NTM
- Optimize backward pass (SortedMap, single-pass)
