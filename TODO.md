# Backlog

## High Priority

| Item | Difficulty | Notes |
|------|-----------|-------|
| Reinforcement learning example | L–XL | Policy gradient (e.g. REINFORCE on CartPole or grid world). Requires: environment interface, episode rollout, discounted return computation, policy gradient loss (`-log_prob * reward`). May need: `Categorical` distribution sampling from logits, baseline variance reduction. Could reuse `Train.runTraining` with episodes as "epochs" |
| Transformer multi-head + layer norm | M | Current impl is single-head, no layer norm. Add: multi-head splitting (reshape + narrow per head), layer normalization, positional encoding, learned embeddings |
| MLX backend | XL | Apple Metal GPU via [mlx-c](https://github.com/ml-explore/mlx-c). New `backend_mlx.c` implementing `backend.h`. Build-time selection: `make BACKEND=mlx backend`. Would give GPU acceleration on Apple Silicon with the same Idris code. The `backend.h` abstraction was designed for this |

## Medium Priority

| Item | Difficulty | Notes |
|------|-----------|-------|
| Convolutional layers | L | Conv1D/Conv2D with autograd. Natural next layer type for image tasks |
| Regularisation/normalisation layers | M | Dropout, layer norm, batch norm. Required for transformers and deeper models |
| More Tensor functions | M | Partially done: `tensor_cat2`, `tensor_narrow` added for NTM pipeline. Remaining: general concat, reshape, transpose, gather/scatter, batched matmul |

## Low Priority

| Item | Difficulty | Notes |
|------|-----------|-------|
| Broadcasting | XL | Type-safe broadcasting. Key tension: expressiveness vs shape safety guarantees. See `docs/static-vs-dynamic-graphs.md` |
| Static graph optimizations | L–XL | Compile-time operator fusion, memory planning via dependent types. See `docs/static-vs-dynamic-graphs.md` |
| DNC (Differentiable Neural Computer) | XL | Graves et al. 2016 — temporal link matrix, dynamic memory allocation, multiple read heads |
| `fromDouble` persistent leak | S | ~15KB/epoch, ~140MB over 10k NTM epochs. Manageable at current scale (peak 348MB). Only matters for 50k+ epoch runs |
| Reshaping layers | M | No current use case |

## Done

Architecture & infrastructure:
- Zero `believe_me` policy: all type conversions proven (Nat proofs, erased record proofs, decEq)
- Pure Idris matrix ops: matrixMultiply, transpose, softmaxMatrix, reshapeToMatrix, flattenMatrix
- README.md with static-vs-dynamic graph motivation
- Transformer (single-head causal self-attention, autoregressive character prediction, pure Idris eval path)
- C tape backend (`backend_tape.c`) with build-time backend selection (`BACKEND=tape|torch`)
- Interface-based layer system (`LayerLike` + `AnyLayer` existential)
- Unified training runner (`Train.idr`: `runTraining`, `TrainConfig`, `EarlyStopConfig`)
- Declarative arg parsing (`ArgSpec` + `parseArgs`)
- Uniform example output formatting (banners, progress, timing, RESULT lines)
- Unit test suite (`make test`, `make test-backend-tape`)
- PyTorch reference benchmarks (`pytorch/` directory)

Layers & models:
- Linear, RNN, LSTM, NTM (copy + associative recall)
- Softmax, LogSoftmax, Sigmoid activations
- Xavier/He/LeCun weight initialization

Autograd & optimization:
- Tape-based autograd (Wengert list) — originally Chez Scheme, now C backend
- Fused backward rules: OP_MV, OP_LSTM_GATES, OP_NTM_READ_HEAD, OP_NTM_INTERP_WRITE, OP_VECMAT, OP_CAT, OP_NARROW
- SGD, RMSprop (with momentum), Adam optimizers (native C + Idris-side)
- Global gradient norm/value clipping
- Learning rate schedules (cosine annealing, one-cycle)
- Per-element optimizer buffers (RMSprop/Adam)

NTM-specific:
- NTM tensor pipeline (LSTM handle pass-through, direct FC calls, tensor_narrow, tensor_cat2) — 380ms → 110ms/epoch
- Fused NTM addressing ops with gradient chain
- Memory leak fixes (arena_reset, grad array cleanup, persistent scalar tracking)
- Windowed convergence early stopping
- Two-phase training (encode then decode)
- Curriculum training module (multi-stage)

Performance:
- NTM-copy: ~110ms/epoch (faster than old C backend's ~120ms)
- Arena allocator with chunked linked list (no realloc)
- Consecutive-data cache in tensor_stack_from_array
- Memory diagnostics (`backend_memory_report()`)
