# Backlog

## High Priority

| Item | Difficulty | Notes |
|------|-----------|-------|
| Revisit native autograd (torch/MLX) | M–L | Currently all 3 backends use a custom Wengert list (tape) for autograd. Torch and MLX have built-in autograd that could be faster, more numerically stable, and support higher-order gradients. Investigate: (1) performance gap between custom tape vs native autograd, (2) which ops benefit most, (3) whether hybrid approach is feasible (native autograd for standard ops, custom for fused NTM ops). Would require rearchitecting the backward pass in backend_mlx.cpp and backend_torch.cpp |
| Reinforcement learning example | L–XL | Policy gradient (e.g. REINFORCE on CartPole or grid world). Requires: environment interface, episode rollout, discounted return computation, policy gradient loss (`-log_prob * reward`). May need: `Categorical` distribution sampling from logits, baseline variance reduction. Could reuse `Train.runTraining` with episodes as "epochs" |

## Medium Priority

| Item | Difficulty | Notes |
|------|-----------|-------|
| C test suite on MLX/torch backends | M | `make test-backend-tape` passes all tests. MLX: `tensor_item` returns 0.0 for non-requires_grad scalars (lazy eval timing), crashes in fused MV optimizer test — passes with `fprintf` interleaved (timing-dependent). Torch: crashes after fused MV optimizer → LSTM test sequence (stale pointer from `free_intermediates` / `tensor_free` interaction). All examples work on all 3 backends — issues are in the C test suite's manual `tensor_free` patterns, not in the Idris training path. Root cause: `free_intermediates` model assumes strict create-forward-backward-step-cleanup cycle with no tensor reuse |
| Batch dimension support for attention | M | `tensor_bmm` exists for projections. Remaining: batch the per-sequence attention block (Q@K^T, softmax, attn@V) with block-diagonal masking. Would eliminate the last per-sequence loop (~576 FFI calls/epoch). Impact limited by Chez runtime overhead — see `docs/design-decisions.md` performance analysis |
| Convolutional layers | L | Conv1D/Conv2D with autograd. Natural next layer type for image tasks |
| Regularisation/normalisation layers | M | Dropout, batch norm. Layer norm done. Required for deeper models |
| More Tensor functions | M | Partially done: `tensor_cat2`, `tensor_narrow` added for NTM pipeline. Remaining: general concat, reshape, transpose, gather/scatter |

## Low Priority

| Item | Difficulty | Notes |
|------|-----------|-------|
| Broadcasting | XL | Type-safe broadcasting. Key tension: expressiveness vs shape safety guarantees. See `docs/static-vs-dynamic-graphs.md` |
| Static graph optimizations | L–XL | Compile-time operator fusion, memory planning via dependent types. See `docs/static-vs-dynamic-graphs.md` |
| DNC (Differentiable Neural Computer) | XL | Graves et al. 2016 — temporal link matrix, dynamic memory allocation, multiple read heads |
| `fromDouble` persistent leak | S | ~15KB/epoch, ~140MB over 10k NTM epochs. Manageable at current scale (peak 348MB). Only matters for 50k+ epoch runs |
| Reshaping layers | M | No current use case |
| Chez Scheme runtime overhead | S–XL | Chez GC, thunk evaluation, and allocation account for ~50ms/epoch (vs 2ms in C). Not FFI marshaling — reducing FFI call count from 4,384 to ~1,220 only saved 6ms. Options: explore Idris→C backend (bypass Chez entirely), or accept ~3x gap vs PyTorch on CPU |

## Done

Architecture & infrastructure:
- Model serialization: SafeTensors format (`param_save`/`param_load`, `optimizer_save`/`optimizer_load`), Idris `Checkpoint` module (`saveModel`/`loadModel`), Python interop verified, optimizer state with Adam/RMSprop buffer round-trip
- Cross-backend transfer example (`Example/Transfer.idr`): train→save→continue→save→infer across tape/MLX/torch, `make transfer-demo` orchestrates all 3 phases
- Zero `believe_me` policy: all type conversions proven (Nat proofs, erased record proofs, decEq)
- Pure Idris matrix ops: matrixMultiply, transpose, softmaxMatrix, reshapeToMatrix, flattenMatrix
- README.md with static-vs-dynamic graph motivation
- Transformer (single-head causal self-attention, autoregressive character prediction, pure Idris eval path)
- C tape backend (`backend_tape.c`) with build-time backend selection (`BACKEND=tape|torch`)
- Interface-based layer system (`LayerLike` + `AnyLayer` existential)
- Unified training runner (`Train.idr`: `runTraining`, `TrainConfig`, `EarlyStopConfig`)
- Declarative arg parsing (`ArgSpec` + `parseArgs`)
- Uniform example output formatting (banners, progress, timing, RESULT lines)
- Unified test infrastructure: `make test-all` runs Idris unit tests + C backend tests on all available backends + specialized C tests (safetensors, NTM grad, NTM timestep) + integration tests (`test-examples` validates RESULT lines) + PyTorch reference tests. Backend detection via cached dylibs. Consolidated `test_backend.c` (was split across `test_backend.c` and `test_backend_tape.c`)
- PyTorch reference benchmarks (`pytorch/` directory)

Layers & models:
- Linear, RNN, LSTM, NTM (copy + associative recall)
- Multi-head Transformer (Pre-LN, learned embeddings, sinusoidal PE, layer norm, per-head weights with sum-not-concat)
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
- Tensor-level forward path (`applyVarTensor`, `forwardVarTensor`) — eliminates scalar packing at layer boundaries
- `epochNativeTensorPre` + `TensorDataPoint` — zero-copy data flow from generator to C
- C-side one-hot encoding (`tensor_one_hot`) — eliminates per-element FFI for data prep
- Transformer: 160ms → 56ms/epoch (2.9x). C backend: 2ms. Remaining ~54ms is Chez Scheme runtime overhead (GC, thunk evaluation, list allocation), not FFI marshaling — see `docs/design-decisions.md`
- Batched transformer forward (`transformerForwardBatch`): projections/FF/norms batched as `[B*seqLen, dim]`, per-sequence loop only for attention. Fixed double-nameLayer bug (stale weight handles)
- Backend profiling (`backend_profile_reset`/`backend_profile_report`)
- NTM-copy: ~110ms/epoch (faster than old C backend's ~120ms)
- Arena allocator with chunked linked list (no realloc)
- Consecutive-data cache in tensor_stack_from_array
- Memory diagnostics (`backend_memory_report()`)

MLX backend:
- `backend_mlx.cpp` — Apple Metal GPU via MLX C++ API, tape-based autograd on MLX arrays
- All 6 examples work on all 3 backends (tape, MLX, torch) via tensor path
- NTM Copy: tape 100%/98%, MLX 91%/87%, torch (untested recent fixes)
- NTM ops decomposed into primitives with per-op backward rules
- Fused OP_NORMALIZE for numerically stable attention normalization
- `all_tensors` self-registration: every Tensor tracked, non-persistent freed at `tape_reset()`
- State tensor persistence (`tensor_create_state_*` sets `persistent=1`)
- TensorPair tracking and cleanup on both MLX and torch
- RMSprop optimizer implemented (was missing — NTM used RMSprop, weights never updated)
- OP_SELECT backward, OP_POW exponent gradient, broadcast `reduce_grad()`
- Conv1d_circular d_kernel backward sign fix
- Torch: persistent view tensors, RNN `toDoubleLayer` tensor path
- Multi-element unary ops in tape backend (neg, abs, exp, log, sqrt, sigmoid, tanh)
- OP_LOG_SOFTMAX backward rule added to tape backend
