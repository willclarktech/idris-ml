# Backlog

## High Priority

| Item | Difficulty | Notes |
|------|-----------|-------|
| Core PyTorch feature parity | L–XL | The gaps that matter most for practical use, roughly in priority order: |

**Layers & ops:**
- Embedding layer (S): `nn.Embedding` — lookup table instead of one-hot + linear. Current GPT/Transformer uses one-hot [seqLen, vocab] × linear, wasting O(vocab) per token. Embedding is O(1) lookup + gradient via scatter. Uses the new `tensor_gather`/`tensor_scatter_add`
- GELU activation (S): standard in modern transformers (GPT-2+). `x * 0.5 * (1 + erf(x / sqrt(2)))`. Quick to add alongside existing ReLU/sigmoid/tanh tensor dispatch
- GRU layer (M): lighter recurrent unit than LSTM (2 gates vs 4). Common for smaller sequence models. Same `LayerLike` pattern as LSTM
- Average pooling (S): `AvgPool1D`/`AvgPool2D` — mean instead of max. Needed for global average pooling (common in modern CNNs). Simpler backward than max pool (uniform gradient)
- Cross-attention (M): Q from decoder, K/V from encoder. Extends existing multi-head attention. Enables encoder-decoder (seq2seq, machine translation)
- Transposed convolution (M): `ConvTranspose1D`/`2D` for upsampling (VAEs, GANs, segmentation). Backward of conv is forward of transposed conv

**Infrastructure:**
- CUDA support (M–L): torch backend should work on CUDA with `tensor_to_device("cuda")` — the binding exists but is untested. MLX is Metal-only. Tape backend is CPU-only. Need: verify torch CUDA path, add device selection to examples, benchmark GPU vs CPU. May need Linux CI for testing
- General DataLoader pattern (M): current examples generate/load data ad-hoc. A reusable batched data pipeline would reduce boilerplate. Shuffle, batch, repeat
- Group normalization (S): normalizes over channel groups, no batch dependency. Works at any dim (no Peano Nat issue if group size is small). Better alternative to batch norm for this project

## Medium Priority

(empty)

## Low Priority

| Item | Difficulty | Notes |
|------|-----------|-------|
| Opaque type-level Nats | M–XL | Idris 2 Peano Nats hang the compiler for dims > ~1000. Need machine-backed type-level naturals (like GHC TypeLits). Engage with Idris 2 upstream. Blocks identity layers (dropout, batch norm) at large dims. See `docs/gotchas.md` |
| Broadcasting | XL | Type-safe broadcasting. Key tension: expressiveness vs shape safety guarantees. See `docs/static-vs-dynamic-graphs.md` |
| Static graph optimizations | L–XL | Compile-time operator fusion, memory planning via dependent types. See `docs/static-vs-dynamic-graphs.md` |
| DNC (Differentiable Neural Computer) | XL | Graves et al. 2016 — temporal link matrix, dynamic memory allocation, multiple read heads |
| `fromDouble` persistent leak | S | Partially fixed: `tensor_create_scalar` and `tensor_create` non-grad tensors are now non-persistent on MLX (freed by tape_reset). Remaining: Chez Scheme GC doesn't call `tensor_free`, so non-persistent tensors accumulate within one epoch until optimizer_step. ~15KB/epoch overhead, manageable |
| Reshaping layers | M | No current use case |
| Chez Scheme runtime overhead | S–XL | Chez GC, thunk evaluation, and allocation account for ~50ms/epoch (vs 2ms in C). Not FFI marshaling — reducing FFI call count from 4,384 to ~1,220 only saved 6ms. Options: explore Idris→C backend (bypass Chez entirely), or accept ~3x gap vs PyTorch on CPU |

## Done

Architecture & infrastructure:
- Batched attention: per-sequence attention loop eliminated. All sequences processed in parallel via 3D ops (`bmm_3x3`, `softmax_3d`, `transpose_last2`). FFI calls reduced from B×H×12 to H×8 per block. New ops: `tensor_bmm_3x3` ([B,m,n]×[B,n,k]), `tensor_softmax_3d`, `tensor_transpose_last2`, `tensor_expand_mask`
- Native autograd: torch already used native autograd (2-line backward). MLX migrated from 480 lines of hand-written backward rules to replay-based native autograd via `mlx::vjp` — zero backward rules, MLX handles all gradient computation. Tape backend keeps manual tape (no framework). Adding a new op on MLX: ~2 lines (forward replay case), backward is free
- Model serialization: SafeTensors format (`param_save`/`param_load`, `optimizer_save`/`optimizer_load`), Idris `Checkpoint` module (`saveModel`/`loadModel`), Python interop verified, optimizer state with Adam/RMSprop buffer round-trip
- Cross-backend transfer example (`Example/Transfer.idr`): train→save→continue→save→infer across tape/MLX/torch, `make example-transfer-demo` orchestrates all 3 phases
- Zero `believe_me` policy: all type conversions proven (Nat proofs, erased record proofs, decEq)
- Pure Idris matrix ops: matrixMultiply, transpose, softmaxMatrix, reshapeToMatrix, flattenMatrix
- README.md with static-vs-dynamic graph motivation
- Transformer (single-head causal self-attention, autoregressive character prediction, pure Idris eval path)
- C tape backend (`backend_tape.c`) with build-time backend selection (`BACKEND=tape|torch`)
- Interface-based layer system (`LayerLike` + `AnyLayer` existential)
- Unified training runner (`Train.idr`: `runTraining`, `TrainConfig`, `EarlyStopConfig`)
- Declarative arg parsing (`ArgSpec` + `parseArgs`)
- Uniform example output formatting (banners, progress, timing, RESULT lines)
- Unified test infrastructure: `make test-all` runs Idris unit tests + C backend tests on all available backends + specialized C tests (safetensors, NTM grad, NTM timestep) + integration tests (`test-examples` validates RESULT lines) + PyTorch reference tests. Backend detection via cached dylibs. Consolidated `test_backend.c` — all tests pass on all 3 backends
- PyTorch reference benchmarks (`pytorch/` directory)

Layers & models:
- Linear, RNN, LSTM, NTM (copy + associative recall)
- Multi-head Transformer (Pre-LN, learned embeddings, sinusoidal PE, layer norm, per-head weights with sum-not-concat)
- REINFORCE on CartPole (`Example/Reinforce.idr`): pure Idris CartPole environment (Gymnasium-compatible physics), REINFORCE with mean-return baseline, `categoricalSample` in Sampler.idr, tensor-level `applyVarTensor` for tanh/sigmoid activations. Converges to 200.0 greedy eval on all 3 backends. PyTorch reference in `pytorch/torch_ref/models/reinforce.py`
- MNIST CNN (`Example/Mnist.idr`): LeNet-style Conv2D(1->16,k=5) -> ReLU -> MaxPool(2) -> Conv2D(16->32,k=5) -> ReLU -> MaxPool(2) -> Linear(512->10). Type-safe spatial dimension chain via `ConvOutDim`/`PoolOutDim` type-level functions. First example with external data (MNIST .idx files). Conv2D + MaxPool2D ops on all 3 backends. ReLU tensor-level activation. PyTorch reference in `pytorch/torch_ref/models/mnist_cnn.py`
- GPT char-level LM (`Example/Gpt.idr`): character-level language model on embedded Shakespeare corpus (1342 chars, 36-char vocab). Reuses TransformerState with batched forward, autoregressive text generation. PyTorch reference in `pytorch/torch_ref/models/gpt.py`
- Dropout layer (`Layer/Dropout.idr`): inverted dropout with training/eval mode toggle via `setTraining`/`setNetworkTraining`. C ops on all 3 backends
- Batch norm layer (`Layer/BatchNorm.idr`): per-channel normalization with running stats, training/eval mode. C ops on all 3 backends. Note: Peano Nat ceiling limits use to dims ≤ ~500
- Conv1D + MaxPool1D (`Layer/Conv.idr`): 1D convolution and pooling with type-safe `ConvOutDim`/`PoolOutDim`. SeqClassify example classifies synthetic waveforms (sine/square/triangle)
- ReLU tensor-level activation in `Layer/Activation.idr`
- Tensor function wrappers: gather, scatter_add, squeeze, clone, sum_dim, dim/size queries. FFI bindings for previously unbound C ops
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
