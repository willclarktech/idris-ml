# Backlog

## High Priority

| Item | Difficulty | Notes |
|------|-----------|-------|
| CUDA support | M–L | Torch backend should work via `tensor_to_device("cuda")` — untested. Test script ready: `scripts/test_cuda_colab.sh`. See `docs/develop/cuda-testing.md`. Device type system ready: `Variable (CUDA 0)` compiles, `toDevice` + FFI bindings exist |

## Low Priority

| Item | Difficulty | Notes |
|------|-----------|-------|
| Opaque type-level Nats | M–XL | Idris 2 Peano Nats hang the compiler for dims > ~1000. Need machine-backed type-level naturals (like GHC TypeLits). Engage with Idris 2 upstream. Blocks identity layers (dropout, batch norm) at large dims. See `docs/gotchas.md` |
| Broadcasting | XL | Type-safe broadcasting. Key tension: expressiveness vs shape safety guarantees. Dex's typed index sets are the most promising model for a dependently-typed setting. See `docs/static-vs-dynamic-graphs.md` |
| Static graph optimizations | L–XL | Compile-time operator fusion, memory planning via dependent types. See `docs/static-vs-dynamic-graphs.md` |
| `fromDouble` persistent leak | S | Partially fixed: `tensor_create_scalar` and `tensor_create` non-grad tensors are now non-persistent on MLX (freed by tape_reset). Remaining: Chez Scheme GC doesn't call `tensor_free`, so non-persistent tensors accumulate within one epoch until optimizer_step. ~15KB/epoch overhead, manageable |
| Reshaping layers | M | No current use case |
| Chez Scheme runtime overhead | S–XL | Chez GC, thunk evaluation, and allocation account for ~50ms/epoch (vs 2ms in C). Not FFI marshaling — reducing FFI call count from 4,384 to ~1,220 only saved 6ms. Options: Idris→C backend (below), or accept ~3x gap vs PyTorch on CPU for small models |
| Idris 2 C backend | XL | Compile Idris 2 to C instead of Chez Scheme. Would eliminate the ~50ms/epoch runtime overhead entirely, making idris-ml match the C benchmark numbers end-to-end. Requires upstream Idris 2 support or a custom code generator. Largest potential perf win but highest effort |
| CodeMirror Idris 2 mode | S–M | Jupyter kernel uses `"codemirror_mode": "haskell"` as a fallback. Investigate whether an Idris 2 CodeMirror grammar exists or could be written (CodeMirror 6 Lezer grammar). Would give proper syntax highlighting in JupyterLab |

**Explicitly not planned:** distributed training (infrastructure, not library), mixed precision/quantization (performance optimisation), model zoo (compositions of existing primitives), TorchScript (our type system is the compile-time analysis), bidirectional RNN (transformers have obsoleted), exotic losses (compose from primitives)

## Done

Architecture & infrastructure:
- Type-safe device placement: `Variable (0 d : Device)` phantom parameter prevents mixing CPU/CUDA/MPS tensors at compile time. Zero runtime cost (erased). `toDevice` for intentional transfers. FFI bindings for `tensor_to_device`/`tensor_device`. 21 library files + 18 example/test files updated
- General DataLoader (`DataLoader.idr`): `mkGeneratorLoader` for synthetic data, `mkIndexedLoader` for file-backed datasets with shuffled epoch iteration (Fisher-Yates via C, position tracking via IORef). MNIST updated to use shuffled loading
- Jupyter kernel (`jupyter/`): pexpect-based REPL wrapper providing interactive notebook experience. `Notebook.Prelude` re-exports all modules. Cell parser auto-routes `:t`/`:doc`/`:exec`/`:let`/expressions. FFI works via dylib copy to `_tmpchez_app/`. Session recovery on crash. `make jupyter-install` + `make jupyter-lab`. Notebooks in `tutorials/` (01-06: concepts) and `models/` (9 per-architecture walkthroughs). `make test-notebooks` runs all headless
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
- Linear, RNN, LSTM, NTM (copy + associative recall), DNC (copy + associative recall). DNC convergence validated on R=1 and R=4: PyTorch ref 100% on both tasks, Idris tracks PyTorch trajectory. R=4 multi-head verified (compile + run, no code changes needed). See `docs/develop/dnc-convergence-results.md`
- Multi-head Transformer (Pre-LN, embedding lookup, sinusoidal PE, layer norm, per-head weights with sum-not-concat). Input: token indices `[seqLen]`, output: logits `[seqLen * vocabSize]`
- REINFORCE on CartPole (`Example/Reinforce.idr`): pure Idris CartPole environment (Gymnasium-compatible physics), REINFORCE with mean-return baseline, `categoricalSample` in Sampler.idr, tensor-level `applyVarTensor` for tanh/sigmoid activations. Converges to 200.0 greedy eval on all 3 backends. PyTorch reference in `pytorch/torch_ref/models/reinforce.py`
- MNIST CNN (`Example/Mnist.idr`): LeNet-style Conv2D(1->16,k=5) -> ReLU -> MaxPool(2) -> Conv2D(16->32,k=5) -> ReLU -> MaxPool(2) -> Linear(512->10). Type-safe spatial dimension chain via `ConvOutDim`/`PoolOutDim` type-level functions. First example with external data (MNIST .idx files). Conv2D + MaxPool2D ops on all 3 backends. ReLU tensor-level activation. PyTorch reference in `pytorch/torch_ref/models/mnist_cnn.py`
- GPT char-level LM (`Example/Gpt.idr`): character-level language model on embedded Shakespeare corpus (1342 chars, 36-char vocab). Reuses TransformerState with embedding lookup, AdamW, autoregressive text generation. PyTorch reference in `pytorch/torch_ref/models/gpt.py`
- Dropout layer (`Layer/Dropout.idr`): inverted dropout with training/eval mode toggle via `setTraining`/`setNetworkTraining`. C ops on all 3 backends
- Batch norm layer (`Layer/BatchNorm.idr`): per-channel normalization with running stats, training/eval mode. C ops on all 3 backends. Note: Peano Nat ceiling limits use to dims ≤ ~500
- Conv1D + MaxPool1D (`Layer/Conv.idr`): 1D convolution and pooling with type-safe `ConvOutDim`/`PoolOutDim`. SeqClassify example classifies synthetic waveforms (sine/square/triangle)
- ReLU, GELU, LeakyReLU, SiLU/Swish tensor-level activations in `Layer/Activation.idr`
- Embedding layer (`Layer/Embedding.idr`): token index lookup via gather/scatter_add. O(1) per token vs O(vocab) for one-hot
- GRU layer (`Layer/Gru.idr`): 2-gate recurrent unit (lighter than LSTM). C ops on all 3 backends
- Residual layer wrapper (`Layer/Residual.idr`): `output = input + inner(input)`. Enables ResNet-style composition
- Average pooling 1D + 2D (`Layer/Conv.idr`): AvgPool1DState, AvgPool2DState
- Grouped/depthwise convolution: `tensor_conv1d_grouped`, `tensor_conv2d_grouped` with groups parameter
- Transposed convolution 1D + 2D: `tensor_conv_transpose1d`, `tensor_conv_transpose2d` for upsampling
- Cross-attention: `tensor_cross_attention(Q, K, V, mask, scale)` for encoder-decoder architectures
- Tensor function wrappers: gather, scatter_add, squeeze, clone, sum_dim, min/max reductions, dim/size queries, embedding. FFI bindings for previously unbound C ops
- Loss functions: MSE, BCE, BCE with logits, cross-entropy, NLL, L1 (MAE), Huber/Smooth L1, KL divergence (standard + log-space)
- Training/eval mode: `setTraining`/`setNetworkTraining` in LayerLike interface (for dropout, batch norm)

Autograd & optimization:
- Tape-based autograd (Wengert list) — originally Chez Scheme, now C backend
- Fused backward rules: OP_MV, OP_LSTM_GATES, OP_GRU_CELL, OP_NTM_READ_HEAD, OP_NTM_INTERP_WRITE, OP_VECMAT, OP_CAT, OP_NARROW, OP_CONV1D, OP_CONV2D, OP_MAX_POOL1D, OP_MAX_POOL2D, OP_DROPOUT, OP_BATCH_NORM, OP_EMBEDDING, OP_GELU, OP_LEAKY_RELU, OP_SILU
- SGD, RMSprop (with momentum), Adam, AdamW (decoupled weight decay) optimizers (native C + Idris-side)
- Global gradient norm/value clipping
- Learning rate schedules (cosine annealing, one-cycle, warmup + cosine, composable warmup wrapper, step LR, exponential LR)
- Per-element optimizer buffers (RMSprop/Adam/AdamW)
- Gradient accumulation: `nativeBackwardOnly` + `nativeOptimizerStep` for split backward/step, `epochNativeTensorPreAccum` convenience function
- Group normalization: `tensor_group_norm` for per-channel-group normalization
- Softmax, LogSoftmax, Sigmoid activations
- Per-parameter LR overrides: `optimizer_set_param_lr` / `setParamLR` (tape + MLX backends). Enables transfer learning (freeze layers, layer-wise LR)
- Xavier/He/LeCun weight initialization

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
- Operator-level benchmarks (`csrc/bench_ops.c`): matmul, matvec, element-wise, softmax, conv2d, training step at multiple sizes. PyTorch reference (`bench_ops.py`) + comparison table (`compare_ops.py`). `make bench-ops-compare`. See `docs/benchmarks.md`

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
