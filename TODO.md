# Backlog

## High Priority

| Item | Difficulty | Notes |
|------|-----------|-------|
| Cleaner port/adapter model for layers | L | Currently Layer.idr contains every layer type and all dispatch logic in one mutual block; consider typeclass-based extensibility |

## Medium Priority

| Item | Difficulty | Notes |
|------|-----------|-------|
| More Tensor functions (eg concatenation) | M | |
| Reshaping layers | M | |

## Low Priority

| Item | Difficulty | Notes |
|------|-----------|-------|
| Reinforcement learning demo | L | Simple RL example (e.g. policy gradient on CartPole or grid world) to demonstrate autograd beyond supervised/sequence tasks |
| DNC (Differentiable Neural Computer) | XL | Graves et al. 2016 successor to NTM — temporal link matrix, dynamic memory allocation, multiple read heads |
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
- Bounded NTM gamma (focus sharpening) — later replaced by unbounded softplus gamma
- Global gradient norm clipping (`adamGlobalClip`)
- O(n) topoSort (accumulator-based)
- Hyperparameter sweep script (`scripts/sweep.sh`)
- Tape-based autograd (Wengert list) with Chez FFI storage
- Buffer-backed tensor ops + C FFI (Phase 1: matmul/dot, 1.3-1.9x speedup)
- Persistent weight buffers + bulk tape registration (Phase 3: ~1.14x NTM speedup)
- Early stopping (patience-based + NaN detection in Backprop)
- Hyperparameters type (Config record in NtmCopy.idr)
- Learning rate schedules (one-cycle, cosine annealing in Schedule.idr)
- C-backed softmax/logSoftmax (Phase 2 of buffer-backed tensors)
- Xavier/He/LeCun weight initialization (Init.idr)
- NTM debug/diagnostics module (Debug.idr, `--diagnose` flag)
- Random data generation module (Generate.idr, `SequenceTask` port/adapter)
- NTM memory init: sigmoid(xavier_random) matching PyTorch's sigmoid(FC_bias)
- Curriculum training (3 stages: len 1-3, 1-5, 1-8) — no longer required for PyTorch-aligned NTM
- Gradient clip norm 50.0 (Collier & Beel default)
- 3-element shift kernel + hot-start addressing
- NTM learned initial addressing (backprop through head weights + readHeadOutput)
- NTM associative recall example (content-based addressing)
- Unit test suite (`make test` / `make test-c`)
- Unified NTM head ops via NormalizationFunction parameter (Memory.idr)
- C-backed NTM memory ops (batchCosineSimilarityVar, readOpVar, writeOpVar — ~1.8x NTM speedup)
- Gaussian/normal distribution sampling (Sampler.idr, composable with init strategies)
- Automatic parameter naming (`autoName` in Layer.idr, type-based prefixes with collision-free scoping)
- PyTorch benchmarks (`pytorch/` directory — correctness tests, timing benchmarks, side-by-side comparison)
- NTM recall convergence verification (3 experiments: RMSprop baseline, Adam, Adam+2items — see `docs/ntm-convergence-results.md`)
- Convergence script CLI args (`--recall-controller`, `--recall-optimizer`, `--recall-clip`, etc.)
- NTM documentation extraction (`docs/ntm.md` — architecture, convergence, failure modes)
- LSTM layer (`LstmLayer` constructor in Layer.idr — gate computation, zero forget bias, cell state extraction)
- Interpolation write (`interpolationWrite` in Memory.idr, `interpolationWriteVar` C-backed in Variable.idr)
- Softplus gamma (`forwardReadHeadUnbounded` in Memory.idr — `gamma = 1 + softplus(x)`, unbounded)
- RMSprop optimizer with value clipping (`rmsprop`, `rmspropValueClip` in Optimizer.idr)
- Binary vector data format + two-phase training (`TwoPhaseDataPoint`, `epochTwoPhase`, `copyTaskBinary`, `recallTaskBinary`)
- PyTorch-aligned NTM architecture (LSTM controller, separate head FCs from cell state, output FC from hidden++read, interpolation write)
- Current RSS tracking via `mach_task_info` (`getCurrentRssMB` in Variable.idr)
- RMSprop momentum (`rmspropValueClipMomentumDense` in Optimizer.idr, `rmsprop_vc_momentum_step` in C)
- Realistic 1K NTM-copy benchmark (fresh data + GC, `benchNtmCopy1k` / `bench_ntm_copy_1k`)
- C-backed BCE with logits (`bceWithLogitsVar`, tag 26 — fused sigmoid + BCE, single tape entry per output vector)
- C-backed LSTM cell op (`lstmCellVar`, tag 24 — fused bias+gates+cell/hidden update, single tape entry)
- C-backed addressing ops (`interpolateVar`/`shiftVar`/`focusVar`, tags 21-23 — replace ~1400 scalar entries per head)
- Buffer-passing MatVec→LstmCell (`matrixVectorMultiplyVarBufOut` + `lstmCellVarFromBufs` — bypass Variable materialization)
- Shadow ConstOps (tag 25 — gradient slots without values/pids, set via C bulk `tape_set_shadow_tags`)
- Dense optimizer (`DenseOptimizer`/`DenseOptimizerState` — C arrays indexed by pid_id, ~47% faster for NTM)
- C-bulk delta application (`applyDeltasAndSyncLayer`/`applyDeltasAndSyncNetwork` — bypass emap+sync)
- Persistent NtmMemBuf (C struct across timesteps, per-sequence reset, epoch-cached tape registration)
- Bias WeightBuf (LinearLayer/LstmLayer bias buffers, fused MatVec+Bias kernel)
- Learned LSTM h0/c0 (Xavier uniform init, named as WeightBufs, matching PyTorch nn.Parameter)
- Windowed convergence early stopping (`esThreshold`/`esWindow`/`esPatience` — replaces patience-based for NTM)
- Periodic forced GC for long NTM training (`forceGC` every 10 epochs)
- NTM recall batch=1 default (matching reference implementations; batch=16 for copy)
- NTM recall benchmark in bench-compare (`benchNtmRecall` / `bench_ntm_recall`)
- Periodic bit accuracy logging in NTM recall training loop
- NTM recall convergence (batch=1 converges; batch=16 plateaus due to gradient dilution across variable-structure sequences)
- PyTorch focus() NaN fix (clamp raised weights to prevent underflow when near-uniform weights + large gamma → 0/0)
