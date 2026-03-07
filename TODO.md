# Backlog

## High Priority

| Item | Difficulty | Notes |
|------|-----------|-------|
| Cleaner port/adapter model for layers | L | Currently Layer.idr contains every layer type and all dispatch logic in one mutual block; consider typeclass-based extensibility |
| RMSprop momentum | S | PyTorch reference uses momentum=0.9 with RMSprop; current impl has no momentum term |

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
- Bounded NTM gamma (focus sharpening)
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
- NTM constant memory init 1e-6 (Collier & Beel stability)
- Controller output clipping [-20, 20] (`clampVar` in Variable.idr)
- Curriculum training (3 stages: len 1-3, 1-5, 1-8)
- Gradient clip norm 5.0 → 50.0 (Collier & Beel default)
- 3-element shift kernel + hot-start addressing
- NTM tanh memory bounding (Collier & Beel stability)
- NTM learned initial addressing (backprop through head weights + readHeadOutput)
- NTM associative recall example (content-based addressing)
- Unit test suite (`make test` / `make test-c`)
- Unified NTM head ops via NormalizationFunction parameter (Memory.idr)
- C-backed NTM memory ops (batchCosineSimilarityVar, readOpVar, writeOpVar — ~1.8x NTM speedup)
- Gaussian/normal distribution sampling (Sampler.idr, composable with init strategies)
- Automatic parameter naming (`autoName` in Layer.idr, type-based prefixes with collision-free scoping)
- PyTorch benchmarks (`bench/` directory — correctness tests, timing benchmarks, side-by-side comparison)
- NTM recall convergence verification (3 experiments: RMSprop baseline, Adam, Adam+2items — see `docs/ntm-convergence-results.md`)
- Convergence script CLI args (`--recall-controller`, `--recall-optimizer`, `--recall-clip`, etc.)
- NTM documentation extraction (`docs/ntm.md` — architecture, convergence, failure modes)
- LSTM layer (`LstmLayer` constructor in Layer.idr — gate computation, forget bias init, cell state extraction)
- Interpolation write (`interpolationWrite` in Memory.idr, `interpolationWriteVar` C-backed in Variable.idr)
- Softplus gamma (`forwardReadHeadUnbounded` in Memory.idr — `gamma = 1 + softplus(x)`, unbounded)
- RMSprop optimizer with value clipping (`rmsprop`, `rmspropValueClip` in Optimizer.idr)
- Binary vector data format + two-phase training (`TwoPhaseDataPoint`, `epochTwoPhase`, `copyTaskBinary`, `recallTaskBinary`)
- PyTorch-aligned NTM architecture (LSTM controller, separate head FCs from cell state, output FC from hidden++read, interpolation write)
- Current RSS tracking via `mach_task_info` (`getCurrentRssMB` in Variable.idr)
