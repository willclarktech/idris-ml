# Backlog

## High Priority

| Item | Difficulty | Notes |
|------|-----------|-------|
| CUDA support | M–L | Torch backend should work via `tensor_to_device("cuda")` — untested. Test script ready: `scripts/test_cuda_colab.sh`. See `docs/develop/cuda-testing.md`. Device type system ready: `Variable (CUDA 0)` compiles, `toDevice` + FFI bindings exist |
| Batched Variable forward path for RL | M | A2C/PPO/SAC Idris examples currently do per-step `forwardVarTensor` calls during the update phase (one forward per transition in the batch). Each call creates its own tape entries; a batch of 64 produces 64× the ops of a single batched forward. This makes Idris ~20× slower than PyTorch per update step and forces CI to use short rollouts. Adding a batched forward helper (shape `[batch, i] → [batch, o]` on the tensor path, with backward accumulation) would close most of the remaining compute-speed gap and let us match PyTorch's convergence ceilings (e.g. PPO-on-Pendulum at rollout=2048 → -353; currently shipping rollout=400 → -1200 for tractable CI) |
| Hyperparameter optimization — revisit | M | Current `scripts/sweep.sh` is grid-search only. This is fine for 1–2 hyperparams but infeasible for PPO/SAC which have 5+ tightly-coupled knobs (lr, γ, λ, clip ε, entropy coef, batch, rollout length, hidden sizes). First deliverable: rewrite the sweep driver to run random search with ASHA-style early termination, reusing the existing `make example-<name> <FLAG>=<val>` contract and RESULT-line parser. Second deliverable: evaluate Optuna integration as an external Python harness invoking the same make targets. Benefits both RL tuning and existing sweeps (transformer, NTM) |
| Publish to package managers | S–M | Make idris-ml installable via standard Idris 2 package channels. Investigate: (1) [pack](https://github.com/stefan-hoeck/idris2-pack) — the main Idris 2 package manager, needs a `pack.toml`; (2) add to the [pack-db](https://github.com/stefan-hoeck/idris2-pack-db) collection. Note: the C backend (`libidrisml`) needs special handling — pack doesn't natively support native library deps. May need a `postinstall` script or `make backend` instruction. Also consider publishing to GitHub Releases with prebuilt dylibs for common platforms |
| Torch backend: missing `tensor_alloc_ints` symbol | S–M | GPT, MNIST, and SeqClassify all crash on the torch backend with `Exception in foreign-procedure: no entry for "tensor_alloc_ints"` (surfaced by test-examples' new failure-dump). The FFI export is present in tape and MLX but missing from `packages/backends/backend_torch.cpp`. Add it mirroring the tape implementation. Verify with `make BACKEND=torch example-gpt GPT_ARGS="--epochs 200"` and the torch rows of `make test-examples` |

## Medium Priority

| Item | Difficulty | Notes |
|------|-----------|-------|
| Survey Glaive Research for usable ideas | S | [glaive-research.org](https://glaive-research.org/) is a nonprofit applying category theory to AI verification, explicitly publishing in Idris with dependent types. Overlap with idris-ml is substantial — shape-indexed tensors, type-safe automatic differentiation, categorical composition — and they may have working formalisms we can borrow. Review their Q4 2025 report and the "Building a Neural Network from First Principles using Free Categories and Para(Optic)" post, extract any directly adoptable ideas (free-category composition for layer chains, Para/Optic view of backprop, applicative-functor generalized transformers, categorical autodiff semantics), and file follow-up tickets for the ones worth implementing. Deliverable: a short `docs/develop/glaive-survey.md` with per-idea "adopt / borrow interface / inspiration only / out of scope" verdicts |
| Privacy-preserving ML (PPML) | M–L | Examples and layers for differential privacy (DP-SGD, gradient clipping + noise), secure aggregation, and homomorphic encryption (HE-friendly activations). Inspired by OpenMined/PySyft. Idris 2's type system could enforce privacy budgets at compile time (e.g., epsilon-tracking via phantom types). Start with DP-SGD example (clip per-sample gradients + Gaussian noise in optimizer), then HE-compatible polynomial activations |
| Static graph optimizations | L–XL | Compile-time operator fusion, memory planning via dependent types. Idris 2's type system knows tensor shapes at compile time — could fuse sequences like matmul+bias+relu into single kernels, plan memory reuse for same-shape intermediates, and eliminate dead computations. See `docs/static-vs-dynamic-graphs.md` |
| test-examples.expect threshold calibration | S | Thresholds in `test-examples.expect` were set from a single tape run on one machine. Run `make test-examples` across all three backends on a clean checkout and tighten anything that's too loose or loosen anything causing noise failures. Particular audit targets: MNIST at 5-epoch smoke (accuracy bound of 0.12 is just above random-chance 0.10 — jittery), and RL partial-convergence bounds (a2c, sac) where seed variance can straddle the threshold |
| NTM/DNC multi-seed convergence target | M | `test-examples-convergence` excludes NTM/DNC today because their default 9300+ epoch runs × 3 seeds is prohibitive. But NTM batch=1 seed sensitivity is exactly the canonical regression mode (gotchas.md:185 — seed=42 converges, seed=123 doesn't). Either (a) a reduced-epoch NTM/DNC lane that still catches hard divergence, or (b) a nightly job at full epochs. Picks up where `test-examples-convergence` stops |
| CI workflow iteration | S | `.github/workflows/test.yml` is a first-pass draft. Expect one round of tweaks after the first CI run: Chez Scheme package name on Ubuntu (may be `chezscheme-9.5` rather than plain `chezscheme`), Idris 2 bootstrap `SCHEME=` detection, cache path correctness (we cache `/usr/local/bin/idris2` + `/usr/local/lib/idris2` — verify `sudo make install` lands there), and whether `make install` works without interactive prompts |
| MLX (and eventually torch) backends in CI | M | Initial CI is tape-only. Add MLX matrix entry on `macos-latest` (`pip install mlx` + set `MLX_SITE`) to exercise the MLX path in CI. Libtorch is a bigger install and deserves a separate PR — defer. The existing graceful backend-skip in test-examples means each added backend is additive |

## Low Priority

| Item | Difficulty | Notes |
|------|-----------|-------|
| Opaque type-level Nats | M–XL | Idris 2 Peano Nats hang the compiler for dims > ~1000. Need machine-backed type-level naturals (like GHC TypeLits). Engage with Idris 2 upstream. Blocks identity layers (dropout, batch norm) at large dims. See `docs/gotchas.md` |
| Broadcasting | XL | Type-safe broadcasting. Key tension: expressiveness vs shape safety guarantees. Dex's typed index sets are the most promising model for a dependently-typed setting. See `docs/static-vs-dynamic-graphs.md` |
| `fromDouble` persistent leak | S | Partially fixed: `tensor_create_scalar` and `tensor_create` non-grad tensors are now non-persistent on MLX (freed by tape_reset). Remaining: Chez Scheme GC doesn't call `tensor_free`, so non-persistent tensors accumulate within one epoch until optimizer_step. ~15KB/epoch overhead, manageable |
| Reshaping layers | M | No current use case |
| Double DQN / Dueling DQN | S | Variants atop base DQN. Double DQN uses the online network to select actions and the target network to evaluate them (reduces maximization bias). Dueling DQN splits the Q-head into value + advantage streams. File once base DQN is stable |
| Prioritized experience replay / N-step returns / Rainbow | M | Value-based enhancements atop DQN. Prioritized replay samples transitions proportional to TD-error; N-step returns bootstrap after N steps instead of 1; Rainbow combines PER + N-step + Double + Dueling + noisy nets + distributional. Worth it only if we want to push DQN toward SOTA on harder envs |
| Q-learning on Taxi and FrozenLake | S | Same algorithm as the CliffWalking example, different envs. Low marginal algorithmic value but useful for Jupyter tutorials (FrozenLake stochastic slippery dynamics visualize well, Taxi exercises the 500-state table) |
| Value / Policy iteration (model-based tabular) | M | Requires exposing transition probabilities from stochastic gym envs (FrozenLake, Blackjack). Gym-side change — add `transitions : state -> action -> List (Double, state)` to the Env interface, or a separate `ModelBasedEnv` interface. Then Bellman-optimality iteration is ~30 lines |
| Tabular exploration bandits (UCB, Thompson sampling) | S | Multi-armed bandit algorithms. No current env motivates since toy-text envs are full MDPs, not bandits. Would need a `MultiArmedBandit` env (Gaussian or Bernoulli arms) in `Gym.ToyText` first |
| SAC on LunarLander-continuous | S | Follow-up once Box2D envs land. SAC's existing continuous-control pathway should apply directly to LunarLander-continuous with minimal changes |
| Box2D gym envs | L | LunarLander, BipedalWalker, CarRacing. Require a 2D rigid-body physics engine (contacts, joints, rotation). LunarLander is the simplest (~500 lines of rigid-body physics in pure Idris). BipedalWalker and CarRacing need articulated bodies and tile-based tracks respectively |
| Atari + MuJoCo gym envs | XL | Atari needs ROM emulation (ALE equivalent); MuJoCo needs an MJCF-compatible physics engine. Out of scope for pure Idris short term |
| Idris 2 C backend (custom) | XL | If RefC doesn't work out: custom Idris 2→C code generator optimized for tensor workloads. Much higher effort than RefC investigation above |
| CodeMirror Idris 2 mode | S–M | Jupyter kernel uses `"codemirror_mode": "haskell"` as a fallback. Investigate whether an Idris 2 CodeMirror grammar exists or could be written (CodeMirror 6 Lezer grammar). Would give proper syntax highlighting in JupyterLab |
| RefC backend adoption | S–M | Blocked on upstream Idris 2 RefC bugs. Our codebase is RefC-ready (zero Scheme FFI, Compat.Random, shims for missing runtime functions). RefC crashes in `idris2_trampoline` on nested ADTs (Tensor Functor map over Vect of STensor). Simple C FFI programs work. Revisit when Idris 2 releases post-0.8.0 with updated RefC runtime. See `docs/develop/refc-investigation.md` and `docs/develop/refc-upstream-bug.md` |
| Upstream Idris 2 RefC bug report | S | File issue on idris-lang/Idris2 for: RefC 0.8.0 trampoline crash on nested algebraic data types (Functor map over Vect of constructors). Repro, ASAN trace, and draft report in `docs/develop/refc-upstream-bug.md`. Also: (1) contrib System.Random needs `%foreign "C:..."` for RefC compat, (2) 0.8.0 runtime missing `idris2_negate_Double` etc. (fixed in main) |
| Unify `*_ARGS` convention for example targets | S | Each example target declares its own override variable (`GPT_ARGS`, `MNIST_ARGS`, `DQN_ARGS`, ...). test-examples hardcodes the name-to-variable mapping in a `case` statement. If someone renames `DQN_ARGS` → `DQN_EXTRA_ARGS`, test-examples silently falls back to defaults — no error, just slow CI. Either standardise on one name (e.g. `EXTRA_ARGS`) across all example-* targets, or declare the mapping as Make variables adjacent to the targets so drift breaks loudly |

**Explicitly not planned:** distributed training (infrastructure, not library), mixed precision/quantization (performance optimisation), model zoo (compositions of existing primitives), TorchScript (our type system is the compile-time analysis), bidirectional RNN (transformers have obsoleted), exotic losses (compose from primitives)

## Done

Package refactor:
- Moved `Generate.idr` (synthetic task-data generators: copy/recall/pattern/reversal/sorting) out of `idris-ml` core and into `idris-ml-examples/src/`, alongside its example consumers. New `packages/idris-ml-examples/test/` harness (mirrors idris-gym's test layout) hosts the moved `Test.Generate` suite; `make test-examples-unit` runs it. New `install-examples` Makefile target installs `idris-ml-examples` as a library so the test harness can depend on it. `Notebook.Prelude` drops the `import public Generate` re-export (the one notebook that referenced Generate did so in prose only, not a live cell). Shrinks the core library's surface and makes the separation between general-purpose training primitives and task-specific scaffolding explicit

RL algorithms suite:
- Tabular (Round 1): Q-learning on CliffWalking (`Example/QLearning.idr`, converges to -13 optimal), SARSA on CliffWalking (`Example/Sarsa.idr`, -19 safer path, classic off-policy vs on-policy contrast), first-visit Monte Carlo on Blackjack (`Example/MonteCarlo.idr`, win_rate ≈ 0.42 vs 0.28 random). Q-table reuses `Tensor [|S|, |A|] Double` to keep storage consistent with deep examples
- DQN on CartPole (`Example/Dqn.idr`, greedy eval 200/200) — online Q + `toDoubleNetwork` snapshot as frozen target, `RL.ReplayBuffer` for experience replay, epsilon-greedy with linear decay
- A2C on CartPole (`Example/A2c.idr`) — separate actor + critic with scope-prefixed paramIds (see below), GAE, mini-step rollouts; Idris 4/7 vs PyTorch 3/7 seed convergence at aligned config
- PPO on Pendulum (`Example/Ppo.idr`) — Gaussian policy with learnable log_std, clipped surrogate, K-epoch mini-batch updates; ships at rollout=400 CI config (greedy=-1572 vs PyTorch -1197 at same) — full convergence needs rollout=2048 (deferred: batched Variable forward)
- SAC on Pendulum (`Example/Sac.idr`) — tanh-squashed Gaussian actor, twin Q-networks, hard target sync (Polyak τ=0.005 deferred)
- Shared RL primitives: `packages/idris-ml/src/RL/ReplayBuffer.idr` (ring buffer with IOArray, uniform sampling; 9 unit tests) and `packages/idris-ml/src/RL/Gae.idr` (Schulman 2015 GAE, pure, 9 unit tests with hand-computed reference values)
- `Train.runTrainingIO` — IO-based epoch-function variant of `runTraining` for examples that need sampling / RNG threading inside the epoch (used by DQN/A2C/PPO/SAC; pure `runTraining` now thin-wraps it)
- **ParamId scoping fix**: A2C/PPO/SAC all construct multiple Variable-CPU networks registered with a single optimizer. Naive `autoName net` on each puts them in conflict: both register `ll0_weights` in the backend param registry, and the second call overwrites the first's entry — silently zeroing the first network's gradient flow. `autoNameScoped "actor_" net` (inlined per-example for now, see `Example/A2c.idr`) threads a prefix through to each layer's `nameLayer` so the consolidated weight tensors get distinct registry keys. This was a latent bug masked by single-seed convergence claims; CLAUDE.md now requires multi-seed convergence testing to prevent recurrence
- Policy docs: `docs/develop/reference-alignment.md` + CLAUDE.md strengthened with "no silent architectural pivots" rule and multi-seed convergence requirement
- PyTorch references: `torch_ref/models/{q_learning,sarsa,monte_carlo,dqn,a2c,ppo,sac}.py` + tests under `correctness/`, all passing

Architecture & infrastructure:
- `idris-gym` Gymnasium-parity API: `Env state action obs` interface with 4-tuple `step` (Outcome = Continue/Terminated/Truncated, Info dict), `Space` ADT (Discrete/Box/MultiBin/MultiDisc), `Gym.Rng` pure SplitMix64, `Gym.Wrapper.{TimeLimit, Record, Normalize, Action}` helper-function wrappers, `Gym.Vector` sync vector env. Envs: Classic Control (CartPole, MountainCar, MountainCarContinuous, Pendulum, Acrobot) + Toy Text (FrozenLake, CliffWalking, Taxi, Blackjack). 88 unit tests in `make test-gym`. Box2D/Atari/MuJoCo deferred (need physics engines or ROM emulators)
- Monorepo restructure: 6 packages under `packages/` — `idris-ml` (core lib), `idris-gym` (pure Idris RL environments), `idris-ml-examples` (examples), `backends` (C/C++), `jupyter` (kernel), `pytorch` (reference). Local install via `make install` + `IDRIS2_PACKAGE_PATH`. `Gym.Env` interface + `Gym.CartPole` extracted from Reinforce
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
- `test-examples` hardening: (a) crash and missing-RESULT FAIL paths now dump the captured output with a `  | ` prefix instead of silently discarding it — the bug that let gpt/mnist/seq-classify "crashes" stay undiagnosable; (b) `EXAMPLE_TIMEOUT` (default 600s) wraps each inner make via `timeout`/`gtimeout`, rc=124 reported distinctly from a crash; (c) `test-examples.expect` sidecar file threshold-checks RESULT values (e.g. `loss < 0.5`), with auto-conversion for fraction values like `sort_acc=6/6` and correct NaN rejection — presence-only RESULT no longer counts as pass. Threshold logic lives in `scripts/check-result.sh`
- `test-examples` word-split fix: `extra_args="GPT_ARGS=--epochs 200"` was passed unquoted to `$(MAKE)`, so shell word-splitting sent GNU Make 3.81 two tokens (`GPT_ARGS=--epochs` with value truncated + stray `200` goal) and Make errored with "No rule to make target 200". Root cause of the tape-backend crashes for every example whose ARGS contained a space. Switched to `case` + `"$$extra_args"` quoting
- Cross-backend Transfer in test-examples: added `example-transfer` (per-backend save/load sanity) and special-cased `example-transfer-demo` to run once per test-examples invocation when all 3 backends built, exercising the real tape → mlx → torch safetensors handoff
- `test-examples-convergence` target: RL subset (reinforce, q-learning, sarsa, mc, dqn, a2c, ppo, sac) at 3 seeds (42, 123, 7) against the same expect-file thresholds, requires ≥ 2/3 passes. Catches "ships at seed=42 only" regressions without the 5-seed runtime cost. NTM/DNC deferred (would need dedicated slower job)
- GitHub Actions workflow: `.github/workflows/test.yml` matrix across `ubuntu-latest` + `macos-latest` (both free unlimited on public repos), tape backend only. Idris 2 v0.8.0 built from source with actions/cache
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
