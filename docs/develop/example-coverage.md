# Example coverage

The set of examples in `packages/idris-ml-examples/src/Example/` plus
`packages/pytorch/torch_ref/scripts/` is the authoritative demonstration
of what `idris-ml` can do. This doc tracks (a) which architectures × problem
classes are covered, (b) per-example smoke + convergence runtimes and
thresholds, (c) gaps to fill, (d) redundancies, (e) wrong-shape problems.

Maintained by HPO Tracks B1, B2, and B6. Per-example runtime + threshold
columns are filled in by B2; coverage gaps in §3 drive B6.

> **Scope note (2026-07-27)**: the table and matrix below predate the
> newer examples (the HF inference + fine-tune family, DoubleDqn, the
> precision/dtype demos, the bench/diagnostic utilities, …) — they cover
> 27 of the 50 files in `packages/idris-ml-examples/src/Example/`
> (`ls packages/idris-ml-examples/src/Example/*.idr | wc -l` is the
> ground truth). The rows that exist are current; the newer examples are
> tracked by the smoke table in `scripts/test-e2e-examples.sh` +
> `test-examples.expect` rather than here.

## 1. Per-example detail

The training-mode column predates the `fit` driver: every example now
routes through `Ml.Fit` (`fitSupervised` for the supervised rows, a
custom `EpochStep` for the recurrent / two-phase / RL rows). The
`Backprop.idr` `epoch*` family the column originally mapped to was
deleted with the other old surfaces at the example-migration sweep
(closed 2026-06-19; see CHANGELOG).

Smoke args come from the per-example `smoke_args` table in
`scripts/test-e2e-examples.sh` (the `test-e2e-examples` recipe body).
Smoke threshold = `test-examples.expect`,
convergence threshold = `test-examples-convergence.expect`. Times marked
**measured** were observed in development; **estimated** numbers are
derived from documented per-epoch costs × default epoch count, and should
be replaced with measurements as part of B3 dogfood runs.

| Example | Architecture | Problem | Mode | Smoke args | Smoke time (tape) | Convergence threshold | Convergence time (tape) | Notes |
|---|---|---|---|---|---|---|---|---|
| Supervised | FC: 2→8→1 | XOR-style classification | Supervised | --epochs 5 | <5 s (estimated) | loss < 0.5 | ~3 s (measured) @ 1000 ep | |
| Rnn | RNN: 1→20→1 | Sine-wave regression | Recurrent | --epochs 5 | <5 s (estimated) | loss < 0.5 | ~5 s (estimated) @ 1000 ep | |
| Lstm | LSTM: 1→4→1 | Synthetic timeseries | Recurrent | --epochs 5 | <5 s (estimated) | loss < 0.7 | ~8 s (estimated) @ 2000 ep | |
| Gru | GRU: 1→4→1 | Synthetic timeseries (mirrors Lstm) | Recurrent | --epochs 5 | <5 s (measured) | loss < 0.05 | ~25 s (measured) @ ~1300 ep / seed=42 | **B6 added 2026-04-30**: 5/5 both backends, lr_find 1.0× cross-backend agreement at 0.475. Uses simplified GRU variant (r computed but unused) — see `reference-alignment.md` |
| Transformer | 2 blocks, 4 heads, dModel=16 (was 32, B4 2026-04-30) | Sequence sorting | Supervised (batched) | --epochs 5 | ~5 s (estimated) | sort_acc ≥ 0.8 | ~25 s (measured) @ 1000 ep, both backends 5/5 at 6/6 sort_acc | uses LayerNorm + Embedding + Attention |
| SeqClassify | embed→Conv1D→pool→FC | Synthetic waveform classification | Supervised | --epochs 5 | ~5 s (estimated) | loss < 0.5 | ~5 s (estimated) @ 1000 ep | |
| Mnist | LeNet (Conv2D×2 + FC) | MNIST digit classification | Supervised (full-pass) | --epochs 1 | ~2 min (estimated, full pass) | accuracy ≥ 0.85 | ~10 min (measured) @ 5 ep | uses Dropout; smoke runs 1 full pass |
| Gpt | Transformer (2 blocks, dModel=64) | Char-LM (embedded by default) | Supervised | --epochs 3 | ~10 s (measured) | bpc < 5.0 | ~40 s (measured) @ 30 ep, embedded; full convergence (~30 min @ 1000 ep, tinyshakespeare, val_bpc < 3.5) lives in `make example-gpt-full` | B3-fixes 2026-04-30: default shrunk to embedded/30 (~30 s), warmup proportional |
| NtmCopy | LSTM(100) + NTM(N=128, M=20) | Memory copy (seqLen 1-20) | TwoPhase BCE | --epochs 5 | ~30 s (estimated) | acc_short ≥ 0.9 | ~35 ms/ep × ≤ 50K ep, ~30 min full no-ES, expect ~15-20 min with WindowedPercentile ES (2026-05-08) | batch=1 (reverted from 16 to match Graves+vlgiitr references) |
| NtmAssociativeRecall | LSTM(100) + NTM(N=128, M=20) | Memory recall (K=2-6) | TwoPhase BCE | --epochs 5 | ~30 s (estimated) | acc_k2 ≥ 0.8 | similar to NtmCopy | batch=1 (reverted from 16 to match Graves+vlgiitr references; recall task documented as requiring batch=1, see gotchas.md:184) |
| DncCopy | LSTM(100) + DNC(N=32, M=20, R=1) | Memory copy (seqLen 1-10) | TwoPhase BCE | --epochs 5 --max-len 3 --batch 1 | ~10 s (measured during DNC revert) | acc_short ≥ 0.8 | 1733 ms/ep × 2K ep ≈ 58 min (measured); est. 13 h @ 46K to full | reverted from N=128 batch=16 — see `dnc-convergence-results.md`. Layer-perf rewrite is a separate TODO entry |
| DncAssociativeRecall | LSTM(100) + DNC(N=32, M=20, R=1) | Memory recall (K=2-6) | TwoPhase BCE | --epochs 5 --max-items 2 --batch 1 | ~10 s (estimated) | acc_k2 ≥ 0.6 | similar to DncCopy | |
| Reinforce | FC: 4→128→2 | CartPole (RL) | Custom REINFORCE | --epochs 10 | ~5 s (estimated) | avg_return ≥ 150 | ~100 s (measured) @ 2000 ep / seed=42 | |
| Dqn | FC: 4→64→64→2 | CartPole (RL) | Custom DQN | --epochs 10 | ~10 s (estimated) | avg_return ≥ 100 | ~1-3 min (estimated) @ 300 ep | batched per-batch forward in batchLossBatched (commit `3c24cd3`) |
| MountainCar | FC: 2→64→64→3 | MountainCar-v0 (RL) | Custom DQN + |v|-shaping | --epochs 5 | ~5 s (measured) | avg_return ≥ −160 | ~17 min (measured) @ 500 ep | B6 2026-04-30; 5/5 Idris (-152 to -102, mean -114) + 5/5 PyTorch (-150 to -106, mean -123); shaping=10·\|v'\| as dense intermediate signal |
| MountainCarCont | actor 2→64→64→1, twin Q 3→64→64→1 | MountainCarContinuous-v0 (RL, continuous) | Custom SAC + |v|-shaping | --epochs 5 | ~5 s (measured) | avg_return ≥ 85 | Idris ~11 min (measured, ES at ~step 6-8K); PyTorch ~5 min @ 30K | B6 2026-04-30; 5/5 Idris (94.86-95.47, mean 95.27) + 5/5 PyTorch (88.3-96.6, mean 92.96); same shaping rule as discrete MC; PyTorch needs full 30K to break through, Idris ES-terminates at ~7K |
| A2c | FC: 4→64→64→{2,1} (split) | CartPole (RL) | Custom A2C+GAE | --epochs 50 | ~30 s (estimated) | avg_return ≥ 150 | ~29 ms/ep × 5K ep ≈ 2.5 min (measured) | aligned to PyTorch — separate actor+critic per `reference-alignment.md` |
| Ppo | FC: 6→64→64→3 (categorical) + 6→64→64→1 critic | Acrobot (RL, discrete) | Custom PPO+GAE | --epochs 5 | ~30 s (estimated) | avg_return ≥ −150 | ~6 s/ep × 100 ep ≈ 10 min (measured); 5/6 Idris seeds -63 to -94 | **B3-fixes 2026-04-30**: env swapped Pendulum → Acrobot (discrete). PyTorch 5/5, Idris 5/6 (seed=4 reproducible bad-init collapse) |
| Sac | FC actor + 2× Q-nets | Pendulum (RL) | Custom SAC | --epochs 100 | ~10 s (estimated) | avg_return ≥ −500 | ~91 ms/ep × ~24K ep ≈ 36 min (measured) | uses polyak target sync |
| Sarsa | Q-table [12, 4] | CliffWalking (tabular) | Custom SARSA | full default 1000 ep | <1 s (measured) | avg_return ≥ −120 | <1 s (measured) | |
| QLearning | Q-table [12, 4] | CliffWalking (tabular) | Custom Q-learning | full default 1000 ep | <1 s (measured) | avg_return ≥ −120 | <1 s (measured) | |
| MonteCarlo | Q-table [400, 2] | Blackjack (tabular) | First-visit MC | full default 100K ep | ~1 s (estimated) | win_rate ≥ 0.3 | ~1 s (estimated) | |
| FrozenLake | Q-table [16, 4] | FrozenLake slippery 4×4 (tabular, stochastic) | Custom Q-learning | full default 10K ep | ~2 s (measured) | avg_return ≥ 0.4 | ~2 s (measured) | B6 2026-04-30; 5/5 Idris (0.66–0.80) + 5/5 PyTorch (0.56–0.75) |
| Taxi | Q-table [500, 6] | Taxi-v3 (tabular, deterministic) | Custom Q-learning | full default 20K ep | ~7 s (measured) | avg_return ≥ 5 | ~7 s (measured) | B6 2026-04-30; 5/5 Idris + 5/5 PyTorch all hit optimal +8 |
| Transfer | FC transfer cascade | Synthetic supervised tasks | Supervised | full default | TBD (measure in B3) | loss < 0.5 | ~5 s (estimated) | uses Checkpoint save/load across backends |
| Bench | Multi-model | Internal benchmark | Mixed | n/a | n/a | n/a | ~3 s (measured) | not in test-examples |
| Profile | Single-model | Internal benchmark | Supervised | n/a | n/a | n/a | per-epoch ms | not in test-examples |

### Smoke-time goals vs reality

Rule of thumb: **smoke ≤ 30 s per example on tape**. Examples currently
near or above the budget:

- **Mnist** at `--epochs 1` is ~2 min (full 60K-image pass). Keeping as-is
  — MNIST's per-epoch cost is the demonstration; shrinking to a partial
  pass would hide the data-loader behavior.
- **NtmCopy / NtmAssociativeRecall / A2c / Ppo** at their smoke configs
  are ~30 s each. Borderline; each has structural reasons not to shrink
  further (NTM needs ≥ a few epochs for two-phase state to settle; A2C
  needs ≥ a few episodes for any RL signal; PPO smoke is already 5 epochs
  × 1024 rollout × K=10 ≈ 80K inner updates on Acrobot).

All other examples sit comfortably below 30 s.

## 2. Layer × problem coverage matrix

Cells reference the example(s) demonstrating each combination.
Empty cell = no current example.

| Layer ↓ \\ Problem → | Synth-reg | Synth-cls | Synth-seq | MNIST | LM | Mem-copy | Mem-recall | RL-discrete | RL-cont | Tabular |
|---|---|---|---|---|---|---|---|---|---|---|
| **FC (Linear)** |  | Supervised, Transfer | SeqClassify |  |  |  |  | Reinforce, Dqn, A2c, Ppo | Sac |  |
| **RNN** | Rnn |  |  |  |  |  |  |  |  |  |
| **LSTM** |  | Lstm |  |  |  | NtmCopy, DncCopy | NtmAssociativeRecall, DncAssociativeRecall |  |  |  |
| **GRU** |  | Gru |  |  |  |  |  |  |  |  |
| **Transformer** |  |  | Transformer |  | Gpt |  |  |  |  |  |
| **Conv1D** |  |  | SeqClassify |  |  |  |  |  |  |  |
| **Conv2D** |  |  |  | Mnist |  |  |  |  |  |  |
| **NTM** |  |  |  |  |  | NtmCopy | NtmAssociativeRecall |  |  |  |
| **DNC** |  |  |  |  |  | DncCopy | DncAssociativeRecall |  |  |  |
| **BatchNorm** |  |  |  |  |  |  |  |  |  |  |
| **LayerNorm** |  |  | Transformer |  | Gpt |  |  |  |  |  |
| **Dropout** |  |  |  | Mnist |  |  |  |  |  |  |
| **Embedding** |  |  | Transformer |  | Gpt |  |  |  |  |  |
| **Residual** |  |  |  |  |  |  |  |  |  |  |
| **Attention** |  |  | Transformer |  | Gpt |  |  |  |  |  |
| **Q-table** |  |  |  |  |  |  |  |  |  | QLearning, Sarsa, MonteCarlo |

Architecture aliases:
- "FC" = layered `Linear` + activation (no recurrence/conv/attention)
- "Q-table" = `Tensor [|S|, |A|] Double`, no neural net
- BatchNorm/LayerNorm/Dropout/Embedding/Residual/Attention are sub-components,
  not standalone layers — coverage rows mean "exercised within this example",
  not "demonstrated in isolation"

## 3. Coverage gaps (drives B6)

### Layers without any example
- ~~**GRU**~~ — covered by Gru (B6 calibration spike, 2026-04-30; mirrors Lstm pattern-prediction at lr=0.5; 5/5 Idris + 5/5 PyTorch convergence). Side benefit: `Layer/Gru.idr` shipped without an `applyGeneric` implementation; this ticket added the pure-Double forward path needed by `evaluateRecurrent`.
- **Residual** — `Layer/Residual.idr` exists but unused. Most natural fit: a small ResNet-style FC block on MNIST or a deeper Transformer demonstration.
- **BatchNorm** — `Layer/BatchNorm.idr` exists but unused (Conv-based examples use it as a sub-component sporadically). A dedicated example is low priority.

### Gym envs without any example
Five of the nine shipped Gym envs have no example:
- ~~**MountainCar** (discrete)~~ — covered by MountainCar (B6 2026-04-30; DQN with |v|-magnitude reward shaping; 5/5 both backends, Idris -152 to -102 / PyTorch -150 to -106). Required prerequisite was the batched-forward DQN refactor (`3c24cd3`).
- ~~**MountainCarCont** (continuous)~~ — covered by MountainCarCont (B6 2026-04-30; SAC with |v|-magnitude reward shaping; 5/5 both backends, Idris mean 95.27 / PyTorch mean 92.96).
- ~~**Acrobot** (discrete)~~ — covered by Ppo (B3-fixes 2026-04-30; PPO env swap from Pendulum).
- ~~**Taxi** (tabular discrete)~~ — covered by Taxi (B6 2026-04-30; tabular Q-learning at α=0.1 γ=0.99 ε=0.1 over 20K episodes; 5/5 both backends hit optimal +8).
- ~~**FrozenLake** (tabular discrete, stochastic)~~ — covered by FrozenLake (B6 2026-04-30; tabular Q-learning on slippery 4×4 at α=0.1 γ=0.99 ε=0.3 over 10K episodes; 5/5 both backends with mean avg_return ~0.71).

### Cross-cutting gaps
- **Synthetic regression** has only Rnn. A simple FC regression example (Supervised does classification) would round out the smallest demonstrations.
- **Language modeling** has only Gpt. No autoregressive RNN-LM example — would demonstrate LSTM/GRU in an LM setting and contrast with the Transformer LM.
- **Embedding-only** demonstration — currently always inside Transformer/Gpt. A small word-embedding example (e.g., learning Word2Vec-style co-occurrence vectors) would demonstrate the layer in isolation.

## 4. Redundancies (consider consolidating)

Multiple examples on the same architecture×problem, intentionally:

- **NtmCopy + DncCopy** (and the matching Recall pair): same memory tasks, different memory architectures. Keep both — demonstrate the architectural progression NTM → DNC.
- **Sarsa + QLearning on CliffWalking**: same env, on-policy vs off-policy contrast. Keep both — pedagogically valuable.
- **Reinforce + Dqn + A2c on CartPole**: same env, three different algorithm families (policy gradient, value-based, actor-critic). Keep all three.
- **Ppo on Acrobot + Sac on Pendulum**: discrete vs continuous control, on-policy vs off-policy. Keep both.

No redundancies to consolidate — each pair/triplet demonstrates a distinct
architectural or algorithmic point, and the alignment policy means we'd be
removing valuable references for the PyTorch comparison too.

## 5. Wrong-shape problems (B2 decisions)

Three examples have a problem-config mismatch that prevents convergence at
default args even though the architecture is correct:

- ~~**Ppo on Pendulum**~~ — **resolved (B3-fixes, 2026-04-30)**: env swapped to
  Acrobot (discrete-action). At `rollout=1024, 100 rollouts, lr=3e-4`,
  PyTorch hits avg_return -63 to -106 across 5 seeds; Idris seed=42
  reaches -94. Convergence threshold updated from `≥ -800`
  (partial-convergence) to `≥ -150` (real convergence with margin).
- **Gpt on tinyshakespeare** at 1000 epochs ≈ 30 min on tape. Too slow
  for a default `make example-gpt` invocation; smoke uses the embedded
  1.3 KB corpus at 3 epochs (~10 s) for safety. **B2 decision**:
  shrink the *default* `make example-gpt` to use the embedded corpus +
  fewer epochs (~30 s target), and add a separate `example-gpt-full`
  target or document `--corpus tinyshakespeare --epochs 1000` as the
  convergence run. Keep the convergence-test threshold (`val_bpc < 3.5`)
  pinned to the full config since that's where it's been measured.
- **Dnc Copy/Recall** at PyTorch-aligned config (N=128, batch=16,
  max-len=20): tape backend can't run end-to-end in reasonable time
  (~5 min/epoch, ~10 days for the trajectory PyTorch finishes in 13 min).
  Already reverted on 2026-04-29 to N=32 batch=1 max-len=10. **B2 decision**:
  no change — the existing `DNC layer perf — re-enable PyTorch-aligned
  config` TODO entry is the right place to re-enable; B2 doesn't retune.

No other examples have a wrong-shape problem at their current defaults.

### Threshold updates needed

The audit didn't surface threshold mismatches between
`test-examples-convergence.expect` and what the examples actually achieve;
documented numbers (`reference-alignment.md`, `dnc-convergence-results.md`,
`ntm-convergence-results.md`) match the threshold rows. No
`*.expect` updates required as part of B2.

## 6. Status / next steps

- **B1**: complete. Coverage matrix + gaps filed in TODO under HPO B6.
- **B2**: complete (this audit). Three wrong-shape problems documented in §5
  for B3/B6 to act on; no `*.expect` updates required at current defaults.
  TBD cells in §1 will be filled in by B3 with measured numbers as each
  example is dogfooded.
- **B3**: next. Per-example LR + schedule tuning using `lr_find` (HPO A2).
  Records each tuning in `docs/develop/hyperparameter-tuning-2026.md`.
- **B4**: network-structure (width/depth) tuning using the generalized
  sweep harness (HPO A3).
- **B5**: multi-seed validation at the new defaults; updates
  `docs/develop/reference-alignment.md` pass rates.
- **B6**: fills coverage gaps from §3 — one new example per ticket. Also
  acts on the §5 swap recommendations (PPO env swap, GPT default shrink).
