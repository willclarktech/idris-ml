# Example coverage

The set of examples in `packages/idris-ml-examples/src/Example/` plus
`packages/pytorch/torch_ref/scripts/` is the authoritative demonstration
of what `idris-ml` can do. This doc tracks (a) which architectures × problem
classes are covered, (b) per-example smoke + convergence runtimes and
thresholds, (c) gaps to fill, (d) redundancies, (e) wrong-shape problems.

Maintained by HPO Tracks B1, B2, and B6. Per-example runtime + threshold
columns are filled in by B2; coverage gaps in §3 drive B6.

## 1. Per-example detail

The training-mode column maps to the epoch function in
`packages/idris-ml/src/Backprop.idr` (`epochNative`,
`epochRecurrentNative`, `epochTwoPhaseBceNative`,
`epochNativeTensorPreAccum`) or the example's own custom epoch (RL).

| Example | Architecture | Problem | Mode | Smoke time | Convergence target | Convergence time | Notes |
|---|---|---|---|---|---|---|---|
| Supervised | FC: 2→8→1 | XOR-style classification | Supervised | TBD | TBD | TBD | |
| Rnn | RNN: 1→20→1 | Sine-wave regression | Recurrent | TBD | TBD | TBD | |
| Lstm | LSTM: 1→4→1 | Synthetic timeseries | Recurrent | TBD | TBD | TBD | |
| Transformer | 2 blocks, 4 heads, dModel=32 | Sequence sorting | Supervised (batched) | TBD | TBD | TBD | uses LayerNorm + Embedding + Attention |
| SeqClassify | embed→Conv1D→pool→FC | Synthetic waveform classification | Supervised | TBD | TBD | TBD | |
| Mnist | LeNet (Conv2D×2 + FC) | MNIST digit classification | Supervised (full-pass) | TBD | acc ≥ 0.99 | ~10 min @ 5 ep | uses Dropout |
| NtmCopy | LSTM(100) + NTM(N=128, M=20) | Memory copy (seqLen 1-20) | TwoPhase BCE | TBD | TBD | TBD | |
| NtmAssociativeRecall | LSTM(100) + NTM(N=128, M=20) | Memory recall (K=2-6) | TwoPhase BCE | TBD | TBD | TBD | |
| DncCopy | LSTM(100) + DNC(N=32, M=20, R=1) | Memory copy (seqLen 1-10) | TwoPhase BCE | TBD | acc_short ≥ 0.7 | ~1.1 s/ep, ≥ 13h to full | reverted from N=128 batch=16 — see `docs/develop/dnc-convergence-results.md` |
| DncAssociativeRecall | LSTM(100) + DNC(N=32, M=20, R=1) | Memory recall (K=2-6) | TwoPhase BCE | TBD | TBD | TBD | |
| Gpt | Transformer (2 blocks, dModel=64) | Char-LM (tinyshakespeare) | Supervised | TBD | TBD | ~30 min @ 1000 ep | uses cosineWithWarmup schedule manually; A1 should let it use the new beforeEpoch hook |
| Reinforce | FC: 4→128→2 | CartPole (RL) | Custom REINFORCE | TBD | avg_return ≥ 195 / 5 seeds | ~100 s | |
| Dqn | FC: 4→64→64→2 | CartPole (RL) | Custom DQN | TBD | avg_return ≥ 150 / 5 seeds | TBD | |
| A2c | FC: 4→64→64→{2,1} (split) | CartPole (RL) | Custom A2C+GAE | TBD | avg_return ≥ 150 / 4-of-7 seeds | TBD | aligned to PyTorch — separate actor+critic per `docs/develop/reference-alignment.md` |
| Ppo | FC: 3→64→64→1 | Pendulum (RL) | Custom PPO+GAE | TBD | TBD | TBD | **wrong-shape**: rollout=400 doesn't converge; PyTorch needs rollout=2048. See §5 |
| Sac | FC actor + 2× Q-nets | Pendulum (RL) | Custom SAC | TBD | avg_return ≥ -250 / 5 seeds | TBD | uses polyak target sync |
| Sarsa | Q-table [12, 4] | CliffWalking (tabular) | Custom SARSA | TBD | TBD | <1 s | |
| QLearning | Q-table [12, 4] | CliffWalking (tabular) | Custom Q-learning | TBD | TBD | <1 s | |
| MonteCarlo | Q-table [400, 2] | Blackjack (tabular) | First-visit MC | TBD | win_rate ≥ 0.40 | <1 s | |
| Transfer | FC transfer cascade | Synthetic supervised tasks | Supervised | TBD | TBD | TBD | uses Checkpoint save/load across backends |
| Bench | Multi-model | Internal benchmark | Mixed | n/a | n/a | ~3 s | |
| Profile | Single-model | Internal benchmark | Supervised | n/a | n/a | per-epoch ms | |

## 2. Layer × problem coverage matrix

Cells reference the example(s) demonstrating each combination.
Empty cell = no current example.

| Layer ↓ \\ Problem → | Synth-reg | Synth-cls | Synth-seq | MNIST | LM | Mem-copy | Mem-recall | RL-discrete | RL-cont | Tabular |
|---|---|---|---|---|---|---|---|---|---|---|
| **FC (Linear)** |  | Supervised, Transfer | SeqClassify |  |  |  |  | Reinforce, Dqn, A2c | Ppo, Sac |  |
| **RNN** | Rnn |  |  |  |  |  |  |  |  |  |
| **LSTM** |  | Lstm |  |  |  | NtmCopy, DncCopy | NtmAssociativeRecall, DncAssociativeRecall |  |  |  |
| **GRU** |  |  |  |  |  |  |  |  |  |  |
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
- **GRU** — `packages/idris-ml/src/Layer/Gru.idr` is shipped but no example uses it. Most natural fit: a GRU-on-synthetic-timeseries mirror of the existing Lstm example, to demonstrate the simpler recurrent unit. Alternative: GRU controller in NTM (research curiosity, not aligned with PyTorch references).
- **Residual** — `Layer/Residual.idr` exists but unused. Most natural fit: a small ResNet-style FC block on MNIST or a deeper Transformer demonstration.
- **BatchNorm** — `Layer/BatchNorm.idr` exists but unused (Conv-based examples use it as a sub-component sporadically). A dedicated example is low priority.

### Gym envs without any example
Five of the nine shipped Gym envs have no example:
- **MountainCar** (discrete) — classic exploration challenge; pairs naturally with DQN or REINFORCE.
- **MountainCarCont** (continuous) — Box action space; pairs with PPO or SAC.
- **Acrobot** (discrete) — sparse reward; harder than CartPole, exercises the discrete-action algorithm suite on a non-trivial env.
- **Taxi** (tabular discrete) — 500-state table; pairs with Q-learning to extend the tabular suite.
- **FrozenLake** (tabular discrete, stochastic) — slippery dynamics demonstrate stochastic-MDP handling; pairs with Q-learning or Monte Carlo.

### Cross-cutting gaps
- **Synthetic regression** has only Rnn. A simple FC regression example (Supervised does classification) would round out the smallest demonstrations.
- **Language modeling** has only Gpt. No autoregressive RNN-LM example — would demonstrate LSTM/GRU in an LM setting and contrast with the Transformer LM.
- **Embedding-only** demonstration — currently always inside Transformer/Gpt. A small word-embedding example (e.g., learning Word2Vec-style co-occurrence vectors) would demonstrate the layer in isolation.

## 4. Redundancies (consider consolidating)

Multiple examples on the same architecture×problem, intentionally:

- **NtmCopy + DncCopy** (and the matching Recall pair): same memory tasks, different memory architectures. Keep both — demonstrate the architectural progression NTM → DNC.
- **Sarsa + QLearning on CliffWalking**: same env, on-policy vs off-policy contrast. Keep both — pedagogically valuable.
- **Reinforce + Dqn + A2c on CartPole**: same env, three different algorithm families (policy gradient, value-based, actor-critic). Keep all three.
- **Ppo + Sac on Pendulum**: same env, on-policy vs off-policy continuous control. Keep both.

No redundancies to consolidate — each pair/triplet demonstrates a distinct
architectural or algorithmic point, and the alignment policy means we'd be
removing valuable references for the PyTorch comparison too.

## 5. Wrong-shape problems (decisions for B2)

These examples have a problem-config mismatch that prevents convergence at
default args, even though the architecture is correct:

- **Ppo on Pendulum** at the current `RolloutLen=400`: doesn't converge to a
  meaningful policy; PyTorch reference also flagged in
  `docs/develop/reference-alignment.md` as needing `rollout=2048` (~15 hours
  on tape backend) for proper convergence. **B2 decision needed**: swap
  Pendulum → discrete-action env (Acrobot or CartPole-as-continuous) so the
  example actually demonstrates clipped-surrogate convergence at a
  CPU-feasible rollout length.

- **Gpt on tinyshakespeare** at the current 1000 epochs: ~30 min on tape, too
  slow for a default `make example-gpt`. The architecture is fine; the
  problem is corpus size + epoch count. **B2 decision needed**: shrink
  default to a smaller corpus or fewer epochs. Keep the full
  tinyshakespeare-1000 config as a separate convergence-only target.

- **Dnc Copy/Recall** at full PyTorch-aligned config (N=128, batch=16,
  max-len=20): tape backend can't run end-to-end in reasonable time
  (~5 min/epoch). Already reverted on Apr-29 to N=32 batch=1 max-len=10;
  re-alignment blocked on the existing **DNC layer perf** TODO entry, not
  retuned in B2.

## 6. Status / next steps

- **B1 (this doc)**: structure populated, runtime/threshold cells marked TBD pending B2.
- **B2**: per-example smoke timing + threshold audit. Fills in the TBD cells in §1, decides §5 swaps, updates `test-examples-convergence.expect` per example.
- **B3**: per-example LR + schedule tuning using `lr_find` (HPO A2). Records each tuning in `docs/develop/hyperparameter-tuning-2026.md`.
- **B4**: network-structure (width/depth) tuning using the generalized sweep harness (HPO A3).
- **B5**: multi-seed validation at the new defaults; updates `docs/develop/reference-alignment.md` pass rates.
- **B6**: fills coverage gaps from §3 — one new example per ticket.
