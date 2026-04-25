# Reference Implementation Alignment

## Policy

Idris examples and their PyTorch references must use **identical defaults** for all hyperparameters, architecture, and initialization. When a discrepancy is found, adopt **whichever is the better practice** in BOTH implementations.

When adding or changing an example, always update both Idris and PyTorch to match.

## Alignment Changes (2026-04)

### Idris defaults changed to match PyTorch

| Example | Parameter | Before | After |
|---------|-----------|--------|-------|
| NTM Copy | Batch size | 1 | 16 |
| NTM Recall | Batch size | 1 | 16 |
| DNC Copy | Batch size | 1 | 16 |
| DNC Copy | Memory size N | 32 | 128 |
| DNC Copy | Max seq length | 10 | 20 |
| DNC Recall | Batch size | 1 | 16 |
| DNC Recall | Memory size N | 32 | 128 |
| LSTM | Learning rate | 0.1 | 0.03 |
| LSTM | Seed | 123456 | 42 |
| Supervised | Seed | 123456 | 42 |
| RNN | Seed | 123456 | 42 |
| MNIST | Epochs | 2000 | 100 |
| NTM/DNC Copy/Recall | Eval test size | 20 | 100 |

### Idris layer implementations changed

| Layer | Change | Rationale |
|-------|--------|-----------|
| Transformer embedding | zeros → xavier uniform | Zero breaks symmetry; xavier is standard |
| LSTM hidden/cell init | xavier random → zeros | Zeros is standard, matches PyTorch |

### PyTorch references changed to match Idris (best practice)

| Reference | Change | Rationale |
|-----------|--------|-----------|
| GPT | Adam → AdamW (wd=0.01) | Weight decay prevents overfitting in LMs |
| GPT | Temperature 0.8 → 1.0 | Standard eval |
| MNIST CNN | Added dropout (0.25/0.5) | Regularization best practice |
| MNIST | Epochs 10 → 100 | Reasonable for comparison with early stopping |
| SeqClassify CNN | Added dropout (0.5) | Regularization best practice |
| LSTM | Added forget gate bias=1.0 | Jozefowicz et al. 2015, helps gradient flow |

### Added

| Item | Description |
|------|-------------|
| Reinforce script | `pytorch/torch_ref/scripts/reinforce.py` — was missing |

### LSTM hidden size (resolved)

Idris LSTM ties hidden=output in `LstmState`. Was using `{i=1, o=1}` (hidden=1) while PyTorch used `LinearLSTMCell(1, 4, 1)` (hidden=4 + output projection). Fixed by using `lstmLayer {i=1, o=4}` + `linearLayer {i=4, o=1}` to match.

## Alignment Changes (2026-04-22) — RL suite

### A2C divergence (resolved below)

During initial A2C port, the Idris side was pivoted to a **combined single-chain network** (output vector = `[logit_0, logit_1, value]`) because Idris' `Network` type is a linear chain and can't express PyTorch's branching actor-head + critic-head on a shared trunk. The pivot was not mirrored in the PyTorch reference, which retained the branching architecture. Hyperparameters also drifted: Idris ended up at `lr=3e-3, entropy=0.05`, PyTorch at `lr=7e-4, entropy=0.01`.

**Fix**: PyTorch `a2c.py` rewritten to use the same combined-chain architecture as Idris. Both sides now use `lr=3e-3, entropy=0.05, rollout=10, gamma=0.99, lam=0.95, value_coef=0.5`. Both converge to greedy eval ~200 on CartPole.

### PPO divergence (resolved below)

Same failure mode. Idris PPO used combined chain + `rollout=200, K=3, full-batch`; PyTorch ref used separate actor + critic + `rollout=2048, K=10, batch=64`. Architectural divergence hid whether Idris' plateau at -1500 was a config issue or an implementation bug.

**Fix**: PyTorch `ppo.py` rewritten to use combined chain (state-independent learnable `log_std`, mean and value on the same output head), and both sides adopt `rollout=2048, K=10, batch=64`. The stronger PyTorch settings are the baseline because Idris-matched settings (short rollout + no mini-batching) demonstrably do not converge for either side.

### Process note

This incident prompted a strengthening of the alignment policy in CLAUDE.md — see "Architectural alignment — DO NOT pivot silently." The key rule: if Idris' Network chain can't express the PyTorch architecture, **update PyTorch to match Idris**, not keep both diverged.

## Status

All known discrepancies resolved. Both implementations now use identical defaults.
