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

## Status

All known discrepancies resolved. Both implementations now use identical defaults.
