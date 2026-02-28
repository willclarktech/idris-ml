# NTM Recall Convergence Results

Experiments run with `bench/bench/scripts/convergence.py` to verify PyTorch NTM recall convergence under different configurations.

## Experiment A: Baseline (RMSprop, 100K iterations)

**Config**: LSTM controller, N=128, M=20, RMSprop lr=1e-4, value clip ±10, items=[2,6], seed=42

```
(running)
```

## Experiment B: Adam optimizer

**Config**: LSTM controller, N=128, M=20, Adam lr=1e-3, value clip ±10, items=[2,6], seed=42

```
(running)
```

## Experiment C: Adam + fixed 2 items

**Config**: LSTM controller, N=128, M=20, Adam lr=1e-3, value clip ±10, items=[2,2], seed=42

```
(running)
```
