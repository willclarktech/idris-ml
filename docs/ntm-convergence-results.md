# NTM Recall Convergence Results

Experiments run with `bench/bench/scripts/convergence.py` to verify PyTorch NTM recall convergence under different optimizer and curriculum configurations.

Architecture: LSTM controller (hidden=100), N=128 memory slots, M=20 memory width, separate head FCs, output = FC(controller_hidden + read_vector). BCELoss. Value clip ±10.

## Summary

| Experiment | Optimizer | Items | Final loss | 2-item acc | 3-item acc | 6-item acc |
|-----------|----------|-------|-----------|-----------|-----------|-----------|
| A (baseline) | RMSprop lr=1e-4 | [2,6] | 0.389 | 100% | 92.8% | 68.9% |
| B (Adam) | Adam lr=1e-3 | [2,6] | 0.367 | 100% | 97.2% | 64.4% |
| C (Adam+2items) | Adam lr=1e-3 | [2,2] | 0.000 | 100% | 55.6% | 57.2% |
| **vlgiitr ref** | RMSprop lr=1e-4 | [2,6] | **0.000** | **100%** | **100%** | **100%** |

**Key finding**: The vlgiitr reference implementation converges fully (100% accuracy, 2-6 items) with the same optimizer and hyperparameters as our Experiment A. Our model (A-C) plateaus at 65-93% on 3-6 items. The difference is architectural, not optimizer/curriculum related. See [Experiment F (vlgiitr reference)](#experiment-f-vlgiitr-reference) for details.

## Experiment A: Baseline (RMSprop, 100K iterations)

**Config**: LSTM controller, N=128, M=20, RMSprop lr=1e-4 alpha=0.95 momentum=0.9, value clip ±10, items=[2,6], seed=42

**Loss curve** (every 10K iterations):
```
iter  10000: loss=0.6218
iter  20000: loss=0.5861
iter  30000: loss=0.5576
iter  40000: loss=0.5223
iter  50000: loss=0.4695
iter  60000: loss=0.4599
iter  70000: loss=0.4271
iter  80000: loss=0.4128
iter  90000: loss=0.3985
iter 100000: loss=0.3894
```

Loss decreases steadily but slowly — still improving at 100K. Random baseline for BCE with items=[2,6] is ~0.69.

**Evaluation accuracy**:
```
2 items: 100.0%
3 items:  92.8%
4 items:  74.4%
5 items:  77.8%
6 items:  68.9%
```

## Experiment B: Adam optimizer

**Config**: LSTM controller, N=128, M=20, Adam lr=1e-3, value clip ±10, items=[2,6], seed=42

**Loss curve** (every 10K iterations):
```
iter  10000: loss=0.6235
iter  20000: loss=0.5788
iter  30000: loss=0.5483
iter  40000: loss=0.4860
iter  50000: loss=0.4235
iter  60000: loss=0.4402
iter  70000: loss=0.4136
iter  80000: loss=0.4043
iter  90000: loss=0.3807
iter 100000: loss=0.3665
```

Adam converges ~6% lower loss than RMSprop at 100K (0.367 vs 0.389). Both are still improving.

**Evaluation accuracy**:
```
2 items: 100.0%
3 items:  97.2%
4 items:  69.4%
5 items:  77.2%
6 items:  64.4%
```

Adam matches RMSprop on 2 items but achieves better 3-item accuracy (97% vs 93%). Performance on 4-6 items is similar (both ~65-78%).

## Experiment C: Adam + fixed 2 items

**Config**: LSTM controller, N=128, M=20, Adam lr=1e-3, value clip ±10, items=[2,2], seed=42

**Loss curve** (every 10K iterations):
```
iter  10000: loss=0.2003
iter  20000: loss=0.0144
iter  30000: loss=0.0003
iter  40000: loss=0.0005
iter  50000: loss=0.0001
iter  60000: loss=0.0003
iter  70000: loss=0.0014
iter  80000: loss=0.0000
iter  90000: loss=0.0000
iter 100000: loss=0.0002
```

Converges to near-zero loss by ~25K iterations. Occasional spikes (e.g., 0.008 at 92.5K) suggest rare difficult sequences, but the model recovers instantly.

**Evaluation accuracy**:
```
2 items: 100.0%
3 items:  55.6%
4 items:  61.7%
5 items:  43.9%
6 items:  57.2%
```

Perfect on trained 2-item sequences. No generalization to 3+ items — the model overfits to the 2-item structure.

## Conclusions

1. **The vlgiitr reference converges fully**: 100% accuracy on 2-6 items with RMSprop, no curriculum needed. This proves the task is solvable with the right architecture.

2. **Our model has an architectural problem, not an optimizer/curriculum problem**: Experiments A-B plateau at 65-93% with the same optimizer that gives 100% in the vlgiitr reference. Curriculum (Exp C) doesn't help either.

3. **Three architectural differences are the prime suspects**: separate head FCs (vs monolithic), cell state input (vs hidden state), and interpolation write (vs erase+add). These should be investigated in order of priority.

4. **RMSprop is unstable but self-correcting**: the vlgiitr reference experienced two catastrophic crashes (loss spiking to 0.86) during training but recovered each time within ~5K iterations. This suggests RMSprop with value clipping ±10 is adequate but brittle.

## Architecture change: non-learnable addressing (post-experiments A-C)

Experiments A-C used `nn.Parameter` for initial addressing weights (learnable, hot-start on slot 0) with `project_addressing()` clamping after each optimizer step. Investigation of the vlgiitr/ntm-pytorch reference revealed three differences:

| Aspect | Experiments A-C | vlgiitr reference |
|--------|----------------|-------------------|
| Addressing init | `nn.Parameter` (learnable) | `torch.zeros(N)` (plain tensor) |
| Read head output init | `nn.Parameter(torch.zeros(m))` | `kaiming_uniform_` tensor |
| `project_addressing()` | Clamp + renormalize after every step | Not needed |

The learnable addressing likely contributed to the 4-6 item plateau: gradients flow back through cloned addressing parameters, and the optimizer pushes them toward degenerate solutions. The `project_addressing()` post-step is a band-aid that also interferes with gradient flow (uses `torch.no_grad()`). In vlgiitr, addressing starts from zeros every sequence and is entirely determined by the controller.

Experiments D and E below use the fixed architecture matching vlgiitr.

## Experiment D: Fixed architecture, RMSprop (vlgiitr match)

**Config**: Non-learnable addressing (zeros), kaiming read output init, LSTM controller, N=128, M=20, RMSprop lr=1e-4 alpha=0.95 momentum=0.9, value clip ±10, items=[2,6], seed=42

*(pending — run with `make bench-convergence-recall`)*

## Experiment E: Fixed architecture, Adam

**Config**: Non-learnable addressing (zeros), kaiming read output init, LSTM controller, N=128, M=20, Adam lr=1e-3, value clip ±10, items=[2,6], seed=42

*(pending — run with `cd bench && uv run python -u -m bench.scripts.convergence --task recall --recall-optimizer adam --recall-lr 1e-3`)*

## Experiment F: vlgiitr reference (ground truth)

**Config**: vlgiitr/ntm-pytorch reference run directly with their code. LSTM controller (hidden=100), N=128, M=20, 1 head, RMSprop lr=1e-4 alpha=0.95 momentum=0.9, value clip ±10, items=[2,6), seed=42. Script: `bench/reference/vlgiitr/run_recall.py`. Log: `logs/vlgiitr-recall.log`.

**Loss curve** (every 10K iterations):
```
iter  10000: loss=0.0691
iter  20000: loss=0.0212
iter  30000: loss=0.0200
iter  40000: loss=0.0264
iter  50000: loss=0.0001
iter  60000: loss=0.0000
iter  70000: loss=0.0032
iter  80000: loss=0.0174
iter  90000: loss=0.0000
iter 100000: loss=0.0000
```

Convergence is rapid: loss drops from 0.69 to 0.07 by iter 10K. Two catastrophic crashes occurred (iter ~12.5K to 0.52, iter ~34.5K to 0.86) where training temporarily reset to random-level loss, but RMSprop recovered both times within ~5K iterations. After 60K, loss is essentially zero with rare small spikes.

Training completed in 27.1 min (61.5 it/s).

**Evaluation accuracy (10 trials each)**:
```
2 items: 100.0%
3 items: 100.0%
4 items: 100.0%
5 items: 100.0%
6 items: 100.0%
```

**Comparison with our model**: The vlgiitr reference achieves 100% on all item counts with the same optimizer and hyperparameters that give our model only 69-93% on 3-6 items. The difference must be architectural.

### Key architectural differences (investigation targets)

| Aspect | vlgiitr reference | Our model (Exp A-C) | Priority |
|--------|-------------------|---------------------|----------|
| Head param FCs | 6 separate FCs per head | 2 monolithic FCs (one per head) | **HIGH** |
| Head input | LSTM **cell state** (c) | LSTM **hidden state** (h) | **HIGH** |
| Write mechanism | `w*data + (1-w)*mem` (interpolation) | `mem*(1-w*e) + w*a` (erase+add) | **HIGH** |
| Memory init | Learned FC + sigmoid | Constant 1e-6 | Medium |
| Tanh bounding | None | Applied after every write | Medium |
| Controller state init | Learned FC(dummy) | Learned nn.Parameter | Low |
| FC init | xavier gain=1.4, bias N(0,0.01) | xavier gain=1.0, bias zeros | Low |
| Output FC init | kaiming_uniform_ | xavier_uniform_ | Low |
| γ (sharpening) | 1 + softplus(x) → [1,∞) | 1 + 4*sigmoid(x) → [1,5] | Low |

The top three differences are likely the root cause:
1. **Separate head FCs** allow each head parameter (key, β, g, s, γ, erase, add) to have its own input-to-output mapping, giving more capacity
2. **Cell state input** gives heads access to the full LSTM internal state rather than the gated hidden state
3. **Interpolation write** is simpler (no separate erase vector) and may be easier to learn

## Next steps

- Investigate the top-3 architectural differences systematically (separate head FCs, cell state, write mechanism)
- Run experiments D and E with fixed addressing if still needed
- Port vlgiitr write mechanism (interpolation) to our model as experiment
