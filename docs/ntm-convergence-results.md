# NTM Recall Convergence Results

Experiments run with `bench/bench/scripts/convergence.py` to verify PyTorch NTM recall convergence under different optimizer and curriculum configurations.

Architecture: LSTM controller (hidden=100), N=128 memory slots, M=20 memory width, separate head FCs, output = FC(controller_hidden + read_vector). BCELoss. Value clip ±10.

## Summary

| Experiment | Optimizer | Items | Final loss | 2-item acc | 3-item acc | 6-item acc |
|-----------|----------|-------|-----------|-----------|-----------|-----------|
| A (baseline) | RMSprop lr=1e-4 | [2,6] | 0.389 | 100% | 92.8% | 68.9% |
| B (Adam) | Adam lr=1e-3 | [2,6] | 0.367 | 100% | 97.2% | 64.4% |
| C (Adam+2items) | Adam lr=1e-3 | [2,2] | 0.000 | 100% | 55.6% | 57.2% |

**Key finding**: Both A and B learn 2-3 item recall well (92-100%) within 100K iterations, but plateau for 4-6 items. Experiment C converges fully on 2-item sequences but doesn't generalize to more items. This confirms **curriculum training is needed** for the full range.

Adam (B) converges slightly faster than RMSprop (A) and achieves better 3-item accuracy (97% vs 93%), but neither solves 4+ items without curriculum.

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

1. **The architecture works**: the model can fully solve 2-item recall (100% accuracy). This confirms the LSTM controller + separate head FCs + output_fc(cat(hidden, read)) architecture is correct.

2. **Adam slightly better than RMSprop**: 3% lower final loss, 4 percentage points better on 3-item accuracy. But neither optimizer alone solves 4+ items.

3. **Curriculum is essential**: training on items=[2,6] directly plateaus. The model learns 2-3 item patterns but struggles to generalize to higher counts. A staged curriculum (start with 2 items, advance to 3, then 4-6) should enable full convergence, as demonstrated by the idris-ml NTM copy task.

4. **More iterations may help**: both A and B loss curves are still declining at 100K. Running to 200-500K might reach better accuracy, but curriculum is the more principled approach.

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

## Next steps

- Run experiments D and E with fixed architecture
- Clone vlgiitr/ntm-pytorch reference and run their recall task as ground truth
- If D/E still plateau, investigate curriculum on top of fixed architecture
- Compare with reference implementation results (vlgiitr claims to generalize to 20+ items at 100K iterations)
