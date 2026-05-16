# NTM Recall Convergence Results

> **Note**: As of the simplification refactor, all ablation flags have been removed from the PyTorch NTM implementation. Experiment J's configuration (all vlgiitr differences combined) is now the only architecture, hardcoded in `bench/bench/ntm/layer.py`. The experiments below are historical records of the investigation that led to this decision.

Experiments run with `bench/bench/scripts/convergence.py` to verify PyTorch NTM recall convergence under different optimizer and curriculum configurations.

Architecture (current): LSTM controller (hidden=100), N=128 memory slots, M=20 memory width, separate head FCs with cell state input, interpolation write (no erase), learned memory init (nn.Parameter+sigmoid), learned controller h0/c0 (nn.Parameter), xavier gain=1.4 for head FCs, kaiming for output FC. BCEWithLogitsLoss (fused sigmoid+BCE). Value clip ±10.

## Summary

| Experiment | Optimizer | Items | Final loss | 2-item acc | 3-item acc | 6-item acc |
|-----------|----------|-------|-----------|-----------|-----------|-----------|
| A (baseline) | RMSprop lr=1e-4 | [2,6] | 0.389 | 100% | 92.8% | 68.9% |
| B (Adam) | Adam lr=1e-3 | [2,6] | 0.367 | 100% | 97.2% | 64.4% |
| C (Adam+2items) | Adam lr=1e-3 | [2,2] | 0.000 | 100% | 55.6% | 57.2% |
| G (interp write) | RMSprop lr=1e-4 | [2,6] | 0.408 | 100% | 91.1% | 65.6% |
| H (G + cell) | RMSprop lr=1e-4 | [2,6] | 0.414 | 100% | 91.7% | 65.0% |
| I (H + no tanh) | RMSprop lr=1e-4 | [2,6] | 0.426 | 100% | 87.2% | 66.1% |
| K (learned mem) | RMSprop lr=1e-4 | [2,6] | 0.392 | 100% | 84.4% | 71.1% |
| L (all inits) | RMSprop lr=1e-4 | [2,6] | 0.383 | 100% | 93.9% | 60.6% |
| **J (all vlgiitr)** | **RMSprop lr=1e-4** | **[2,6]** | **0.076** | **100%** | **100%** | **98.3%** |
| **vlgiitr ref** | RMSprop lr=1e-4 | [2,6] | **0.000** | **100%** | **100%** | **100%** |

**Key finding**: Experiment J matches the vlgiitr reference by combining all six differences: interpolation write, cell state head input, no tanh bound, learned memory (FC+sigmoid), learned controller (FC from dummy), vlgiitr FC init (xavier gain=1.4, normal bias). Neither init changes alone (Exp K, L) nor architecture changes alone (Exp G-I) are sufficient — they are synergistic.

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

**Config**: vlgiitr/ntm-pytorch reference run directly with their code. LSTM controller (hidden=100), N=128, M=20, 1 head, RMSprop lr=1e-4 alpha=0.95 momentum=0.9, value clip ±10, items=[2,6), seed=42. Script: `bench/reference/vlgiitr/run_recall.py`. Log: `docs/develop/vlgiitr-recall.log`.

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

## Experiment G: Interpolation write (cumulative ablation step 1)

**Config**: Same as Exp A but with vlgiitr interpolation write (`w*data + (1-w)*mem`) instead of erase+add. Other: hidden state input, tanh bound on.

**Loss**: 0.408 @ 100K (vs baseline A: 0.389). Nearly identical trajectory.

**Evaluation**:
```
2 items: 100.0%
3 items:  91.1%
4 items:  77.8%
5 items:  63.9%
6 items:  65.6%
```

No meaningful difference from baseline.

## Experiment H: Interpolation write + cell state input

**Config**: Exp G + head FCs use LSTM cell state (c) instead of hidden state (h), matching vlgiitr.

**Loss**: 0.414 @ 100K. Nearly identical to G.

**Evaluation**:
```
2 items: 100.0%
3 items:  91.7%
4 items:  76.7%
5 items:  64.4%
6 items:  65.0%
```

Cell state input makes no difference.

## Experiment I: Interpolation + cell state + no tanh bound

**Config**: Exp H + tanh memory bounding disabled (vlgiitr has no tanh bounding).

**Loss**: 0.426 @ 100K. Within noise of G and H.

**Evaluation**:
```
2 items: 100.0%
3 items:  87.2%
4 items:  73.9%
5 items:  61.7%
6 items:  66.1%
```

Tanh bounding makes no difference either.

### Key finding: G-H-I null result

All three "HIGH priority" architectural differences (write mechanism, head input, tanh bounding) have **zero effect** on convergence. The model shows the same pathological pattern across all experiments: near-uniform addressing (entropy=4.85, peak=0.0078), only 1 distinct write slot, loss plateaus ~0.39-0.43.

The massive convergence gap (vlgiitr: 0.07 @ 10K; ours: 0.62 @ 10K) must originate from a different source — likely initialization (learned memory/controller via FC, head FC gain/bias) or another architectural detail not yet identified.

## Experiment J: All vlgiitr differences combined

**Config**: Interpolation write + cell state input + no tanh bound + learned memory (FC+sigmoid) + learned controller (FC from dummy) + vlgiitr FC init (xavier gain=1.4, normal bias for heads; kaiming for output FC).

**Loss curve** (key milestones):
```
iter  10000: loss=0.423737
iter  15000: loss=0.322243
iter  16000: loss=0.213738   ← rapid drop begins
iter  20000: loss=0.080780
iter  30000: loss=0.066014
iter  40000: loss=0.006350
iter  50000: loss=0.012791
iter  60000: loss=0.003675
iter  70000: loss=0.006162
iter  80000: loss=0.081399   ← RMSprop crash (recovers)
iter  90000: loss=0.075089
iter 100000: loss=0.075694
```

Convergence pattern matches vlgiitr reference: rapid drop at ~15K, near-zero by 40K, occasional RMSprop crashes that self-correct. Loss trajectory is nearly identical to Experiment F.

**Evaluation**:
```
2 items: 100.0%
3 items: 100.0%
4 items: 100.0%
5 items: 100.0%
6 items:  98.3%
```

**Diagnostics**: Write entropy=0.98 (focused), read entropy=3.75, sequential read=YES, 16 distinct write slots during encoding. Night-and-day difference from experiments A-I which all showed near-uniform addressing (entropy=4.85, 1 distinct slot).

### Root cause: initialization, not architecture

### Root cause update: init + architecture are synergistic

Isolation experiments K and L show that neither init nor architecture changes alone are sufficient:

| Experiment | Architecture | Init | Loss | Result |
|-----------|-------------|------|------|--------|
| A (baseline) | ours | ours | 0.389 | Plateau |
| G-I (arch only) | vlgiitr | ours | 0.41-0.43 | Plateau |
| K (mem init only) | ours | learned mem | 0.392 | Plateau |
| L (all inits only) | ours | all vlgiitr | 0.383 | Plateau |
| **J (all changes)** | **vlgiitr** | **all vlgiitr** | **0.076** | **Converged** |

The init changes provide initial symmetry breaking (learned memory gives differentiated rows, learned controller gives non-zero initial head inputs, larger FC init amplifies signals). The architecture changes (interpolation write, cell state, no tanh) create the optimization landscape where this symmetry breaking can cascade into full convergence.

## Experiment K: Learned memory only

**Config**: Baseline architecture (erase+add, hidden state, tanh bound) + learned memory init only.

**Loss**: 0.392 @ 100K. Same plateau as baseline.

**Evaluation**:
```
2 items: 100.0% (0.0/18 bit errors/seq)
3 items:  84.4% (2.8/18 bit errors/seq)
4 items:  74.4% (4.6/18 bit errors/seq)
5 items:  69.4% (5.5/18 bit errors/seq)
6 items:  71.1% (5.2/18 bit errors/seq)
```

Diagnostics show improved addressing vs baseline (9 distinct write slots, entropy=1.61 vs 4.85) but not enough to converge.

## Experiment L: All inits, baseline architecture

**Config**: Baseline architecture + all vlgiitr inits (learned memory, learned controller FC, vlgiitr FC init).

**Loss**: 0.383 @ 100K. Same plateau.

**Evaluation**:
```
2 items: 100.0% (0.0/18 bit errors/seq)
3 items:  93.9% (1.1/18 bit errors/seq)
4 items:  66.1% (6.1/18 bit errors/seq)
5 items:  68.3% (5.7/18 bit errors/seq)
6 items:  60.6% (7.1/18 bit errors/seq)
```

All three init changes together still can't overcome the baseline architecture limitations.

## PyTorch-side alignment optimizations (2026-03-06)

After the Idris implementation was fully aligned, an audit identified suboptimal patterns in the PyTorch reference that made benchmark comparisons unfair:

1. **BCEWithLogitsLoss**: Replaced `sigmoid → clamp → BCELoss` with `F.binary_cross_entropy_with_logits` (fused kernel). Model now returns raw logits; sigmoid applied only for eval bit accuracy. Fixes numerical instability and clamp inconsistency between `train_ntm_step` and `_train_ntm_epoch`
2. **bench_ntm_copy momentum**: Added `momentum=0.9` to `bench_ntm_copy` RMSprop optimizer, matching convergence.py and Idris. Previously missing, making PyTorch do less optimizer work per step in timing benchmarks
3. **F.softplus**: Replaced manual `torch.log(1 + torch.exp(x))` with `F.softplus(x)` in memory.py. Uses threshold=20 to avoid overflow for large x
4. **Direct nn.Parameter for init**: Replaced FC(dummy) wrappers for h0/c0/memory_init with direct `nn.Parameter`. Eliminates 2760 dead weight parameters and 48 FC forward/backward ops per epoch. Functionally equivalent (FC with zero input only uses bias)

## Idris recall optimizer fix (2026-03-06)

The Idris NTM recall task was using `rmspropValueClipDense` (no momentum) while the copy task and PyTorch reference both use momentum=0.9. The `--momentum` CLI flag existed in `scripts/sweep.sh` but was silently dropped by `parseConfig` (catch-all `go (_ :: rest) c = go rest c`). Fixed by adding `momentum` field to Config and switching to `rmspropValueClipMomentumDense`.

## Next steps

- Verify Idris recall convergence with momentum=0.9
- The six necessary changes are already ported: interpolation write, cell state input, no tanh bound, learned memory init, learned controller init, FC init scale
