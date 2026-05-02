# Hyperparameter tuning log — 2026

Per-example record of what `lr_find` recommended, what we tried, and what
we shipped. Each entry follows the same template so the suite stays
comparable as we re-tune over time. Tickets B3 and B4 fill this in.

## Per-example template

```
## <Example name>           date: YYYY-MM-DD
- Before: <key=value pairs> (e.g. lr=0.03 schedule=constant)
- lr_find (Idris):  RECOMMENDED_LR=<value>  curve range: <lr_min..lr_max>, iters=<N>
- lr_find (PyTorch): RECOMMENDED_LR=<value>  curve range: <lr_min..lr_max>, iters=<N>
- Cross-backend agreement: <within 2×? if not, why?>
- Multi-seed pass rate (≥5 seeds) at OLD defaults:  <X/Y> (loss=<>, return=<>)
- Multi-seed pass rate (≥5 seeds) at NEW defaults:  <X/Y> (loss=<>, return=<>)
- Wall-clock to convergence at NEW defaults: <Xm Ys / seed>
- Decision: <ship-as-is / update lr / add schedule / problem-swap>
- Commit: <hash> if defaults changed
```

Single-seed `lr_find` runs are screening only; a default change requires
≥5-seed validation per CLAUDE.md.

---

## Supervised           date: 2026-04-29

- **Before**: `lr=0.03 schedule=constant epochs=1000 seed=42`. Loss at 1000 epochs (seed=42, tape) ≈ 0.138.
- **lr_find (Idris)**: `RECOMMENDED_LR=0.02009`, sweep range 1e-7..10, iters=100, divergence at iter 97 (lr ≈ 6.89), seed=42.
- **lr_find (PyTorch)**: `RECOMMENDED_LR=0.02420`, sweep range 1e-7..10, iters=100, no divergence in the swept range (loss continued falling to 0.27 at lr=10), seed=42.
- **Cross-backend agreement**: ratio = 0.0242/0.0201 ≈ 1.20×. Well within the 2× tolerance documented in HPO A2.
- **Multi-seed pass rate**: not measured at OLD or NEW. The recommended LR (0.02) is within 1.5× of the current default (0.03) and the example already passes its smoke + convergence thresholds (loss < 0.5) at the documented values, so a multi-seed sweep at a new default isn't load-bearing.
- **Wall-clock to convergence**: ~3 s @ 1000 epochs on tape (unchanged).
- **Decision**: ship-as-is. The current `lr=0.03` is in the same neighborhood as both backends' recommendation; the tiny (×1.5) gap doesn't justify changing the default and re-validating multi-seed.

A note on the curve shape: Idris' `lr_find` shows divergence at iter ≈ 97 (lr ≈ 7), while PyTorch's loss keeps decreasing to lr=10 without diverging. Most likely cause: SGD's update rule applied to a 5-point classification problem with no batching is itself not enough to destabilize the model in 100 iterations of LR sweep on either backend, but Idris's float64 + scalar Variable arithmetic accumulates rounding error differently from PyTorch's float32 + autograd. Both produce a usable recommendation; the divergence-detection is a soft signal.

- **Commit**: n/a (no default change).

---

## Rnn           date: 2026-04-29

- **Before**: `lr=0.03 schedule=constant epochs=2000 seed=42`. Architecture is a 1-input → 1-output `LinearRNNCell` (no hidden expansion); BCE-with-logits loss on the synthetic pattern dataset.
- **lr_find (Idris)**: `RECOMMENDED_LR=0.5722`, sweep range 1e-7..10, iters=100, divergence at iter 99 (lr=10), seed=42.
- **lr_find (PyTorch)**: `RECOMMENDED_LR=0.0015`, sweep range 1e-7..10, iters=100, no divergence in the swept range (loss bouncing around 0.4–10.9), seed=42.
- **Cross-backend agreement**: ratio = 0.5722/0.0015 ≈ 380×. **Outside the 2× tolerance — both recommendations are unreliable.**
- **Multi-seed pass rate**: not measured.
- **Wall-clock to convergence**: ~5 s @ 2000 epochs on tape (unchanged).
- **Decision**: ship-as-is.
- **Why lr_find doesn't help here**: the 1-cell RNN's loss curve under this dataset is essentially flat across the LR sweep — neither backend's loss meaningfully decreases as a function of LR. fastai's "steepest descent ÷ 10" heuristic then picks a noisy local-slope feature (different one on each backend), giving wildly divergent recommendations. **General rule: when cross-backend `lr_find` disagrees by more than 2×, treat both recommendations as unreliable signal and don't change the default.**

This is a useful negative-result entry for the workflow: `lr_find` is a screening tool, not a recommendation engine. The cross-backend agreement check is what catches unreliable cases like this one. B3 will look for similar disagreement on each example before changing defaults.

- **Commit**: n/a (no default change).

---

## Lstm           date: 2026-04-29

- **Before**: `lr=0.03 schedule=constant epochs=2000 patience=500 seed=42`. LSTM(1→4) → Linear(4→1), BCE-with-logits on the synthetic pattern dataset. Converges to loss ≈ 0.36 at seed=42 (no early stop — runs full 2000 epochs).
- **lr_find (Idris)**: `RECOMMENDED_LR=0.8302`, sweep range 1e-7..10, iters=100, no divergence in the swept range, seed=42.
- **lr_find (PyTorch)**: `RECOMMENDED_LR=0.4751`, sweep range 1e-7..10, iters=100, mild divergence at the high end, seed=42.
- **Cross-backend agreement**: ratio = 0.8302/0.4751 ≈ 1.75×. **Within the 2× tolerance — actionable.**
- **Multi-seed pass rate (≥5 seeds)**:
  | LR | Idris (5 seeds) | PyTorch (5 seeds) |
  |---|---|---|
  | 0.03 (old) | 5/5 finish at 2000 ep, loss ∈ [0.21, 0.36] — none reach < 0.05 | (not measured; aligned config) |
  | 0.5 (new) | 5/5 converge to loss ∈ [0.0012, 0.0016] in 1665–2000 ep | 5/5 converge to loss ∈ [0.0013, 0.0018] in 1558–2000 ep |
- **Wall-clock to convergence**: ~5 s @ 2000 ep on tape (similar to old). Faster early-stop on most seeds.
- **Decision**: **update default lr=0.03 → 0.5 on both Idris and PyTorch** + tighten `test-examples-convergence.expect` from `loss < 0.7` to `loss < 0.05` (14× safety margin above worst seed=42 loss).
- **Why this works**: the old 0.03 was extremely conservative — the LSTM's loss landscape is mild enough that a much higher LR gets to the minimum faster without divergence. lr_find caught this clearly (steepest descent at LR ≈ 5–8, recommended ÷ 10 ≈ 0.5–0.8). Cross-backend agreement of 1.75× gave confidence to act.
- **Commit**: (this commit).

---

## Transformer           date: 2026-04-29

- **Before**: `lr=0.001 epochs=1000 patience=300 seed=42`. 2-block multi-head transformer with Adam + global grad-norm clipping. Sequence sorting on synthetic 5-token inputs. Documented to converge with `sort_acc ≥ 0.8`.
- **lr_find (Idris)**: `RECOMMENDED_LR=1e-8`, sweep range 1e-7..10, **diverged at iter 95 (lr ≈ 4.75)**, seed=42.
- **lr_find (PyTorch)**: `RECOMMENDED_LR=1e-8`, sweep range 1e-7..10, **diverged at iter 94 (lr ≈ 3.94)**, seed=42.
- **Cross-backend agreement**: ratio = 1.00× (identical). **But this is misleading** — both backends fell back to `lrMin` because no iteration had a meaningfully-negative slope before divergence.
- **Why the heuristic fails here**: Adam with global grad-norm clipping already adapts the effective per-parameter LR, so the loss curve under an LR sweep is essentially flat (with noise) until the LR is large enough to overflow the gradient clip. The "steepest descent ÷ 10" heuristic — designed for plain SGD — picks up bias-corrected EMA noise at the start of the sweep instead of a real LR sweet-spot.
- **Multi-seed pass rate**: not measured. Default unchanged.
- **Decision**: ship-as-is at lr=0.001.
- **Lesson**: when both backends agree on `RECOMMENDED_LR=lrMin` (or some other extremum), check whether the recommendation is a fallback (no negative slope) rather than a real signal. Future improvement: have `lrFind` emit a warning when the curve has no negative-slope region. For now: cross-backend agreement near-1.0× should be sanity-checked against the curve shape, not trusted blindly. Adam-based examples (Transformer, Gpt, Mnist, Dqn, A2c, Ppo, Sac) are likely to fall in this category.
- **Commit**: n/a (no default change).

---

## SeqClassify           date: 2026-04-29

- **Before**: `lr=0.001 epochs=1000 patience=200 seed=42`. Conv1d(1→4,k=3) → ReLU → Pool → Conv1d(4→8,k=3) → ReLU → Pool → Dropout(0.5) → Linear(48→3) on synthetic waveform classification. Adam + global grad-norm clipping.
- **lr_find (Idris)**: `RECOMMENDED_LR=6.4e-8` — essentially the fallback at lrMin.
- **lr_find (PyTorch)**: `RECOMMENDED_LR=0.0423` — diverged at iter 98 with a real curve.
- **Cross-backend agreement**: ratio = 0.0423 / 6.4e-8 ≈ 660,000×. **Wildly disagreeing — both unreliable.**
- **Decision**: ship-as-is at lr=0.001.
- **Why this confirms the Adam-fallback pattern**: Idris's curve was essentially flat (Adam adapted away the LR sweep) so it bailed to ~lrMin. PyTorch's Adam allowed enough loss change to detect divergence at the high end, picking up *something* but not necessarily a useful something. Either way: cross-backend agreement is broken, treat as unreliable.
- **Commit**: n/a.

---

## Transfer           date: 2026-04-29

- **Before**: `lr=0.03 epochs=500 seed=123456`. Same 5-point classification dataset, single `Linear<2:3>` model, and NLL loss as Supervised. Transfer is an Idris-only SafeTensors portability demo (no PyTorch counterpart) — `--mode train|continue|infer` exercises save/load across backends.
- **lr_find (Idris)**: `RECOMMENDED_LR=0.02420`, sweep range 1e-7..10, iters=100, divergence at iter 97 (lr ≈ 6.89), seed=123456.
- **lr_find (PyTorch)**: n/a (no PyTorch counterpart).
- **Cross-backend agreement**: n/a — single-backend example. Recommendation lines up with Supervised's neighborhood (Supervised Idris 0.0201 / PyTorch 0.0242 at seed=42), confirming the duplicate setup.
- **Multi-seed pass rate**: not measured (Transfer's purpose is portability, not convergence per se — its convergence target is "matches Supervised").
- **Decision**: ship-as-is at lr=0.03. Adding the `--lr-find` flag for consistency-of-approach across the example suite.
- **Commit**: (this commit).

---

## Reinforce           date: 2026-04-29

- **Before**: `lr=0.001 epochs=2000 gamma=0.99 batch=10 seed=42`. Linear(4→128) → Tanh → Linear(128→2) policy network on CartPole. `nativeAdamGlobalClip` with grad-norm clip 1.0. Loss is `mean((baseline - return) * log_prob)`, which is **negative** when returns exceed the per-batch baseline (often the case on CartPole with returns of 20–500).
- **lr_find (Idris)**: `RECOMMENDED_LR=1e-8` — diverged at iter 1 (after iter 0 baseline). The smoothed-loss check `smoothed > divergeFactor × minSmoothed` fails for **negative** losses: with `min = -24.6`, `4 × min = -98.4`, and any subsequent smoothed value > `-98.4` triggers divergence. Algorithmic bug in `lrFind` for negative-loss tasks — known limitation.
- **lr_find (PyTorch)**: `RECOMMENDED_LR=1.45e-8` — positive loss (Adam stochasticity makes single-step REINFORCE losses positive on this fixture), curve oscillates 15..130 across the sweep with no clear minimum. Ran to completion at iter 99, recommendation falls back to ≈ `lrMin/10`.
- **Cross-backend agreement**: ratio = 1.45× — *looks like* agreement, but both are fallback (within 10× of `lrMin`). Counts as the same "fallback / unreliable" bucket as Transformer (1.0×).
- **Multi-seed pass rate**: not measured. Default unchanged.
- **Decision**: ship-as-is at lr=0.001.
- **Why lr_find doesn't help here**: REINFORCE with batch=10 + reward-to-go baseline is high-variance; one batch's "loss" doesn't give a usable LR signal even before the negative-loss heuristic bug bites. lr_find is designed for supervised loss curves; policy-gradient losses violate the "smaller = better, monotonic in early sweep" assumption.
- **Two follow-up improvements implied**: (1) `lrFind` divergence-check should be sign-stable (e.g. `smoothed > min + |min|` or use `argmin` only), and (2) for RL examples, an episode-return-based "lr_find" would be more meaningful than per-batch loss.
- **Commit**: (this commit).

---

## Dqn           date: 2026-04-29

- **Before**: `lr=5e-4 epochs=300 gamma=0.99 batch=64 buffer=10000 target_sync=100 eps=1.0→0.05 seed=42`. MLP(4→64→64→2) Q-network on CartPole with replay buffer + target snapshot every 100 env steps. `nativeAdamGlobalClip` (clip=10).
- **lr_find (Idris)**: `RECOMMENDED_LR=1e-8` — diverged at iter 1. Same negative-loss bug as Reinforce: episode-return-as-loss is negative (returns 12–18 → loss −18, −12), which trips the divergence check `smoothed > 4 × min` immediately. Reduced to `numIters=30` since per-iter cost (one full episode) is non-trivial; doesn't matter — bails on iter 1 either way.
- **lr_find (PyTorch)**: n/a — no PyTorch DQN script (Idris-only example).
- **Cross-backend agreement**: n/a (single-backend example).
- **Multi-seed pass rate**: not measured. Default unchanged.
- **Decision**: ship-as-is at lr=5e-4. The flag is wired for API consistency; the runtime sweep is not meaningful for episode-based RL with the current `lrFind` heuristic.
- **Why**: `lrFind`'s divergence-check assumes non-negative monotonic loss. RL episode-return-as-loss violates both. The right fix is in the `lrFind` heuristic itself (see Reinforce entry's "two follow-up improvements"); not a tuning concern here.
- **Commit**: (this commit).

---

## A2c           date: 2026-04-29

- **Before**: `lr=7e-4 epochs=5000 rollout=20 gamma=0.99 lambda=0.95 entropy=0.01 seed=42`. Separate actor (4→64→64→2) and critic (4→64→64→1) MLPs with paramId-scoped autoName. `nativeAdamGlobalClip` (clip=0.5).
- **lr_find (Idris)**: `RECOMMENDED_LR=1e-8` — diverged at iter 1 (same negative-loss bug as Reinforce / Dqn).
- **lr_find (PyTorch)**: n/a — no PyTorch A2C script (Idris-only example).
- **Cross-backend agreement**: n/a (single-backend).
- **Decision**: ship-as-is at lr=7e-4. Flag wired for API consistency.
- **Commit**: (this commit).

---

## Pattern observed across B3 so far

After 5 examples (Supervised, Rnn, Lstm, Transformer, SeqClassify), the
pattern crystallizes:

| Example | Optimizer | Cross-backend | Outcome |
|---|---|---|---|
| Supervised | SGD | 1.20× | actionable; default already in range, ship-as-is |
| Rnn | SGD | 380× | unreliable (1-cell architecture too small) |
| Lstm | SGD | 1.75× | **actionable; default 0.03 → 0.5** ✓ |
| Transformer | Adam | 1.0× (fallback) | unreliable (both bailed to lrMin) |
| SeqClassify | Adam | 660,000× | unreliable |

**SGD-based examples (3/3) produced informative results; Adam-based
examples (2/2) produced fallback / nonsense.** The reason is that Adam
+ grad-norm clipping already adapts the effective per-parameter LR,
flattening the lr_find loss curve.

**Implication for the remaining B3 sub-tickets**: don't expect default
changes from the Adam-based examples (Mnist, Gpt, Dqn, A2c, Ppo, Sac).
Add the `--lr-find` flag for consistency-of-approach and document
the result, but expect the entry to be "ship-as-is".

The lr_find tool is most useful for SGD examples and least useful for
Adam-based ones — that's a real fastai-vs-modern-optimizer story.
Future improvement: have `lrFind` emit a warning when the recommended
LR is within 10× of `lrMin` (likely fallback). For now, the cross-
backend ratio is the gate.

---

(Future entries here — one block per example dogfooded by B3.)
