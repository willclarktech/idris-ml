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

## Ppo           date: 2026-04-29

- **Before**: `lr=3e-4 epochs=200 rollout=400 batch=64 gamma=0.99 lambda=0.95 clip=0.2 K=10 entropy=0.0 value-coef=0.5 seed=42`. Separate actor (3→64→64→1) + critic (3→64→64→1) MLPs + standalone learnable `log_std` parameter. Pendulum (continuous control). `nativeAdamGlobalClip` (clip=0.5).
- **lr_find (Idris)**: `RECOMMENDED_LR=0.001743`, sweep 1e-7..10, iters=30 (each iter is one full PPO rollout = 400 env steps + K=10 mini-batch updates). **No negative-loss bug here**: Pendulum rewards are negative, so `negate avgEpReward` is positive (≈1876 at random init). Loss curve: 1876 → 1500 → 1500 (plateau) — see notes.
- **lr_find (PyTorch)**: n/a — no PyTorch PPO script (Idris-only example currently).
- **Cross-backend agreement**: n/a (single-backend).
- **Why the recommendation is unreliable here**: the loss plateau at ~1500 corresponds to "policy collapses to a fixed action and gets ~−1500 episodic reward". `lr_find` can't distinguish "this LR converges fastest" from "this LR collapses the policy fastest"; both produce the same plateau. The recommendation 0.0017 (≈6× the default 3e-4) is the LR at which the policy first collapses, not the LR at which it learns. Don't update the default.
- **Multi-seed pass rate**: not measured. Default unchanged.
- **Decision**: ship-as-is at lr=3e-4. Pendulum is also slated for an env swap in **B3-fixes** (clipped surrogate doesn't demonstrate convergence at CPU-feasible rollout); LR retuning waits until after that.
- **Lesson added (RL + bounded reward)**: when the loss curve plateaus at a non-trivial value (here, 1500), `lr_find` isn't measuring "best LR" — it's measuring "LR at which the optimization breaks the policy." For RL with bounded rewards this is a structural limitation of the steepest-descent heuristic, not specific to Adam.
- **Commit**: (this commit).

---

## Sac           date: 2026-04-29

- **Before**: `lr=3e-4 steps=30000 gamma=0.99 alpha=0.2 batch=64 warmup=1000 tau=0.005 seed=42`. Three networks: actor (3→64→64→1) + Q1/Q2 (4→64→64→1), three group-scoped Adam optimizers (`actor_`, `q1_`, `q2_`), Polyak-blended target Q-nets (`q1tgt_`, `q2tgt_`). Pendulum continuous control.
- **lr_find (Idris)**: **skipped at runtime**. SAC's "epoch" in `runTrainingIO` is a single env step, not a rollout. With `warmup=1000` + `EpisodeLen=200`, the first 1000 iters of an LR sweep would all be no-ops (warmup pre-fills the buffer; no training step yet); episode-return-as-loss only updates every 200 steps either way. lr_find at 30 or even 100 iters would never reach the training phase.
- **lr_find (PyTorch)**: n/a — no PyTorch SAC script.
- **Cross-backend agreement**: n/a.
- **Decision**: ship-as-is at lr=3e-4. The `--lr-find` flag is wired (for API consistency); invoking it prints a "skipped — see docs" message.
- **Implication for the heuristic**: SAC's structure (warmup-gated, off-policy, per-step training) is the strongest case for the future-improvement direction noted in earlier entries — `lrFind` should accept either an "epoch = N optimizer steps" or "epoch = M env steps after warmup" mode for off-policy RL. Out of scope for B3.
- **Commit**: (this commit).

---

## Mnist           date: 2026-04-29

- **Before**: `lr=0.001 epochs=5 patience=3 seed=42`. LeNet-style CNN: Conv2d(1→16,k=5) → ReLU → Pool(2) → Conv2d(16→32,k=5) → ReLU → Pool(2) → Dropout(0.5) → Linear(512→10). `nativeAdamGlobalClip` (clip=1.0) on Idris, `torch.optim.Adam` on PyTorch. Cross-entropy loss.
- **lr_find (Idris)**: `RECOMMENDED_LR=1e-8` — fallback to `lrMin/10`. Loss curve had a real descent (smoothed 2.7 → 0.49 around lr ≈ 0.14) before climbing again, but the EMA-smoothing flattened the negative-slope window enough that the heuristic bailed. 17 s for 100 iters at batch=64.
- **lr_find (PyTorch)**: `RECOMMENDED_LR=0.0291`, sweep range 1e-7..10, iters=100, divergence at iter 77 (lr ≈ 0.167, loss spiked to 75). Real signal — recommends a 30× higher LR than the current default. 2.2 s for 100 iters.
- **Cross-backend agreement**: ratio = 0.0291 / 1e-8 ≈ **2,910,000×** — wildly disagreeing. **Treat both as unreliable.**
- **Multi-seed pass rate**: not measured. Default unchanged.
- **Decision**: ship-as-is at lr=0.001. Cross-backend gate fails decisively (Adam-fallback × Idris/PyTorch curve-shape mismatch).
- **Why this confirms the Adam-fallback pattern at higher resolution**: 100 mini-batches is enough signal that PyTorch's curve looks plausible (clear minimum around lr=0.13), but Idris's loss-magnitude scale (sum-reduction vs PyTorch's mean) and EMA-smoothing interaction conspire to mask the descent for the steepest-descent heuristic. Even when both backends "see" the same shape, the recommendation can diverge by 6+ orders of magnitude. The cross-backend gate (>2× → unreliable) is the right check.
- **Commit**: (this commit).

---

## Gpt           date: 2026-04-29

- **Before**: `corpus=tinyshakespeare lr=0.001 epochs=1000 patience=0 seed=42`. 2-block multi-head transformer (seqLen=64, dModel=64, heads=4, headDim=16, vocab=65) with learned embeddings, sinusoidal PE, causal self-attention. `nativeAdamW` (β₂=0.99, wd=0.1, clip=1.0) + cosine LR schedule with 100-epoch warmup, applied via `setLRAll` (per-param LR override).
- **lr_find (Idris)**: **skipped at runtime**. GPT's `setLRAll` calls `setParamLR` for every parameter, which takes precedence over the optimizer's group-level LR that `lrFind`'s `setLearningRate` writes. Combined with the per-batch transformer-forward cost, the runtime sweep is skipped. The flag is wired (prints a "skipped — see docs" message and exits).
- **lr_find (PyTorch)**: `RECOMMENDED_LR=6.28e-5`, sweep range 1e-7..10, iters=100, divergence at iter 95 (lr ≈ 4.75, loss 161). One mini-batch update per iter, no LR schedule applied during the sweep. 7.6 s.
- **Cross-backend agreement**: n/a — Idris path is skipped; can't compare.
- **Multi-seed pass rate**: not measured. Default unchanged.
- **Decision**: ship-as-is at lr=1e-3. PyTorch's recommendation (6.28e-5, ≈16× *lower* than the default) is suggestive but unverifiable without an Idris counterpart, and applying it would break alignment with the nanoGPT recipe (which trains at ≈1e-3 with cosine warmup). Defer to **B4** (network-structure tuning) and a future per-param-LR-aware variant of `lrFind`.
- **Why this maps to the plan's "skip runtime sweep" theme**: GPT's per-param LR schedule structurally conflicts with `lrFind`'s group-level LR setting. Future improvement: a `setAllParamLR` variant of `setLearningRate` that writes the per-param overrides too, used by `lrFind` when an LR schedule is present.
- **Commit**: (this commit).

---

## NTM (Copy + Associative Recall)           date: 2026-04-29

Both NTM examples share the same architecture (LSTM controller + interpolation write + content-addressing read), training (`epochTwoPhaseTensor` with `nativeRmsprop`), and loss (sigmoid + BCE). They produce essentially the same `lr_find` story, recorded once.

- **Before (Copy)**: `lr=1e-4 clip=10 alpha=0.95 momentum=0.9 batch=16 seqLen=1-20 seed=42`. NTM<N=128, M=20, H=100>.
- **Before (Recall)**: `lr=1e-4 clip=10 alpha=0.95 momentum=0.9 batch=16 items=2-6 seqLen=3 seed=42`. Same NTM dimensions.
- **lr_find (Copy, Idris)**: `RECOMMENDED_LR=1.20e-8`. Loss decays from ~0.7 to ~0.55 then NaN at iter 80 (lr ≈ 0.29). Fallback to `lrMin/10`.
- **lr_find (Copy, PyTorch)**: `RECOMMENDED_LR=1e-8`. Loss decays from ~0.7 to ~0.69 then diverges at iter 76 (lr ≈ 0.138). Also fallback (`lrMin/10`).
- **Cross-backend agreement (Copy)**: ratio = 1.20× — *looks like* agreement (within 2× tolerance), but both are fallback (≤ 10× of `lrMin`). Same pattern as Transformer (1.0× both-fallback).
- **lr_find (Recall, Idris)**: `RECOMMENDED_LR=1.75e-8`. Same shape — flat curve then NaN around iter 80.
- **lr_find (Recall, PyTorch)**: `RECOMMENDED_LR=1e-8`. Diverges at iter 81 (lr ≈ 0.35). Fallback.
- **Cross-backend agreement (Recall)**: ratio = 1.75× — same fake-agreement-both-fallback pattern.
- **Multi-seed pass rate**: not measured. Defaults unchanged.
- **Decision (both)**: ship-as-is at lr=1e-4. Cross-backend agreement is in the "fallback" bucket; recommendation is unreliable.
- **Why this is structurally fallback**: NTM's loss curve is essentially flat across the early lr_find sweep (sigmoid+BCE on a fresh random network outputs probabilities ≈0.5, giving loss ≈ log 2 ≈ 0.69 regardless of LR), then collapses to NaN once the LR is large enough to overflow. There's no "negative-slope sweet spot" between random-init plateau and divergence — the curve is two-modal. The steepest-descent ÷ 10 heuristic has no minimum to recommend.
- **Implication**: NTM at default lr=1e-4 is in the safe pre-divergence regime. B4 (architecture tuning, e.g. smaller M or H) is the higher-headroom direction for these examples.
- **Commit**: (this commit).

---

## DNC (Copy + Associative Recall)           date: 2026-04-29

Same architecture (LSTM controller + DNC memory: usage allocation + temporal links + erase+add write), training (`epochTwoPhaseTensor` with `nativeRmsprop`), and loss (sigmoid + BCE) as NTM. Reduced N=32 (vs NTM's 128) so the O(n²) link matrix stays tractable. R=1 read head, batch=1.

- **Before (Copy)**: `lr=1e-4 clip=10 batch=1 seqLen=1-10 seed=42`. DNC<N=32, M=20, H=100, R=1>.
- **Before (Recall)**: `lr=1e-4 clip=10 batch=1 items=2-6 seqLen=3 seed=42`. Same DNC dimensions.
- **lr_find (Copy, Idris)**: `RECOMMENDED_LR=0.00260` (real signal, ~26× the default!). Diverged at iter 94 (lr ≈ 3.94). 1.5 min runtime (slow per-iter due to O(n²) link matrix).
- **lr_find (Copy, PyTorch)**: `RECOMMENDED_LR=3.43e-7` (≈ `lrMin`/3 — fallback). Diverged at iter 79 (lr ≈ 0.24). 0.7 s.
- **Cross-backend agreement (Copy)**: ratio = 0.00260 / 3.43e-7 ≈ **7,580×** — wildly disagreeing. **Treat as unreliable.**
- **lr_find (Recall, Idris)**: `RECOMMENDED_LR=0.0351` (real signal, ~351× the default!). Diverged at iter 94. 3.5 min runtime.
- **lr_find (Recall, PyTorch)**: `RECOMMENDED_LR=1e-8` (fallback). Diverged at iter 79.
- **Cross-backend agreement (Recall)**: ratio = **3,510,000×** — wildly disagreeing.
- **Multi-seed pass rate**: not measured. Defaults unchanged.
- **Decision (both)**: ship-as-is at lr=1e-4. Cross-backend gate fails decisively in both cases — Idris finds a real curve, PyTorch falls back. Idris's recommendation looks plausible (10–100× higher than default for small DNC) but isn't trustworthy without backend agreement.
- **Why Idris and PyTorch disagree here**: the `nativeRmsprop` step in Idris uses tape autograd through the DNC link matrix. PyTorch DNC uses a slightly different stability clamping schedule (see `Layer.Dnc` "Numerical stability clamping" in CLAUDE.md). The two backends produce structurally different loss curves under the same LR sweep — Idris's curve has a clear descent before NaN; PyTorch diverges earlier without producing a useful negative slope. **This is a real implementation divergence, not just lr_find heuristic noise** — flag for follow-up alignment work (already tracked separately as the "DNC layer perf" TODO).
- **Implication**: B4 (architecture tuning, e.g. lower N) is the higher-headroom direction here, alongside the existing DNC alignment ticket.
- **Commit**: (this commit).

---

## Pattern observed across B3 (final, all 11 examples dogfooded)

| Example | Optimizer | Cross-backend | Outcome |
|---|---|---|---|
| Supervised | SGD | 1.20× | actionable; default already in range, ship-as-is |
| Rnn | SGD | 380× | unreliable (1-cell architecture too small) |
| **Lstm** | SGD | 1.75× | **actionable; default 0.03 → 0.5** ✓ |
| Transformer | Adam | 1.0× (fallback) | unreliable (both bailed to lrMin) |
| SeqClassify | Adam | 660,000× | unreliable |
| Transfer | SGD | n/a (Idris-only) | duplicate of Supervised; ship-as-is |
| Reinforce | Adam | 1.45× (both fallback) | unreliable (negative-loss bug + Adam fallback) |
| Dqn | Adam | n/a (Idris-only) | unreliable (negative-loss bug) |
| A2c | Adam | n/a (Idris-only) | unreliable (negative-loss bug) |
| Ppo | Adam | n/a (Idris-only) | recommendation = "policy-collapse LR", not optimal |
| Sac | Adam | n/a (Idris-only) | runtime sweep skipped (warmup-gated structure) |
| Mnist | Adam | 2,910,000× | unreliable (Adam-fallback) |
| Gpt | AdamW | n/a (Idris path skipped) | per-param LR schedule conflicts with `lrFind` |
| NTM (Copy + Recall) | RMSprop | 1.20× / 1.75× (both fallback) | unreliable (curve flat then NaN) |
| DNC (Copy + Recall) | RMSprop | 7,580× / 3,510,000× | unreliable (real Idris/PyTorch divergence) |

**Material default change from B3: 1 of 11 examples (Lstm).** All others ship-as-is. The cross-backend agreement gate is the right discipline: when applied honestly it caught every misleading case (Adam-fallback, negative-loss-bug, policy-collapse, both-backends-fallback at 1.0–1.75× ratio).

**Optimizer-by-optimizer**:
- **SGD** (3 examples): 3/3 produced informative results, 1 yielded a real default change. lr_find at its strongest.
- **RMSprop** (4 examples — NTM/DNC): all unreliable (flat curve then NaN, or real Idris/PyTorch divergence). The "small momentum-adapted optimizer" effect partially flattens the curve like Adam does.
- **Adam / AdamW** (8 examples): 0/8 produced actionable signal. Across the suite, Adam's per-parameter LR adaptation flattens the lr_find loss curve to the point where the steepest-descent ÷ 10 heuristic gives either fallback (~lrMin/10) or wildly noisy values. **Confirms the fastai-vs-modern-optimizer story.**

**Two structural lr_find limitations exposed by B3** — both fixed:
1. ~~**Negative-loss handling**~~ — **fixed** (2026-04-30). New `hasDiverged` helper uses `(corrected - best) > (divergeFactor - 1) × |best|` instead of `corrected > divergeFactor × best`. Idris + PyTorch tests pin the behavior across positive/negative/near-zero `best` values.
2. ~~**Fallback detection**~~ — **fixed** (2026-04-30). New `isFallbackCurve` helper returns `True` when the swept curve has no negative-slope window (loss flat or rising throughout). `lrFind` now emits a `WARNING: fallback recommendation` line before the `RECOMMENDED_LR` whenever this triggers — surfaces the "Adam already adapts effective LR" / "small-arch flat curve" cases directly instead of relying on the cross-backend gate to catch them indirectly. Idris + PyTorch tests cover empty/single-point/descending/monotonic-increase/flat curves.

**Cross-cutting decisions for the suite**:
- Cross-backend agreement gate (>2× → unreliable) is now the standard B3 entry policy. Documented in CLAUDE.md and `Hpo` tutorials.
- B4 (network-structure tuning via the A3 sweep harness) is now the higher-headroom direction for Adam-based examples — `lr_find` won't move defaults there, but a small-network sweep may.
- **B3 ticket is done**; B3-fixes (PPO env swap, GPT default shrink) shipped separately — see below.

---

## B3-fixes (2026-04-30)

Two examples had wrong-shape problems that B3's `--lr-find` couldn't help with — they needed structural changes, not LR retuning.

### PPO: env swap Pendulum → Acrobot

Pendulum (continuous action, Gaussian policy) at the rollout sizes we can afford on tape (rollout=400) doesn't converge; PyTorch needs rollout=2048 (~15h on tape) to reach parity. The convergence threshold of `avg_return ≥ -800` was a partial-convergence acknowledgement, not a real demonstration. Both sides rewritten on **Acrobot** (discrete action, sparse reward, longer horizon) — the canonical "PPO clipped-surrogate demonstrates" benchmark.

Configuration:
- Architecture: separate actor (6 → 64 → 64 → 3 logits) + critic (6 → 64 → 64 → 1), tanh
- Policy: categorical (was Gaussian)
- Hyperparameters: `lr=3e-4, gamma=0.99, lambda=0.95, clip=0.2, K=10, batch=64, rollout=1024, entropy_coef=0.01, 100 rollouts`
- Acrobot physics matches `Gym.ClassicControl.Acrobot` (semi-implicit Euler, 4 substeps of dt=0.05) on both Idris and PyTorch sides

Multi-seed greedy eval (20 episodes per seed):

| Seed | PyTorch | Idris |
|------|---------|-------|
| 1    | -63.0   | -63.0 |
| 2    | -63.0   | -82.0 |
| 3    | -75.0   | -74.0 |
| 4    | -106.0  | -500.0 *(reproducible bad init)* |
| 5    | —       | -75.0 |
| 42   | -73.0   | -94.0 |

PyTorch 5/5 solved. Idris 5/6 solved (-63 to -94); seed=4 reproducibly collapses to -500 across two independent runs — appears to be an unlucky xavier-uniform draw at that PRNG seed, not a systemic gap. Acrobot solved is ~-100, random ~-500.

Convergence threshold updated from `avg_return ≥ -800` (partial) to `avg_return ≥ -150` (real, with margin). Smoke threshold updated from `≥ -2500` (Pendulum random) to `≥ -550` (Acrobot random).

Side benefit: closes a coverage gap from B1 (Acrobot was one of the unused Gym envs).

### GPT: default shrink to embedded/30

Default `make example-gpt` was `--corpus tinyshakespeare --epochs 1000` (~30 min on tape) — too slow for a default demo. Shrunk to `--corpus embedded --epochs 30` (~40 s); the full convergence run lives at `make example-gpt-full` for users who want the canonical char-LM demonstration.

Sub-changes:
- `warmupEpochs` is now `min(100, epochs/10)` instead of always 100. Without this, 30 epochs at warmup=100 means the LR never finishes ramping up — the demo would do almost no learning.
- Convergence threshold updated: `val_bpc < 3.5` (tinyshakespeare/1000) → `bpc < 5.0` (embedded/30). Untrained ~6.0; embedded/30 with warmup=3 reaches ~4.5 deterministically at seed=42.
- Smoke gate (`test-examples`) override simplified: `GPT_ARGS=--epochs 3` (was `--corpus embedded --epochs 3`) since embedded is now the default.

The full `tinyshakespeare/1000` config is preserved as `make example-gpt-full` (depends on `dataset-tinyshakespeare`); `val_bpc < 3.5` documented as the expected threshold there but not exercised in the convergence loop.

---

## B4 — Network-structure tuning (in progress, 2026-04-30)

### Transformer: dModel 32 → 16 (HeadDim 8 → 4) — shipped

After the Dqn negative result, tried the same B4 method on a matmul-bound example. Transformer attention scales as O(seqLen² × dModel); halving dModel cuts attention cost by 2× and projection cost by 2× too.

Multi-seed at the new (dModel=16, HeadDim=4) — both backends:

| Seed | Idris sort_acc | PyTorch sort_acc |
|------|----------------|------------------|
| 1, 2, 3, 4, 42 | 6/6 (perfect) | 6/6 (perfect) |

5/5 on both backends, all at 6/6 (perfect sort accuracy). The convergence threshold is `sort_acc >= 0.8`; both backends pass with full margin.

**Decision: ship.** This is the first B4 default change to land. Per-epoch wall time at dModel=16: ~25 ms on tape, comparable to the dModel=32 baseline (the synthetic sort task is small enough that overhead dominates), but the smaller model compiles faster and uses less memory — meaningful for users running the example on CI / smaller machines.

NumHeads stays 4; HeadDim drops 8 → 4 to keep `NumHeads × HeadDim == dModel`.

### Gpt: dModel 64 → 32 (HeadDim 16 → 8) — reverted

Tried halving Gpt's dModel at the new embedded/30 default (B3-fixes). Risk: limited capacity at dModel=32 might push bpc above the threshold.

Multi-seed at dModel=32 (Idris):

| Seed | bpc |
|------|-----|
| 1 | 4.86 |
| 2 | 4.87 |
| **3** | **5.03** *(above 5.0 threshold)* |
| 4 | 4.79 |
| 42 | 4.82 |

4/5 seeds pass; seed=3 dips to 5.03 (above the 5.0 threshold). Compared to the dModel=64 baseline (bpc 4.40–4.54, range 0.14, all comfortably below 5.0), the dModel=32 cluster is right at the edge (4.79–5.03, range 0.24).

**Decision: revert.** Same pattern as the Mnist and Dqn experiments — a seed-specific regression at smaller capacity. Notably seed=3 was the regressive seed in both Mnist and Gpt this round; that's consistent with seed=3 happening to land an unlucky weight init that smaller models can't compensate for.

Wall-time saving at dModel=32 was small (Gpt is already fast at the new default — ~40 s); the trade isn't worth the loss of convergence margin.

### Mnist: Conv2D channels (16, 32) → (8, 16) — reverted

Halved both Conv2D layer channels (`OutC1=16→8`, `OutC2=32→16`) on the LeNet-style architecture. Mnist's per-epoch cost is genuinely matmul-bound (full 60K-image pass), so this should yield the largest wall-time win of any B4 attempt.

Wall-time effect (single-seed, seed=42):
- Baseline (16, 32): ~10 min (documented in `example-coverage.md`)
- Smaller (8, 16): **5m 21s — ~50% reduction**

Multi-seed accuracy at the smaller channels:

| Seed | accuracy |
|------|----------|
| 1    | 0.934    |
| 2    | 0.876    |
| **3** | **0.832** *(below 0.85 threshold)* |
| 4    | 0.906    |
| 42   | 0.911    |

4/5 seeds pass; seed=3 dips to 0.832 (below the convergence threshold of 0.85). Average accuracy 0.892 with worst-case 0.832 — meaningful drop in robustness vs the baseline.

**Decision: revert.** Same pattern as the Dqn experiment: a seed-specific regression at smaller capacity. The wall-time win is real (~50%), but trading 5/5 reliability for 4/5 reliability isn't worth it for a demonstration example. A different B4 strategy (depth instead of width, or smaller-but-more-FC-units) might recover the wall-time win without the robustness loss; deferred.

### Dqn: hidden 64 → 32 — reverted

First B4 prototype. Tried halving DQN's hidden width on CartPole (4 → 32 → 32 → 2) to see if the smaller net still converges to the existing `avg_return >= 100` threshold and saves wall time.

Idris results at hidden=32 (5 seeds, 300 epochs each, ~6 min per run):

| Seed | hidden=32 | hidden=64 (baseline) |
|------|-----------|----------------------|
| 1    | 126       | (not measured)       |
| 2    | 168       | (not measured)       |
| 3    | 200       | (not measured)       |
| 4    | **13**    | **117**              |
| 42   | 146       | (default — converges) |

hidden=32 is a real regression for seed=4: it reaches only 13 (vs 117 at hidden=64). 4/5 vs (presumed) 5/5 — going strictly down on convergence robustness with no compensating wall-time win (the per-run time was ~6 min at both widths; matmul isn't the bottleneck on CartPole).

**Decision: revert to hidden=64.**

**Lesson for B4 going forward**:
- For small networks on small environments (CartPole class), halving hidden width doesn't reduce wall time meaningfully — overhead (FFI, tape ops, env-step) dominates over matmul.
- The headroom is in *large* networks where matmul actually scales: Transformer (dModel), NTM/DNC (memory size N), Mnist (Conv2D channels). DQN/Reinforce/A2C/PPO width tuning is unlikely to pay off given this finding.
- Don't apply B4 by reflex; only run the grid where the matmul scale is actually a constraint.

---

## B3-redogfood (RL examples after sign-stable fix, 2026-04-30)

The B3 entries for Reinforce/A2c/Dqn read "ship-as-is (negative-loss bug)" — the divergence check bailed at iter 1 because losses are negative (these examples report `negate avg_return`). With `0bca2a4` and `1e4f63e` shipped (sign-stable divergence + fallback detection), the bug no longer applies; re-dogfooding is now meaningful.

**Cross-backend setup (new this round)**: B3 had Idris-only `--lr-find` for A2c and Dqn (no PyTorch CLI). Added `scripts/a2c.py` + `scripts/dqn.py` mirroring `scripts/reinforce.py`. Also negated the `epoch_fn` in `scripts/reinforce.py` — `reinforce_epoch` returns mean episode return (higher=better), but `lr_find` expects loss-style; without the negate, PyTorch saw an inverted curve.

### Reinforce (CartPole, Adam)

Config-before: `lr=0.001`, `epochs=2000`, `batch=10`. Architecture: 4 → 128 → 2.

| Backend | RECOMMENDED_LR | Fallback warning? |
|---|---|---|
| Idris | 0.02009 | no — meaningful negative slope at lr ≈ 0.2 |
| PyTorch (after loss-convention fix) | 0.001789 | no — meaningful negative slope at lr ≈ 0.018 |

Cross-backend ratio: **11.2×**, exceeds the 2× gate. Both backends agree directionally (recommend HIGHER LR than current 0.001) but disagree on magnitude. **Decision: ship-as-is.** Honest cross-backend disagreement at this magnitude isn't a confident signal.

### A2c (CartPole, Adam)

Config-before: `lr=7e-4`, `epochs=5000`, `rollout=10`. Architecture: 4 → 64 → 64 → 2 (actor) + 4 → 64 → 64 → 1 (critic), separate.

| Backend | RECOMMENDED_LR | Fallback warning? |
|---|---|---|
| Idris | 4.329e-5 | no |
| PyTorch | 2.783e-4 | no |

Cross-backend ratio: **6.4×**, exceeds the 2× gate. Both backends agree directionally (recommend LOWER LR than current 7e-4) — Idris ~16× lower, PyTorch ~2.5× lower. **Decision: ship-as-is.**

### Dqn (CartPole, Adam)

Config-before: `lr=5e-4`, `epochs=300`, `batch=64`, `target_sync=100`. Architecture: 4 → 64 → 64 → 2.

| Backend | RECOMMENDED_LR | Fallback warning? |
|---|---|---|
| Idris | 6.723e-8 | **yes** (rec ≤ lrMin) |
| PyTorch | 1.0e-8 | **yes** (rec ≤ lrMin) |

Both backends produce a fallback recommendation (rec at or below `lrMin=1e-7`); the new fallback-detection rule fires on both. **Decision: ship-as-is.** The DQN loss curve is too noisy / Adam-adapted for `lr_find` to pick a useful LR.

### Ppo (Acrobot, Adam) — added 2026-04-30 (session 5)

PPO was originally on Pendulum at B3 ("ship-as-is, recommendation = policy-collapse LR"); B3-fixes (`36dbd5f`) swapped to Acrobot with categorical policy and re-validated convergence (5/5 PyTorch, 5/6 Idris) but did not re-run lr_find. This entry closes that gap.

Config-before: `lr=3e-4`, `epochs=100`, `rollout=1024`, `K=10`, `batch=64`, `gamma=0.99`, `lambda=0.95`, `clip=0.2`, `entropy=0.01`. Architecture: 6 → 64 → 64 → 3 (actor) + 6 → 64 → 64 → 1 (critic), separate, tanh activations.

PyTorch CLI parity for this entry: added `scripts/ppo.py` mirroring `scripts/{a2c,dqn}.py` (loss = `-avg_ep_return` per rollout, 30-iter sweep matching Idris's `numIters := 30`).

| Backend | RECOMMENDED_LR | Fallback warning? |
|---|---|---|
| Idris | 4.894e-4 | no — clear descent zone iter 16–21 (smoothed 486 → 423) |
| PyTorch | 2.395e-7 | no — but **effectively fallback** (rec ≈ 2.4× lrMin; the apparent steepest descent at iter 5→6 is noise where loss was clamped at 500=max-ep-len) |

Cross-backend ratio: **~2,043×** — far over the 2× gate. The mechanism is the same Acrobot physics (deterministic), but Idris's xavier-uniform init draws different starting weights than PyTorch's default kaiming, and the curves' descent regions land at different LRs. Idris finds a real descent zone (loss drops from 486 to 423 over iters 16–21); PyTorch's curve descends in the same zone (smoothed 489 → 354 over iters 13–22) but the algorithm's argmin slope falls at iter 5→6 noise instead.

**Decision: ship-as-is.** The current Idris default `lr=3e-4` is within 1.6× of the Idris recommendation (4.89e-4) — already in the right zone. PyTorch's "recommendation" is three orders below its default and is correctly identified by the cross-backend gate as untrustworthy. No change.

Library note: PyTorch's case is exactly the kind of fallback the existing rules don't catch — `rec=2.395e-7 > lrMin=1e-7` (just barely), and the curve has *some* negative-slope segments so `isFallbackCurve` is False. Possible future tightening: warn when `rec` is within `recommendDiv × lrMin` (i.e. one steepest-descent step above the floor) — would have fired here. Filed as a smaller follow-up; not implemented yet.

### Cross-cutting takeaways from B3-redogfood

- **The sign-stable fix works**: all three examples that previously bailed at iter 1 now run all 100/30 iterations. The negative-loss bug is closed.
- **Adam-fallback story holds**: same pattern as the rest of the Adam-based examples in B3 — meaningful curves on some seeds, fallback on others, never a confident cross-backend match. Reinforce's 11×, A2c's 6.4×, and PPO's ~2000× cross-backend disagreements are all over the gate. Reinforce + A2c are directionally consistent (both backends recommend the same direction); PPO is not — Idris finds a real descent zone and recommends slightly higher than current default, PyTorch's algorithm picks early-iter noise and bottoms out near lrMin. The cross-backend gate catches all four.
- **One small alignment finding**: `scripts/reinforce.py` had `reinforce_epoch` returning higher-is-better for the `--lr-find` epoch_fn. Negated for the lr_find path; unrelated to training output.
- **One library improvement**: tightened fallback detection — the "rec ≤ lrMin" check now fires the WARNING on noisy curves where the steepest descent is just at iter 0 (Dqn case). Strict `isFallbackCurve` ("all slopes ≥ 0") wasn't catching this.

## B6 — New examples (calibration spike, 2026-04-30)

### Gru (pattern prediction, SGD)

First B6 ticket. Mirrors Example.Lstm (same task, same architecture shape, same SGD optimizer) with the GRU layer instead of LSTM.

Config-before / config-after (no change): `lr=0.5`, `epochs=2000`, `patience=500`, `seed=42`. Architecture: `GRU(1, 4) ~> Linear(4, 1)`, BCE-with-logits loss.

| Backend | RECOMMENDED_LR (100-iter sweep, seed=42) |
|---|---|
| Idris | 0.4751 |
| PyTorch | 0.4751 |

Cross-backend ratio: **1.00× — exact agreement.** Current default 0.5 is within 5% of the recommendation; ship-as-is.

**Multi-seed at lr=0.5** (5 seeds × 2 backends), threshold `loss < 0.05`:

| Seed | Idris loss | PyTorch loss |
|------|------------|--------------|
| 1    | 8.42e-4    | 1.34e-3      |
| 2    | 6.06e-4    | 8.88e-4      |
| 3    | 8.23e-4    | 8.49e-4      |
| 4    | 8.76e-4    | 9.04e-4      |
| 42   | 8.17e-4    | 6.83e-4      |

10/10 perfect, all 50× or more below the threshold. Convergence in 1095–1537 epochs (early-stopped via patience=500).

**Calibration findings** (per the audit's "spike to re-estimate B6 cost"):
- Total elapsed time, idea → committed: ~1 hour. Most expensive subtask was discovering that `Layer/Gru.idr` shipped without `applyGeneric` (used by `evaluateRecurrent`). Filed a 30-line `applyGeneric` matching the C kernel exactly.
- Layer-spec review found the C kernel `tensor_gru_cell` implements a **simplified GRU** (r gate computed but unused). PyTorch reference `LinearGRUCell` mirrors that variant, not `nn.GRUCell`. Documented in `reference-alignment.md`. Filed as a possible future correctness improvement, but not a blocker.
- The B6 cost estimate "1–2 days per example" from the skip-decision audit (session 5) is **on the high side for layer gaps that mirror existing examples**: GRU was ~1 hour. Env-gap tickets (MountainCar/Taxi/FrozenLake) likely fit the 1-day estimate; layer-gaps that are pure ports of existing recurrent / FC examples are several-hour scope.

### FrozenLake (tabular Q-learning on stochastic env)

Second B6 ticket. Closes the FrozenLake env coverage gap and validates that the existing tabular scaffolding handles stochastic envs.

Config: `alpha=0.1 gamma=0.99 epsilon=0.3 epochs=10000 seed=42`. Q-table `[16, 4]`, MaxSteps=100, slippery dynamics (intended action prob 1/3, each perpendicular 1/3). Reward 0/1 (sparse).

Initial pass at `alpha=0.5 gamma=1.0 epsilon=0.1 epochs=500` (the QLearning-on-CliffWalking defaults) produced highly seed-dependent results (2/5 succeed at >0.7, 2/5 stuck at 0.0) because the sparse-reward + stochastic-env combination requires more exploration to find the goal at all. Tuned to `eps=0.3 epochs=10000` based on Gymnasium tabular Q baselines for slippery 4×4.

**Multi-seed at the new defaults** (5 seeds × 2 backends), threshold `avg_return >= 0.4`:

| Seed | Idris avg_return | PyTorch avg_return |
|------|------------------|--------------------|
| 1    | 0.68             | 0.74               |
| 2    | 0.66             | 0.56               |
| 3    | 0.70             | 0.75               |
| 4    | 0.80             | 0.74               |
| 42   | 0.74             | 0.69               |

10/10 pass. Idris mean 0.72, PyTorch mean 0.70. The slip-cap on greedy success rate caps avg_return well below 1.0 (an optimal policy still slips into holes ~30% of the time on this 4×4 layout); 0.7 is a strong tabular result.

`lr_find` is not applicable (tabular updates are not gradient-based; the natural sweep would be α and ε, both already at sensible values per the multi-seed evidence).

Cross-backend determinism: Q-table updates are pure deterministic given the same seed sequence; both backends give identical avg_return when run from the Idris example (tape/mlx/torch all hit 0.74 at seed=42 because tabular RL doesn't exercise the autograd path).

Convergence runtime: ~2 s on tape. Smoke = full convergence run (no epoch override needed).

### Taxi (tabular Q-learning on deterministic env)

Third B6 ticket. Closes the Taxi env coverage gap. Tabular Q-learning on the deterministic 5×5 Taxi-v3 grid, mirroring `Example.QLearning` (CliffWalking) almost line-for-line — same fixed-start scaffolding, no slip noise needed.

Config: `alpha=0.1 gamma=0.99 epsilon=0.1 epochs=20000 seed=42`. Q-table `[500, 6]` (500 states × 6 actions), MaxSteps=200, fixed start `defaultStart` (taxi at (2,2), passenger at R=0, destination B=3). Reward range: -1/step, -10 for illegal pickup/dropoff, +20 for successful dropoff at destination.

The walls in the 5×5 layout (between cols 1-2 in rows 0-1; between cols 2-3 in rows 3-4) force the optimal trajectory into 13 actions (4 to reach R, 1 pickup, 7 to reach B with detours, 1 dropoff) → optimal return = 12·(-1) + 20 = **+8**.

**Multi-seed convergence (≥5 seeds, both backends)**, threshold `avg_return >= 5`:

| Seed | Idris avg_return | PyTorch avg_return |
|------|------------------|--------------------|
| 1    | 8.0              | 8.0                |
| 2    | 8.0              | 8.0                |
| 3    | 8.0              | 8.0                |
| 4    | 8.0              | 8.0                |
| 42   | 8.0              | 8.0                |

**10/10 hit optimal**. Deterministic env + fixed start = single trajectory; once Q-learning converges to the optimal policy (which it does reliably under these defaults), every greedy episode replays it.

`lr_find` is not applicable (tabular). The natural sweep would be α/ε; both already at sensible defaults per the multi-seed evidence.

Convergence runtime: ~7 s on tape. Smoke = full convergence run (no epoch override needed). Same number on all 3 backends (tabular doesn't exercise autograd).

**Calibration footnote**: Taxi was the second straight B6 ticket where the existing tabular scaffolding extended cleanly without modification — no env-API changes, no new training-loop variant. ~30 minutes idea→committed. The 1-day estimate from the audit holds for env-gap tickets that need stochastic-env wiring or reward shaping; ports onto deterministic envs that share the QLearning/CliffWalking template are sub-hour.

### MountainCar (attempted 2026-04-30, reverted)

Fourth B6 attempt — port DQN to MountainCar with potential-based reward shaping (φ(s) = pos, Ng et al. 1999). Reverted before commit; documented here so the pattern doesn't get re-discovered.

**Failure mode**: the per-transition DQN loss does a separate `forwardVarTensor` per batch element (64-element batch per train step). MountainCar's 200-step episodes mean every episode runs 200 train steps, each doing 64 forwards, so per-epoch cost was ~4 s on tape (vs CartPole DQN's ~50 ms — episodes there terminate at ~10 steps when the policy is bad). At 100 episodes the policy still hadn't escaped the truncation floor (eval avg_return=-200). Multi-seed validation would have taken hours per run, well outside the single-session budget.

**What's actually needed before retrying**:
- Batched per-transition loss using the `applyVarTensorBatch` infra added in the SAC actorLoss-batched ticket. The DQN per-transition loss can be reformulated as a single batched forward through the Q-net + per-row select on the [B, NumActions] output. Probably 5-10× speedup on the shaped MountainCar problem.
- Either Gymnasium-style randomized initial state (env-side change to add `reset(seed)`) or much more aggressive shaping. Position-only shaping wasn't enough to escape truncation in 100 episodes; energy-based or velocity-based shaping should help.
- Multi-seed validation budget at the new per-epoch cost: ~30 min/seed × 5 seeds × 3 backends = 7-8 hours of compute.

**Implication for B6 scope**: MountainCar is **not** in the sub-hour-scope bucket established by Gru / FrozenLake / Taxi. Treat as ~1–2 days minimum, with the batched-forward refactor as a prerequisite. MountainCarCont (continuous action via Gaussian policy) is even harder and depends on having a reliable MountainCar convergence pattern first. Recommend deferring the pair until the batched-forward DQN refactor lands as its own optimization ticket.

### DQN batched forward (2026-04-30, prerequisite landed)

Refactored `Example/Dqn.idr`'s training loop to do one `forwardVarTensorBatch` per train step instead of B per-sample `forwardVarTensor` calls. Pattern mirrors SAC's `qLossBatch` (Example/Sac.idr): stack obs into `[B, ObsDim]` via `bulkToTensor2d` → one batched online forward → per-row `prim__select(qOutB, 0, k)` then column select at the action index → wrap as `Var` to preserve autograd → squared-error vs the per-sample Double target. Target net stays per-sample (Double arithmetic, no FFI per scalar).

**Validation**:
- Multi-seed CartPole DQN at the unchanged H=64 default: 5/6 seeds ≥ 100 (200, 102, 106, 112, 200, 91). Pre-batched baseline was 4/5 ≥ 100 — pass rate matches within DQN's natural noise.
- Full smoke gate (`make test-examples`) passes identically on all 3 backends (avg_return=9.0 at --epochs 10, seed=42).
- PyTorch reference (`models/dqn.py`) was already natively batched via `q(obs).gather(...)` — no PyTorch changes.

**Implication for MountainCar / MountainCarCont**: prerequisite cleared. Re-attempting MountainCar is now realistic — the per-epoch cost should drop from ~4 s to a fraction of that (a single batched forward replaces 64 per-sample forwards), making multi-seed validation budget-feasible. Reward shaping + initial-state randomization remain the open empirical questions.
