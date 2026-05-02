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

(Future entries here — one block per example dogfooded by B3.)
