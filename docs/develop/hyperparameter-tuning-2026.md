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

(Future entries here — one block per example dogfooded by B3.)
