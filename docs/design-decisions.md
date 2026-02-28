# Design Decisions

## Tape-based autograd (Wengert list)

The autograd uses a flat tape (Wengert list) stored as five parallel Chez Scheme vectors (tags, arg1, arg2, values, paramIds) via `top-level-value`. Variables are indices into this tape. Each arithmetic operation appends an entry recording the op tag, input indices, and forward value.

The backward pass is a single reverse scan of the tape with O(1) gradient accumulation via a mutable FFI array, replacing the previous O(n log n) `SortedMap`-based approach. Only parameter entries (non-empty paramId) are collected into the output `SortedMap String Double`.

This replaced the earlier closure-based graph where each Variable carried `back : Double -> List Double` closures and `children : List Variable`. The tape approach eliminates ~2,200 heap-allocated closures per NTM forward pass, avoids pointer-chasing in topological sort, and reduces per-node size from ~120 bytes to ~40 bytes.

Benchmark (100 NTM training epochs, `src/Example/Bench.idr`):

| Version | Time | Speedup |
|---------|------|---------|
| Closure-based | 37,318 ms | — |
| Tape-based | 14,466 ms | 2.6x |

## Tape generation and staleness

After `collectGrads`, the tape is reset (size=0, gen++). Variables from the previous epoch have stale `tapeGen`. The `ensureOnTape` function detects staleness via generation mismatch and re-registers the variable as a fresh Const entry with its current `.value`. This is transparent to consumers — all code uses Variables through typeclass instances.

A stale parameter used N times in one forward pass creates N independent Const entries. Gradients accumulate correctly via `mergeWith (+)` on paramId during collection.

## FFI CSE prevention

Idris 2 compiles zero-argument `%noinline` definitions as constants evaluated once at module load time. The `tapeGeneration` wrapper must take a varying argument (the tape index) passed through to the FFI call so the compiler treats each call as distinct. Without this, `tapeGeneration` returns a stale value and `ensureOnTape` never re-registers parameters, breaking gradient computation across epochs.

## Optimizer state threading

`trainFrom` and `trainRecurrentFrom` return `(Network, OptimizerState)` to support:
- **Staged training**: print intermediate losses between training blocks
- **Optimizer switching**: e.g. SGD warm-start followed by Adam fine-tuning
- **Checkpoint/resume**: save and restore optimizer momentum state

The simpler `train`/`trainRecurrent` variants discard the state for one-shot training.

## logSoftmax + nllLoss over softmax + crossEntropy

Separate softmax + cross-entropy creates intermediate gradients of `1/pp` in the autograd graph (where `pp` is a softmax probability that can be as small as 1e-6, giving gradients up to 1e6). Even though the mathematically correct combined gradient `softmax(x) - target` is bounded in [-1, 1], the autograd graph doesn't know about this cancellation and propagates the huge intermediates backward.

The log-softmax formulation computes `x - log(sum(exp(x)))` directly, avoiding tiny probabilities entirely. The NLL loss `-(target * logProb)` has no `log` operation, so no `1/pp` gradient. This was the key fix for NTM convergence: the deep computation graph (controller -> memory addressing -> read/write operations) amplified the intermediate gradient explosions.

## Cross-entropy epsilon (1e-6)

The epsilon in `crossEntropy` prevents `log(0)` in the forward pass. Chosen at 1e-6 as a balance: small enough not to distort probabilities, large enough to keep gradients bounded (1/1e-6 = 1e6 vs 1/1e-7 = 1e7 with the old value).

## Gradient clipping: per-parameter vs global norm

The optimizer provides two clipping strategies:

**Per-parameter clipping** (`adam`): bounds each gradient to `[-maxGrad, maxGrad]` independently. Simple and sufficient for feedforward/RNN models.

**Global norm clipping** (`adamGlobalClip`): scales all gradients uniformly so the L2 norm doesn't exceed `maxNorm`. This preserves gradient *direction*, which matters for attention/recurrent models where parameters must coordinate (e.g., NTM key vectors and shift distributions). Per-parameter clipping distorts direction — it can clip the key gradient but not the shift gradient, causing the model to shift to the wrong location.

For the NTM, global norm clipping replaced per-parameter clipping to fix periodic loss spikes caused by gradient direction distortion.

## applyDeltas uses record update

`applyDeltas` updates only the `.value` field via record update syntax `{ value := v.value - d } v`, preserving the existing `tapeIdx`, `tapeGen`, and `paramId`. The stale `tapeIdx` is harmless — `ensureOnTape` will re-register the parameter with the updated value when the next epoch's forward pass runs.

## Detached max in logSoftmax

The max subtraction `x - max(x)` for numerical stability must use a detached constant (created via `fromDouble . cast`, which extracts the Double value and creates a fresh leaf Variable). If the max Variable retained its backward links, every `x - maxVal` subtraction would send a `-1` gradient back through the max, corrupting the gradient of the max element.

## Bounded NTM head parameters

The NTM focus sharpening parameter γ (gamma) must be bounded. The original formulation `softplus(x) + 1` gives γ ∈ [1, ∞), which causes vanishing gradients for non-dominant memory positions: if a weight is 0.1, then `0.1^γ` for large γ becomes negligible (0.1^20 = 1e-20), and the gradient through `w^γ` includes a factor of `γ * w^(γ-1)` which vanishes.

The fix uses `1 + 4 * sigmoid(x)` to bound γ ∈ [1, 5]. At the upper bound, `0.1^5 = 1e-5` — small but with survivable gradients. This lets the NTM learn to sharpen attention without permanently losing gradient signal for secondary memory positions.

## FFI side-effect threading

`let _ = ffiCall` in Idris 2 is dropped by the compiler since the result is unused. FFI functions with side effects must return a value consumed by subsequent computation. In the backward pass, `prim__gradAdd` returns its `AnyPtr` handle, enabling handle threading: `g' = prim__gradAdd g idx val`, where `g'` is passed to the next call. This guarantees evaluation order without `IO`.

## Hyperparameter tuning protocol

Manual hyperparameter tuning is an anti-pattern that wastes hours on random adjustments. The correct order is:

1. **Fix algorithmic issues first** — bounded gamma, global gradient clipping, efficient topoSort
2. **Use systematic search** — `scripts/sweep.sh` grid search with parallel execution
3. **Never manually loop** — if a training run fails, check the algorithmic level before adjusting hyperparameters
