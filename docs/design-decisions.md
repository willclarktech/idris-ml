# Design Decisions

## Autograd via Variable records

Variables carry their own backward function (`back : Double -> List Double`) and `children` list rather than using a global tape. Each operation constructs a new Variable with closures that capture the local partial derivatives. This makes the graph GC-friendly: once training completes and the loss Variable is dropped, the entire computation graph is freed.

## Node ID counter via FFI

`nextNodeId` uses a Chez Scheme top-level value instead of `IORef` or `State`. Idris 2's CSE can merge `unsafePerformIO` calls, causing multiple Variables to share the same node ID. The FFI counter uses Chez's `top-level-value` mechanism which is re-evaluated on every call, guaranteeing unique IDs. See `src/Variable.idr` for the implementation.

## Single-pass backward with SortedMap

`collectGrads` does one topological sort followed by one accumulation pass, collecting per-parameter gradients in a `SortedMap String Double`. The topological sort uses `SortedSet Nat` of node IDs to memoize visited nodes, preventing exponential traversal of the DAG (shared nodes are visited once, not once per path).

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

## updateParam / applyDeltas creates fresh Variables

After each optimizer step, parameters get new `Var` nodes with empty `back` and `children`. This breaks the reference to the old computation graph, allowing GC to free it. Without this, memory would grow linearly with training epochs as old graphs remain referenced.

## Detached max in logSoftmax

The max subtraction `x - max(x)` for numerical stability must use a detached constant (created via `fromDouble . cast`, which extracts the Double value and creates a fresh leaf Variable). If the max Variable retained its backward links, every `x - maxVal` subtraction would send a `-1` gradient back through the max, corrupting the gradient of the max element.

## Bounded NTM head parameters

The NTM focus sharpening parameter γ (gamma) must be bounded. The original formulation `softplus(x) + 1` gives γ ∈ [1, ∞), which causes vanishing gradients for non-dominant memory positions: if a weight is 0.1, then `0.1^γ` for large γ becomes negligible (0.1^20 = 1e-20), and the gradient through `w^γ` includes a factor of `γ * w^(γ-1)` which vanishes.

The fix uses `1 + 4 * sigmoid(x)` to bound γ ∈ [1, 5]. At the upper bound, `0.1^5 = 1e-5` — small but with survivable gradients. This lets the NTM learn to sharpen attention without permanently losing gradient signal for secondary memory positions.

## Accumulator-based topological sort

The original `topoSort` used `acc ++ nodes` (O(n) per append → O(n²) total). For NTM computation graphs with thousands of nodes, this dominated backward pass time. The fix uses an accumulator parameter with cons (`v :: acc'`) for O(1) per node → O(n) total. The `reverse` call in `collectGrads` still works correctly since cons naturally produces leaves-last order.

## Hyperparameter tuning protocol

Manual hyperparameter tuning is an anti-pattern that wastes hours on random adjustments. The correct order is:

1. **Fix algorithmic issues first** — bounded gamma, global gradient clipping, efficient topoSort
2. **Use systematic search** — `scripts/sweep.sh` grid search with parallel execution
3. **Never manually loop** — if a training run fails, check the algorithmic level before adjusting hyperparameters
