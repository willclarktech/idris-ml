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

## Gradient clipping (per-parameter)

The optimizer's `clipGrad` function bounds each parameter's gradient to `[-maxGrad, maxGrad]` independently. Per-parameter clipping is simple and sufficient for feedforward/RNN models. For the NTM's deeper computation graph, the logSoftmax fix was necessary rather than just tighter clipping, because clipping at the parameter level doesn't prevent intermediate gradient explosions within the forward pass.

## updateParam / applyDeltas creates fresh Variables

After each optimizer step, parameters get new `Var` nodes with empty `back` and `children`. This breaks the reference to the old computation graph, allowing GC to free it. Without this, memory would grow linearly with training epochs as old graphs remain referenced.

## Detached max in logSoftmax

The max subtraction `x - max(x)` for numerical stability must use a detached constant (created via `fromDouble . cast`, which extracts the Double value and creates a fresh leaf Variable). If the max Variable retained its backward links, every `x - maxVal` subtraction would send a `-1` gradient back through the max, corrupting the gradient of the max element.
