# LLM inference end-to-end: a walkthrough of the Llama example

This doc walks through `Example/LlamaInference.idr` — idris-ml's
canonical "this runs an actual LLM" demo. It loads
`unsloth/Llama-3.2-1B` from a HuggingFace `.safetensors`
checkpoint, decodes 8 greedy tokens given the prompt "The capital of
France is", and prints "Paris" as the first generated token. Reads as
a tour of the moving parts the row "LLM-class example (Llama-7B-shape
inference)" was about (closed 2026-05-29).

The audience is contributors who already know the codebase's
backbone (`Tensor`, `Nn/*`, `Checkpoint`) and want to
see how the typed-shape and dependently-typed-Nat machinery composes
into a real LLM forward pass — i.e. what idris-ml looks like when
pushed at LLM scale.

## How to run it

Three lanes work today; pick by `BACKEND` and device:

```bash
# mlx-gpu (the fast lane, the "GPU > CPU" showcase)
BACKEND=mlx MLX_DEVICE=gpu MLX_DTYPE=F32 make example-hf-llama-inference    # ~46 s
BACKEND=mlx MLX_DEVICE=gpu MLX_DTYPE=BF16 make example-hf-llama-inference   # native Metal BF16 (2026-05-31)
BACKEND=mlx MLX_DEVICE=gpu MLX_DTYPE=F16  make example-hf-llama-inference   # native Metal F16  (2026-05-31)

# torch-mps F32 (libtorch + Metal)
BACKEND=torch TORCH_DEVICE=mps make example-hf-llama-inference             # ~5 min

# torch-mps BF16 (real reduced precision; runtime ≈ F32 on M4 — see perf-changes.md 2026-05-31)
BACKEND=torch TORCH_DEVICE=mps TORCH_DTYPE=BF16 make example-hf-llama-inference

# tape F32 (CPU; ~7.5 GB working set; OOMs at default F64 — see TODO row)
TAPE_DTYPE=F32 make example-hf-llama-inference                             # ~1 m
```

The checkpoint is downloaded via the `models/` rule the first time
you run it. `unsloth/Llama-3.2-1B` is a public mirror of Meta's
weights — no HF token or license-accept required.

## Architecture (Llama 3.2 1B)

The 1B model has 16 decoder blocks, hidden size 2048, vocabulary
128,256, 32 attention heads with **grouped-query attention (GQA)**
of ratio 4 (8 key/value heads), head dim 64, an SwiGLU MLP with
intermediate size 8192, **rotary position embeddings (RoPE)** with
NTK-aware scaling clamped to maxPos 8192, and **RMSNorm** (ε=1e-5)
applied pre-norm before attention and MLP. The vocabulary projection
is tied to the input embedding (`lm_head.weight` is
`embed_tokens.weight`). None of these dims are hardcoded:
`Transformers.Llama.fromPretrained` reads them from the checkpoint's
`config.json` and returns `(cfg ** model)`, so the model's type
carries the file's shapes. The driver is
[`Example/LlamaInference.idr`](../../packages/idris-ml-examples/src/Example/LlamaInference.idr).

Three pieces sit in core idris-ml (`packages/idris-ml/`) because they
are Llama-family-generic, not HF-specific:

- [`Nn/RmsNorm.idr`](../../packages/idris-ml/src/Nn/RmsNorm.idr)
  — root-mean-square normalisation with a learned per-channel gain.
- [`Nn/RoPE.idr`](../../packages/idris-ml/src/Nn/RoPE.idr) —
  rotary position embedding with the Llama-3 NTK frequency rescaling
  (`buildLlamaRoPETables` precomputes the cos/sin tables once at
  model construction; matches the PyTorch reference within 1e-12).
- [`Nn/SwiGLU.idr`](../../packages/idris-ml/src/Nn/SwiGLU.idr)
  — gated MLP `down(gate(x) * silu(up(x)))` with three linear weights.

The HF-specific assembly — param naming, per-block storage shapes,
the forward pass — lives in
[`packages/idris-transformers/src/Transformers/Llama.idr`](../../packages/idris-transformers/src/Transformers/Llama.idr).
Per [`packages/idris-transformers/CONVENTIONS.md`](../../packages/idris-transformers/CONVENTIONS.md):
core ships the architectural primitives, the transformers package
ships the HF-name-aligned adapter. The module IS the adapter
(expressed in code); there is no separate remap table.

## The forward pass, one type at a time

The headline forward function:

```idris
hfLlamaForwardLm : {0 ex : Executor} -> UserExecutorTraining ex => UserExecutorCore ex
                => RuntimeDType dt => Linked ex => Compatible ex dt
                => {seq, vocab, hidden, numLayers, numHeads, numKvHeads, headDim, intermediate, maxPos : Nat}
                -> (eps : Double)
                -> LlamaModelState vocab hidden numLayers (numHeads * headDim) (numKvHeads * headDim) intermediate ex dt g
                -> RoPETables maxPos headDim ex dt
                -> Tensor [seq] ex dt g                -- input token IDs
                -> IO (Tensor [seq, vocab] ex dt g)    -- per-position logits
```

What's load-bearing in that signature:

- **`ex : Executor`** is the open executor kind (any type with a
  `UserExecutorCore` instance). For Llama, examples bind `ex` to
  `ExampleDevice`, which the Makefile generates per
  `(BACKEND, *_DEVICE)` cell (`TapeExecutor`, `TorchExecutor TMps`,
  `MlxExecutor MGpu`, …). See `docs/develop/design-decisions.md`
  "Open `d` parameter".
- **`dt : DType`** is the open dtype kind. `Compatible ex dt` gates
  admissible pairs at construction — e.g.
  `Compatible (MlxExecutor MGpu) F64` deliberately doesn't exist
  (Metal F32-only).
- **`Linked ex`** is the compile-time linkage gate: a tape-only build
  cannot even spell `MlxExecutor _` here, because no
  `Linked (MlxExecutor _)` instance is emitted by that build's
  `HwConfig`. See `docs/develop/device-availability-gating.md`.
- **`(numHeads * headDim)`** in the model state's Q-projection size:
  Idris-2 reduces this at the type level for concrete values (32 ×
  64 = 2048), and the caller pins the multiplication once at the
  state's construction site, then drops the proof obligation
  thereafter. The companion `(numKvHeads * headDim) = 512` slot is
  the GQA story — fewer KV heads than Q heads.

Inside `hfLlamaForwardLm`, the pipeline is:

```idris
emb       <- applyEmbedLookup model.embedTokens tokens
hMid      <- applyBlocks      … model.blocks tables emb
hFinal    <- applyRmsNorm2d   eps model.finalNorm hMid
logits    <- tlinear2d        zeroBias hFinal model.embedTokens.weight
```

The `model.embedTokens.weight` tensor `[vocab, hidden]` is reused as
the LM projection — that's the *weight tying* the HF model assumes.
The type system doesn't enforce tying; we just feed the same tensor
to two places. Shape-checking catches the rest: if you accidentally
passed a `[hidden, vocab]` weight, the matmul wouldn't type-check.

`applyBlocks` is a recursive walk over the 16 `LlamaBlock`s — each
block does pre-norm + attention + residual + pre-norm + MLP +
residual. The interesting structural piece is in `applyAttention`.

## GQA, expressed in dependent types

Grouped-query attention has 32 query heads and 8 key/value heads;
each KV head serves 4 query heads. The HF storage shape for the
projections is:

```
q_proj.weight  :  [numHeads   * headDim, hidden]   =  [2048, 2048]
k_proj.weight  :  [numKvHeads * headDim, hidden]   =  [ 512, 2048]
v_proj.weight  :  [numKvHeads * headDim, hidden]   =  [ 512, 2048]
o_proj.weight  :  [hidden,  numHeads * headDim]    =  [2048, 2048]
```

The `LlamaAttentionState` record carries the three Linear-no-bias
weights with those exact shapes. The forward, after projection,
reshapes Q into `[seq, numHeads, headDim]` and K/V into `[seq,
numKvHeads, headDim]`. The per-head attention loop iterates over Q
heads, mapping each `q_head_i` to the KV head `i / 4`. The mapping
is a runtime integer divide — it would be a great place for a
type-level `Mod` proof, but Idris-2's Peano `Nat` is too slow at
the relevant sizes (see [`docs/develop/gotchas.md`](gotchas.md)
"Data.Nat stdlib functions are recursive at runtime too"), so the
example uses cast-to-`Int` and runtime divide.

What the type system *does* catch:

- Forgetting to apply RoPE to Q and K after the projection (the
  tensor would be `[seq, headDim]` after slicing a head, but the
  downstream causal-mask kernel expects the post-RoPE rotated
  variant — same shape, but the *value* would be wrong; this one
  the type system can't catch and the test gates do).
- Swapping numHeads and numKvHeads in the per-head split (the type
  signature forces `Tensor [seq, numHeads, headDim]` on the Q side
  and `Tensor [seq, numKvHeads, headDim]` on the KV side; the
  per-head narrow at index `i < numHeads` can't be applied to a
  `[seq, numKvHeads, _]` tensor with `numKvHeads < numHeads`
  without a runtime proof).
- Forgetting the residual add (the result type wouldn't match the
  block-output type; the next block's pre-norm would refuse it).

## RoPE: tables built once, applied per layer

```idris
buildLlamaRoPETables : … -> IO (RoPETables maxPos headDim ex dt g)
```

builds `[maxPos, headDim/2]` cos/sin tables at model construction
time (see
[`Nn/RoPE.idr:212-223`](../../packages/idris-ml/src/Nn/RoPE.idr)).
The tables are reused across every forward, every layer, every
head. `applyRope` slices `[seq, halfDim]` rows from the tables per
forward (NOT per layer; the same slice serves all 16 layers in one
call) and applies the rotation to Q and K element-wise.

The shape arithmetic — head dim 64 splits into two `[seq, 32]`
halves, rotated, recombined — is expressed as plain `splitAt` /
`tconcat` over the typed dimensions. No `believe_me`. Shape
arithmetic that Idris-2 can't reduce at the type level (because
`headDim / 2` is multiplicative-Nat territory) goes through
`TVec`/`TMat` aliases in `Tensor.idr` — see
[`docs/develop/gotchas.md`](gotchas.md) "Tensor [4 * o] hangs
Idris-2 type-checker".

## Greedy decode

The decode loop in `Example/LlamaInference.idr` is cache-aware
(`genLoopCached` — the only generation path; the no-cache `genLoop`
was dropped 2026-06-04 as correctness-equivalent at double the Chez
elaboration cost). The seed step feeds the full prompt into empty
per-layer KV caches; each later step feeds only the
previously-generated token, threading the caches through:

```idris
go _      acc _    Z     = pure acc
go caches acc feed (S k) = do
  perfReset {ex=ExampleExecutor}
  (caches', mNext) <- genStepCached cfg model tables caches feed
  ops <- perfOpCount {ex=ExampleExecutor}
  putStrLn ("[perf] step " ++ show (length acc) ++ ": " ++ show ops ++ " ops")
  case mNext of
    Nothing => do
      putStrLn "  (argmax produced out-of-range token; stopping)"
      pure acc
    Just next => go caches' (acc ++ [next]) [next] k
```

Two pieces deserve attention:

- **The KV caches** (`Transformers.KVCache`) mean a steady-state
  step's forward processes one new token, with attention reading the
  cached K/V for earlier positions, instead of re-running the whole
  growing sequence. `Nn/RoPE.idr`'s `positionOffset` parameter
  supplies the absolute position for those single-token steps.
- **`[perf] step N: K ops`** is the per-forward op-submission counter
  (commit `e9763d0`). On torch it counts every `from_tensor()` wrap
  (one per graph node); on tape + mlx it's a no-op stub returning 0.
  Surfaced through perf-run.sh alongside `[stage]` lines. Used to
  confirm the torch-mps per-MTLCommandBuffer dispatch ceiling
  (~19,400 ops × ~1.89 ms/op on the pre-fusion Llama-1B forward,
  matching the measured wall).

The loop terminates cleanly: the `drainManagedHandles +
releaseAllPersistent` exit-cleanup chain runs once after the loop
returns, before `main` exits, so the per-backend multi-GB free pass
happens inside `main` rather than in the libc/libtorch exit-time
allocator destructor cascade.

## Backend pluggability

The example's only mention of a concrete backend is `ExampleDevice`
+ `ExampleDType`, both generated per build from the template
[`packages/idris-ml-examples/src/BuildConfig.idr.in`](../../packages/idris-ml-examples/src/BuildConfig.idr.in).
Switching backends is just a different `make install`. The same
source compiles to tape / torch-cpu / torch-mps / torch-cuda /
mlx-cpu / mlx-gpu via the `(BACKEND, *_DEVICE)` cell table in
`CLAUDE.md`. The Idris-2 type system fixes types at elaboration, so
the env var is observed at build time and baked into the generated
`BuildConfig.idr` (also why `Linked` instances are similarly
generated). The model and the forward function are not aware of any
of this — they take an open `ex : Executor` parameter and rely on the
`UserExecutorCore / UserExecutorTraining / Compatible / Linked`
instance ladder.

## What's not done yet (filed separately)

- **7B-class inference** — out of scope on the current 16 GB VM
  (7B × F16 ≈ 14 GB params alone). Re-investigate when running on a
  box with ≥32 GB RAM.
- **torch-mps wall vs PyTorch Python** — the per-op submission
  counter confirmed torch-mps is dispatch-bound; the fix is matching
  PyTorch's fused-op catalogue (SDPA, RMSNorm, SwiGLU, and the
  embedding fusion have shipped), tracked in TODO's "Match PyTorch's
  fused-op catalogue on torch backend" row.
- **Tape F64 large-LM OOM** — default tape (F64) doesn't fit
  Llama-1B on a 16 GB VM; use `TAPE_DTYPE=F32`. Tracked in TODO's
  "Tape F64 large-LM OOM (HfLlama + HfBitNet)" row.

## References

- HF `transformers` Llama implementation:
  `https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py`
- mlx-lm Llama implementation (for the mlx target):
  `https://github.com/ml-explore/mlx-examples/tree/main/llms`
- Touvron et al., *LLaMA: Open and Efficient Foundation Language
  Models*, 2023.
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer
  Models from Multi-Head Checkpoints*, 2023.
- Su et al., *RoFormer: Enhanced Transformer with Rotary Position
  Embedding*, 2021.
