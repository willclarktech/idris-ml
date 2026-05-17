# Loading HuggingFace checkpoints in idris-ml

`idris-transformers` is a separate package that lets you load
HuggingFace `.safetensors` checkpoints into typed Idris models —
without writing rename tables or fiddling with per-head reshape
logic. The trick: each HF architecture is one Idris module whose
param names and storage shapes match HF's on-disk format exactly,
so the existing `loadModel` from `idris-ml`'s `Checkpoint` works
out of the box.

```idris
import HfBert
import Checkpoint

model <- hfBertModel {d=ExampleDevice} {dt=ExampleDType}
                     {vocab=30522} {hidden=128} {numLayers=2}
                     {numHeads=2}  {intermediate=512}
                     {maxPos=512}  {typeVocab=2}
                     "bert"
True <- loadModelAllowCast {d=ExampleDevice}
          "model.safetensors"
```

That's it. No remap table. No shape adapter. The module IS the
adapter, expressed as type-checked code.

## Why is this a separate package?

Core `idris-ml` ships a transformer
([`Layer/Transformer.idr`](../../packages/idris-ml/src/Layer/Transformer.idr))
designed as a from-scratch teaching reference: attention is stored
as `Vect numHeads (LinearState ...)` — one per head, decomposition
explicit in types. That makes the math obvious but is a different
storage convention from any HF checkpoint, which fuses Q/K/V (or
splits them differently per architecture). Bridging the gap at the
loader layer (a generic rename + shape-split machinery) was
considered and rejected: the per-head ↔ fused split would carry C
state forever, encode architectural decisions in lookup tables,
and doesn't generalise across HF families (Llama / GPT-2 / BERT /
T5 all use different name schemes).

HuggingFace's own Python `transformers` library doesn't do that
either — every architecture is its own class whose layer names and
storage layouts match its `state_dict()`. `idris-transformers`
mirrors that.

## Currently supported models

| Module | HF target | Status |
| --- | --- | --- |
| `HfBert` | `google/bert_uncased_L-2_H-128_A-2` (and BERT-family checkpoints sharing the same naming) | ready |
| `HfGpt2` | `sshleifer/tiny-gpt2` and GPT-2 family | follow-up (fused QKV via `c_attn.weight` + the HF Conv1D transpose wart) |
| `HfLlama` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` and Llama family | follow-up (gated on Row 7 — needs RMSNorm + RoPE + GQA + SwiGLU layer primitives in core) |

## Worked example: load and run BERT-tiny

The repo ships `Example/HfBertInference.idr` and a one-line make
target that does the whole thing end-to-end:

```bash
make example-hf-bert-inference
```

This:

1. Downloads `google/bert_uncased_L-2_H-128_A-2` (weights + `vocab.txt`)
   via the `scripts/hf-download.sh` helper into
   `packages/idris-transformers/models/` (gitignored local cache —
   `make clean-models` removes it).
2. Builds the Idris example.
3. Loads the checkpoint via plain `loadModel` (44 params: 39 encoder +
   pooler + 5 MLM-head).
4. Runs **fill-in-the-mask** on three short sentences and prints the
   top-5 predictions per `[MASK]`:

```
BERT fill-in-the-mask — google/bert_uncased_L-2_H-128_A-2
==========================================================

Input:  paris is the capital of [MASK] .
Top-5:  france (+12.65), paris (+11.66), spain (+10.90), madrid (+10.82), brussels (+10.18)

Input:  i went to the [MASK] to buy bread .
Top-5:  kitchen (+8.94), bread (+8.76), money (+8.71), cash (+7.91), fridge (+7.91)

Input:  the man worked as a [MASK] .
Top-5:  man (+8.37), photographer (+7.95), teenager (+7.93), woman (+7.65), lawyer (+7.25)
```

The three sentences are pre-tokenized (hand-picked IDs hardcoded in
the example) because there's no WordPiece tokenizer in Idris yet —
that's gated on Row 7 (the LLM-class example). vocab.txt is read at
runtime for the predicted-id → string decode.

To verify the *forward-pass* output matches HF transformers' Python
answer (independent of the MLM head — exercises the encoder + pooler
on the same path):

```bash
make test-hf-bert-roundtrip
```

This regenerates a Python oracle (`scripts/save_oracle.py`) and runs
`scripts/compare_inference.py` against the Idris binary invoked with
`--dump-pooled` (which switches output to the 128-dim pooled `[CLS]`
vector on `[101, 7592, 102]` — one float per line). The gate asserts
element-wise agreement within F32 tolerance (`< 1e-3`; local runs see
~4e-4).

## The `hf-download.sh` helper

```bash
# Default: fetch model.safetensors
bash packages/idris-transformers/scripts/hf-download.sh <repo>

# Sharded models: pass the index file; the script follows the
# weight_map and downloads each shard.
bash packages/idris-transformers/scripts/hf-download.sh \
    <repo> model.safetensors.index.json

# Private / gated models:
export HF_TOKEN=hf_xxx...
bash packages/idris-transformers/scripts/hf-download.sh \
    private-org/gated-model
```

Files land in `packages/idris-transformers/models/<repo>/`,
which is gitignored. `make clean-models` wipes the whole cache.

## Writing your own HF-aligned module

The full rules live in
[`packages/idris-transformers/CONVENTIONS.md`](../../packages/idris-transformers/CONVENTIONS.md).
Short version:

1. **Param names match HF's `state_dict()` exactly** —
   `bert.encoder.layer.0.attention.self.query.weight` literally,
   not `..._weights` or `..._weight`.
2. **Storage shapes match HF on disk.** If HF fuses Q/K/V into one
   tensor, you store it fused too; the per-head reshape happens at
   forward time as a `Tensor.reshape` view, not as a different
   storage choice.
3. **No new layer primitives in this package.** Composed from core
   `idris-ml`'s `Layer.*` (Embedding, LayerNorm, Linear, GELU,
   Residual, …). Anything missing goes into core first.
4. **One model per file** — `HfBert.idr` is BERT; `HfGpt2.idr`
   will be GPT-2; etc. No cross-imports between `Hf*` modules.
5. **One smart constructor** per module
   (`hfBertModel : … -> IO (BertModelState …)`), matching core's
   `*LayerAny` pattern.
6. **The module IS the rename adapter.** If your forward output
   disagrees with the upstream Python reference by a meaningful
   amount, the bug is in the module — there's no separate
   translation table to blame.

## What's not supported yet

- **Tokenizer integration.** The example feeds in pre-tokenized
  IDs; building token IDs from a string requires a SentencePiece /
  BPE / WordPiece implementation, which is gated on the LLM-class
  example row (see `TODO.md`).
- **Training / fine-tuning.** `saveModel` would write HF-native
  names back out trivially, but no current use case drives a
  fine-tuning workflow on HF-aligned modules. The forward pass is
  the v1 deliverable.
- **GPT-2 + Llama.** Follow-up rows tracked in `TODO.md`.

## Cross-references

- [`packages/idris-transformers/CONVENTIONS.md`](../../packages/idris-transformers/CONVENTIONS.md)
  — design rules for `Hf*` modules.
- [`packages/idris-transformers/README.md`](../../packages/idris-transformers/README.md)
  — package overview + build instructions.
- [`packages/idris-ml-examples/src/Example/HfBertInference.idr`](../../packages/idris-ml-examples/src/Example/HfBertInference.idr)
  — the worked example.
- [`packages/idris-transformers/scripts/save_oracle.py`](../../packages/idris-transformers/scripts/save_oracle.py)
  / [`compare_inference.py`](../../packages/idris-transformers/scripts/compare_inference.py)
  — the Python-side oracle generation + comparator.
- HF reference source for BERT param naming:
  [`transformers/models/bert/modeling_bert.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py).
