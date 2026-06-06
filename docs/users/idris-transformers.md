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

model <- hfBertModel {ex=ExampleDevice} {dt=ExampleDType}
                     {vocab=30522} {hidden=128} {numLayers=2}
                     {numHeads=2}  {intermediate=512}
                     {maxPos=512}  {typeVocab=2}
                     "bert"
True <- loadModelAllowCast {ex=ExampleDevice}
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
| `HfGpt2` | `hf-internal-testing/tiny-random-gpt2` (and GPT-2 family) | ready |
| `HfLlama` | `meta-llama/Llama-3.2-1B` (base, gated; license-accept + `HF_TOKEN` required) | ready (forward pass; KV cache follow-up) |

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

## Fine-tuning HF-loaded models

As of 2026-06-07 the fine-tuning surface is in. Three primitives
work together:

1. **Subset-load** — `loadModelPrefix path pfx` in
   [`packages/idris-ml/src/Checkpoint.idr`](../../packages/idris-ml/src/Checkpoint.idr)
   loads only the safetensors keys whose name starts with `pfx`,
   leaving every other registered param untouched. Use to warm-start
   a backbone (`"bert."`) while keeping a fresh classification head
   at its random init.
2. **Freeze-by-prefix** — `freezeByPrefix opt pfx` /
   `unfreezeByPrefix opt pfx` in
   [`packages/idris-ml/src/Train/Freeze.idr`](../../packages/idris-ml/src/Train/Freeze.idr)
   walks the registry and sets the per-param LR override to 0 for
   every name starting with `pfx`. Composes with a single optimizer
   — no two-optimizer plumbing.
3. **Classification head** —
   `hfBertForSequenceClassification bertPfx classifierPfx` in
   [`HfBertForClassification.idr`](../../packages/idris-transformers/src/HfBertForClassification.idr)
   returns a `BertForSequenceClassificationState` whose params
   register under `<bertPfx>.*` (backbone) + `classifier.weight` /
   `classifier.bias` (head, HF-canonical naming). The forward
   composes `hfBertForward` (pooled `[CLS]`) with a 1-D `tlinear`
   into `[numClasses]` logits.

Putting it together:

```idris
model <- hfBertForSequenceClassification {numClasses=3} "bert" "classifier"
_     <- loadModelPrefix "models/google/bert_uncased_L-2_H-128_A-2/model.safetensors" "bert."
let opt = nativeAdamW lr 0.9 0.999 1.0e-8 0.01 1.0
freezeByPrefix opt "bert."  -- optional: head-only training

-- runTrainingIO with a custom epoch fn (HF models aren't Network-shaped)
runTrainingIO (epochBert opt) genBatch trainCfg model
```

The full worked example is
[`Example/BertClassifyFinetune.idr`](../../packages/idris-ml-examples/src/Example/BertClassifyFinetune.idr)
(tiny BERT + synthetic 3-class task, converges to 100% accuracy in
seconds on all three backends; multi-seed 5/5 on tape). The paired
PyTorch reference is
[`bert_classify_finetune.py`](../../packages/pytorch/torch_ref/scripts/bert_classify_finetune.py).

### Real-text dataset support (SST-2 / IMDb / GLUE)

As of 2026-06-07, the real-text fine-tuning path is in. Three
primitives layer on the synthetic example above:

1. **Attention mask on the forward** — `hfBertForward`,
   `hfBertMlmForward`, and `hfBertSeqClassifyForward` take an
   optional final `Maybe (Tensor [seqLen, seqLen] ex dt g)`
   argument. When `Just`, `primMaskedFill scores mask (-1.0e20)`
   runs between matmul and softmax in every attention layer. Entries
   `>= 0.5` are treated as "mask out"; `Nothing` is bit-identical
   to the pre-RT1 unmasked path.
2. **Tokenized dataset loader** —
   [`HfDataset.idr`](../../packages/idris-transformers/src/HfDataset.idr)
   exports `loadHfDataset : String -> IO (List TokenizedExample)`,
   `padToSeqLen : Nat -> Nat -> TokenizedExample -> (Vect seqLen Nat,
   Vect seqLen Double, Nat)`, and `toAttentionMask2d : Vect seqLen
   Double -> Vect (seqLen*seqLen) Double` (1D HF-convention mask →
   row-major flat 2D matrix). Format is a simple TSV per line:
   `<label>\t<id1,id2,…>` — no JSON parser dep.
3. **Downloader** —
   [`scripts/hf-download-dataset.sh`](../../packages/idris-transformers/scripts/hf-download-dataset.sh)
   wraps HF `datasets.load_dataset` + `transformers.AutoTokenizer
   .encode` in the existing pytorch uv venv, writes the TSV into
   `data/hf-datasets/<repo>/<split>.tsv`. Pre-tokenizes at download
   time so the Idris side never pays the ~1s/call subprocess
   startup of `Tokenizer.idr`. Make wrapper:

   ```bash
   make data-sst2       # fetches train + validation splits
   make clean-datasets  # removes the cached data/hf-datasets/
   ```

The worked example is
[`Example/BertClassifySst2Finetune.idr`](../../packages/idris-ml-examples/src/Example/BertClassifySst2Finetune.idr).
Warm-starts the `google/bert_uncased_L-2_H-128_A-2` backbone via
`loadModelPrefixAllowCast`, loads SST-2 via `loadHfDataset`, pads to
seqLen=32 + builds the 2D mask via `toAttentionMask2d`, forwards
through `hfBertSeqClassifyForward _ _ _ _ (Just mask)`. Paired
PyTorch ref:
[`bert_classify_sst2_finetune.py`](../../packages/pytorch/torch_ref/scripts/bert_classify_sst2_finetune.py).

Default config (`--max-train 256 --max-dev 256 --epochs 3`) is
tuned for fast iteration; the subset converges below HF's tutorial
threshold (Idris tape ~59%, torch ~61%, mlx-cpu ~56%; PyTorch
~52% on the same subset). Full SST-2 + 3 epochs at lr=2e-5
matches HF's documented ~80%+, but takes ~hours on Idris tape.

### GPT-2 LM continued pretraining

As of 2026-06-07 the GPT-2 LM continued-pretraining path ships as a
worked example. See
[`Example/Gpt2LmFinetune.idr`](../../packages/idris-ml-examples/src/Example/Gpt2LmFinetune.idr).

Architecture: distilgpt2 (vocab=50257, hidden=768, layers=6, heads=12,
headDim=64, intermediate=3072, maxPos=1024).

Corpus: Tiny Shakespeare (1.1MB) tokenized via distilgpt2's BPE into
`data/tinyshakespeare/input.distilgpt2.tokens` (a flat list of ~338K
token IDs). New helper script
[`scripts/tokenize-text-corpus.sh`](../../packages/idris-transformers/scripts/tokenize-text-corpus.sh)
runs the tokenization once via the pytorch uv venv.

```bash
make data-tinyshakespeare-distilgpt2   # tokenize once
make example-gpt2-lm-finetune          # train (50 default steps)
```

The training loop forwards through `hfGpt2ForwardLm` (which composes
the encoder with the tied `wte` LM head) and computes per-position
cross-entropy against the shifted-by-1 next-token target. Loss
function `gpt2LmLoss` mirrors `Example/Gpt`'s `allPositionsCELoss`:
logSoftmax2d + elementwise multiply against the one-hot + sum +
negate + mean over seqLen. Target one-hot is built via the existing
`primOneHot` primitive (flat `[seqLen*vocab]`) reshaped to 2D so the
loss multiplies against `[seqLen, vocab]` logits directly.

Paired reference:
[`gpt2_lm_finetune.py`](../../packages/pytorch/torch_ref/scripts/gpt2_lm_finetune.py).
Same backbone, same token file, same sliding-window sampling, same
AdamW(lr=5e-5, wd=0.01, clip=1.0). Both sides drop loss into the
4.0-5.0 range after 50 steps of single-example batches, starting from
distilgpt2's pretrained baseline of ~5.5 on this corpus.

### BERT MLM continued pretraining

As of 2026-06-07 the BERT MLM continued-pretraining path ships as a
worked example. See
[`Example/BertMlmFinetune.idr`](../../packages/idris-ml-examples/src/Example/BertMlmFinetune.idr).

Architecture: `google/bert_uncased_L-2_H-128_A-2` (same backbone as
the SST-2 example) with the MLM head loaded (44 total params: 39
backbone + 5 head).

Corpus: Tiny Shakespeare tokenized via BERT WordPiece into
`data/tinyshakespeare/input.bert-tiny.tokens` (~289K tokens). Same
tokenizer script as the GPT-2 LM example; different tokenizer.

```bash
make data-tinyshakespeare-bert-tiny   # tokenize once
make example-bert-mlm-finetune        # train (default 100 steps)
```

The training loop:

1. Samples a random sliding window of SeqLen=32 tokens.
2. Applies HF's 80/10/10 masking at 15% probability per position
   (80% → `[MASK]`=103, 10% → random vocab id, 10% → keep original;
   CLS/SEP never masked).
3. Forwards through `hfBertMlmForward` → `[SeqLen, Vocab]` logits.
4. Computes cross-entropy ONLY at masked positions (via a `[SeqLen,
   Vocab]` target one-hot that's zero-row-padded at unmasked
   positions, then summed and normalized by `numMasked`).

The `bertMlmLoss` function uses the same logSoftmax2d + multiply +
sum + negate chain as the GPT-2 example, but with a per-position
mask multiplier on the target so only masked rows contribute to the
sum. PyTorch ref uses HF's `labels` convention (`-100` at unmasked
positions; `CrossEntropyLoss(ignore_index=-100)` skips them).

### Today's limits + parked follow-ups

- **LoRA / parameter-efficient fine-tuning.** TODO.

## What's not supported yet

- **LoRA / parameter-efficient fine-tuning.** TODO.

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
