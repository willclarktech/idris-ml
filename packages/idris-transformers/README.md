# idris-transformers

HuggingFace-aligned model library on top of [idris-ml](../idris-ml/).

Mirrors the layout of HuggingFace's Python `transformers` package: each
HF architecture lives in its own module under `src/Transformers/*.idr`, with
parameter names and storage shapes matching HF's on-disk safetensors
format. Loading a foreign HF checkpoint is plain `fromPretrained` (or
`load`) from `idris-ml`'s `Checkpoint` module —
no rename map, no shape-split adapter at the loader layer. The module
itself is the adapter, expressed as type-checked Idris.

## Status

| Module | HF checkpoint target | Correctness gate |
| --- | --- | --- |
| `Transformers.Bert` | `google/bert_uncased_L-2_H-128_A-2` (~17 MB) | matches HF forward to **4e-4** (`make test-e2e-bert-roundtrip`) |
| `Transformers.Gpt2` | `distilgpt2` (~350 MB) | matches to 1e-3 (`make test-e2e-gpt2-roundtrip`) |
| `Transformers.Llama` | `unsloth/Llama-3.2-1B` (~2.5 GB BF16, public Meta mirror) | macro forward / RoPE / param-load (`make test-e2e-llama-roundtrip`) |
| `Transformers.BitNet` | `microsoft/bitnet-b1.58-2B-4T` | argmax-match + macro tolerance (`make test-e2e-bitnet-roundtrip`) |

Each gate regenerates a PyTorch oracle and compares per element in CI, so "matches PyTorch" is
verified on every publication push, not asserted.

## Fine-tuning

`Transformers.BertForClassification` adds a classification head (`classifier.weight`/`.bias`)
over the backbone; `Transformers.BertLora` / `Transformers.LoraIO` provide peft-compatible LoRA
adapters. Combine with `Checkpoint.load {only := Just pfx}` (subset warm-start) and
`Train.Freeze`'s `freezeGroup opt =<< namesMatching (isPrefixOf "bert.")` (freeze a
backbone by name group). Worked examples:
`make example-bert-classify-finetune` and `make example-bert-classify-sst2-lora` (see
[idris-ml-examples](../idris-ml-examples/)).

## Conventions

See [`CONVENTIONS.md`](CONVENTIONS.md) for the design rules every
`Transformers.*`-prefixed module follows (param names match HF exactly, storage
shapes match on-disk, no new layer primitives in this package, one
model per file, single smart constructor, module IS the rename
adapter).

## Build

```bash
make install-transformers          # type-check + install to local prefix
make test-unit-idris-transformers  # run the unit test harness
```

The package is also installed as part of `make install` (after
`install-gym`, before `install-examples`).

## Downloading HF checkpoints

`scripts/hf-download.sh <repo> [filename]` wraps `curl -L --fail`,
follows the `weight_map` for sharded models, and honors `HF_TOKEN`
for private/gated models. See the script header for usage.
