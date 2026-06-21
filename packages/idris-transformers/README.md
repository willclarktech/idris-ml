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

| Module | HF checkpoint target | Status |
| --- | --- | --- |
| `Transformers.Bert` | `google/bert_uncased_L-2_H-128_A-2` (~17 MB) | ready |
| `Transformers.Gpt2` | `hf-internal-testing/tiny-random-gpt2` (~150 KB) | ready |
| `Transformers.Llama` | `unsloth/Llama-3.2-1B` (~2.5 GB BF16, public Meta mirror) | ready (forward pass; KV cache follow-up — see `TODO.md`) |

## Conventions

See [`CONVENTIONS.md`](CONVENTIONS.md) for the design rules every
`Transformers.*`-prefixed module follows (param names match HF exactly, storage
shapes match on-disk, no new layer primitives in this package, one
model per file, single smart constructor, module IS the rename
adapter).

## Build

```bash
make transformers-install   # type-check + install to local prefix
make test-transformers      # run the unit test harness
```

The package is also installed as part of `make install` (after
`install-gym`, before `install-examples`).

## Downloading HF checkpoints

`scripts/hf-download.sh <repo> [filename]` wraps `curl -L --fail`,
follows the `weight_map` for sharded models, and honours `HF_TOKEN`
for private/gated models. See the script header for usage.
