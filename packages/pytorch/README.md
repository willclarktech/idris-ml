# pytorch (reference oracle)

PyTorch reference implementations in pure Python. **This is not shipped code** — it's the
correctness oracle that idris-ml is validated against. Every example's Idris version is checked
against its `torch_ref` counterpart, and the HF roundtrip gates compare idris-ml's forward pass
to PyTorch's per element. Package name: `idris-ml-torch-ref`.

The alignment contract — Idris examples and these references must use identical defaults for every
hyperparameter — is the [reference-alignment policy](../../docs/develop/reference-alignment.md).
When a discrepancy is found, the better practice is adopted on **both** sides in the same commit.

## Layout

```
torch_ref/
  scripts/      per-example train/infer runners (supervised, lstm, hf_bert_inference, …)
  models/       tabular RL references (q_learning, sarsa, sac, monte_carlo)
  correctness/  cross-language correctness gates (pytest)
  training/     shared training-loop machinery
  ntm/  dnc/    memory-augmented references
  bench_ops.py  bench_layers.py   operator/layer microbenchmarks
```

## Make targets

```bash
make ref-setup              # uv sync --dev (create the venv, install torch/transformers/…)
make ref-<name>             # run a reference script, e.g. make ref-hf-bert
make test-e2e-pytorch-ref   # the cross-language correctness pytest suite
make ref-lint               # ruff check torch_ref/
make ref-typecheck          # pyright (strict) over torch_ref/
```

The HF roundtrip gates (`make test-e2e-{bert,gpt2,llama,bitnet}-roundtrip`) regenerate the oracle
here, run the Idris example, and compare — so "matches PyTorch" is verified, not asserted.
