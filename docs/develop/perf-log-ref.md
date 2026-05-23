# `perf-log-ref.jsonl` — reference / third-party baseline measurements

Append-only log of *baseline* perf measurements from external tools
(PyTorch Python, mlx-lm, llama.cpp, HuggingFace transformers, etc.).
Companion to `docs/develop/perf-log.jsonl` which logs *our* runs.

## Why a separate file?

`perf-log.jsonl` gets one entry per meaningful idris-ml run (per the
"no expensive run without a perf log" rule in CLAUDE.md). That file
grows hundreds of entries over a working week, keyed by commit hash.

`perf-log-ref.jsonl` is the opposite shape: a handful of entries per
external tool/configuration, keyed by `(ref_tool, ref_version,
workload)`. We measure these *occasionally*, when:

- a new reference target becomes relevant (e.g. new mlx-lm release
  worth comparing against);
- a structural perf claim needs validating (e.g. "is X our ceiling
  or PyTorch's ceiling?");
- we're sizing a perf TODO row and want to know how far we are from
  parity before deciding scope.

Keeping the two files separate means `perf-log.jsonl`'s commit-keyed
churn doesn't drown the rare-but-load-bearing reference numbers.

## Schema

One JSON object per line.

| Field | Type | Notes |
|---|---|---|
| `ts` | ISO 8601 UTC | Measurement timestamp |
| `date` | YYYY-MM-DD | For grouping |
| `kind` | `"reference"` | Distinguishes from `kind: "run"` / `"baseline"` in `perf-log.jsonl` |
| `example` | string | Matches the example key in `perf-log.jsonl` (e.g. `"hf-llama"`, `"hf-bert"`) so cross-file joins are mechanical |
| `ref_tool` | string | `"pytorch-python"`, `"mlx-lm"`, `"llama.cpp"`, `"hf-transformers"`, etc. |
| `ref_version` | string | Library version (e.g. `"2.4.0"`, `"main@<sha>"`, `"2.x via transformers"`) |
| `device` | `"cpu"` / `"mps"` / `"cuda"` / `"gpu"` (mlx) | Same vocabulary as `perf-log.jsonl` |
| `dtype` | `"F32"` / `"F64"` / `"BF16"` / `"F16"` | Same vocabulary as `perf-log.jsonl` |
| `workload` | string | One-line description (model + decode budget + cache config) |
| `wall_ms` | int | Total wall (load + generate) |
| `wall_human` | string | Human-readable wall |
| `run_generate_ms` | int | Just the generation phase (excludes load) — load times are often dominated by disk / network and aren't the perf signal we care about for inference |
| `run_generate_human` | string | Human-readable generate wall |
| `stages` | array of `{label, elapsed_s}` | Per-stage timestamps from the reference tool's output, where available |
| `note` | string | Why this measurement was taken; cross-references to commits in `perf-log.jsonl` |

## Candidate reference targets

Not measured today; worth capturing when relevant:

- **mlx-lm Llama-3.2-1B on MPS GPU** — closest pure-Metal reference
  for the Llama-1B workload. Compare against our mlx-gpu wall (~46s
  per `perf-baseline.md`). Run via the `mlx-lm` CLI:
  `mlx_lm.generate --model meta-llama/Llama-3.2-1B-Instruct --prompt
  "The capital of France is" --max-tokens 8`.
- **llama.cpp Llama-3.2-1B on Metal** — third reference point.
  Quantized weights (Q4_K_M / Q8_0) typical, so dtype field would be
  `"Q4_K_M"` etc.
- **HuggingFace transformers + torch.compile** — does `model =
  torch.compile(model)` change PyTorch Python's MPS wall? Compare
  against the eager numbers we just captured.
- **PyTorch Python on torch-cpu** — for CPU lane reference, paired
  with our torch-cpu wall.

## Querying

Side-by-side with our own walls:

```bash
jq -c 'select(.example == "hf-llama")' docs/develop/perf-log.jsonl \
  > /tmp/ours.jsonl
jq -c 'select(.example == "hf-llama")' docs/develop/perf-log-ref.jsonl \
  > /tmp/theirs.jsonl
# Then eyeball or feed to a comparison script.
```
