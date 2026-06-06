# scripts/

User-facing CLI surface for the idris-ml monorepo: CI gates, perf
benchmarking, codegen, dataset acquisition, and dev helpers. Per-package
scripts (HF inference oracles, dtype verifiers, PyTorch references) live
under `packages/*/scripts/`.

## Directory map

```
scripts/
├── README.md                       this file
├── mltools/                        internal Python package — shared library
│   ├── __init__.py
│   ├── perf_log.py                 perf-log Entry hub + writer CLI
│   ├── header_parser.py            backend.h enum / FFI-symbol parsing
│   └── sweep_grid.py               hyperparameter grid expansion + CSV
├── codegen/                        FFI codegen + lint cluster
│   ├── ffi_manifest.py             single source of truth for Tensor-touching FFIs
│   ├── gen-executor-instances.py   regenerate Executor/{Tape,Torch,Mlx}.idr blocks
│   ├── gen-rename-headers.py       regenerate packages/backends/rename_<b>.h
│   ├── ffi-convert-to-scheme.py    rewrite C %foreign decls into Scheme wrappers
│   ├── check-ffi-wrap-template.py  lint wrap-handle invariants per FFI
│   └── check-non-io-side-effects.py lint %foreign side-effect typing
├── perf_lib.sh                     sourced by perf-*.sh; common shell helpers
├── perf-run.sh                     kind=run    (full convergence + structured log)
├── perf-baseline.sh                kind=baseline (single example, in-script marker)
├── perf-sweep.sh                   kind=baseline (Tier 3 cross-backend sweep)
├── perf-fast.sh                    kind=op_bench (Tier 1 Axes A/B, ≤ 5 min)
├── perf-nightly.sh                 kind=op_bench (Tier 2 + Axes C/D)
├── perf-run-quiet.sh               caffeinate + nice wrapper around perf-run.sh
├── render-benchmarks.py            BENCHMARKS.md from perf-log.jsonl
├── check-perf-regression.py        median-window regression gate on perf-log.jsonl
├── check-paired-defaults.py        Idris ↔ torch_ref config drift gate
├── check-executor-method-drift.py  Executor interface drift gate
├── check-result.sh                 example RESULT-line threshold validator
├── check-gradmode-aliasing.sh      type-system negative-test gates
├── check-gradmode-gate.sh
├── check-int-overflow-cast-gate.sh
├── check-lossy-cast-gate.sh
├── coverage-gap-probe.py           OP_* / FFI test-coverage relational join
├── gen-ci-workflow.py              .github/workflows/test.yml regenerator
├── sweep.py                        generic hyperparameter sweep harness
├── sweeps/                         JSON grid specs for sweep.py
├── dataset_mnist.sh                MNIST download → data/mnist/
├── dataset_tinyshakespeare.sh      tinyshakespeare → data/tinyshakespeare/
├── test-checkpoint-resume.sh       Train.idr resume smoke test
└── test_cuda_colab.sh              one-off CUDA tester for Colab boxes
```

## Conventions

**Bash for orchestration, Python for meaning.** Bash is the right glue for
`make → binary → env-var → exit code`. The moment a script needs to *understand*
data — parse JSON, classify against thresholds, construct structured records —
it should be Python. The shared library for that is `mltools/`.

**Single source of truth per concept.**
  - `mltools/perf_log.py` — perf-log Entry construction (read by writers + readers)
  - `codegen/ffi_manifest.py` — Tensor-touching FFI manifest (read by 4 codegen tools)
  - `mltools/header_parser.py` — backend C/C++ parsing primitives
  - `mltools/sweep_grid.py` — grid expansion + CSV assembly

**Shell standards.**
  - Shebang: `#!/usr/bin/env bash` (portable; not `#!/bin/bash`)
  - `set -euo pipefail` for self-contained scripts; the type-system negative-test
    gates (`check-gradmode-*.sh`, `check-*-cast-gate.sh`) deliberately use only
    `set -u` because they invoke idris2 expecting compile failure and branch on
    its exit code
  - `perf_lib.sh` is sourced, not executed — sets no shell options of its own

**Python standards.**
  - Shebang: `#!/usr/bin/env python3` for entry points
  - Imports from `mltools.*` use `sys.path.insert(0, str(ROOT / "scripts"))`
    or `PYTHONPATH=$REPO/scripts` (perf_lib.sh sets the latter for bash callers)
  - argparse for CLI, dataclasses for structured types, `subprocess.run` for
    process spawning, `concurrent.futures.ProcessPoolExecutor` for parallel work

**Output destinations.**
  - `docs/develop/perf-log.jsonl` — append-only canonical perf store
    (schema: `docs/develop/perf-log.md`)
  - `BENCHMARKS.md` — auto-regenerated from perf-log.jsonl
  - `build/<BUILD_KEY>/coverage-gap-*.csv` — coverage-gap-probe output
  - `results/sweep-<name>.csv` — sweep.py output

## See also

  - `docs/develop/perf-log.md` — JSONL schema + querying cookbook
  - `docs/develop/coverage-policy.md` — what coverage-gap-probe gates
  - `docs/develop/testing.md` — test layer map
  - `CLAUDE.md` Build Commands / Performance optimization workflow sections
