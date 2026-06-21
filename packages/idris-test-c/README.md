# idris-test-c

Cross-cutting C test infrastructure for idris-ml's backend layer.

## What lives here

Test files whose source-of-truth isn't a single C file — integration
tests, framework smoke, oracle ladders, shared training-loop machinery.
Pure C; no Idris source.

| File | Kind | What it covers |
|---|---|---|
| `src/test_criterion_smoke.c` | Criterion smoke | Framework smoke (verifies Criterion links + per-test process isolation). |
| `src/test_autograd.c` | Criterion | Autograd + native SGD/RMSprop step semantics. |
| `src/test_linalg.c` | Criterion | Linear-algebra + structural tensor ops. |
| `src/test_nn_layers.c` | Criterion | Linear / norm / conv / pool / dropout / embedding layers. |
| `src/test_activations.c` | Criterion | softmax / leaky-relu / silu / softplus. |
| `src/test_losses.c` | Criterion | MSE / cross-entropy backward. |
| `src/test_lstm.c` | Criterion | LSTM cell + gradient-chain. |
| `src/test_tensor_misc.c` | Criterion | Miscellaneous tensor (safetensors roundtrip). |
| `src/test_dtype_scaffolding.c` | Criterion | Cross-backend dtype dispatch / runtime tag layout. |
| `src/test_param_registry.c` | Criterion | Shared param registry (cross-backend infra). |
| `src/test_clip_grad_norm.c` | Criterion | Optimizer clip-grad-norm path. |
| `src/test_optimizers.c` | Criterion | SGD / RMSprop / Adam / AdamW step semantics. |
| `src/test_safetensors.c` | Criterion | SafeTensors save/load round-trips (param + optimizer state; bf16/f16/i32 under `BACKEND_TORCH`). |
| `src/test_ntm_grad.c` | Criterion (single Test()) | NTM backward chain at realistic scale. |
| `src/test_ntm_timestep.c` | Criterion (single Test()) | NTM single timestep (LSTM + FC + addressing + output). |
| `src/test_mlx_compile.c` | Criterion | mlx::compile integration (MLX-only, `#ifdef BACKEND_MLX`). |
| `include/test_helpers.h` | Header | Backend-aware tolerance + readout helpers. |
| `include/port_assert.h` | Header | `ASSERT_NEAR` / `ASSERT_TRUE` shims + FD/VAL tolerances + `heap_copy` for the split per-area suites. |

## What does NOT live here

Tests with a clean 1:1 source-file pair live next to their source under
`packages/backends/backend_{tape,torch,mlx}/<subsystem>/` — e.g.
`backend_tape/core/elementwise/test_add.c` next to `add.c`. Criterion's
auto-discovery via the Makefile's `find` glob picks them up either way;
colocation is the readability default. See
[`docs/develop/testing-taxonomy.md`](../../docs/develop/testing-taxonomy.md)
for the hybrid-layout rule.

## How tests are built

The Makefile globs `src/*.c` and links everything into one Criterion
binary at `build/<KEY>/test_criterion_smoke`. The former standalone
main() tests (`test_safetensors`, `test_ntm_*`, `test_mlx_compile`)
were converted to Criterion suites on 2026-06-05 and ride the same
glob; their dedicated `test-unit-*` make recipes are gone.

Include paths: the build adds `-Ipackages/backends` and
`-Ipackages/idris-test-c/include`, so test files use bare `#include
"backend.h"` / `#include "test_helpers.h"` without relative paths.

## Run targets

```
make test-unit-c                  # Criterion suite (all common + colocated)
make test-unit-c-tape             # same, forcing BACKEND=tape
make test-unit-c-torch            # same, forcing BACKEND=torch
make test-unit-c-mlx              # same, forcing BACKEND=mlx
```
