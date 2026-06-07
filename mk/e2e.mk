# mk/e2e.mk — example smoke/convergence matrices. EXAMPLES/BACKENDS
# lists + thin shell-outs to scripts/test-e2e-examples.sh and
# scripts/test-convergence.sh.

# Examples run on every built backend. Keep in sync with packages/idris-ml-examples/src/Example/.
# Excluded intentionally:
#   Bench, Profile — no RESULT lines (covered by bench-compare / example-profile).
EXAMPLES := example-supervised example-rnn example-lstm example-gru example-transformer example-gpt example-matmul-bench example-mnist example-seq-classify example-ntm-copy example-ntm-associative-recall example-dnc-copy example-dnc-recall example-reinforce example-q-learning example-sarsa example-monte-carlo example-frozen-lake example-taxi example-dqn example-mountain-car example-mountain-car-cont example-a2c example-ppo example-sac example-checkpoint
# 5-lane matrix. `mlx-gpu` (BACKEND=mlx MLX_DEVICE=gpu) and `torch-mps`
# (BACKEND=torch TORCH_DEVICE=mps) are virtual lanes that exercise the
# F32 code paths (per BuildConfig.idr); tape / mlx / torch build at F64.
BACKENDS := tape mlx mlx-gpu torch torch-mps

# Crash-only smoke gate: every example × lane, 3-10 epochs each,
# safety-net thresholds in test-examples.expect. Catches crashes / NaN /
# divergence / missing RESULT keys; does NOT require any model to learn.
# See docs/develop/testing.md for the full testing-layer overview.
#
# FAIL_FAST=1 bails on the first failure (handy for the iteration loop);
# the default empty value runs the whole matrix so a final confirmation
# surfaces every failure at once.
FAIL_FAST ?=

# Readiness gate for the example-precision-demo post-matrix step.
# Defaults on; flip to 0 only when temporarily skipping the demo
# (e.g. while debugging the multi-backend hop). Folds away once
# the demo has lived through a few stable CI runs.
PRECISION_DEMO_READY ?= 1
# Recipe body lives in scripts/test-e2e-examples.sh (incl. the smoke_args
# table + the ARGS-var naming guard). The `+` prefix is load-bearing: it
# keeps the jobserver/MAKEFLAGS alive for the script's $MAKE sub-builds.
test-e2e-examples:
	+@MAKE='$(MAKE)' EXAMPLES='$(EXAMPLES)' BACKENDS='$(BACKENDS)' \
		EXAMPLE_TIMEOUT='$(EXAMPLE_TIMEOUT)' FAIL_FAST='$(FAIL_FAST)' \
		PRECISION_DEMO_READY='$(PRECISION_DEMO_READY)' \
		bash scripts/test-e2e-examples.sh

all-backends: test-e2e-examples

# Run every example to convergence at full default epochs, single seed=42,
# tape backend, with tight thresholds from test-examples-convergence.expect.
# Hours of wall time (NTM/DNC dominate). Intended for release validation,
# not CI. See docs/develop/testing.md for the testing-layer overview.
# 4h per-example cap. DNC-copy at default 50K epochs now runs in ~1.7h on
# tape (~130ms/epoch post the 2026-05-02 tensor-handle rewrite — see
# `dnc-perf-baseline.md`). Other examples are well under this cap.
CONVERGENCE_TIMEOUT ?= 14400
CONVERGENCE_EXPECT := test-examples-convergence.expect

# Recipe body lives in scripts/test-convergence.sh; `+` prefix as above.
test-convergence:
	+@MAKE='$(MAKE)' EXAMPLES='$(EXAMPLES)' \
		CONVERGENCE_TIMEOUT='$(CONVERGENCE_TIMEOUT)' \
		CONVERGENCE_EXPECT='$(CONVERGENCE_EXPECT)' \
		bash scripts/test-convergence.sh
