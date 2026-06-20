# mk/e2e.mk — example smoke/convergence matrices. EXAMPLES/BACKENDS
# lists + thin shell-outs to scripts/test-e2e-examples.sh and
# scripts/test-convergence.sh.

# Examples run on every built backend. Keep in sync with packages/idris-ml-examples/src/Example/.
# Excluded intentionally:
#   Bench, Profile — no RESULT lines (covered by bench-compare / example-profile).
.PHONY: test-e2e-examples all-backends test-convergence test-convergence-campaign

EXAMPLES := example-supervised example-rnn example-lstm example-gru example-transformer example-gpt example-matmul-bench example-mnist example-seq-classify example-ntm-copy example-ntm-associative-recall example-dnc-copy example-dnc-recall example-reinforce example-q-learning example-sarsa example-monte-carlo example-frozen-lake example-taxi example-dqn example-double-dqn example-mountain-car example-mountain-car-cont example-a2c example-ppo example-sac example-checkpoint
# 5-lane matrix. `mlx-gpu` (BACKEND=mlx MLX_DEVICE=gpu) and `torch-mps`
# (BACKEND=torch TORCH_DEVICE=mps) are virtual lanes that exercise the
# F32 code paths (per BuildConfig.idr); tape / mlx / torch build at F64.
# torch-mps is Darwin-only: on Linux it shares the plain torch dylib, so
# the lane's `make backend` succeeds (no build-failure skip) and every
# example then crashes at MPS construction. The mlx lanes stay in the
# Linux list on purpose — `make BACKEND=mlx backend` $(error)s off-macOS,
# which routes them through the harness's build-failure skip and thereby
# also disables the tape+mlx+torch demos (checkpoint/precision).
ifeq ($(UNAME),Darwin)
BACKENDS := tape mlx mlx-gpu torch torch-mps
else
BACKENDS := tape mlx mlx-gpu torch
endif

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

# Multi-seed convergence CAMPAIGN: the same runner with SEEDS + a
# resumable TSV (CONVERGENCE_OUT) → per-example pass-rate table (the
# "multi-seed convergence is required" alignment policy). Cheapest-first
# example order so the fast/medium table lands early; NTM/DNC grind last.
# Resumable — re-run after a kill and it continues from the TSV.
CONVERGENCE_SEEDS ?= 42 1 2 3 4
CONVERGENCE_OUT   ?= docs/develop/convergence-campaign.tsv
# Note: example-transfer / example-precision-demo are deliberately NOT here —
# they're cross-backend correctness demos with no convergence metric, and the
# campaign runs single-backend (where they clean-skip via HAVE_ALL_MULTI_BACKENDS
# anyway). They stay covered by the dedicated multi-backend lane (test-examples
# with BACKEND=tape,torch,mlx + PRECISION_DEMO_READY=1).
CONVERGENCE_CAMPAIGN_EXAMPLES := example-supervised example-rnn example-lstm \
	example-gru example-transformer example-seq-classify \
	example-gpt example-q-learning example-sarsa example-monte-carlo \
	example-frozen-lake example-taxi example-mnist example-reinforce \
	example-dqn example-double-dqn example-mountain-car example-mountain-car-cont example-a2c \
	example-ppo example-sac example-ntm-copy example-ntm-associative-recall \
	example-dnc-copy example-dnc-recall

test-convergence-campaign:
	+@MAKE='$(MAKE)' EXAMPLES='$(CONVERGENCE_CAMPAIGN_EXAMPLES)' \
		SEEDS='$(CONVERGENCE_SEEDS)' \
		CONVERGENCE_TIMEOUT='$(CONVERGENCE_TIMEOUT)' \
		CONVERGENCE_EXPECT='$(CONVERGENCE_EXPECT)' \
		CONVERGENCE_OUT='$(CONVERGENCE_OUT)' \
		bash scripts/test-convergence.sh
