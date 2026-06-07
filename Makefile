# idris-ml root Makefile — wiring + top-level aggregates only.
# The build logic lives in mk/*.mk fragments, included in
# dependency order below (config first: knobs + MAKECMDGOALS
# sniffing must precede every consumer). Target names are a
# public API (CI spec, perf scripts, docs) — keep them stable.

# `make` builds backend + type-checks all packages. `make all` also runs tests.
.DEFAULT_GOAL := check-all

include mk/config.mk
include mk/backends.mk
include mk/genconfig.mk
include mk/lint.mk
include mk/tests.mk
include mk/install.mk
include mk/examples.mk
include mk/bench.mk
include mk/ref.mk
include mk/jupyter.mk
include mk/e2e.mk

# `clean` removes everything `make install` / `make backend` can regenerate
# from source: every backend set's tree under `build/`, the coverage tree
# under `build-cov/`, and the legacy pre-per-set `.idris2/` install prefix
# (orphan from before commit-XXX). Does NOT touch downloaded deps:
# `vendored/` (third-party C source), `data/` (datasets), `models/` (HF
# checkpoints) — those are network-expensive and out of scope for clean.
# Use `clean-all` to nuke those too.
clean:
	rm -rf build/
	rm -rf build-cov/
	rm -rf .idris2/

# Active backend set's tree only — `BACKEND=tape make clean-set` removes
# `build/tape-mlxcpu-torchcpu/` but leaves other set caches alone. Use
# when a single set is in a weird state and a full `clean` would discard
# other sets' warm caches unnecessarily.
clean-set:
	rm -rf $(BUILD)

# Everything that's gitignored: build artifacts + vendored third-party
# source + downloaded datasets + downloaded HF model checkpoints.
# Network-expensive (re-running `make backend` will re-clone vendored/,
# re-running examples will re-download datasets, and HF models are
# gigabytes). Reach for this when freeing disk space or before a deep
# refactor; otherwise plain `clean` is enough.
clean-all: clean clean-models
	rm -rf vendored/
	rm -rf data/

# Downloaded HuggingFace checkpoints, tokenizer vocab files, and the
# generated test oracle. Kept out of plain `clean` because re-downloading
# is slow; run this explicitly when you need to free disk space or force
# a fresh fetch.
clean-models:
	rm -rf models/
	# Legacy location (pre-2026-05-27 refactor); remove if leftover.
	rm -rf packages/idris-transformers/models/

# Downloaded + tokenized HF datasets (under data/hf-datasets/). Same
# rationale as clean-models — slow to re-download, so kept out of
# plain `clean`. Hand-curated test fixtures
# (packages/idris-transformers/test-fixtures/) stay untouched.
clean-datasets:
	rm -rf data/hf-datasets/

# Run everything: unit + gym + examples-unit + multi-backend criterion +
# specialized + e2e examples + PyTorch ref + jupyter. Multi-hour aggregate;
# not a CI gate. Subsequent phases will collapse this into per-layer
# aggregators (test-unit / test-integration / test-e2e) — for now it
# chains the layer aggregators directly.
test-all:
	@echo "=== Unit layer (Idris core + gym + transformers + examples + Criterion + NTM unit) ==="
	$(MAKE) test-unit
	@echo ""
	@echo "=== C backend tests (all available backends) ==="
	@for b in tape mlx torch; do \
		echo "--- test-unit-c [$$b] ---"; \
		$(MAKE) BACKEND=$$b test-unit-c 2>&1 && echo "" || echo "FAILED or SKIPPED: $$b"; \
	done
	@echo "=== Integration layer ==="
	$(MAKE) test-integration
	@echo ""
	@echo "=== E2E tests (examples on all backends) ==="
	$(MAKE) test-e2e-examples
	@echo ""
	@if command -v uv >/dev/null 2>&1 && [ -f packages/pytorch/pyproject.toml ]; then \
		echo "=== PyTorch reference tests ==="; \
		$(MAKE) test-e2e-pytorch-ref; \
	else \
		echo "=== PyTorch reference tests SKIPPED (uv not found) ==="; \
	fi
	@echo ""
	@if command -v pytest >/dev/null 2>&1 && [ -f packages/jupyter/pyproject.toml ]; then \
		echo "=== Jupyter kernel tests ==="; \
		$(MAKE) test-e2e-jupyter; \
	else \
		echo "=== Jupyter kernel tests SKIPPED (pytest or jupyter not found) ==="; \
	fi
	@echo ""
	@if [ -d packages/jupyter/.venv ] && $(JUPYTER_VENV)/bin/jupyter --version >/dev/null 2>&1; then \
		echo "=== Notebook execution tests ==="; \
		$(MAKE) test-e2e-notebooks; \
	else \
		echo "=== Notebook execution tests SKIPPED (jupyter not installed) ==="; \
	fi
	@echo ""
	@echo "=== All tests complete ==="

# Type-check notebook prelude package
check-notebook: install-core
	cd packages/idris-ml-notebook && IDRIS2_PREFIX=$(IDRIS2_LOCAL) idris2 --build-dir $(CURDIR)/$(BUILD)/ttc-idris-ml-notebook --build idris-ml-notebook.ipkg

# Build backend + type-check all packages + build every example
# executable. The exhaustive "everything compiles" gate; dominated by
# `check-examples` wall-clock (~20-60 min cold; warm is faster).
# For a quicker preflight that skips the per-example elaboration, use
# `check` (libraries only, a few minutes).
check-all: check check-examples

# Verify everything: check-all + run all tests
all: check-all test-all

.PHONY: all check check-libs check-idris-ml check-gym check-transformers check-notebook check-examples check-all \
        test test-unit test-unit-idris-ml test-unit-idris-transformers \
        test-unit-gym test-unit-examples test-unit-multi-backend test-all dataset-mnist dataset-tinyshakespeare \
        test-unit-c test-unit-c-tape test-unit-c-mlx test-unit-c-torch \
        test-integration test-integration-lint-rename-headers test-integration-lint-ffi-wrap-template \
        test-integration-lint-non-io-side-effects test-integration-lint-paired-defaults \
        test-integration-lint-hf-llama-inference test-integration-lint-ci-workflow \
        test-integration-lint-benchmarks bench bench-fast bench-nightly bench-full \
        all-backends \
        test-integration-typegate-gradmode \
        test-integration-typegate-gradmode-aliasing test-integration-typegate-lossy-cast \
        test-integration-typegate-int-overflow-cast test-integration-checkpoint-resume \
        test-integration-jupyter-cellparser \
        test-coverage test-coverage-backend test-coverage-backend-tape test-coverage-backend-mlx \
        test-coverage-backend-torch test-coverage-gap-probe \
        test-e2e test-e2e-examples test-e2e-pytorch-ref test-e2e-jupyter test-e2e-notebooks test-e2e-cuda \
        test-e2e-hf-bert-roundtrip test-e2e-hf-gpt2-roundtrip test-e2e-hf-bitnet-roundtrip \
        test-e2e-hf-llama-roundtrip test-e2e-hf-llama-generate-roundtrip \
        test-e2e-transformers-oracle-bert test-e2e-transformers-oracle-gpt2 \
        test-e2e-transformers-oracle-llama test-e2e-transformers-oracle-llama-generate \
        test-e2e-rope-oracle \
        test-convergence \
        check check-gym check-notebook check-examples install install-core install-gym install-notebook install-examples \
        example-supervised example-rnn example-lstm example-gru \
        example-ntm-copy example-ntm-associative-recall example-dnc-copy example-dnc-recall \
        example-reinforce example-q-learning example-sarsa example-monte-carlo example-frozen-lake example-taxi \
        example-dqn example-mountain-car example-mountain-car-cont example-a2c example-ppo example-sac \
        example-gpt example-gpt-full example-matmul-bench example-mnist example-seq-classify example-transformer \
        ref-gpt \
        example-transfer example-checkpoint example-checkpoint-demo \
        example-bench example-profile sweep sweep-quick clean \
        backend print-torch ref-setup ref-supervised ref-rnn ref-lstm ref-gru ref-ntm-copy \
        ref-ntm-recall ref-dnc-copy ref-dnc-recall \
        ref-transformer ref-hf-bert ref-hf-gpt2 ref-hf-llama \
        bench-py bench-compare bench-ops bench-ops-py bench-ops-compare \
        bench-layers bench-layers-py ref-lint \
        ref-typecheck ref-convergence ref-convergence-copy ref-convergence-recall \
        jupyter-install jupyter-lab
