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

# `clean` removes everything `make install` / `make backend` / the test
# builds can regenerate from source: every backend set's tree under
# `build/`, the coverage tree under `build-cov/`, the per-package `build/`
# dirs written by the pack-driven test builds (`pack build *-tests.ipkg`
# defaults to the package-local builddir), and the legacy pre-per-set
# `.idris2/` install prefix (orphan from before c5c78ee9, 2026-05-17).
# Does NOT touch downloaded deps: `vendored/` (third-party C source),
# datasets, `models/` (HF checkpoints), or the Python venvs — those are
# network-expensive and out of scope for clean. Use `clean-all` to nuke
# those too.
clean:
	rm -rf build/
	rm -rf build-cov/
	rm -rf .idris2/
	rm -rf packages/*/build/

# Active backend set's tree only — `BACKEND=tape make clean-set` removes
# `build/tape-mlxcpu-torchcpu/` but leaves other set caches alone. Use
# when a single set is in a weird state and a full `clean` would discard
# other sets' warm caches unnecessarily.
clean-set:
	rm -rf $(BUILD)

# Everything that's gitignored: build artifacts + vendored third-party
# source + downloaded datasets + downloaded HF model checkpoints +
# Python venvs + run-output dirs. Network-expensive (re-running
# `make backend` will re-clone vendored/, re-running examples will
# re-download datasets, HF models are gigabytes, and `make ref-setup` /
# `make jupyter-install` rebuild the venvs). Reach for this when freeing
# disk space or before a deep refactor; otherwise plain `clean` is
# enough. Deliberately untouched: `.claude/` (session/project memory).
clean-all: clean clean-models clean-datasets clean-venvs
	rm -rf vendored/
	rm -rf logs/ results/ .tmp/

# Downloaded HuggingFace checkpoints, tokenizer vocab files, and the
# generated test oracle. Kept out of plain `clean` because re-downloading
# is slow; run this explicitly when you need to free disk space or force
# a fresh fetch.
clean-models:
	rm -rf models/
	# Legacy location (pre-2026-05-27 refactor); remove if leftover.
	rm -rf packages/idris-transformers/models/

# Downloaded datasets: everything under `data/` (MNIST, tinyshakespeare,
# tokenized HF datasets) plus the PyTorch ref's own download root
# `packages/pytorch/data/` — torch_ref resolves its relative "data/mnist"
# from the package cwd, not the repo root. Same rationale as clean-models
# — slow to re-download, so kept out of plain `clean`. Hand-curated test
# fixtures (packages/idris-transformers/test-fixtures/) stay untouched.
clean-datasets:
	rm -rf data/
	rm -rf packages/pytorch/data/

# Python virtualenvs — recreated by `make ref-setup` (pytorch) and
# `make jupyter-install` (jupyter). The largest disk consumers after
# models/ (~1.3 GB combined), so worth a dedicated target.
clean-venvs:
	rm -rf packages/pytorch/.venv/
	rm -rf packages/jupyter/.venv/

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

.PHONY: all check-all test-all check-notebook \
        clean clean-set clean-all clean-models clean-datasets clean-venvs
