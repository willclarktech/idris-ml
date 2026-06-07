# mk/bench.mk — benchmarking suite. bench-py/compare, operator-level
# bench_ops, layer benches, the bench-fast/nightly/full tiers, and
# the perf-regression lint gates.

.PHONY: bench-py bench-compare bench-ops bench-ops-py bench-layers \
        bench-layers-py bench-fast bench-deep bench-full bench \
        test-integration-lint-benchmarks \
        test-integration-lint-perf-regression lint-perf-run \
        bench-ops-compare

bench-py:
	cd packages/pytorch && uv run python -m torch_ref.benchmark $(BENCH)

bench-compare: example-bench
	cd packages/pytorch && uv run python -m torch_ref.compare

# Build bench_ops linked against the active backend.
# bench_ops.c calls unsuffixed names (tensor_create, tensor_mm, ...); the
# dylib only exports `_<backend>` suffixed symbols since the unified-name
# alias machinery was retired. Splice in rename_$(PRIMARY).h so the C
# preprocessor rewrites every call site to the primary's suffixed name.
$(BUILD)/bench_ops: $(BACKENDS_DIR)/bench_ops.c backend | $(BUILD)
	cc -o $(BUILD)/bench_ops $(BACKENDS_DIR)/bench_ops.c -include $(BACKENDS_DIR)/rename_$(PRIMARY).h -L$(BUILD) -lidrisml -Wl,-rpath,$(CURDIR)/$(BUILD) -lm

# Build bench_ops for a specific backend (e.g., make bench-ops-build-tape)
bench-ops-build-%: $(BACKENDS_DIR)/bench_ops.c | $(BUILD)
	@$(MAKE) --no-print-directory BACKEND=$* backend 2>/dev/null
	cc -o $(BUILD)/bench_ops_$* $(BACKENDS_DIR)/bench_ops.c -include $(BACKENDS_DIR)/rename_$*.h -L$(BUILD) -lidrisml -Wl,-rpath,$(CURDIR)/$(BUILD) -lm

bench-ops: $(BUILD)/bench_ops
	./$(BUILD)/bench_ops

bench-ops-py:
	cd packages/pytorch && uv run python -m torch_ref.bench_ops

# Axis B — Idris-level single-layer fwd+bwd microbench. Counterpart to
# bench-ops (Axis A, C-kernel) one rung up: measures the FFI + tape wrap
# + autograd graph cost at the layer-composition level. Output line
# format matches Axis A so scripts/perf-fast.sh parses both with one
# regex; the entries get `axis="B"` tagged before emission to perf-log.
bench-layers: install
	idris2 $(IDRIS_FLAGS) -o layers-bench $(EXAMPLE_SRC)/Example/LayersBench.idr
	cp $(LIB) $(BUILD)/exec/layers-bench_app/
	$(STDBUF) ./$(BUILD)/exec/layers-bench $(LAYERS_BENCH_ARGS)

bench-layers-py:
	cd packages/pytorch && uv run python -m torch_ref.bench_layers

# Compare all available backends vs PyTorch.
# Each iteration rebuilds libidrisml.dylib with only one backend as
# primary (BACKEND=$$b → single-element list), then copies it to a
# backend-named filename for the bench_ops binary to link against.
# Under multi-link this is a real rebuild per backend, but bench_ops is
# operator-level so we want isolated per-backend timings anyway.
####################################################################
# Principled perf benchmark suite (testing-taxonomy Axis A / B / C / D).
# Three cadence tiers:
#   bench-fast    — Tier 1, CI, <= 5 min (Axis A + B).
#   bench-deep    — Tier 2, CI per publication push, <= 20 min (Axes A+B+C+D).
#   bench-full    — Tier 3, manual / pre-tag (the cross-backend sweep).
# `bench` aliases `bench-fast` (the daily-driver default).
# All three append to docs/develop/perf-log.jsonl and regenerate
# BENCHMARKS.md via scripts/render-benchmarks.py. Framework details:
# docs/develop/testing-taxonomy.md (Axis A/B/C/D + selection rule).
####################################################################

bench-fast:
	bash scripts/perf-fast.sh

bench-deep:
	bash scripts/perf-deep.sh

bench-full:
	bash scripts/perf-sweep.sh

# Default `bench` aggregator — alias for the fast tier. For deeper
# perf coverage use `bench-deep` or `bench-full`.
bench: bench-fast

# CI preflight: BENCHMARKS.md must agree with perf-log.jsonl.
test-integration-lint-benchmarks:
	python3 scripts/render-benchmarks.py --check

# CI preflight: perf-regression advisory gate. Reads perf-log.jsonl,
# computes a median-of-last-5 baseline per (axis, label, runtime),
# classifies the latest as OK / WARN (>15%) / FAIL (>40%) vs that
# baseline. Always exits 0 today (advisory); a later commit will
# flip the FAIL threshold to exit 1.
test-integration-lint-perf-regression:
	python3 scripts/check-perf-regression.py

# Companion `--mode run` gate over kind="run" example-perf entries.
# Groups by (example, backend, args), medians the last 10 runs,
# fails (exit 1) when the latest is > 100% slower than that
# baseline AND PERF_GATE=1 is set. Without PERF_GATE the script
# always exits 0 — informational only — so a noisy initial
# rollout doesn't break CI. Once the noise profile is calibrated,
# the CI step (or a follow-up commit) sets PERF_GATE=1 to promote
# to hard-fail.
lint-perf-run:
	python3 scripts/check-perf-regression.py --mode run

bench-ops-compare:
	@for b in tape mlx torch; do \
		$(MAKE) --no-print-directory BACKEND=$$b backend 2>/dev/null || continue; \
		cp $(BUILD)/libidrisml.$(LIB_EXT) $(BUILD)/libidrisml_$$b.$(LIB_EXT); \
		cc -o $(BUILD)/bench_ops_$$b $(BACKENDS_DIR)/bench_ops.c \
			$(BUILD)/libidrisml_$$b.$(LIB_EXT) -Wl,-rpath,$(CURDIR)/$(BUILD) -lm -lc++ 2>/dev/null \
		|| true; \
	done
	cd packages/pytorch && uv run python -m torch_ref.compare_ops
