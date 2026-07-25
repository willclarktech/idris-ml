# mk/tests.mk — unit-test layer + aggregators. Criterion C suites
# (+ASAN, +coverage), microbenches, check-examples, Idris package
# suites, and the test / test-unit / test-integration / test-e2e /
# test-coverage aggregators.

# Criterion-driven test suite (per-test process isolation + JUnit XML).
# Today only ships a smoke test (test_criterion_smoke.c) verifying the
# framework links and runs. Phase 1 (per /Users/admin/.claude/plans/modular-petting-minsky.md)
# migrates the per-op suites into packages/backends/test/<backend>/...,
# which this target will discover and link.
#
# Criterion is provided by nix (nixpkgs `criterion` + `criterion.dev`).
# Include / lib paths derived from the user nix-profile; an explicit
# CRITERION_PREFIX= overrides for non-nix environments.
.PHONY: test-unit-c test-unit-c-tape test-unit-c-mlx test-unit-c-torch \
        test-unit-c-asan-tape test-unit-c-asan-mlx \
        test-unit-c-asan-torch test-unit-c-asan test-coverage-backend \
        test-coverage-backend-tape test-coverage-backend-mlx \
        test-coverage-backend-torch test-coverage-all test-coverage-gap-probe \
        reach-dump test-coverage-reach-gap \
        bench-rank3-broadcast bench-rank3-broadcast-wrapped print-torch \
        check-examples test-unit test-unit-idris test test-integration \
        test-e2e test-coverage test-unit-idris-ml \
        test-unit-multi-backend test-unit-gym test-unit-args \
        test-unit-idris-transformers bench-gym test-unit-examples \
        test-unit-fmt

# Criterion discovery. Order: explicit CRITERION_PREFIX override, then
# pkg-config (the nix dev shell + most distros ship criterion.pc), then a
# prefix probe (nix profile / brew / apt). pkg-config is preferred because
# it returns the correct include + lib + rpath even for the coverage
# lane's raw clang, which bypasses the nix cc wrapper's NIX_CFLAGS_COMPILE
# and so can't find buildInputs implicitly (the non-coverage build's
# wrapped cc can). The probe's nix-profile fallback keeps the error as
# "criterion.h not found" rather than a cryptic -I/include.
CRITERION_PC_LIBS := $(shell pkg-config --libs criterion 2>/dev/null)
ifdef CRITERION_PREFIX
CRITERION_CFLAGS := -I$(CRITERION_PREFIX)/include
CRITERION_LDFLAGS := -L$(CRITERION_PREFIX)/lib -lcriterion -Wl,-rpath,$(CRITERION_PREFIX)/lib
else ifneq ($(strip $(CRITERION_PC_LIBS)),)
CRITERION_CFLAGS := $(shell pkg-config --cflags criterion 2>/dev/null)
CRITERION_LDFLAGS := $(CRITERION_PC_LIBS)
else
CRITERION_DETECT := $(firstword $(wildcard $(HOME)/.nix-profile/include/criterion /opt/homebrew/include/criterion /usr/include/criterion))
ifeq ($(CRITERION_DETECT),)
CRITERION_PREFIX := $(HOME)/.nix-profile
else
CRITERION_PREFIX := $(patsubst %/include/criterion,%,$(CRITERION_DETECT))
endif
CRITERION_CFLAGS := -I$(CRITERION_PREFIX)/include
CRITERION_LDFLAGS := -L$(CRITERION_PREFIX)/lib -lcriterion -Wl,-rpath,$(CRITERION_PREFIX)/lib
endif
# Runtime flags forwarded to the criterion binary on the command line.
# Examples:
#   CRITERION_FLAGS='--filter=smoke/hello'         # run one test
#   CRITERION_FLAGS='--verbose'                    # per-assertion log
#   CRITERION_FLAGS='-j1 --no-early-exit'          # disable forking (debugger-friendly)
#   CRITERION_FLAGS='--xml=foo.xml --tap=foo.tap'  # multi-format reports
# `--xml=build/test-criterion-<b>.xml` is appended by the recipe so JUnit
# output always lands at a predictable path for CI; user-supplied
# CRITERION_FLAGS are prepended so they take precedence if duplicated.
CRITERION_FLAGS ?=

# Overridable test-binary compiler (COV_CC_OVERRIDES sets clang on the
# Linux coverage lane; gcc rejects the clang-only coverage flags).
TEST_CC ?= cc

TEST_C_DIR := packages/idris-test-c
# Discover Criterion suites. Test location encodes semantic coupling
# (see docs/develop/testing-taxonomy.md "Test file layout"):
#  - backend_{tape,torch,mlx}/<subsystem>/test_<topic>.c — backend-specific
#    tests (one backend's internals/numerics), always #ifdef BACKEND_<NAME>.
#  - $(TEST_C_DIR)/src/ops/<subsystem>/test_<op>.c — contract tests of the
#    public backend.h FFI; run against whatever dylib is primary.
#  - $(TEST_C_DIR)/src/ — cross-cutting infra (framework smoke, NTM
#    integration, oracle ladder, param registry, clip-grad-norm, optimizers).
# All three backend dirs are globbed regardless of primary; non-primary
# bodies compile away via their #ifdef. Test files are always `.c` — the
# dylib source globs exclude them (backends.mk).
CRITERION_BACKEND_TEST_SRCS := $(shell find $(BACKENDS_DIR)/backend_tape -name 'test_*.c' 2>/dev/null) \
                               $(shell find $(BACKENDS_DIR)/backend_torch -name 'test_*.c' 2>/dev/null) \
                               $(shell find $(BACKENDS_DIR)/backend_mlx -name 'test_*.c' 2>/dev/null) \
                               $(shell find $(TEST_C_DIR)/src -name '*.c' -not -name 'test_criterion_smoke.c' 2>/dev/null)
CRITERION_TEST_SRCS := $(TEST_C_DIR)/src/test_criterion_smoke.c $(CRITERION_BACKEND_TEST_SRCS)
TEST_C_INCLUDES := -I$(BACKENDS_DIR) -I$(TEST_C_DIR)/include

# Under ASan, point LeakSanitizer at the suppression file for tape's
# by-design persistent allocations (lsan_teardown.c frees the rest
# before the exit-time check). Empty in non-ASan runs.
LSAN_SUPP := $(TEST_C_DIR)/lsan.supp
TEST_RUN_ENV := $(if $(ASAN),LSAN_OPTIONS=suppressions=$(abspath $(LSAN_SUPP)))

test-unit-c: $(CRITERION_TEST_SRCS) $(BACKEND_RENAME_H) backend | $(BUILD)
	$(TEST_CC) -o $(BUILD)/test_criterion_smoke $(EXTRA_CFLAGS) -include $(BACKEND_RENAME_H) $(TEST_C_INCLUDES) $(CRITERION_TEST_SRCS) -DBACKEND_$(shell echo $(PRIMARY) | tr a-z A-Z) $(CRITERION_CFLAGS) -L$(BUILD) -lidrisml -Wl,-rpath,$(BUILD) $(EXTRA_LDFLAGS) $(CRITERION_LDFLAGS) $(BACKEND_LDFLAGS) -lm
	$(TEST_RUN_ENV) ./$(BUILD)/test_criterion_smoke $(CRITERION_FLAGS) --xml=$(BUILD)/test-criterion-$(PRIMARY).xml

test-unit-c-tape:
	$(MAKE) BACKEND=tape test-unit-c

test-unit-c-mlx:
	$(MAKE) BACKEND=mlx test-unit-c

test-unit-c-torch:
	$(MAKE) BACKEND=torch test-unit-c

# AddressSanitizer + UndefinedBehaviorSanitizer pass over the C test
# suite. Builds the backend + criterion binary together with
# `-fsanitize=address,undefined -fno-omit-frame-pointer -O1 -g` and
# links the same sanitizer runtimes, into a distinct `BUILD_KEY=...
# -asan/` tree (via the ASAN axis in BUILD_KEY) so the warm tree
# stays clean. Per-backend variants below; the aggregate runs only
# the lanes whose backend is available locally (mlx wants macOS,
# torch wants libtorch installed).
#
# First runs may surface latent UB sites — file each finding bucket
# as a follow-up TODO row; fix the cheap ones inline. The CI lane
# starts with `continue-on-error: true` until the first cleanup
# commits land, then promotes to hard-fail.
ASAN_CFLAGS := -fsanitize=address,undefined -fno-omit-frame-pointer -O1 -g
ASAN_LDFLAGS := -fsanitize=address,undefined

test-unit-c-asan-tape:
	$(MAKE) ASAN=1 EXTRA_CFLAGS='$(ASAN_CFLAGS)' EXTRA_LDFLAGS='$(ASAN_LDFLAGS)' BACKEND=tape test-unit-c

test-unit-c-asan-mlx:
	$(MAKE) ASAN=1 EXTRA_CFLAGS='$(ASAN_CFLAGS)' EXTRA_LDFLAGS='$(ASAN_LDFLAGS)' BACKEND=mlx test-unit-c

test-unit-c-asan-torch:
	$(MAKE) ASAN=1 EXTRA_CFLAGS='$(ASAN_CFLAGS)' EXTRA_LDFLAGS='$(ASAN_LDFLAGS)' BACKEND=torch test-unit-c

# Default aggregate: runs only tape (the always-available backend).
# CI matrix entries invoke the per-backend variants directly so each
# runner sees only its own lane.
test-unit-c-asan: test-unit-c-asan-tape

# Coverage build. Recompiles backend + test binary with gcov instrumentation
# (`--coverage` = -fprofile-arcs -ftest-coverage) into build-cov/ (separate
# from build/ so a coverage run doesn't pollute the normal dylib + vice versa).
# Runs the full Criterion suite (which writes .gcda counter files next to the
# build-cov object tree), then reads them with gcovr → a Cobertura XML
# (build-cov/cov-<b>.xml, uploaded to Codecov) + an HTML report.
#
# The stack is gcov-based (gcovr), NOT llvm source-based, so genuinely
# untestable lines can carry an INLINE `// GCOVR_EXCL_LINE — <reason>` marker
# (see gcovr.cfg + docs/develop/coverage-policy.md). Codecov is the CI gate
# (codecov.yml: per-backend flags + patch/project status checks).
#
# COV_BUILD is per-backend (build-cov-tape / -torch / -mlx): gcov keys .gcno to
# the exact compiled object, so a shared tree reused across backends would mix
# one backend's .gcno with another's freshly-written .gcda → "mismatched number
# of counters" corruption. Per-backend trees also let each keep a warm
# instrumented build (like the main BUILD_KEY trees).
COV_BUILD := build-cov-$(PRIMARY)
# Keep -O0 here: tried -O1, but libtorch's template-heavy headers blew up
# compile time on the torch coverage lane (8:58 → 21:58 cold). -O0 stays as the
# cheaper option. The big win is -j$(NPROC) on the recursive make below.
COV_CFLAGS := --coverage -O0 -g
COV_LDFLAGS := --coverage

# File/path exclusions now live in the committed gcovr.cfg at the repo root
# (gcov-native `exclude =` regexes), auto-loaded by gcovr from --root. Per-file
# torch/mlx passthrough exclusions live in codecov.yml `ignore:`. The old
# llvm-cov COV_IGNORE_REGEX is retired.

# gcov coverage reads clang's gcov-format .gcno/.gcda via `llvm-cov gcov`. We
# force the clang toolchain on BOTH platforms so the gcov reader is uniformly
# `llvm-cov gcov` (gcc's `--coverage` would emit a gcc gcov format needing gcc's
# `gcov` reader instead). On macOS the system cc is already Apple clang; on Linux
# the default cc is gcc, so force the (wrapped, glibc-consistent) nix clang via
# the per-backend CC vars (command-line overrides beat backends.mk's := ).
ifeq ($(UNAME),Darwin)
COV_CC_OVERRIDES :=
else
# COV_CLANG/COV_CLANGXX default to bare clang/clang++ but the nix dev shell
# exports them as the *wrapped* clang (absolute store path) so the coverage
# binary is built against nix glibc + the nix dynamic linker. Without that
# it's a system-linker binary that links nix criterion, whose libanl.so.1
# needs nix glibc 2.42 → "GLIBC_ABI_DT_X86_64_PLT not found" at runtime.
COV_CLANG ?= clang
COV_CLANGXX ?= clang++
COV_CC_OVERRIDES := tape_CC=$(COV_CLANG) torch_CC=$(COV_CLANGXX) mlx_CC=$(COV_CLANGXX) LINK_CC=$(COV_CLANGXX) TEST_CC=$(COV_CLANG) SHARED_CC=$(COV_CLANG)
# The wrapped clang injects nix hardening's -D_FORTIFY_SOURCE=2, but coverage
# is -O0 (see COV_CFLAGS note), so glibc features.h fires `#warning
# _FORTIFY_SOURCE requires compiling with optimization` once per TU. Fortify
# just no-ops at -O0; silence the (clang-only) preprocessor warning rather
# than bend the optimization level the torch lane can't afford.
COV_CFLAGS += -Wno-\#warnings
endif

# The gcov reader: Apple's behind xcrun on macOS, the nix `llvm` package's on
# Linux (matches the wrapped nix clang the lane forces there). gcovr is told to
# invoke it via --gcov-executable. gcovr itself + file/path excludes come from
# the dev shell + the committed gcovr.cfg.
ifeq ($(UNAME),Darwin)
GCOV_TOOL := xcrun llvm-cov gcov
else
GCOV_TOOL := llvm-cov gcov
endif
GCOVR := gcovr

test-coverage-backend:
	$(MAKE) -j$(NPROC) BUILD=$(COV_BUILD) \
	  EXTRA_CFLAGS="$(COV_CFLAGS)" \
	  EXTRA_LDFLAGS="$(COV_LDFLAGS)" \
	  BACKEND=$(BACKEND) \
	  $(COV_CC_OVERRIDES) \
	  $(COV_BUILD)/test_criterion_smoke
	@command -v gcovr >/dev/null 2>&1 || { echo "gcovr not found — run inside 'nix develop'"; exit 1; }
	@# Clear stale per-run counters (.gcda); .gcno persist from the compile.
	find $(COV_BUILD) -name '*.gcda' -delete
	@# Drop foreign-backend object/.gcno trees that a prior multi-link coverage
	@# build (BACKEND=tape,mlx,torch) may have left under this primary's tree.
	@# gcovr scans the whole COV_BUILD dir, so a stale .gcno from another backend
	@# counts as uncovered and inflates the denominator. build-cov-<primary> is
	@# single-backend by construction, so any non-primary backend tree is
	@# contamination. Two dir families carry a backend suffix: backend_<b>/ (the
	@# backend's own sources) and shared_<group>_<b>/ (shared TUs compiled once
	@# per backend in TRAINING_ADAPTER_BACKENDS — these have .gcno but no .gcda
	@# under a foreign primary, so they only ever inflate "valid"). Purge both for
	@# every non-primary backend in the built-in set.
	for b in $(filter-out $(PRIMARY),$(MULTI_BACKEND_REQUIRED)); do \
	  find $(COV_BUILD) -type d -name "*_$$b" -exec rm -rf {} +; \
	done
	@# -j1 serializes Criterion's per-test forks. The LLVM gcov writer still
	@# emits noisy "profiling: ... cannot merge" lines for two heavily-forked
	@# TEST files (test_activations, test_dtype_scaffolding) — both excluded from
	@# the report, so product-code counts are unaffected; we filter the spam.
	@# This is a measurement run, not the test gate (that's test-unit-c), so a
	@# failing test doesn't abort it (|| true).
	./$(COV_BUILD)/test_criterion_smoke -j1 --xml=$(COV_BUILD)/test-criterion-$(PRIMARY).xml 2>&1 \
	  | grep -v '^profiling:' || true
	@rm -rf $(COV_BUILD)/html && mkdir -p $(COV_BUILD)/html
	@echo ""
	@echo "=== Coverage report ($(PRIMARY)) ==="
	$(GCOVR) --root $(PWD) --gcov-executable '$(GCOV_TOOL)' \
	  --cobertura $(COV_BUILD)/cov.xml --cobertura-pretty \
	  --html-details $(COV_BUILD)/html/index.html \
	  --txt --print-summary \
	  $(COV_BUILD)
	@echo ""
	@echo "Coverage HTML:      file://$(PWD)/$(COV_BUILD)/html/index.html"
	@echo "Coverage Cobertura: $(COV_BUILD)/cov.xml (uploaded to Codecov in CI)"

# Build-only the criterion suite with coverage flags so the
# test-coverage-backend recipe can run it (writing .gcda) before gcovr reads.
# Matches the test-unit-c build recipe — link the full
# discovered suite, not just the smoke shell.
# TEST_CC is overridable so the coverage path can force clang on Linux
# (EXTRA_CFLAGS carries the --coverage instrumentation flags).
TEST_CC := cc
$(COV_BUILD)/test_criterion_smoke: $(CRITERION_TEST_SRCS) $(BACKEND_RENAME_H) $(LIB) | $(COV_BUILD)
	$(TEST_CC) -o $@ $(EXTRA_CFLAGS) -include $(BACKEND_RENAME_H) $(TEST_C_INCLUDES) $(CRITERION_TEST_SRCS) -DBACKEND_$(shell echo $(PRIMARY) | tr a-z A-Z) $(CRITERION_CFLAGS) -L$(BUILD) -lidrisml -Wl,-rpath,$(PWD)/$(BUILD) $(EXTRA_LDFLAGS) $(CRITERION_LDFLAGS) $(BACKEND_LDFLAGS) -lm

# No `$(COV_BUILD):` dir rule here — the coverage recipe runs a sub-make with
# BUILD=$(COV_BUILD), so backends.mk's `$(BUILD): mkdir -p $(BUILD)` already
# creates build-cov/. A second rule for the same target made `make` warn
# "overriding recipe for target 'build-cov'" in the coverage sub-make.

test-coverage-backend-tape:
	$(MAKE) BACKEND=tape test-coverage-backend

test-coverage-backend-mlx:
	$(MAKE) BACKEND=mlx test-coverage-backend

test-coverage-backend-torch:
	$(MAKE) BACKEND=torch test-coverage-backend

# Single-command overview across all three backends. Coverage is inherently
# per-backend (each backend's sources are only exercised when it is primary;
# the dylib's unified symbols resolve to the primary, so a multi-link
# `BACKEND=tape,mlx,torch test-coverage-backend` parks mlx+torch sources at ~0%
# and reports a meaningless ~55% aggregate). This runs the three per-backend
# builds in turn — each into its own build-cov-<b>/ tree — then tabulates the
# root lines-covered/lines-valid from each Cobertura cov.xml. There is no merged
# denominator on purpose: the three numbers ARE the overview.
# Sequential sub-makes (NOT prerequisites): two concurrent coverage builds under
# -j would race for memory and interleave output. Each writes its own
# build-cov-<b>/ tree, so serializing here is the only ordering needed.
test-coverage-all:
	$(MAKE) test-coverage-backend-tape
	$(MAKE) test-coverage-backend-mlx
	$(MAKE) test-coverage-backend-torch
	@echo ""
	@echo "=== Coverage overview (per-backend; line%) ==="
	@printf '%-8s %10s %10s %8s\n' backend covered valid lines
	@for b in tape mlx torch; do \
	  xml="build-cov-$$b/cov.xml"; \
	  if [ -f "$$xml" ]; then \
	    line=$$(grep -m1 'lines-valid=' "$$xml"); \
	    cov=$$(printf '%s' "$$line" | sed -n 's/.*lines-covered="\([0-9]*\)".*/\1/p'); \
	    val=$$(printf '%s' "$$line" | sed -n 's/.*lines-valid="\([0-9]*\)".*/\1/p'); \
	    pct=$$(awk "BEGIN{ if ($$val>0) printf \"%.1f%%\", 100*$$cov/$$val; else print \"n/a\" }"); \
	    printf '%-8s %10s %10s %8s\n' "$$b" "$$cov" "$$val" "$$pct"; \
	  else \
	    printf '%-8s %10s %10s %8s\n' "$$b" - - "MISSING"; \
	  fi; \
	done

# Static coverage gap probe — no build required. Emits CSV reports of
# OP_* tags + extern "C" symbols vs test-file mentions. Output land in
# $(BUILD)/coverage-gap-{ops,symbols}.csv. Hard gate: nonzero exit on any
# uncovered OP_* or zero-hit FFI symbol (see coverage-policy.md).
test-coverage-gap-probe:
	@python3 scripts/coverage-gap-probe.py $(BUILD)

# Idris reachability gap-finder (ADVISORY v1 — exit 0). Compiles the
# idris-ml test main + every example with `--dumpcases`, producing a
# tree-shaken list of definitions reachable from each entry point. The
# probe then diffs that union against the source universe and reports
# definitions no test or example exercises. Not a coverage % — a gap
# finder. See docs/develop/reachability-policy.md.
#
# The test main builds through its ipkg (`--build`) so the hedgehog /
# idris-test dep chain resolves via the warm install (same path as
# test-unit-idris-ml); examples compile standalone with $(IDRIS_FLAGS).
# `--dumpcases` requires real codegen (`--cg chez -o`); examples that
# don't compile under the active $(BACKEND) are skipped (advisory).
reach-dump: install
	@mkdir -p $(BUILD)/reach
	@echo "--- reach-dump: idris-ml test main ---"
	cd packages/idris-ml && $(IDRIS2) --dumpcases $(CURDIR)/$(BUILD)/reach/test.cases \
		--build idris-ml-tests.ipkg >/dev/null
	@echo "--- reach-dump: examples ---"
	@for f in $(wildcard $(EXAMPLE_SRC)/Example/*.idr); do \
		name=$$(basename $$f .idr); \
		if $(IDRIS2) $(IDRIS_FLAGS) --dumpcases $(BUILD)/reach/$$name.cases \
			--cg chez -o reach-$$name $$f >/dev/null 2>&1; then \
			echo "  ok   $$name"; \
		else \
			echo "  SKIP $$name (did not compile under BACKEND=$(BACKEND))"; \
		fi; \
	done
	@echo "reach-dump: $$(ls $(BUILD)/reach/*.cases 2>/dev/null | wc -l | tr -d ' ') dumps in $(BUILD)/reach/"

test-coverage-reach-gap: reach-dump
	@python3 scripts/reach-gap-probe.py $(BUILD)

# Specialized C test suites. The NTM + mlx-compile tests live under
# packages/idris-test-c/src/ (cross-cutting integration; no 1:1 source
# pair). They're standalone main()s (NOT Criterion) so they get their
# own recipes rather than folding into test-unit-c.
# (test_safetensors.c was converted to Criterion under Test(safetensors, ...)
# and folded into the auto-discovered suite.)
# #402 rank-3 broadcast microbenchmark. Links directly against libtorch
# (no libidrisml / no FFI) to baseline what torch::mul takes on a
# strided rank-3 broadcast — the hot pattern in applyRopeAllHeads.
# Comparing this number against our wrapper's per-op cost (~10-26 ms/op
# observed) and PyTorch Python's (~2 ms/op observed) localises whether
# the gap is in our FFI/wrapper or in libtorch's MPS path.
# Pair with `time_rank3_broadcast.py` for cross-language confirmation.
# Requires BACKEND=torch (so TORCH_INC + TORCH_LIB resolve).
bench-rank3-broadcast: $(BACKENDS_DIR)/bench_rank3_broadcast.cpp | $(BUILD)
	$(torch_CC) $(torch_CFLAGS) -o $(BUILD)/bench_rank3_broadcast \
		$(BACKENDS_DIR)/bench_rank3_broadcast.cpp \
		$(torch_LDFLAGS_$(UNAME))
	@echo "--- bench_rank3_broadcast: device=cpu ---"
	./$(BUILD)/bench_rank3_broadcast cpu
	@echo "--- bench_rank3_broadcast: device=mps ---"
	./$(BUILD)/bench_rank3_broadcast mps

# #402 wrapper-direct rank-3 broadcast microbenchmark. Links libidrisml
# and calls tensor_mul_torch in the same tight loop as
# bench_rank3_broadcast.cpp. The delta between the two numbers is the
# C-side wrapper cost (from_tensor's new + intermediates push + counter
# bump); any further gap up to HfLlama's observed ~10-26 ms/op lives
# above the C boundary (Scheme wrap / Idris autograd / typeclass
# dispatch). Requires the libidrisml dylib at $(LIB), which depends on
# `make backend BACKEND=torch TORCH_DEVICE=mps`.
bench-rank3-broadcast-wrapped: $(BACKENDS_DIR)/bench_rank3_broadcast_wrapped.cpp $(LIB) | $(BUILD)
	$(torch_CC) $(torch_CFLAGS) -I$(BACKENDS_DIR) \
		-o $(BUILD)/bench_rank3_broadcast_wrapped \
		$(BACKENDS_DIR)/bench_rank3_broadcast_wrapped.cpp \
		-L$(BUILD) -lidrisml -Wl,-rpath,$(BUILD) \
		$(torch_LDFLAGS_$(UNAME))
	@echo "--- bench_rank3_broadcast_wrapped: device=cpu ---"
	./$(BUILD)/bench_rank3_broadcast_wrapped cpu
	@echo "--- bench_rank3_broadcast_wrapped: device=mps ---"
	TORCH_DEVICE=mps ./$(BUILD)/bench_rank3_broadcast_wrapped mps

print-torch:
	@echo "LIBTORCH_PATH=$(LIBTORCH_PATH)"
	@echo "TORCH_INC=$(TORCH_INC)"
	@echo "TORCH_LIB=$(TORCH_LIB)"
	@echo "BACKEND=$(BACKEND)"
	@echo "LIB=$(LIB)"

# Type-check examples (builds each as executable, which is the real check)
check-examples: install
	@for f in $(EXAMPLE_SRC)/Example/*.idr; do \
		mod=$$(basename "$$f" .idr); \
		case "$$mod" in \
			Transfer|MlxStreamDemo) \
				echo "Skipping Example.$$mod (cross-backend: names non-linked devices, so it only compiles under a multi-backend BACKEND — checked via its own target)"; \
				continue ;; \
			IndexOps) \
				echo "Skipping Example.$$mod (torch-only: targsort returns I64, no Compatible instance on tape/mlx — checked via example-index-ops)"; \
				continue ;; \
		esac; \
		slug=$$(echo "$$mod" | tr 'A-Z' 'a-z'); \
		echo "Building Example.$$mod..."; \
		$(IDRIS2) $(IDRIS_FLAGS) --build-dir $(BUILD) -o "check-$$slug" "$$f" || exit 1; \
	done
	@echo "All examples type-check."

# Unit test layer — see docs/develop/testing-taxonomy.md.
#
# Canonical aggregator: every unit-layer leaf across the active
# backend. Split by language: `test-unit-idris` covers every
# Idris-side package suite, `test-unit-c` is the Criterion C suite.
# `test-unit-multi-backend` is intentionally NOT here — it forces
# BACKEND=tape,torch,mlx which isn't always feasible on the active
# build set. Wall-clock on a warm tape build: ~3-5 min.
# Run locally pre-commit: `make test` (alias of `test-unit`).
test-unit: test-unit-idris test-unit-c

# All Idris-side unit suites (across packages).
test-unit-idris: test-unit-idris-ml test-unit-gym test-unit-args test-unit-idris-transformers test-unit-examples test-unit-fmt

# Default `test` aggregator — alias for the unit-test layer (the
# fast tier that's safe to run pre-commit). For broader gates use
# `test-integration`, `test-e2e`, or `test-all`.
test: test-unit

# Integration test layer — see docs/develop/testing-taxonomy.md.
#
# Canonical aggregator: every integration-layer leaf (negative type-check
# gates, source-code linters, multi-module integration probes that don't
# run a full training loop). Run locally when you touched a type-level
# guarantee or the FFI wrap convention. Adding a new integration-layer
# test means adding the target name to this list.
test-integration: \
		test-integration-lint-rename-headers \
		test-integration-lint-ci-coverage \
		test-integration-lint-ffi-wrap-template \
		test-integration-lint-non-io-side-effects \
		test-integration-lint-paired-defaults \
		test-integration-lint-example-pairing \
		test-integration-lint-paired-metrics \
		test-integration-lint-init-manifest \
		test-integration-lint-data-manifest \
		test-integration-lint-convergence-expect-coverage \
		test-integration-lint-llama-inference \
		test-integration-lint-bitnet-inference \
		test-integration-lint-hf-finetune \
		test-integration-lint-ci-workflow \
		test-integration-lint-benchmarks \
		test-integration-lint-perf-regression \
		test-integration-typegate-gradmode \
		test-integration-typegate-lossy-cast \
		test-integration-typegate-int-overflow-cast \
		test-integration-typegate-backend-linked \
		test-integration-typegate-linear-model \
		test-integration-typegate-seq-shape \
		test-integration-lint-prim-ratchet \
		test-integration-lint-fmt \
		test-integration-checkpoint-resume \
		test-integration-log-level-profile-gate \
		test-integration-jupyter-cellparser \
		test-integration-py-scripts

# E2E test layer — see docs/develop/testing-taxonomy.md.
#
# Canonical aggregator: every e2e-layer leaf (example smoke matrix
# across backend lanes, HF cross-language roundtrips, oracle gates,
# jupyter notebook execution). Run locally when you touched an example
# or training-loop module. ~15 min on tape; oracle/HF steps download
# HF model weights on first run and need HF_TOKEN for the gated models.
# Adding a new e2e-layer test means adding it to this list; the CI
# workflow picks it up via `make test-e2e`.
#
# test-e2e-cuda is opportunistic (Colab/manual; not in this aggregator).
# test-e2e-notebooks needs jupyter-install (heavy); kept out of the
# default chain — invoke it explicitly when adding new notebooks.
test-e2e: \
		test-e2e-examples \
		test-e2e-bert-roundtrip \
		test-e2e-gpt2-roundtrip \
		test-e2e-bitnet-roundtrip \
		test-e2e-llama-roundtrip \
		test-e2e-llama-generate-roundtrip \
		test-e2e-transformers-oracle-bert \
		test-e2e-transformers-oracle-gpt2 \
		test-e2e-transformers-oracle-llama \
		test-e2e-transformers-oracle-llama-generate \
		test-e2e-rope-oracle \
		test-e2e-pytorch-ref \
		test-e2e-jupyter

# Coverage test layer — see docs/develop/testing-taxonomy.md.
#
# Canonical aggregator: the three-axis target (symbol coverage +
# OP_* backward coverage + F32 paired oracle) per docs/develop/coverage-policy.md.
# Advisory-only — the line-% LLVM reports are report-only; the
# three-axis policy gates contribution. Adding a new coverage probe
# means adding the target to this list.
test-coverage: test-coverage-gap-probe test-coverage-backend

# Idris-side unit suite against the active backend. Buckets that
# touch the C surface (GradMode, ManagedHandle, Tensor lifecycle)
# resolve through `{d=TestExecutor}` which the Makefile-generated
# `TestConfig.idr` pins to the active backend.
#
# Test build goes through `pack` so the hedgehog + Test.Property
# dep chain resolves through the curated pack collection (nix's
# idris2 contrib is stripped of Test.Golden / hedgehog; pack's is
# complete). pack-built artifacts land in packages/idris-ml/build/
# next to the ipkg; the libidrisml.dylib needs to ride alongside
# the executable for the FFI to resolve at runtime.
test-unit-idris-ml: backend $(TESTCONFIG_IDR) $(HWCONFIG_IDR) $(HWDEVICES_IDR) $(MLCONFIG_IDR)
	cd packages/idris-ml && pack --no-prompt build idris-ml-tests.ipkg
	cp $(LIB) packages/idris-ml/build/exec/test_app/
	./packages/idris-ml/build/exec/test

# Multi-backend Idris tests — adds Test.Transfer (cross-backend
# `toExecutor` smoke + roundtrip) to the unit-test list. Forces
# BACKEND=torch,tape,mlx so tape / torch / mlx C symbols are all
# linked into one dylib — Test.Transfer references all three by
# name through `UserExecutorTransfer` instance dispatch and would
# crash at FFI resolution under any single-backend build. Torch
# primary so the F32-hop's tcastUnsafe (a RuntimeDType op routed
# via unified C names) lands on a backend that supports F32.
#
# After the Idris cross-backend test runs, each backend's Criterion
# suite is re-invoked. Each sub-`make` relinks libidrisml.dylib for
# that backend's primary (the same single-backend dylib `make
# BACKEND=<b> install` would produce) — so Criterion validates the
# *single-backend* lane for each one, not the multi-link dylib.
test-unit-multi-backend:
	$(MAKE) BACKEND=torch,tape,mlx install
	$(MAKE) BACKEND=torch,tape,mlx _test-unit-multi-backend-build
	$(MAKE) BACKEND=tape test-unit-c
	$(MAKE) BACKEND=torch test-unit-c
	$(MAKE) BACKEND=mlx test-unit-c

# Sub-target invoked by test-unit-multi-backend under BACKEND=torch,tape,mlx
# so $(LIB) / $(TESTCONFIG_IDR) all resolve in the multi-link
# set's tree, not the outer make's BACKEND context.
_test-unit-multi-backend-build: $(TESTCONFIG_IDR) $(HWCONFIG_IDR) $(HWDEVICES_IDR)
	cd packages/idris-ml && pack --no-prompt build idris-ml-tests-multi.ipkg
	cp $(LIB) packages/idris-ml/build/exec/test-multi_app/
	./packages/idris-ml/build/exec/test-multi

# Idris tests for idris-gym package (pure Idris, no backend required).
# Tests ipkg shares sourcedir with the library ipkg (colocated under
# src/Test/), built via pack so hedgehog (test dep) resolves cleanly.
test-unit-gym:
	cd packages/idris-gym && pack --no-prompt build idris-gym-tests.ipkg
	$(STDBUF) ./packages/idris-gym/build/exec/idris-gym-test

# Idris tests for idris-args package (pure Idris, zero deps beyond
# base; no backend required). Same colocated dual-ipkg pattern.
test-unit-args:
	cd packages/idris-args && pack --no-prompt build idris-args-tests.ipkg
	$(STDBUF) ./packages/idris-args/build/exec/idris-args-test

# Idris tests for idris-transformers package. Pure-Idris suite for
# bertParamNames catalogue + an FFI suite that constructs a real
# HfBert and asserts the C-side param registry matches the catalogue
# exactly. The dylib gets copied alongside the test executable so the
# FFI registry calls land on the active backend's symbols (mirrors
# the test-unit-idris-ml recipe).
# The Test.Tokenizer buckets subprocess into hf_tokenize.py, which
# reads vocab files from models/<repo>/ — declare those fixtures as
# prerequisites so the pattern rule in mk/examples.mk fetches them
# when missing (Make's existence check IS the cache; CI run
# 27373449876's Ubuntu leg failed 0/3 on the missing dirs).
TRANSFORMERS_TEST_FIXTURES := \
	$(HF_MODELS_DIR)/google/bert_uncased_L-2_H-128_A-2/config.json \
	$(HF_MODELS_DIR)/distilgpt2/config.json

test-unit-idris-transformers: backend $(HWCONFIG_IDR) $(HWDEVICES_IDR) $(MLCONFIG_IDR) $(IDRIS_TRANSFORMERS_TESTCONFIG_IDR) $(TRANSFORMERS_TEST_FIXTURES)
	cd packages/idris-transformers && pack --no-prompt install-deps idris-transformers-tests.ipkg
	cd packages/idris-transformers && pack --no-prompt build idris-transformers-tests.ipkg
	cp $(LIB) packages/idris-transformers/build/exec/idris-transformers-test_app/
	$(STDBUF) ./packages/idris-transformers/build/exec/idris-transformers-test

# Microbench for idris-gym hot paths (RNG, Blackjack obs, env step+observe).
# Pure Idris, no backend dependency. Useful for Job 4-style env-side
# perf experiments where single-run RL training is too noisy.
#
# Pass bench names (rng, blackjack, pendulum, acrobot, taxi, cliffwalking)
# to run a subset, e.g. `make bench-gym BENCH_ARGS=rng`. Default runs all.
bench-gym:
	cd packages/idris-gym && pack --no-prompt build idris-gym-bench.ipkg
	$(STDBUF) ./packages/idris-gym/build/exec/idris-gym-bench $(BENCH_ARGS)

# Unit tests for idris-ml-examples (runs moved Test.Generate).
# Tests ipkg shares sourcedir with the library ipkg (colocated under
# src/Test/), built via pack so hedgehog resolves cleanly.
test-unit-examples: backend $(BUILDCONFIG_IDR) $(HWCONFIG_IDR) $(HWDEVICES_IDR) $(MLCONFIG_IDR)
	cd packages/idris-ml-examples && pack --no-prompt build idris-ml-examples-tests.ipkg
	cp $(LIB) packages/idris-ml-examples/build/exec/idris-ml-examples-test_app/
	$(STDBUF) ./packages/idris-ml-examples/build/exec/idris-ml-examples-test
