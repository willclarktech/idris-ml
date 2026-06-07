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
        test-coverage-backend-torch test-coverage-gap-probe \
        bench-rank3-broadcast bench-rank3-broadcast-wrapped print-torch \
        check-examples test-unit test-unit-idris test test-integration \
        test-e2e test-coverage test-unit-idris-ml \
        test-unit-multi-backend test-unit-gym test-unit-args \
        test-unit-idris-transformers bench-gym test-unit-examples

# Criterion prefix autodetection: nix profile (local dev), then brew
# (macOS CI), then /usr (Ubuntu CI's libcriterion-dev). Explicit
# CRITERION_PREFIX= still overrides. Falls back to the nix profile when
# nothing is found so the error stays "criterion.h not found" rather
# than a cryptic -I/include.
CRITERION_DETECT := $(firstword $(wildcard $(HOME)/.nix-profile/include/criterion /opt/homebrew/include/criterion /usr/include/criterion))
ifeq ($(CRITERION_DETECT),)
CRITERION_PREFIX ?= $(HOME)/.nix-profile
else
CRITERION_PREFIX ?= $(patsubst %/include/criterion,%,$(CRITERION_DETECT))
endif
CRITERION_CFLAGS := -I$(CRITERION_PREFIX)/include
CRITERION_LDFLAGS := -L$(CRITERION_PREFIX)/lib -lcriterion -Wl,-rpath,$(CRITERION_PREFIX)/lib
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

# Discover Criterion suites. Three locations:
#  - packages/backends/test/common/  — backend-agnostic per-op tests
#    colocated next to their source (forward + backward correctness via
#    the public backend.h FFI; runs against any backend's dylib).
#  - packages/backends/test/<primary>/  — backend-specific tests that
#    touch internals (e.g. tape's OP_* dispatch table) or assert
#    port-struct slot populations specific to that backend's adapter.
#  - packages/idris-test-c/src/  — cross-cutting test infra package
#    (framework smoke, NTM integration tests, mlx-compile, training-loop
#    oracle ladder, param registry, clip-grad-norm, optimizers).
TEST_C_DIR := packages/idris-test-c
# Discover Criterion suites. Tests are colocated alongside source under
# backend_{tape,torch,mlx}/<subsystem>/test_<topic>.c (one test file per
# kernel pair lives next to the tape source — it tests the public
# `tensor_<op>` FFI so it covers all backends regardless of physical
# location). Backend-specific tests gate themselves with `#ifdef
# BACKEND_<NAME>`. Cross-cutting infra (integration tests, oracle
# ladder, framework smoke) lives in $(TEST_C_DIR)/src/. The temporary
# $(BACKENDS_DIR)/test/{common,tape,mlx}/ tree is fading out as Phase 3b
# moves complete.
CRITERION_BACKEND_TEST_SRCS := $(shell find $(BACKENDS_DIR)/backend_tape -name 'test_*.c' 2>/dev/null) \
                               $(shell find $(BACKENDS_DIR)/backend_torch -name 'test_*.c' 2>/dev/null) \
                               $(shell find $(BACKENDS_DIR)/backend_mlx -name 'test_*.c' 2>/dev/null) \
                               $(shell find $(BACKENDS_DIR)/test/common -name '*.c' 2>/dev/null) \
                               $(shell find $(BACKENDS_DIR)/test/$(PRIMARY) -name '*.c' 2>/dev/null) \
                               $(shell find $(TEST_C_DIR)/src -name '*.c' -not -name 'test_criterion_smoke.c' 2>/dev/null)
CRITERION_TEST_SRCS := $(TEST_C_DIR)/src/test_criterion_smoke.c $(CRITERION_BACKEND_TEST_SRCS)
TEST_C_INCLUDES := -I$(BACKENDS_DIR) -I$(TEST_C_DIR)/include

test-unit-c: $(CRITERION_TEST_SRCS) $(BACKEND_RENAME_H) backend | $(BUILD)
	$(TEST_CC) -o $(BUILD)/test_criterion_smoke $(EXTRA_CFLAGS) -include $(BACKEND_RENAME_H) $(TEST_C_INCLUDES) $(CRITERION_TEST_SRCS) -DBACKEND_$(shell echo $(PRIMARY) | tr a-z A-Z) $(CRITERION_CFLAGS) -L$(BUILD) -lidrisml -Wl,-rpath,$(BUILD) $(EXTRA_LDFLAGS) $(CRITERION_LDFLAGS) -lm
	./$(BUILD)/test_criterion_smoke $(CRITERION_FLAGS) --xml=$(BUILD)/test-criterion-$(PRIMARY).xml

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

# Coverage build. Recompiles backend + test binary with
# `-fprofile-instr-generate -fcoverage-mapping` into build-cov/ (separate
# from build/ so a coverage run doesn't pollute the normal dylib + vice
# versa). Runs the full Criterion suite with LLVM_PROFILE_FILE pointing
# to build-cov/profraw/, merges via llvm-profdata, then emits
# `llvm-cov report` + HTML via `llvm-cov show -format=html`.
#
# Report-only — no CI gate yet. The HTML artifact lives at
# build-cov/html-<b>/index.html.
COV_BUILD := build-cov
# Keep -O0 here: tried -O1 (which is compatible with -fcoverage-mapping),
# but libtorch's template-heavy headers blew up compile time on the torch
# coverage lane (8:58 → 21:58 cold). -O0 stays as the cheaper option.
# The big win is -j$(NPROC) on the recursive make below.
COV_CFLAGS := -fprofile-instr-generate -fcoverage-mapping -O0 -g
COV_LDFLAGS := -fprofile-instr-generate

# llvm source-based coverage is clang-only. On macOS the system cc IS
# clang and the llvm tools live behind xcrun. On Linux the default cc
# is gcc, which rejects -fprofile-instr-generate/-fcoverage-mapping
# (first exercised by CI on 2026-06-11 when the coverage matrix moved
# to push — the target was previously dispatch-only and had never run
# on a gcc host), so force the clang toolchain via the per-backend CC
# vars (command-line overrides beat the := assignments in backends.mk)
# and call the llvm tools directly (apt's `llvm` package ships
# unversioned llvm-profdata/llvm-cov).
ifeq ($(UNAME),Darwin)
COV_CC_OVERRIDES :=
COV_LLVM := xcrun
else
COV_CC_OVERRIDES := tape_CC=clang torch_CC=clang++ mlx_CC=clang++ LINK_CC=clang++ TEST_CC=clang SHARED_CC=clang
COV_LLVM :=
endif

test-coverage-backend:
	$(MAKE) -j$(NPROC) BUILD=$(COV_BUILD) \
	  EXTRA_CFLAGS="$(COV_CFLAGS)" \
	  EXTRA_LDFLAGS="$(COV_LDFLAGS)" \
	  BACKEND=$(BACKEND) \
	  $(COV_CC_OVERRIDES) \
	  $(COV_BUILD)/test_criterion_smoke
	@mkdir -p $(COV_BUILD)/profraw
	@rm -f $(COV_BUILD)/profraw/*.profraw
	LLVM_PROFILE_FILE='$(COV_BUILD)/profraw/test_criterion_%p_%m.profraw' \
	  ./$(COV_BUILD)/test_criterion_smoke --xml=$(COV_BUILD)/test-criterion-$(PRIMARY).xml > /dev/null
	$(COV_LLVM) llvm-profdata merge -sparse $(COV_BUILD)/profraw/*.profraw -o $(COV_BUILD)/$(PRIMARY).profdata
	@echo ""
	@echo "=== Coverage report ($(PRIMARY)) ==="
	$(COV_LLVM) llvm-cov report $(COV_BUILD)/libidrisml.$(LIB_EXT) -instr-profile=$(COV_BUILD)/$(PRIMARY).profdata -ignore-filename-regex='($(BACKENDS_DIR)/(cJSON|safetensors|shared_utils|mnist))|(/(usr|nix|opt|Library|System|\.venv)/)|(\.cache/)'
	@rm -rf $(COV_BUILD)/html-$(PRIMARY)
	$(COV_LLVM) llvm-cov show $(COV_BUILD)/libidrisml.$(LIB_EXT) -instr-profile=$(COV_BUILD)/$(PRIMARY).profdata -format=html -output-dir=$(COV_BUILD)/html-$(PRIMARY) -ignore-filename-regex='($(BACKENDS_DIR)/(cJSON|safetensors|shared_utils|mnist))|(/(usr|nix|opt|Library|System|\.venv)/)|(\.cache/)'
	@echo ""
	@echo "Coverage HTML: file://$(PWD)/$(COV_BUILD)/html-$(PRIMARY)/index.html"

# Build-only the criterion suite with coverage flags so the
# test-coverage-backend recipe can set LLVM_PROFILE_FILE before running.
# Matches the test-unit-c build recipe — link the full
# discovered suite, not just the smoke shell.
# TEST_CC is overridable so the coverage path can force clang on Linux
# (EXTRA_CFLAGS carries clang-only instrumentation flags there).
TEST_CC := cc
$(COV_BUILD)/test_criterion_smoke: $(CRITERION_TEST_SRCS) $(BACKEND_RENAME_H) $(LIB) | $(COV_BUILD)
	$(TEST_CC) -o $@ $(EXTRA_CFLAGS) -include $(BACKEND_RENAME_H) $(TEST_C_INCLUDES) $(CRITERION_TEST_SRCS) -DBACKEND_$(shell echo $(PRIMARY) | tr a-z A-Z) $(CRITERION_CFLAGS) -L$(BUILD) -lidrisml -Wl,-rpath,$(PWD)/$(BUILD) $(EXTRA_LDFLAGS) $(CRITERION_LDFLAGS) -lm

$(COV_BUILD):
	mkdir -p $@

test-coverage-backend-tape:
	$(MAKE) BACKEND=tape test-coverage-backend

test-coverage-backend-mlx:
	$(MAKE) BACKEND=mlx test-coverage-backend

test-coverage-backend-torch:
	$(MAKE) BACKEND=torch test-coverage-backend

# Static coverage gap probe — no build required. Emits CSV reports of
# OP_* tags + extern "C" symbols vs test-file mentions. Output land in
# $(BUILD)/coverage-gap-{ops,symbols}.csv. Advisory exit; gating flip
# tracked under W3+W4 in coverage-policy.md.
test-coverage-gap-probe:
	@python3 scripts/coverage-gap-probe.py $(BUILD)

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
		idris2 $(IDRIS_FLAGS) --build-dir $(BUILD) -o "check-$$slug" "$$f" || exit 1; \
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
test-unit-idris: test-unit-idris-ml test-unit-gym test-unit-args test-unit-idris-transformers test-unit-examples

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
		test-integration-lint-hf-llama-inference \
		test-integration-lint-ci-workflow \
		test-integration-lint-benchmarks \
		test-integration-lint-perf-regression \
		test-integration-typegate-gradmode \
		test-integration-typegate-gradmode-aliasing \
		test-integration-typegate-lossy-cast \
		test-integration-typegate-int-overflow-cast \
		test-integration-typegate-backend-linked \
		test-integration-lint-prim-ratchet \
		test-integration-checkpoint-resume \
		test-integration-jupyter-cellparser

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
		test-e2e-hf-bert-roundtrip \
		test-e2e-hf-gpt2-roundtrip \
		test-e2e-hf-bitnet-roundtrip \
		test-e2e-hf-llama-roundtrip \
		test-e2e-hf-llama-generate-roundtrip \
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
test-unit-idris-ml: backend $(TESTCONFIG_IDR) $(HWCONFIG_IDR) $(HWDEVICES_IDR)
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

test-unit-idris-transformers: backend $(HWCONFIG_IDR) $(HWDEVICES_IDR) $(IDRIS_TRANSFORMERS_TESTCONFIG_IDR) $(TRANSFORMERS_TEST_FIXTURES)
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
test-unit-examples: backend $(BUILDCONFIG_IDR) $(HWCONFIG_IDR) $(HWDEVICES_IDR)
	cd packages/idris-ml-examples && pack --no-prompt build idris-ml-examples-tests.ipkg
	cp $(LIB) packages/idris-ml-examples/build/exec/idris-ml-examples-test_app/
	$(STDBUF) ./packages/idris-ml-examples/build/exec/idris-ml-examples-test
