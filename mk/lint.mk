# mk/lint.mk — repo-hygiene gates. rename-header sync, CI-workflow
# spec sync, FFI wrap template, non-IO side effects, paired defaults,
# lint-py*, lint-c*, include-cleaner, and the negative type-gates.

# Regenerate the per-backend rename headers from backend.h. The
# generated files are checked in; `make test-integration-lint-rename-headers`
# (in CI) gates that they stay in sync with backend.h.
.PHONY: rename-headers test-integration-lint-rename-headers \
        test-integration-lint-ci-workflow \
        test-integration-lint-ci-coverage \
        test-integration-lint-ffi-wrap-template \
        test-integration-lint-non-io-side-effects \
        test-integration-lint-paired-defaults test-integration-lint-example-pairing \
        test-integration-lint-paired-metrics test-integration-lint-init-manifest \
        test-integration-lint-data-manifest test-integration-lint-step-oracle \
        test-integration-py-scripts \
        test-integration-lint-convergence-expect-coverage \
        lint-py lint-py-pytorch \
        lint-py-scripts lint-py-transformers lint-py-examples \
        lint-py-jupyter typecheck-py typecheck-py-pytorch \
        typecheck-py-scripts typecheck-py-transformers \
        typecheck-py-examples typecheck-py-jupyter \
        lint-c lint-c-tape lint-c-tape-linux lint-c-include-cleaner \
        lint-c-torch lint-c-mlx test-integration-typegate-gradmode \
        test-integration-typegate-lossy-cast \
        test-integration-typegate-int-overflow-cast \
        test-integration-typegate-backend-linked \
        test-integration-typegate-linear-model \
        test-integration-typegate-seq-shape \
        test-integration-lint-prim-ratchet \
        test-integration-lint-env-reset-literals

rename-headers:
	@python3 scripts/codegen/gen-rename-headers.py

# Ratchet gate: prim__ usage in examples must not grow (per-file
# baseline inside the script; BringYourOwn.idr exempt — it IS the
# bring-your-own-backend recipe). The example-migration sweep drives
# the baseline to zero.
test-integration-lint-prim-ratchet:
	@python3 scripts/check-prim-in-examples.py

# Gate: examples must reset environments through the Env interface, not by
# writing the state literal. Every Idris deep-RL example pinned its rollout
# auto-reset and every evaluation episode to one fixed state while its
# reference called Gymnasium's randomized `env.reset()` (found 2026-08-01) —
# so a greedy eval replayed one trajectory N times and reported it as an
# N-episode mean, and Pendulum was being evaluated from the downward
# equilibrium against the reference's uniformly random angle. No metric or
# init check can see this; the constructor names can.
test-integration-lint-env-reset-literals:
	@python3 scripts/check-env-reset-literals.py

test-integration-lint-rename-headers:
	@python3 scripts/codegen/gen-rename-headers.py --check

# Gate: regenerate .github/workflows/test.yml from
# .github/workflows/test.yml.spec.json and fail if the on-disk file
# diverges. Catches "someone hand-edited the workflow without updating
# the spec" — the spec is the single source of truth for the
# test-invocation block of the workflow. Adding a new gate: append to
# the spec, run scripts/gen-ci-workflow.py, commit both. See Phase 4
# of the test-rationalization epic.
test-integration-lint-ci-workflow:
	@python3 scripts/gen-ci-workflow.py --check

# Gate: cross-check the CI workflows against the make taxonomy. Two
# invariants: every `make <target>` a workflow invokes must resolve
# (catches removed-target landmines), and every test-unit /
# test-integration / test-e2e / test-coverage aggregator leaf must run
# in some workflow or be a named exception / known hole in the spec
# JSON. See the spec's _coverage_comment.
test-integration-lint-ci-coverage:
	@python3 scripts/check-ci-gate-coverage.py

# Verify every Tensor-touching %foreign declaration matches the
# wrap-on-return Scheme template. See docs/develop/tensor-lifecycle.md
# "FFI conventions". The single source of truth for which C symbols are
# Tensor handles is the scripts/codegen/ffi_manifest/ package — both
# the converter and the linter read from it.
test-integration-lint-ffi-wrap-template:
	@python3 scripts/codegen/check-ffi-wrap-template.py

# Lint: flag %foreign declarations whose Idris type is non-IO but whose
# C body has side effects (allocate, mutate, log, append to tape).
# Catches the bug class fixed by the IO refactor (commits leading up to
# e337512) — see the audit doc + `feedback_typeclass_zero_arg_method_eval.md`
# for the underlying mechanism. Known dead surfaces are skip-listed in
# the script until the dead-code cleanup row lands.
test-integration-lint-non-io-side-effects:
	@python3 scripts/codegen/check-non-io-side-effects.py

# Verify Idris example defaults match the paired torch_ref/scripts/*.py defaults.
# Catches the "I changed Idris's default but forgot the matching ref" drift class.
test-integration-lint-paired-defaults:
	@python3 scripts/check-paired-defaults.py

# Verify every convergence-campaign example is paired with a PyTorch reference
# in scripts/paired_examples.py. The defaults and metrics gates only check the
# pairs they are handed, so an example missing from that table is exempt from
# both — example-double-dqn and example-sac were, from the day each landed
# until 2026-07-31.
test-integration-lint-example-pairing:
	@python3 scripts/check-example-pairing.py

# Verify paired examples report the same RESULT metric keys. A metric one side
# reports and the other does not means no threshold can compare them:
# Example.SeqClassify reported training loss while its reference reported
# held-out accuracy, and the Idris side turned out to evaluate nothing at all.
test-integration-lint-paired-metrics:
	@python3 scripts/check-paired-metrics.py

# Verify paired examples build the same parameter shapes with the same init
# distribution. Runs both sides under IDRISML_DUMP_INIT, which makes each dump
# its freshly-constructed parameters to safetensors and exit before training,
# then diffs shapes exactly and init moments within a band. Four init
# divergences reached main before this existed (dense 2026-07-29; conv,
# recurrent and attention 2026-07-31), each found by a human reading two files.
# Needs the uv venv for safetensors, hence the cd.
test-integration-lint-init-manifest:
	@cd packages/pytorch && uv run --no-sync python ../../scripts/check-init-manifest.py

# Verify paired examples train on the same data distribution. Runs both sides
# under IDRISML_DUMP_DATA, which makes each print one batch's shape and moments
# and exit before training. Example.SeqClassify generated noise-free waveforms
# against its reference's N(0, 0.1), and example-mnist fed raw [0,1] pixels
# against the reference's normalised ones — neither visible to any other gate.
test-integration-lint-data-manifest:
	@cd packages/pytorch && uv run --no-sync python ../../scripts/check-data-manifest.py

# Verify paired examples compute the same step from the same starting point.
# The reference writes the fixture it started from (parameters, plus — for
# RL — its recorded draws in a .replay sidecar) and its own post-step
# parameters; Idris loads the fixture, replays the draws via the example's
# --replay flag, takes one step and dumps. The two post-step parameter sets
# must agree to round-off. The only gate that compares arithmetic rather
# than descriptions — the others all pass while the two sides compute
# different things, which is how dropout-at-inference, a double log-softmax
# and a never-updated actor all shipped here.
test-integration-lint-step-oracle:
	@cd packages/pytorch && uv run --no-sync python ../../scripts/check-step-oracle.py

# Verify every convergence-campaign example has a threshold row in the
# convergence expect file. check-result.sh treats a missing row as
# presence-only (exit 0), so a listed example without one records `pass`
# without asserting anything — example-double-dqn did exactly that from
# 2026-06-19 until 2026-07-29.
test-integration-lint-convergence-expect-coverage:
	@bash scripts/check-convergence-expect-coverage.sh
	@bash scripts/check-convergence-expect-coverage.sh \
		test-refs-convergence.expect CONVERGENCE_REF_MODULES

# Lint the Python surface — split per-package: each Python-bearing
# package owns its own `lint-py-<pkg>` target. The top-level
# `lint-py` aggregator depends on all of them. Adding a new
# Python-bearing package = one new `lint-py-<pkg>` target + add
# it as a dep of `lint-py`. The canonical ruff config lives at
# `ruff.toml` at the repo root; ruff's ancestor-walk discovery
# finds it from every .py file, no per-package config needed.
#
# Each target does file-level discovery within its tree (`ruff
# check <tree>` finds every .py recursively), so there's no path
# enumeration to drift out of date. The aggregator is the only
# place where packages are listed, which is the right granularity.
#
# All targets cd into packages/pytorch first because that's where
# the uv venv with ruff lives (dev dep in pyproject.toml's
# [dependency-groups] dev). `uv run --no-sync` so a stale venv
# doesn't re-fetch; CI primes the venv up front.
#
# `lint-` is the fourth top-level verb in the codebase, alongside
# check (compile) / test (behave) / bench (perf). Linting is
# discrete from testing — it's static analysis, doesn't exercise
# behaviour, doesn't need a backend.

lint-py: lint-py-pytorch lint-py-scripts lint-py-transformers lint-py-examples lint-py-jupyter
	@echo "lint-py OK (all Python-bearing packages)"

# ruff-check + vulture only; `ruff format --check` lives in the fmt
# family (`make check-fmt-py`) so format != lint (see mk/fmt.mk).
lint-py-pytorch:
	@cd packages/pytorch && uv run --no-sync --quiet ruff check . && uv run --no-sync --quiet vulture
	@echo "  lint-py-pytorch OK"

lint-py-scripts:
	@cd packages/pytorch && uv run --no-sync --quiet ruff check ../../scripts
	@echo "  lint-py-scripts OK"

# Unit tests for the repo's Python tooling (scripts/tests/): the FFI
# manifest integrity checks, the coverage-gap probe, and the perf-log
# writers (incl. the kind=compile record). Runs in the lint job's uv dev
# venv (pytest is a dev dep). An integration-layer leaf so the CI-coverage
# gate enforces it actually runs — these were previously orphaned (no
# target invoked them, so test_ffi_manifest_integrity sat red unnoticed).
test-integration-py-scripts:
	@cd packages/pytorch && uv run --no-sync --quiet pytest ../../scripts/tests/ -q
	@echo "  test-integration-py-scripts OK"

lint-py-transformers:
	@cd packages/pytorch && uv run --no-sync --quiet ruff check ../idris-transformers/scripts
	@echo "  lint-py-transformers OK"

lint-py-examples:
	@cd packages/pytorch && uv run --no-sync --quiet ruff check ../idris-ml-examples/scripts
	@echo "  lint-py-examples OK"

lint-py-jupyter:
	@cd packages/pytorch && uv run --no-sync --quiet ruff check ../jupyter
	@echo "  lint-py-jupyter OK"

# Typecheck the Python surface — same per-package split as lint-py.
# One pyright (dev dep of packages/pytorch, version pinned by its
# uv.lock) drives every run; per-surface strictness comes from the
# config root pyright resolves: packages/pytorch/pyproject.toml for
# the pytorch tree (cwd discovery — the mechanism ref-typecheck has
# always used), an explicit `-p <dir>` pyrightconfig.json for the
# others (pyright discovers config from the project root, NOT by
# per-file ancestor walk like ruff, so out-of-tree paths need -p).
#
# typecheck-py-pytorch is an alias of ref-typecheck (the public
# name, in mk/ref.mk) so the typecheck-py-* family is uniform for
# CI/docs.

typecheck-py: typecheck-py-pytorch typecheck-py-scripts typecheck-py-transformers \
              typecheck-py-examples typecheck-py-jupyter
	@echo "typecheck-py OK (all Python-bearing packages)"

typecheck-py-pytorch: ref-typecheck
	@echo "  typecheck-py-pytorch OK"

typecheck-py-scripts:
	@cd packages/pytorch && uv run --no-sync --quiet pyright -p ../../scripts
	@echo "  typecheck-py-scripts OK"

typecheck-py-transformers:
	@cd packages/pytorch && uv run --no-sync --quiet pyright -p ../idris-transformers/scripts
	@echo "  typecheck-py-transformers OK"

typecheck-py-examples:
	@cd packages/pytorch && uv run --no-sync --quiet pyright -p ../idris-ml-examples/scripts
	@echo "  typecheck-py-examples OK"

# Depends on the jupyter venv (ipykernel/jupyter_client/pexpect must
# be importable — packages/jupyter/pyproject.toml's [tool.pyright]
# points venvPath at it) but deliberately NOT on jupyter-install,
# which dep-chains the heavy `backend check`. The prerequisite is the
# literal venv path, not $(JUPYTER_VENV): lint.mk is included before
# jupyter.mk, so the variable is still empty when this prerequisite
# list is read (recipes expand at run time, so $(JUPYTER_PIP) is fine).
typecheck-py-jupyter: packages/jupyter/.venv/bin/activate
	@$(JUPYTER_PIP) install -q -e packages/jupyter/.[dev]
	@cd packages/pytorch && uv run --no-sync --quiet pyright -p ../jupyter
	@echo "  typecheck-py-jupyter OK"

# Lint the C / C++ backend surface: clang-format (layout drift
# against the repo-root `.clang-format` — tabs for indent, K&R
# brace style, 100-col limit) + cppcheck (unused functions +
# bug-class warnings, fast) + clang-tidy (dead-store + bugprone +
# misc-unused, slower because libtorch + mlx headers parse on every
# .cpp). Conservative check sets live in `.clang-tidy` and the
# inline cppcheck flags; widening lands as cleanup commits.
#
# Detects tool availability — if cppcheck / clang-tidy aren't on
# PATH locally (e.g. a checkout without `brew install cppcheck llvm`)
# the target prints a useful hint and exits 0 so it doesn't block
# pre-commit. The CI lane unconditionally apt-installs the tools,
# so the gate fires there.
#
# `lint-c-tape` is the fast subset (tape only, ~5s) for local
# pre-commit use. `lint-c` runs the full sweep including the C++
# backends (slow — libtorch headers).
lint-c: lint-c-tape lint-c-torch lint-c-mlx
	@echo "lint-c OK (cppcheck + clang-tidy across all 3 backends; clang-format is make check-fmt-c)"

# cppcheck + clang-tidy only; clang-format lives in the fmt family
# (`make check-fmt-c`, all backends at once) so format != lint.
# cppcheck covers ALL C: the tape backend, the backends/ root + shared/
# implementation files, and the idris-test-c contract/test suites. (torch/mlx
# .cpp sources cppcheck in their own c++ lanes below.) `unknownMacro:*test_*.c`
# silences Criterion's `Test()` macro in test files.
# clang-tidy covers implementation only — NOT test files (Criterion patterns),
# examples (backend_byo.c), benches (bench_*.c), or RTS glue (refc_shims.c,
# needs gmp). LINT_C_CORE_SRCS adds the backend-agnostic root + shared/
# implementation files to the tape clang-tidy pass (compiled with tape flags;
# -I$(CJSON_DIR) for safetensors.c's cJSON).
LINT_C_CORE_SRCS := packages/backends/idx.c packages/backends/log.c packages/backends/probes.c \
                    packages/backends/shared_utils.c packages/backends/safetensors.c \
                    $(SHARED_TRAINING_SRCS)
# $(CJSON_H) prereq: safetensors.c (in LINT_C_CORE_SRCS) #includes cJSON.h,
# which is vendored on demand. The lint job doesn't build the backend, so the
# header would be absent without this — clang-tidy then hard-errors with
# clang-diagnostic-error 'cJSON.h file not found'. Make fetches it first.
lint-c-tape: $(CJSON_H)
	@if command -v cppcheck >/dev/null 2>&1; then \
		cppcheck --quiet --enable=warning --suppress=missingIncludeSystem --suppress=nullPointerOutOfMemory --suppress=nullPointerArithmeticOutOfMemory --suppress=ctunullpointerOutOfMemory --suppress=ctunullpointer --suppress=nullPointerRedundantCheck --suppress=invalidFunctionArg --suppress=returnImplicitInt --suppress=normalCheckLevelMaxBranches --suppress=syntaxError --suppress=nullPointerOutOfResources --suppress=unknownMacro:*test_*.c --error-exitcode=1 --inline-suppr -I packages/backends -I packages/backends/backend_tape -I packages/idris-test-c/include packages/backends/*.c packages/backends/shared/ packages/backends/backend_tape/ packages/idris-test-c/src/ || exit 1; \
	else \
		echo "lint-c-tape: cppcheck not installed (install via 'brew install cppcheck' or 'apt-get install cppcheck'); skipping"; \
	fi
	@if command -v clang-tidy >/dev/null 2>&1; then \
		clang-tidy --quiet $(BACKEND_TAPE_SRCS) $(LINT_C_CORE_SRCS) -- $(CLANG_TIDY_EXTRA_CFLAGS) $(tape_CFLAGS) -I$(CJSON_DIR) -include $(BACKENDS_DIR)/rename_tape.h || exit 1; \
	else \
		echo "lint-c-tape: clang-tidy not installed (install via 'brew install llvm' or 'apt-get install clang-tidy'); skipping"; \
	fi

# Linux-lane approximation of the CI clang-tidy gate, for pre-sync
# feedback when developing on macOS. CI's preflight lints on Ubuntu, where
# clang-tidy analyzes the #else (non-Apple) branches of the tape
# sources — code a macOS lint-c-tape run never parses. Observed
# 2026-06-11: CI-only analyzer findings in conv2d_batched.c's Linux
# branch shipped through a green local gate. This target re-runs
# clang-tidy with __APPLE__ undefined and a non-Accelerate <cblas.h>
# (nix openblas) so those branches get analyzed locally. Caveat:
# local clang-tidy (brew llvm, v21) vs CI's apt clang-tidy (v18) can
# disagree on individual analyzer findings — this is an
# approximation, not a bit-exact mirror of the CI lane.
LINT_LINUX_CBLAS_INC ?= $(shell nix path-info nixpkgs\#openblas.dev 2>/dev/null)/include
lint-c-tape-linux:
	@if ! command -v clang-tidy >/dev/null 2>&1; then \
		echo "lint-c-tape-linux: clang-tidy not installed; skipping"; \
	elif [ ! -f "$(LINT_LINUX_CBLAS_INC)/cblas.h" ]; then \
		echo "lint-c-tape-linux: no cblas.h found (run 'nix build --no-link nixpkgs#openblas.dev' or set LINT_LINUX_CBLAS_INC); skipping"; \
	else \
		clang-tidy --quiet $(BACKEND_TAPE_SRCS) -- $(CLANG_TIDY_EXTRA_CFLAGS) $(tape_CFLAGS) -U__APPLE__ -isystem $(LINT_LINUX_CBLAS_INC) -include $(BACKENDS_DIR)/rename_tape.h || exit 1; \
		clang-tidy --quiet --checks='-*,misc-include-cleaner' -warnings-as-errors='*' $(BACKEND_TAPE_SRCS) -- $(CLANG_TIDY_EXTRA_CFLAGS) $(tape_CFLAGS) -U__APPLE__ -isystem $(LINT_LINUX_CBLAS_INC) || exit 1; \
	fi

# Rename-free misc-include-cleaner gate. Runs clang-tidy with ONLY
# this check and WITHOUT `-include rename_tape.h`, because the
# rename macros rewrite every backend.h-declared symbol (tensor_*,
# param_*, ...) before include-cleaner sees them, hiding their
# provider attribution and producing ~250 architectural FPs against
# `-include`d source. The rename-free invocation reflects the
# semantic view the developer wrote, which is what include-cleaner
# is designed to check.
lint-c-include-cleaner:
	@if command -v clang-tidy >/dev/null 2>&1; then \
		clang-tidy --quiet --checks='-*,misc-include-cleaner' \
		    -warnings-as-errors='*' \
		    $(BACKEND_TAPE_SRCS) -- $(CLANG_TIDY_EXTRA_CFLAGS) $(tape_CFLAGS) || exit 1; \
	else \
		echo "lint-c-include-cleaner: clang-tidy not installed; skipping"; \
	fi

lint-c-torch:
	@if command -v cppcheck >/dev/null 2>&1; then \
		cppcheck --quiet --enable=warning --suppress=missingIncludeSystem --suppress=nullPointerOutOfMemory --suppress=nullPointerArithmeticOutOfMemory --suppress=ctunullpointerOutOfMemory --suppress=ctunullpointer --suppress=nullPointerRedundantCheck --suppress=invalidFunctionArg --suppress=returnImplicitInt --suppress=normalCheckLevelMaxBranches --suppress=syntaxError --suppress=nullPointerOutOfResources --suppress=unknownMacro:*test_*.c --error-exitcode=1 --inline-suppr --language=c++ -I packages/backends -I packages/backends/backend_torch packages/backends/backend_torch/ || exit 1; \
	else \
		echo "lint-c-torch: cppcheck not installed; skipping"; \
	fi
	@echo "lint-c-torch: clang-tidy on libtorch C++ skipped by default; enable via 'make C_LINT_FULL_CLANG_TIDY=1 lint-c-torch'. Runs on Linux and (under the nix default devShell) macOS — mk/config.mk points -isysroot at nix's apple-sdk (IDRISML_MACOS_SDKROOT) so nix clang-tidy's libc++ stops skewing against the host SDK."
	@# libtorch include roots go in as -isystem (subst -I) so their
	@# diagnostics are SUPPRESSED FROM DISPLAY — there were 430k+ shown
	@# before (run 27879495722), drowning the real findings. This does NOT
	@# shrink clang-tidy's "N warnings generated." tally: that counts check
	@# matches across the whole TU AST (libtorch's headers included) BEFORE
	@# the system-header / header-filter display drop, and clang-tidy has no
	@# flag to skip analyzing included headers — so ~13k/TU is generated-
	@# then-hidden regardless of -isystem. The bash -c wrapper streams each
	@# file's output through `grep -v` to drop that one summary line, prints a
	@# `>> <file>` progress marker first (the per-file count lines used to be
	@# the only heartbeat — without a replacement the ~40-min step looks
	@# hung), and propagates clang-tidy's exit via PIPESTATUS so the enforcing
	@# gate still fails on any real (displayed) finding. Per-file parallel
	@# fan-out (GNU xargs -P) because one serial clang-tidy over ~100 TUs blew
	@# the CI job's 30m timeout. CI runs this on the ubuntu lint-full lane;
	@# it also runs locally on macOS now (nix apple-sdk -isysroot, see above).
	@if [ -n "$$C_LINT_FULL_CLANG_TIDY" ] && command -v clang-tidy >/dev/null 2>&1; then \
		printf '%s\n' $(BACKEND_TORCH_SRCS) | xargs -P "$$(nproc 2>/dev/null || echo 4)" -I{} \
			bash -c 'echo ">> $$1"; clang-tidy --quiet "$$1" -- $(CLANG_TIDY_EXTRA_CFLAGS) $(subst -I,-isystem ,$(torch_CFLAGS)) -include $(BACKENDS_DIR)/rename_torch.h 2>&1 | grep -v "warnings generated\.$$"; exit $${PIPESTATUS[0]}' _ {} || exit 1; \
	fi

lint-c-mlx:
	@if command -v cppcheck >/dev/null 2>&1; then \
		cppcheck --quiet --enable=warning --suppress=missingIncludeSystem --suppress=nullPointerOutOfMemory --suppress=nullPointerArithmeticOutOfMemory --suppress=ctunullpointerOutOfMemory --suppress=ctunullpointer --suppress=nullPointerRedundantCheck --suppress=invalidFunctionArg --suppress=returnImplicitInt --suppress=normalCheckLevelMaxBranches --suppress=syntaxError --suppress=nullPointerOutOfResources --suppress=unknownMacro:*test_*.c --error-exitcode=1 --inline-suppr --language=c++ -I packages/backends -I packages/backends/backend_mlx packages/backends/backend_mlx/ || exit 1; \
	else \
		echo "lint-c-mlx: cppcheck not installed; skipping"; \
	fi
	@echo "lint-c-mlx: clang-tidy on mlx C++ skipped by default; enable via 'make C_LINT_FULL_CLANG_TIDY=1 lint-c-mlx'. mlx is macOS-only, so this runs on macOS under the nix default devShell (nix apple-sdk -isysroot, see mk/config.mk) — nix clang-tidy now parses against nix's own SDK instead of the host's."
	@# bash -c wrapper streams clang-tidy through `grep -v` to drop the noisy
	@# "N warnings generated." per-TU summary (counted pre-display-suppression
	@# over mlx's headers), prints a `>> <file>` progress marker, and
	@# propagates the exit via PIPESTATUS; see lint-c-torch for the rationale.
	@if [ -n "$$C_LINT_FULL_CLANG_TIDY" ] && command -v clang-tidy >/dev/null 2>&1; then \
		printf '%s\n' $(BACKEND_MLX_SRCS) | xargs -P "$$(nproc 2>/dev/null || echo 4)" -I{} \
			bash -c 'echo ">> $$1"; clang-tidy --quiet "$$1" -- $(CLANG_TIDY_EXTRA_CFLAGS) $(subst -I,-isystem ,$(mlx_CFLAGS)) -include $(BACKENDS_DIR)/rename_mlx.h 2>&1 | grep -v "warnings generated\.$$"; exit $${PIPESTATUS[0]}' _ {} || exit 1; \
	fi

# Verify the GradMode gate is intact: a NoGrad loss must NOT type-check
# as input to nativeTrainStep. Inverts the idris2 exit code (success =
# compile failed) and matches on the WithGrad/NoGrad error message.
# Depends on `install` so idris-ml is locatable in the local IDRIS2 prefix.
test-integration-typegate-gradmode: install
	@IDRIS2_LOCAL=$(IDRIS2_LOCAL) ./scripts/check-gradmode-gate.sh

# Verify the cross-family lossless-cast gate (DType.Core.LosslessTo)
# refuses a mantissa-shrinking direction (F32 → BF16). Inverts the
# idris2 exit code (success = compile failed) and matches on the
# `LTE 23 7` error so unrelated regressions don't pass the gate.
test-integration-typegate-lossy-cast: install
	@IDRIS2_LOCAL=$(IDRIS2_LOCAL) ./scripts/check-lossy-cast-gate.sh

# Verify the int-overflow lossless-cast gate (F1 of #410, #412)
# refuses I64 → F32 (max int value far exceeds F32 mantissa). Inverts
# the idris2 exit code (success = compile failed) and matches on the
# `LTE 64 25` error so unrelated regressions don't pass the gate.
test-integration-typegate-int-overflow-cast: install
	@chmod +x ./scripts/check-int-overflow-cast-gate.sh
	@IDRIS2_LOCAL=$(IDRIS2_LOCAL) ./scripts/check-int-overflow-cast-gate.sh

# Verify the Backend-bundle availability gate: `Backend ex dt` must not
# resolve for an executor lacking `Linked` (the per-build availability
# gate generated into HwConfig.idr). Inverts the idris2 exit code and
# asserts the search failure names the Linked leaf.
test-integration-typegate-backend-linked: install
	@chmod +x ./scripts/check-backend-bundle-gate.sh
	@IDRIS2_LOCAL=$(IDRIS2_LOCAL) ./scripts/check-backend-bundle-gate.sh

# Verify the linear-model gate: reusing a model handle after it has been
# consumed by `evalL`/`freezeL` must be a compile-time linearity error
# (the typed defence against "freeze/eval a model, then reuse the stale
# handle to train" silent no-ops). Checks the negative test fails with a
# linearity error AND the positive single-use test still compiles.
test-integration-typegate-linear-model: install
	@chmod +x ./scripts/check-linear-model-gate.sh
	@IDRIS2_LOCAL=$(IDRIS2_LOCAL) ./scripts/check-linear-model-gate.sh

# Verify the Seq shape gate: a chain whose hidden dims don't line up must
# fail to compile with an error NAMING BOTH DIMS (the ChainFits witness:
# `Can't find an implementation for ChainFits 256 128`), not the opaque
# `Module ?l` search failure. Also compiles the well-sized positive twin.
test-integration-typegate-seq-shape: install
	@chmod +x ./scripts/check-seq-shape-gate.sh
	@IDRIS2_LOCAL=$(IDRIS2_LOCAL) ./scripts/check-seq-shape-gate.sh
