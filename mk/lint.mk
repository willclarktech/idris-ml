# mk/lint.mk — repo-hygiene gates. rename-header sync, CI-workflow
# spec sync, FFI wrap template, non-IO side effects, paired defaults,
# lint-py*, lint-c*, include-cleaner, and the negative type-gates.

# Regenerate the per-backend rename headers from backend.h. The
# generated files are checked in; `make test-integration-lint-rename-headers`
# (in CI) gates that they stay in sync with backend.h.
.PHONY: rename-headers test-integration-lint-rename-headers \
        test-integration-lint-ci-workflow \
        test-integration-lint-ffi-wrap-template \
        test-integration-lint-non-io-side-effects \
        test-integration-lint-paired-defaults lint-py lint-py-pytorch \
        lint-py-scripts lint-py-transformers lint-py-examples \
        lint-py-jupyter typecheck-py typecheck-py-pytorch \
        typecheck-py-scripts typecheck-py-transformers \
        typecheck-py-examples typecheck-py-jupyter \
        lint-c lint-c-tape lint-c-include-cleaner \
        lint-c-torch lint-c-mlx test-integration-typegate-gradmode \
        test-integration-typegate-gradmode-aliasing \
        test-integration-typegate-lossy-cast \
        test-integration-typegate-int-overflow-cast

rename-headers:
	@python3 scripts/codegen/gen-rename-headers.py

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

# Verify every Tensor-touching %foreign declaration matches the
# wrap-on-return Scheme template. See
# docs/develop/tensor-lifecycle-plan.md "FFI conventions". The single
# source of truth for which C symbols are Tensor handles is
# scripts/codegen/ffi_manifest.py — both the converter and the linter
# read from it.
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

lint-py-pytorch:
	@cd packages/pytorch && uv run --no-sync --quiet ruff check . && uv run --no-sync --quiet ruff format --check . && uv run --no-sync --quiet vulture
	@echo "  lint-py-pytorch OK"

lint-py-scripts:
	@cd packages/pytorch && uv run --no-sync --quiet ruff check ../../scripts && uv run --no-sync --quiet ruff format --check ../../scripts
	@echo "  lint-py-scripts OK"

lint-py-transformers:
	@cd packages/pytorch && uv run --no-sync --quiet ruff check ../idris-transformers/scripts && uv run --no-sync --quiet ruff format --check ../idris-transformers/scripts
	@echo "  lint-py-transformers OK"

lint-py-examples:
	@cd packages/pytorch && uv run --no-sync --quiet ruff check ../idris-ml-examples/scripts && uv run --no-sync --quiet ruff format --check ../idris-ml-examples/scripts
	@echo "  lint-py-examples OK"

lint-py-jupyter:
	@cd packages/pytorch && uv run --no-sync --quiet ruff check ../jupyter && uv run --no-sync --quiet ruff format --check ../jupyter
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
# PATH locally (e.g. a dev box without `brew install cppcheck llvm`)
# the target prints a useful hint and exits 0 so it doesn't block
# pre-commit. The CI lane unconditionally apt-installs the tools,
# so the gate fires there.
#
# `lint-c-tape` is the fast subset (tape only, ~5s) for local
# pre-commit use. `lint-c` runs the full sweep including the C++
# backends (slow — libtorch headers).
lint-c: lint-c-tape lint-c-torch lint-c-mlx
	@echo "lint-c OK (clang-format + cppcheck + clang-tidy across all 3 backends)"

# Shared C/C++ surface — linked into every backend, so re-check
# from each lint-c-<backend> target. rename_*.h is auto-generated
# (gen-rename-headers.py owns its layout) and excluded.
BACKENDS_SHARED_SRCS := $(shell find packages/backends -maxdepth 1 \( -name "*.c" -o -name "*.h" -o -name "*.cpp" -o -name "*.hpp" \) ! -name "rename_*.h" 2>/dev/null) \
                       $(shell find packages/backends/shared \( -name "*.c" -o -name "*.h" -o -name "*.cpp" -o -name "*.hpp" \) 2>/dev/null)

lint-c-tape:
	@if command -v clang-format >/dev/null 2>&1; then \
		clang-format --dry-run -Werror --style=file $$(find packages/backends/backend_tape \( -name "*.c" -o -name "*.h" \) 2>/dev/null) $(BACKENDS_SHARED_SRCS) || exit 1; \
	else \
		echo "lint-c-tape: clang-format not installed (install via 'brew install clang-format' or 'apt-get install clang-format'; macOS Command Line Tools also ships one at /Library/Developer/CommandLineTools/usr/bin/clang-format if that's on PATH); skipping"; \
	fi
	@if command -v cppcheck >/dev/null 2>&1; then \
		cppcheck --quiet --enable=warning --suppress=missingIncludeSystem --suppress=nullPointerOutOfMemory --suppress=nullPointerArithmeticOutOfMemory --suppress=ctunullpointerOutOfMemory --suppress=ctunullpointer --suppress=nullPointerRedundantCheck --suppress=invalidFunctionArg --suppress=returnImplicitInt --suppress=normalCheckLevelMaxBranches --suppress=syntaxError --error-exitcode=1 --inline-suppr -I packages/backends -I packages/backends/backend_tape packages/backends/backend_tape/ || exit 1; \
	else \
		echo "lint-c-tape: cppcheck not installed (install via 'brew install cppcheck' or 'apt-get install cppcheck'); skipping"; \
	fi
	@if command -v clang-tidy >/dev/null 2>&1; then \
		clang-tidy --quiet $(BACKEND_TAPE_SRCS) -- $(CLANG_TIDY_EXTRA_CFLAGS) $(tape_CFLAGS) -include $(BACKENDS_DIR)/rename_tape.h || exit 1; \
	else \
		echo "lint-c-tape: clang-tidy not installed (install via 'brew install llvm' or 'apt-get install clang-tidy'); skipping"; \
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
	@if command -v clang-format >/dev/null 2>&1; then \
		clang-format --dry-run -Werror --style=file $$(find packages/backends/backend_torch \( -name "*.c" -o -name "*.h" -o -name "*.cpp" -o -name "*.hpp" \) 2>/dev/null) $(BACKENDS_SHARED_SRCS) || exit 1; \
	else \
		echo "lint-c-torch: clang-format not installed; skipping"; \
	fi
	@if command -v cppcheck >/dev/null 2>&1; then \
		cppcheck --quiet --enable=warning --suppress=missingIncludeSystem --suppress=nullPointerOutOfMemory --suppress=nullPointerArithmeticOutOfMemory --suppress=ctunullpointerOutOfMemory --suppress=ctunullpointer --suppress=nullPointerRedundantCheck --suppress=invalidFunctionArg --suppress=returnImplicitInt --suppress=normalCheckLevelMaxBranches --suppress=syntaxError --error-exitcode=1 --inline-suppr --language=c++ -I packages/backends -I packages/backends/backend_torch packages/backends/backend_torch/ || exit 1; \
	else \
		echo "lint-c-torch: cppcheck not installed; skipping"; \
	fi
	@echo "lint-c-torch: clang-tidy on libtorch C++ skipped by default; enable via 'make C_LINT_FULL_CLANG_TIDY=1 lint-c-torch' (Linux). macOS+nix is blocked: Apple SDK <sys/resource.h> uint8_t/uint64_t fail to resolve against nix clang-tidy's libc++ pre-include chain — not fixable cheaply."
	@if [ -n "$$C_LINT_FULL_CLANG_TIDY" ] && command -v clang-tidy >/dev/null 2>&1; then \
		clang-tidy --quiet $(BACKEND_TORCH_SRCS) -- $(CLANG_TIDY_EXTRA_CFLAGS) $(torch_CFLAGS) -include $(BACKENDS_DIR)/rename_torch.h || exit 1; \
	fi

lint-c-mlx:
	@if command -v clang-format >/dev/null 2>&1; then \
		clang-format --dry-run -Werror --style=file $$(find packages/backends/backend_mlx \( -name "*.c" -o -name "*.h" -o -name "*.cpp" -o -name "*.hpp" \) 2>/dev/null) $(BACKENDS_SHARED_SRCS) || exit 1; \
	else \
		echo "lint-c-mlx: clang-format not installed; skipping"; \
	fi
	@if command -v cppcheck >/dev/null 2>&1; then \
		cppcheck --quiet --enable=warning --suppress=missingIncludeSystem --suppress=nullPointerOutOfMemory --suppress=nullPointerArithmeticOutOfMemory --suppress=ctunullpointerOutOfMemory --suppress=ctunullpointer --suppress=nullPointerRedundantCheck --suppress=invalidFunctionArg --suppress=returnImplicitInt --suppress=normalCheckLevelMaxBranches --suppress=syntaxError --error-exitcode=1 --inline-suppr --language=c++ -I packages/backends -I packages/backends/backend_mlx packages/backends/backend_mlx/ || exit 1; \
	else \
		echo "lint-c-mlx: cppcheck not installed; skipping"; \
	fi
	@echo "lint-c-mlx: clang-tidy on mlx C++ skipped by default; enable via 'make C_LINT_FULL_CLANG_TIDY=1 lint-c-mlx' (Linux). Same macOS+nix block as torch — Apple SDK headers reject nix clang-tidy."
	@if [ -n "$$C_LINT_FULL_CLANG_TIDY" ] && command -v clang-tidy >/dev/null 2>&1; then \
		clang-tidy --quiet $(BACKEND_MLX_SRCS) -- $(CLANG_TIDY_EXTRA_CFLAGS) $(mlx_CFLAGS) -include $(BACKENDS_DIR)/rename_mlx.h || exit 1; \
	fi

# Verify the GradMode gate is intact: a NoGrad loss must NOT type-check
# as input to nativeTrainStep. Inverts the idris2 exit code (success =
# compile failed) and matches on the WithGrad/NoGrad error message.
# Depends on `install` so idris-ml is locatable in the local IDRIS2 prefix.
test-integration-typegate-gradmode: install
	@IDRIS2_LOCAL=$(IDRIS2_LOCAL) ./scripts/check-gradmode-gate.sh

# Verify the aliasing footgun on `freezeNetwork` is closed by linear
# types: using the pre-freeze Network reference must be a compile error.
test-integration-typegate-gradmode-aliasing: install
	@IDRIS2_LOCAL=$(IDRIS2_LOCAL) ./scripts/check-gradmode-aliasing.sh

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
