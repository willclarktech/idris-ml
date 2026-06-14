# mk/fmt.mk — the cross-language source-formatter family.
#
# `fmt` rewrites every source file in place (Idris + Python + C/C++);
# `check-fmt` fails if anything is unformatted. Each is an umbrella over
# per-language subtargets. Formatting is kept DISTINCT from linting (per
# docs/develop/testing-taxonomy.md "format != lint"): `lint-py`/`lint-c`
# do ruff-check+vulture / cppcheck+clang-tidy only — the format CHECK
# lives here.
#
#   writers:  fmt        = fmt-idris + fmt-py + fmt-c
#   checkers: check-fmt  = test-integration-lint-fmt (idris)
#                          + check-fmt-py + check-fmt-c

.PHONY: fmt fmt-idris fmt-py fmt-c \
        check-fmt test-integration-lint-fmt check-fmt-py check-fmt-c \
        test-unit-fmt idris-fmt-build

# ---- Idris: idris-fmt, the repo's compiler-native formatter ----------

IDRIS_FMT     := ./packages/idris-fmt/build/exec/idris-fmt
# Only tracked .idr — git ls-files excludes generated sources
# (ML/Config.idr, HwConfig.idr, … are .gitignored codegen output) and
# anything under build/. The formatter never touches generated files.
FMT_IDR_FILES := $(shell git ls-files '*.idr')

# Build the formatter executable (cheap once warm).
idris-fmt-build:
	cd packages/idris-fmt && pack --no-prompt build idris-fmt.ipkg

fmt-idris: idris-fmt-build
	@$(IDRIS_FMT) --write $(FMT_IDR_FILES)

# CI gate (test-integration leaf, runs in the test-integration job which
# carries idris2/pack). Non-zero exit if any .idr file is unformatted.
test-integration-lint-fmt: idris-fmt-build
	@$(IDRIS_FMT) --check $(FMT_IDR_FILES) && echo "idris-fmt: all .idr formatted"

# ---- Python: ruff format ---------------------------------------------

# Dirs ruff covers, relative to packages/pytorch (mirrors lint-py-*).
PY_FMT_DIRS := . ../../scripts ../idris-transformers/scripts ../idris-ml-examples/scripts ../jupyter

fmt-py:
	@cd packages/pytorch && uv run --no-sync --quiet ruff format $(PY_FMT_DIRS)
	@echo "ruff format: Python formatted"

check-fmt-py:
	@cd packages/pytorch && uv run --no-sync --quiet ruff format --check $(PY_FMT_DIRS)
	@echo "ruff format --check: all Python formatted"

# ---- C / C++: clang-format -------------------------------------------

# Every backend C/C++ source (tape + torch + mlx + shared). rename_*.h
# is generated (gen-rename-headers.py owns its layout) and excluded.
C_FMT_SRCS := $(shell find packages/backends \( -name "*.c" -o -name "*.h" -o -name "*.cpp" -o -name "*.hpp" \) ! -name "rename_*.h" 2>/dev/null)

fmt-c:
	@if command -v clang-format >/dev/null 2>&1; then \
		clang-format -i --style=file $(C_FMT_SRCS); \
		echo "clang-format: C/C++ formatted"; \
	else \
		echo "fmt-c: clang-format not installed; skipping"; \
	fi

check-fmt-c:
	@if command -v clang-format >/dev/null 2>&1; then \
		clang-format --dry-run -Werror --style=file $(C_FMT_SRCS) || exit 1; \
		echo "clang-format --dry-run: all C/C++ formatted"; \
	else \
		echo "check-fmt-c: clang-format not installed; skipping"; \
	fi

# ---- umbrellas -------------------------------------------------------

fmt: fmt-idris fmt-py fmt-c
	@echo "fmt: all languages formatted in place"

check-fmt: test-integration-lint-fmt check-fmt-py check-fmt-c
	@echo "check-fmt: all languages formatted"

# ---- formatter unit tests --------------------------------------------

# Round-trip oracle + render suite. Colocated dual-ipkg pattern, same as
# test-unit-args. Rolled into test-unit-idris.
test-unit-fmt:
	cd packages/idris-fmt && pack --no-prompt build idris-fmt-tests.ipkg
	$(STDBUF) ./packages/idris-fmt/build/exec/idris-fmt-test
