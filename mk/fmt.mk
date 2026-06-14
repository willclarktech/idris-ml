# mk/fmt.mk — the Idris source formatter (idris-fmt) and its gates.
#
# idris-fmt is the repo's compiler-native Idris 2 formatter (it parses
# with the compiler's own parser and gates every reformat behind a
# round-trip oracle). `make fmt` rewrites every .idr file in place;
# `make check-fmt` (aka test-integration-lint-fmt, the CI gate) fails if
# any file is not already formatted. `make test-unit-fmt` runs the
# formatter's own unit suite.

.PHONY: fmt check-fmt test-integration-lint-fmt test-unit-fmt idris-fmt-build

IDRIS_FMT  := ./packages/idris-fmt/build/exec/idris-fmt
FMT_FILES  := $(shell find packages -name '*.idr' -not -path '*/build/*')

# Build the formatter executable (cheap once warm).
idris-fmt-build:
	cd packages/idris-fmt && pack --no-prompt build idris-fmt.ipkg

# Reformat every .idr file in place.
fmt: idris-fmt-build
	$(IDRIS_FMT) --write $(FMT_FILES)

# CI gate: non-zero exit if any .idr file is not formatted. `check-fmt`
# is the friendly spelling; `test-integration-lint-fmt` is the taxonomy
# name CI invokes (shares the recipe).
check-fmt test-integration-lint-fmt: idris-fmt-build
	$(IDRIS_FMT) --check $(FMT_FILES)

# Formatter unit tests (round-trip oracle + render). Colocated dual-ipkg
# pattern, same as test-unit-args.
test-unit-fmt:
	cd packages/idris-fmt && pack --no-prompt build idris-fmt-tests.ipkg
	$(STDBUF) ./packages/idris-fmt/build/exec/idris-fmt-test
