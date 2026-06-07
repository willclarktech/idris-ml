# mk/install.mk — local-prefix installs (install-core/gym/
# transformers/notebook/examples) + per-package type-check targets
# (check-*).

# Install core library to local prefix (needed before building examples/tests).
# `--build-dir` keys the per-package ttc cache on the active BUILD_KEY so
# `BACKEND=tape` and `BACKEND=torch` (etc.) each have their own warm cache.
#
# Every install-* target deps $(BUILD)/.library-cache-stamp so the
# stale-ttc wipe in its recipe runs BEFORE any `idris2 --install`
# starts. Without this the stamp was only a sibling prereq of the
# `install` aggregate: serially the wipe ran after the installs
# (letting a stale-inlined ttc reach the prefix), and under `make -j`
# it ran concurrently — `rm -rf $(BUILD)/ttc-*` deleting build dirs
# mid-elaboration (observed: install-gym/install-core dying with no
# output inside example-checkpoint-demo under MAKEFLAGS=-j2).
.PHONY: install-core install-gym install-transformers install-notebook \
        install-examples install-test-harness install check-idris-ml \
        check-gym check-transformers check-idris check

install-core: backend $(HWCONFIG_IDR) $(HWDEVICES_IDR) $(BUILD)/.library-cache-stamp
	@cd packages/idris-ml && IDRIS2_PREFIX=$(IDRIS2_LOCAL) idris2 --build-dir $(CURDIR)/$(BUILD)/ttc-idris-ml --install idris-ml.ipkg >/dev/null

# Install gym to local prefix
install-gym: $(BUILD)/.library-cache-stamp
	@cd packages/idris-gym && IDRIS2_PREFIX=$(IDRIS2_LOCAL) idris2 --build-dir $(CURDIR)/$(BUILD)/ttc-idris-gym --install idris-gym.ipkg >/dev/null

# Install idris-transformers (HF-aligned model library) to local prefix.
# Depends on install-core because every Hf* module imports from idris-ml.
install-transformers: install-core
	@cd packages/idris-transformers && IDRIS2_PREFIX=$(IDRIS2_LOCAL) idris2 --build-dir $(CURDIR)/$(BUILD)/ttc-idris-transformers --install idris-transformers.ipkg >/dev/null

# Install notebook prelude to local prefix
install-notebook: install-core
	@cd packages/idris-ml-notebook && IDRIS2_PREFIX=$(IDRIS2_LOCAL) idris2 --build-dir $(CURDIR)/$(BUILD)/ttc-idris-ml-notebook --install idris-ml-notebook.ipkg >/dev/null

# Install idris-ml-examples as a library (needed by its test harness)
install-examples: install-core install-gym install-transformers $(BUILDCONFIG_IDR)
	@cd packages/idris-ml-examples && IDRIS2_PREFIX=$(IDRIS2_LOCAL) idris2 --build-dir $(CURDIR)/$(BUILD)/ttc-idris-ml-examples --install idris-ml-examples.ipkg >/dev/null

# Install the shared test harness (Test.Harness) used by every package's
# test/ suite. Pure-Idris harness PLUS hedgehog adapter. Since
# adding the hedgehog dep, idris-test must be installed via pack
# (nix's idris2 doesn't know hedgehog). The pack-managed test
# ipkgs (idris-ml-tests etc.) pick it up automatically; this
# explicit recipe stays for callers that need a pre-installed
# idris-test in pack's collection (idempotent).
install-test-harness:
	@pack --no-prompt install idris-test >/dev/null

# Install all Idris packages locally. install-test-harness is NOT
# in the chain — pack lazily installs idris-test the first time
# any tests ipkg references it.
install: install-core install-gym install-transformers install-notebook install-examples

# Type-check the idris-ml core library only. Fastest single-package
# gate. The `check` aggregator below is the daily-driver default.
#
# Depends on install-core (not raw backend) so that under `make -j`,
# check-idris-ml is serialized AFTER install-core. Both share the
# `$(BUILD)/ttc-idris-ml` directory; running them in parallel would
# race on the TTC cache. install-core elaborates idris-ml in --install
# mode and writes the TTC files; the --build invocation below is then
# a fast TTC-hit no-op. The downstream win: check-gym, check-transformers,
# and check-notebook run concurrently with this step once install-core
# completes, dropping `make -j4 check` wall-clock to ~slowest-single-
# package elaboration. install-core is a write to IDRIS2_LOCAL/, which
# is already a per-checkout prefix — harmless side effect.
check-idris-ml: install-core
	cd packages/idris-ml && idris2 --build-dir $(CURDIR)/$(BUILD)/ttc-idris-ml --build idris-ml.ipkg

# Type-check gym package
check-gym:
	cd packages/idris-gym && idris2 --build-dir $(CURDIR)/$(BUILD)/ttc-idris-gym --build idris-gym.ipkg

# Type-check idris-transformers package (depends on idris-ml being installed).
check-transformers: install-core
	cd packages/idris-transformers && IDRIS2_PREFIX=$(IDRIS2_LOCAL) idris2 --build-dir $(CURDIR)/$(BUILD)/ttc-idris-transformers --build idris-transformers.ipkg

# Default `check` aggregator — type-check every Idris-side library
# package (core + gym + transformers + notebook). Does NOT build
# example executables; for that, run `check-examples` (~20-60 min)
# or `check-all` (libs + examples). C is built via `make backend` —
# there's no `check-c` because the C compile step IS the type-check.
# Wall-clock on a warm tree is a few minutes; cold is longer.
check-idris: check-idris-ml check-gym check-transformers check-notebook

# Daily-driver alias — same scope as `check-idris`.
check: check-idris
