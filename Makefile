UNAME := $(shell uname)
BUILD := build
BACKEND ?= tape

# MLX stream selection at runtime, also consumed by the BuildConfig
# generation rule below — when PRIMARY=mlx and MLX_DEVICE=gpu, examples
# spell `Tensor [..] (MlxDev MGpu) F32 WithGrad` so the type-level
# claim matches what mlx actually runs (Metal GPU is float32-only per
# the f32 rewrite). Any other configuration falls back to `CPU` + `F64`.
MLX_DEVICE ?= cpu

# Per-backend default seed for examples. Some examples (notably NTM-copy and
# DNC-copy/recall — see docs/develop/gotchas.md) are highly seed-sensitive at
# moderate epoch budgets, and a seed that converges cleanly on one backend can
# stall on another due to ULP-level numerical differences in the gradient
# computation. The seed picked here is the one that converges with the
# current backend's tape-order on convergence-expected examples; users
# override per-example via the example's <FOO>_ARGS variable, e.g.
# `make example-ntm-copy NTM_COPY_ARGS="--seed N"`. Non-seed-sensitive
# examples (supervised, rnn/lstm/gru, transformer, etc.) work at either
# default; the unified per-backend value keeps the surface predictable.
ifeq ($(BACKEND),mlx)
  EXAMPLE_DEFAULT_SEED := 99
else
  EXAMPLE_DEFAULT_SEED := 42
endif
SEED_FLAG := --seed $(EXAMPLE_DEFAULT_SEED)

# Shared-library extension. macOS uses .dylib, everything else .so. Used by
# both the active-backend symlink ($(LIB)) and the per-backend output file
# ($(BACKEND_LIB)) so they match on every platform.
ifeq ($(UNAME), Darwin)
  LIB_EXT := dylib
else
  LIB_EXT := so
endif

# Per-example wall-clock cap for test-examples. Examples exceeding this are
# killed and reported as timeouts. Override with `EXAMPLE_TIMEOUT=900 make ...`.
EXAMPLE_TIMEOUT ?= 600

# Line-buffer Chez output. Without this, stdout fully-buffers when piped or
# redirected and progress logs only appear at process exit. We use stdbuf
# unless its libstdbuf.so is incompatible with the system's dyld (e.g. brew
# coreutils stdbuf on Apple-Silicon GH runners is arm64 but the inserted-
# library loader requires arm64e). Test by injecting it into a no-op `true`:
# success means libstdbuf loads, failure means we fall back to no buffering.
STDBUF := $(shell stdbuf -oL true >/dev/null 2>&1 && echo "stdbuf -oL")

# `make` builds backend + type-checks all packages. `make all` also runs tests.
.DEFAULT_GOAL := check-all

# --- Package paths ---
CORE_SRC := packages/idris-ml/src
GYM_SRC := packages/idris-gym/src
EXAMPLE_SRC := packages/idris-ml-examples/src
TEST_SRC := packages/idris-ml/test/src
BACKENDS_DIR := packages/backends

# Local package install prefix (writable, avoids polluting system Idris2)
IDRIS2_LOCAL := $(CURDIR)/.idris2

# Where the system idris2 was installed — needed to find contrib/base/etc.
# when our install rules override IDRIS2_PREFIX with $(IDRIS2_LOCAL). Returns
# empty before idris2 is on PATH (e.g. during `make backend`); harmless then.
# Nix-built idris2 bakes its own paths so the override is also harmless there.
SYS_IDRIS2_PREFIX := $(shell idris2 --paths 2>/dev/null | sed -n 's/.*Installation Prefix.*"\([^"]*\)".*/\1/p')

ifeq ($(SYS_IDRIS2_PREFIX),)
export IDRIS2_PACKAGE_PATH := $(IDRIS2_LOCAL)/idris2-0.8.0
else
export IDRIS2_PACKAGE_PATH := $(IDRIS2_LOCAL)/idris2-0.8.0:$(SYS_IDRIS2_PREFIX)/idris2-0.8.0
endif

# Idris flags for example/test builds (use installed packages)
IDRIS_FLAGS := --source-dir $(EXAMPLE_SRC) -p contrib -p idris-ml -p idris-gym

# Library source files — any change invalidates top-level build/ttc/ cache.
# Idris 2's interface-hash dependency tracking doesn't invalidate downstream
# TTCs when a module's public interface is unchanged but a where-clause body
# (or other inlined internal) changed. Single-file `idris2 -o <name>` example
# builds then reuse stale build/ttc/Example/*.ttc with old inlined code baked
# in. Wiping build/ttc when any library source is newer than this stamp
# forces a clean rebuild. See docs/develop/gotchas.md.
LIBRARY_SRCS := $(shell find packages/idris-ml/src packages/idris-gym/src -name '*.idr' 2>/dev/null) \
                packages/idris-ml-examples/src/Generate.idr

build/.library-cache-stamp: $(LIBRARY_SRCS)
	@echo "Library source changed — invalidating build/ttc cache"
	@rm -rf build/ttc
	@mkdir -p build
	@touch $@

# --- Backend selection ---
# BACKEND is a comma-separated list of built-in backends to link into
# libidrisml.{so,dylib}. The first item is the **primary** backend —
# its symbols are also exported under unified names (e.g.
# `_tensor_add` aliased to `_tensor_add_<primary>`) so existing Idris
# `%foreign "C:tensor_add,libidrisml"` declarations resolve to it.
# Non-primary backends are reachable only by their suffixed names,
# which Phase 2.x UserDevice instance methods will target directly.
#
# Examples:
#   BACKEND=tape                  — single tape build (default, lean)
#   BACKEND=tape,torch            — Linux full build; both linked
#   BACKEND=tape,torch,mlx        — macOS full build
#   BACKEND=torch                 — torch-only CI lane
#   BACKEND=mlx                   — mlx-only CI lane
empty :=
space := $(empty) $(empty)
comma := ,
BACKEND_LIST := $(subst $(comma),$(space),$(BACKEND))
PRIMARY := $(firstword $(BACKEND_LIST))

# Per-backend property tables. Common compile flags (`-O2 -fPIC
# -include rename_<b>.h`) are applied by the per-backend rule below;
# `<b>_CFLAGS` adds whatever else that backend's compile needs
# (include paths, C++ std). `<b>_LDFLAGS_<UNAME>` is per-platform.

tape_SRC := $(BACKENDS_DIR)/backend_tape.c
tape_CC := cc
# ACCELERATE_NEW_LAPACK is a compile-time #define (gates BLAS API
# version); the framework flag is link-time.
tape_CFLAGS := -DACCELERATE_NEW_LAPACK
tape_LDFLAGS_Darwin := -framework Accelerate
tape_LDFLAGS_Linux := -lm -lblas

# libtorch detection — only when torch is in BACKEND_LIST.
ifneq ($(filter torch,$(BACKEND_LIST)),)
  ifndef LIBTORCH_PATH
    LIBTORCH_PATH := $(shell pkg-config --variable=prefix torch 2>/dev/null)
  endif
  ifndef LIBTORCH_PATH
    LIBTORCH_PATH := $(shell python3 -c "import torch, os; print(os.path.dirname(torch.__file__))" 2>/dev/null)
  endif
  ifndef LIBTORCH_PATH
    LIBTORCH_PATH := $(shell packages/pytorch/.venv/bin/python3 -c "import torch, os; print(os.path.dirname(torch.__file__))" 2>/dev/null)
  endif
  ifndef LIBTORCH_PATH
    $(error libtorch not found. Set LIBTORCH_PATH, install via pkg-config, or run: cd packages/pytorch && uv sync)
  endif
  TORCH_INC := $(LIBTORCH_PATH)/include
  TORCH_INC_API := $(LIBTORCH_PATH)/include/torch/csrc/api/include
  TORCH_LIB := $(LIBTORCH_PATH)/lib
endif

torch_SRC := $(BACKENDS_DIR)/backend_torch.cpp
torch_CC := c++
torch_CFLAGS := -std=c++17 -I$(TORCH_INC) -I$(TORCH_INC_API)
torch_LDFLAGS_Darwin := -L$(TORCH_LIB) -ltorch -ltorch_cpu -lc10 -Wl,-rpath,$(TORCH_LIB)
# GNU ld defaults to --as-needed since binutils 2.x, which drops NEEDED
# tags for libs whose symbols aren't *directly* referenced in our object
# files. libidrisml.so calls torch C++ API which transitively pulls c10/
# torch_cpu symbols at runtime — but the linker can't see those needs at
# static-link time, so it strips the NEEDED tags and dlopen later trips
# on `undefined symbol: _ZTIN3c105ErrorE` (c10::Error RTTI).
# `--no-as-needed` forces NEEDED for the libtorch trio explicitly.
torch_LDFLAGS_Linux := -L$(TORCH_LIB) -Wl,--no-as-needed -ltorch -ltorch_cpu -lc10 -Wl,--as-needed -Wl,-rpath,$(TORCH_LIB)

# MLX detection — only when mlx is in BACKEND_LIST; Apple-only.
#
# mlx is declared as a darwin-only dependency of `idris-ml-torch-ref`
# (`packages/pytorch/pyproject.toml`). The canonical install lives in
# that project's uv-managed venv. To set up:
#
#   cd packages/pytorch && uv sync     # installs mlx into .venv on macOS
#
# Resolution order, picks the first install that has both
# `include/mlx/mlx.h` AND a Metal-capable runtime. Metal capability is
# detected by the presence of `lib/mlx.metallib` — the precompiled
# Metal kernel library. mlx installs without it (e.g. nixpkgs'
# `python3Packages.mlx`, built with `MLX_BUILD_METAL=false`) silently
# fail at runtime with "Cannot set gpu device without gpu backend",
# so we won't auto-pick those.
#
#   1. $MLX_SITE if you set it explicitly (no validation)
#   2. Project venv at `packages/pytorch/.venv/.../mlx` (canonical)
#   3. $UV_CACHE_DIR (or `~/.cache/uv`) — any uv-installed mlx anywhere
#   4. Any importable `mlx` reachable from python3 (conda, system, etc.)
#
# No silent fallback to `nix build nixpkgs#python3Packages.mlx`: that
# side-effects mlx into the nix store regardless of any
# `home-manager` / `nix-darwin` uninstall, and ships a CPU-only build.
ifneq ($(filter mlx,$(BACKEND_LIST)),)
  ifneq ($(UNAME), Darwin)
    $(error MLX backend requires macOS; current UNAME=$(UNAME))
  endif

  # Validation helper: return $1 if it looks like a usable Metal-capable
  # mlx install, empty otherwise. Tests for both headers (`mlx/mlx.h`)
  # and the Metal kernel library (`mlx.metallib`).
  _mlx_validate = $(if $(and $(wildcard $1/include/mlx/mlx.h),$(wildcard $1/lib/mlx.metallib)),$1,)

  ifndef MLX_SITE
    # (2) Project venv. Canonical — declared via packages/pytorch/pyproject.toml.
    MLX_SITE := $(call _mlx_validate,$(wildcard packages/pytorch/.venv/lib/python*/site-packages/mlx))
  endif
  ifeq ($(MLX_SITE),)
    # (3) uv global archive cache. Picks up `uv pip install mlx` even
    # outside the project venv, e.g. for one-off use elsewhere.
    _mlx_uv := $(shell find "$${UV_CACHE_DIR:-$$HOME/.cache/uv}" -path "*/mlx/lib/mlx.metallib" -type f 2>/dev/null | head -n 1)
    MLX_SITE := $(if $(_mlx_uv),$(patsubst %/lib/mlx.metallib,%,$(_mlx_uv)),)
  endif
  ifeq ($(MLX_SITE),)
    # (4) Anything importable from python3 (conda / system site-packages).
    _mlx_py := $(shell python3 -c "import importlib.util as u, os; s=u.find_spec('mlx'); print(s.submodule_search_locations[0] if s and s.submodule_search_locations else '')" 2>/dev/null)
    MLX_SITE := $(call _mlx_validate,$(_mlx_py))
  endif

  ifeq ($(MLX_SITE),)
    $(error No Metal-capable mlx install found. Run `cd packages/pytorch && uv sync` to install the declared mlx dependency into the project venv, or set MLX_SITE=<path/to/mlx> where the path contains both include/mlx/mlx.h and lib/mlx.metallib.)
  endif

  # Absolute paths so `-Wl,-rpath,$(MLX_LIB)` works regardless of the
  # CWD at runtime (the matmul-bench wrapper chdir's into build/exec/...).
  MLX_INC := $(abspath $(MLX_SITE)/include)
  MLX_LIB := $(abspath $(MLX_SITE)/lib)
endif

mlx_SRC := $(BACKENDS_DIR)/backend_mlx.cpp
mlx_CC := c++
mlx_CFLAGS := -std=c++20 -I$(MLX_INC)
# `-framework Accelerate` is still required even though tape may already
# pull it in: ld with duplicate framework refs deduplicates, but ld
# without ANY ref to a transitively-needed framework rejects mlx's
# Accelerate-using symbols at load time (manifested as a scatter-VJP
# autograd failure in DNC examples — see Phase 1 verification).
mlx_LDFLAGS_Darwin := -L$(MLX_LIB) -lmlx -Wl,-rpath,$(MLX_LIB) -framework Accelerate -framework Metal -framework Foundation
# Linux not reachable in practice — the `ifneq ($(UNAME), Darwin)
# $(error ...)` guard above stops the build before we get here on
# Linux. Empty value keeps the Makefile parseable on non-Darwin even
# when mlx isn't in BACKEND_LIST.
mlx_LDFLAGS_Linux :=

# Final dylib path — one file, all listed backends in it, primary aliases.
LIB := $(BUILD)/libidrisml.$(LIB_EXT)

# Primary backend's rename header drives the alias step + shared-source
# rename (shared sources are compiled per-primary because their callers
# use unified names which only resolve to the primary's suffixed defs).
BACKEND_RENAME_H := $(BACKENDS_DIR)/rename_$(PRIMARY).h

# Primary-backend unified-name aliases. Built from the rename header
# by extracting each `#define <unified> <suffixed>` pair and emitting
# a linker alias. Lets Idris-side `%foreign "C:tensor_add,libidrisml"`
# declarations keep working unchanged through Phase 1 even though the
# defining C symbol is now suffixed. Also lets backend-agnostic shared
# objects link against the unified names. Phase 2.x retires this alias
# step as each %foreign moves into a per-instance UserDevice method
# bound to the suffixed name directly.
#
# NF==3 filter skips the rename header's include-guard `#define
# IDRISML_RENAME_*_H` (NF=2) and any other single-arg defines.
ifeq ($(UNAME), Darwin)
  # macOS ld(1) takes a file of `<aliasee> <aliasname>` pairs, with
  # leading underscores per the Mach-O symbol convention.
  BACKEND_ALIAS_FILE := $(BUILD)/aliases_$(PRIMARY).macos.list
  BACKEND_ALIAS_FLAGS := -Wl,-alias_list,$(BACKEND_ALIAS_FILE)
else
  # GNU ld takes `--defsym=<aliasname>=<aliasee>`; one flag per alias.
  BACKEND_ALIAS_FILE :=
  BACKEND_ALIAS_FLAGS := $(shell awk '/^\#define / && NF==3 { printf "-Wl,--defsym=%s=%s ", $$2, $$3 }' $(BACKEND_RENAME_H))
endif

$(BACKEND_ALIAS_FILE): $(BACKEND_RENAME_H) | $(BUILD)
	@awk '/^\#define / && NF==3 { print "_"$$3" _"$$2 }' $< > $@

# Per-backend object outputs.
BACKEND_OBJS := $(foreach b,$(BACKEND_LIST),$(BUILD)/backend_$(b).o)

# Per-backend compile template — generates a rule for each backend's
# backend_<b>.o file. Each backend uses its own CC + CFLAGS + rename
# header; the union of all per-backend LDFLAGS gets passed to the
# final link.
define backend_compile_rule
$(BUILD)/backend_$(1).o: $($(1)_SRC) $(BACKENDS_DIR)/backend.h $(BACKENDS_DIR)/rename_$(1).h | $(BUILD)
	$($(1)_CC) -O2 -fPIC $($(1)_CFLAGS) -include $(BACKENDS_DIR)/rename_$(1).h -c -o $$@ $$<
endef

$(foreach b,$(BACKEND_LIST),$(eval $(call backend_compile_rule,$(b))))

# Shared C sources (serialization, JSON, data loading). These call /
# define `tensor_*`/`mnist_*`/`optimizer_*`/`param_*`/`index_array_get`
# symbols, so they're compiled with the PRIMARY backend's rename header
# so their cross-TU references match the primary's suffixed defs (other
# backends' suffixed defs are reachable but not by these shared TUs).
# cJSON is pure-C (no tensor surface) so it stays backend-agnostic.
SHARED_OBJ := $(BUILD)/safetensors_$(PRIMARY).o $(BUILD)/cJSON.o $(BUILD)/mnist_$(PRIMARY).o $(BUILD)/dataloader_$(PRIMARY).o

$(BUILD)/safetensors_$(PRIMARY).o: $(BACKENDS_DIR)/safetensors.c $(BACKENDS_DIR)/backend.h $(BACKENDS_DIR)/cJSON.h $(BACKEND_RENAME_H) | $(BUILD)
	cc -O2 -fPIC -include $(BACKEND_RENAME_H) -c -o $@ $<

$(BUILD)/cJSON.o: $(BACKENDS_DIR)/cJSON.c $(BACKENDS_DIR)/cJSON.h | $(BUILD)
	cc -O2 -fPIC -c -o $@ $<

$(BUILD)/mnist_$(PRIMARY).o: $(BACKENDS_DIR)/mnist.c $(BACKENDS_DIR)/backend.h $(BACKEND_RENAME_H) | $(BUILD)
	cc -O2 -fPIC -include $(BACKEND_RENAME_H) -c -o $@ $<

$(BUILD)/dataloader_$(PRIMARY).o: $(BACKENDS_DIR)/dataloader.c $(BACKEND_RENAME_H) | $(BUILD)
	cc -O2 -fPIC -include $(BACKEND_RENAME_H) -c -o $@ $<

# Final link compiler: c++ if any C++ backend (torch/mlx) is in the
# list, else cc. Picks the right runtime libraries automatically.
ifneq ($(filter torch mlx,$(BACKEND_LIST)),)
  LINK_CC := c++
else
  LINK_CC := cc
endif

# Union of per-backend link flags for the current platform.
BACKEND_LDFLAGS := $(foreach b,$(BACKEND_LIST),$($(b)_LDFLAGS_$(UNAME)))

# Stamp that records the current BACKEND value. Touched only when the
# value differs from disk; the dylib depends on it so changing BACKEND
# invalidates the previous link even when target file names match.
# `FORCE` runs every invocation so the comparison happens each time.
.PHONY: FORCE
FORCE:

$(BUILD)/.backend-stamp: FORCE | $(BUILD)
	@[ "$$(cat $@ 2>/dev/null)" = "$(BACKEND)" ] || { echo "$(BACKEND)" > $@; }

# Stamp + generated source for the example device/dtype selection. When
# PRIMARY=mlx and MLX_DEVICE=gpu, examples target `(MlxDev MGpu)` + `F32`;
# everything else stays on `CPU` + `F64`. The stamp records the active
# tuple so the generation step only writes the source file when the
# config actually changes (avoiding TTC churn on no-op rebuilds), mirror
# of the .backend-stamp pattern above.
BUILDCONFIG_KEY := $(PRIMARY):$(MLX_DEVICE)
BUILDCONFIG_IDR := packages/idris-ml-examples/src/BuildConfig.idr
BUILDCONFIG_IN  := packages/idris-ml-examples/src/BuildConfig.idr.in

$(BUILD)/.buildconfig-stamp: FORCE | $(BUILD)
	@[ "$$(cat $@ 2>/dev/null)" = "$(BUILDCONFIG_KEY)" ] || { echo "$(BUILDCONFIG_KEY)" > $@; }

$(BUILDCONFIG_IDR): $(BUILDCONFIG_IN) $(BUILD)/.buildconfig-stamp
	@if [ "$(PRIMARY)" = "mlx" ] && [ "$(MLX_DEVICE)" = "gpu" ]; then \
		DEVICE="MlxDev MGpu"; DTYPE="F32"; \
	else \
		DEVICE="CPU"; DTYPE="F64"; \
	fi; \
	sed "s|@DEVICE@|$$DEVICE|g; s|@DTYPE@|$$DTYPE|g" $< > $@
	@echo "[BuildConfig] PRIMARY=$(PRIMARY) MLX_DEVICE=$(MLX_DEVICE) → $$(awk -F' = ' '/^ExampleDevice = / { print $$2; exit }' $@) / $$(awk -F' = ' '/^ExampleDType = / { print $$2; exit }' $@)"

# Final link: all listed backends' .o + shared objects (primary's
# suffix) + primary's unified-name aliases. One dylib, no symlink.
$(LIB): $(BACKEND_OBJS) $(BACKEND_ALIAS_FILE) $(SHARED_OBJ) $(BUILD)/.backend-stamp | $(BUILD)
	$(LINK_CC) -O2 -shared -o $@ $(BACKEND_OBJS) $(SHARED_OBJ) $(BACKEND_LDFLAGS) $(BACKEND_ALIAS_FLAGS)

# Download MNIST dataset
dataset-mnist:
	bash scripts/dataset_mnist.sh

# Download tinyshakespeare corpus (~1 MB, 65-char vocab) for the GPT
# convergence run. Smoke gate uses the small embedded corpus and does
# not need this file.
dataset-tinyshakespeare:
	bash scripts/dataset_tinyshakespeare.sh

# Multi-link: one libidrisml.{so,dylib} with all listed BACKENDs in it.
# Primary backend's symbols are exported under both unified
# (`tensor_add`) and suffixed (`tensor_add_<primary>`) names; other
# backends' symbols are reachable only via their suffixed names.
backend: $(LIB)

# Regenerate the per-backend rename headers from backend.h. The
# generated files are checked in; `make check-rename-headers` (in CI)
# gates that they stay in sync with backend.h.
rename-headers:
	@python3 scripts/gen-rename-headers.py

check-rename-headers:
	@python3 scripts/gen-rename-headers.py --check

# Verify every Tensor-touching %foreign declaration matches the
# wrap-on-return Scheme template. See
# docs/develop/tensor-lifecycle-plan.md "FFI conventions". The single
# source of truth for which C symbols are Tensor handles is
# scripts/lifecycle/ffi_manifest.py — both the converter and the linter
# read from it.
check-ffi-wrap-template:
	@python3 scripts/lifecycle/check-ffi-wrap-template.py

# Lint: flag %foreign declarations whose Idris type is non-IO but whose
# C body has side effects (allocate, mutate, log, append to tape).
# Catches the bug class fixed by the IO refactor (commits leading up to
# e337512) — see the audit doc + `feedback_typeclass_zero_arg_method_eval.md`
# for the underlying mechanism. Known dead surfaces are skip-listed in
# the script until the dead-code cleanup row lands.
check-non-io-side-effects:
	@python3 scripts/lifecycle/check-non-io-side-effects.py

# Backend API test suite — runs against whichever backend is active
test-backend: $(BACKENDS_DIR)/test_backend.c backend | $(BUILD)
	cc -o $(BUILD)/test_backend $(BACKENDS_DIR)/test_backend.c -DBACKEND_$(shell echo $(PRIMARY) | tr a-z A-Z) -L$(BUILD) -lidrisml -Wl,-rpath,$(BUILD) -lm
	./$(BUILD)/test_backend

# Per-backend convenience targets
test-backend-tape:
	$(MAKE) BACKEND=tape test-backend

test-backend-mlx:
	$(MAKE) BACKEND=mlx test-backend

test-backend-torch:
	$(MAKE) BACKEND=torch test-backend

# Specialized C test suites
test-safetensors: $(BACKENDS_DIR)/test_safetensors.c backend | $(BUILD)
	cc -o $(BUILD)/test_safetensors $(BACKENDS_DIR)/test_safetensors.c -L$(BUILD) -lidrisml -Wl,-rpath,$(BUILD) -lm
	./$(BUILD)/test_safetensors

test-ntm-grad: $(BACKENDS_DIR)/test_ntm_grad.c backend | $(BUILD)
	cc -o $(BUILD)/test_ntm_grad $(BACKENDS_DIR)/test_ntm_grad.c -L$(BUILD) -lidrisml -Wl,-rpath,$(BUILD) -lm
	./$(BUILD)/test_ntm_grad

test-ntm-timestep: $(BACKENDS_DIR)/test_ntm_timestep.c backend | $(BUILD)
	cc -o $(BUILD)/test_ntm_timestep $(BACKENDS_DIR)/test_ntm_timestep.c -L$(BUILD) -lidrisml -Wl,-rpath,$(BUILD) -lm
	./$(BUILD)/test_ntm_timestep

# Job 3 Phase B — mx::compile integration tests. MLX-only.
test-mlx-compile: $(BACKENDS_DIR)/test_mlx_compile.c
	$(MAKE) BACKEND=mlx backend
	cc -o $(BUILD)/test_mlx_compile $(BACKENDS_DIR)/test_mlx_compile.c -L$(BUILD) -lidrisml -Wl,-rpath,$(BUILD) -lm
	./$(BUILD)/test_mlx_compile

print-torch:
	@echo "LIBTORCH_PATH=$(LIBTORCH_PATH)"
	@echo "TORCH_INC=$(TORCH_INC)"
	@echo "TORCH_LIB=$(TORCH_LIB)"
	@echo "BACKEND=$(BACKEND)"
	@echo "LIB=$(LIB)"

# Install core library to local prefix (needed before building examples/tests)
install-core: backend
	@cd packages/idris-ml && IDRIS2_PREFIX=$(IDRIS2_LOCAL) idris2 --install idris-ml.ipkg >/dev/null

# Install gym to local prefix
install-gym:
	@cd packages/idris-gym && IDRIS2_PREFIX=$(IDRIS2_LOCAL) idris2 --install idris-gym.ipkg >/dev/null

# Install notebook prelude to local prefix
install-notebook: install-core
	@cd packages/idris-ml-notebook && IDRIS2_PREFIX=$(IDRIS2_LOCAL) idris2 --install idris-ml-notebook.ipkg >/dev/null

# Install idris-ml-examples as a library (needed by its test harness)
install-examples: install-core install-gym $(BUILDCONFIG_IDR)
	@cd packages/idris-ml-examples && IDRIS2_PREFIX=$(IDRIS2_LOCAL) idris2 --install idris-ml-examples.ipkg >/dev/null

# Install all Idris packages locally
install: install-core install-gym install-notebook install-examples build/.library-cache-stamp

# Idris build (type-check core library)
check: backend
	cd packages/idris-ml && idris2 --build idris-ml.ipkg

# Type-check gym package
check-gym:
	cd packages/idris-gym && idris2 --build idris-gym.ipkg

# Verify Idris example defaults match the paired torch_ref/scripts/*.py defaults.
# Catches the "I changed Idris's default but forgot the matching ref" drift class.
check-paired-defaults:
	@python3 scripts/check-paired-defaults.py

# Verify the GradMode gate is intact: a NoGrad loss must NOT type-check
# as input to nativeTrainStep. Inverts the idris2 exit code (success =
# compile failed) and matches on the WithGrad/NoGrad error message.
# Depends on `install` so idris-ml is locatable in the local IDRIS2 prefix.
check-gradmode-gate: install
	@./scripts/check-gradmode-gate.sh

# Verify the aliasing footgun on `freezeNetwork` is closed by linear
# types: using the pre-freeze Network reference must be a compile error.
check-gradmode-aliasing: install
	@./scripts/check-gradmode-aliasing.sh

# Type-check examples (builds each as executable, which is the real check)
check-examples: install
	@for f in $(EXAMPLE_SRC)/Example/*.idr; do \
		mod=$$(basename "$$f" .idr); \
		slug=$$(echo "$$mod" | tr 'A-Z' 'a-z'); \
		echo "Building Example.$$mod..."; \
		idris2 $(IDRIS_FLAGS) -o "check-$$slug" "$$f" || exit 1; \
	done
	@echo "All examples type-check."

# Idris tests
test: install
	idris2 --source-dir $(TEST_SRC) -p contrib -p idris-ml -o test $(TEST_SRC)/Main.idr
	cp $(LIB) build/exec/test_app/
	./build/exec/test

# Idris tests for idris-gym package (pure Idris, no backend required)
test-gym: install-gym
	cd packages/idris-gym/test && idris2 --build test.ipkg
	$(STDBUF) ./packages/idris-gym/test/build/exec/idris-gym-test

# Microbench for idris-gym hot paths (RNG, Blackjack obs, env step+observe).
# Pure Idris, no backend dependency. Useful for Job 4-style env-side
# perf experiments where single-run RL training is too noisy.
#
# Pass bench names (rng, blackjack, pendulum, acrobot, taxi, cliffwalking)
# to run a subset, e.g. `make bench-gym BENCH_ARGS=rng`. Default runs all.
bench-gym: install-gym
	cd packages/idris-gym/test && idris2 --build bench.ipkg
	$(STDBUF) ./packages/idris-gym/test/build/exec/idris-gym-bench $(BENCH_ARGS)

# Unit tests for idris-ml-examples (runs moved Test.Generate)
test-examples-unit: install-examples
	cd packages/idris-ml-examples/test && idris2 --build test.ipkg
	cp $(LIB) packages/idris-ml-examples/test/build/exec/idris-ml-examples-test_app/
	$(STDBUF) ./packages/idris-ml-examples/test/build/exec/idris-ml-examples-test

# Build and run examples (require: make install)
example-supervised: install
	idris2 $(IDRIS_FLAGS) -o supervised $(EXAMPLE_SRC)/Example/Supervised.idr
	cp $(LIB) build/exec/supervised_app/
	./build/exec/supervised $(SEED_FLAG) $(SUPERVISED_ARGS)

example-rnn: install
	idris2 $(IDRIS_FLAGS) -o rnn $(EXAMPLE_SRC)/Example/Rnn.idr
	cp $(LIB) build/exec/rnn_app/
	./build/exec/rnn $(SEED_FLAG) $(RNN_ARGS)

example-lstm: install
	idris2 $(IDRIS_FLAGS) -o lstm $(EXAMPLE_SRC)/Example/Lstm.idr
	cp $(LIB) build/exec/lstm_app/
	./build/exec/lstm $(SEED_FLAG) $(LSTM_ARGS)

example-gru: install
	idris2 $(IDRIS_FLAGS) -o gru $(EXAMPLE_SRC)/Example/Gru.idr
	cp $(LIB) build/exec/gru_app/
	./build/exec/gru $(SEED_FLAG) $(GRU_ARGS)

# BringYourOwn — worked example of a user-supplied backend. Builds
# libbyo.dylib alongside the active libidrisml so the example app
# can dlopen both: the BYO instance dispatches to `byo_*` symbols
# in libbyo, and the built-in CPU instance dispatches to unified
# names in libidrisml. See packages/backends/backend_byo.c +
# Example/BringYourOwn.idr.
$(BUILD)/libbyo.$(LIB_EXT): $(BACKENDS_DIR)/backend_byo.c | $(BUILD)
	cc -O2 -shared -fPIC -o $@ $<

example-bring-your-own: install $(BUILD)/libbyo.$(LIB_EXT)
	idris2 $(IDRIS_FLAGS) -o bring-your-own $(EXAMPLE_SRC)/Example/BringYourOwn.idr
	cp $(LIB) $(BUILD)/libbyo.$(LIB_EXT) build/exec/bring-your-own_app/
	./build/exec/bring-your-own

example-ntm-copy: install
	idris2 $(IDRIS_FLAGS) -o ntm-copy $(EXAMPLE_SRC)/Example/NtmCopy.idr
	cp $(LIB) build/exec/ntm-copy_app/
	$(STDBUF) ./build/exec/ntm-copy $(SEED_FLAG) $(NTM_COPY_ARGS)

example-ntm-associative-recall: install
	idris2 $(IDRIS_FLAGS) -o ntm-associative-recall $(EXAMPLE_SRC)/Example/NtmAssociativeRecall.idr
	cp $(LIB) build/exec/ntm-associative-recall_app/
	$(STDBUF) ./build/exec/ntm-associative-recall $(SEED_FLAG) $(NTM_RECALL_ARGS)

example-dnc-copy: install
	idris2 $(IDRIS_FLAGS) -o dnc-copy $(EXAMPLE_SRC)/Example/DncCopy.idr
	cp $(LIB) build/exec/dnc-copy_app/
	$(STDBUF) ./build/exec/dnc-copy $(SEED_FLAG) $(DNC_COPY_ARGS)

example-dnc-recall: install
	idris2 $(IDRIS_FLAGS) -o dnc-recall $(EXAMPLE_SRC)/Example/DncAssociativeRecall.idr
	cp $(LIB) build/exec/dnc-recall_app/
	$(STDBUF) ./build/exec/dnc-recall $(SEED_FLAG) $(DNC_RECALL_ARGS)

example-transformer: install
	idris2 $(IDRIS_FLAGS) -o transformer $(EXAMPLE_SRC)/Example/Transformer.idr
	cp $(LIB) build/exec/transformer_app/
	./build/exec/transformer $(SEED_FLAG) $(TRANSFORMER_ARGS)

example-tcast-demo: install
	idris2 $(IDRIS_FLAGS) -o tcast-demo $(EXAMPLE_SRC)/Example/TCastDemo.idr
	cp $(LIB) build/exec/tcast-demo_app/
	./build/exec/tcast-demo $(TCAST_DEMO_ARGS)

example-gpt: install
	idris2 $(IDRIS_FLAGS) -o gpt $(EXAMPLE_SRC)/Example/Gpt.idr
	cp $(LIB) build/exec/gpt_app/
	$(STDBUF) ./build/exec/gpt $(SEED_FLAG) $(GPT_ARGS)

# Full-corpus convergence run (~hours on tape). Default `make example-gpt`
# is a ~30s embedded-corpus demo; this target is the real char-LM
# convergence target (matching nanoGPT/train_shakespeare_char.py).
example-gpt-full: install dataset-tinyshakespeare
	idris2 $(IDRIS_FLAGS) -o gpt $(EXAMPLE_SRC)/Example/Gpt.idr
	cp $(LIB) build/exec/gpt_app/
	$(STDBUF) ./build/exec/gpt $(SEED_FLAG) --corpus tinyshakespeare --epochs 1000 $(GPT_ARGS)

example-mnist: install dataset-mnist
	idris2 $(IDRIS_FLAGS) -o mnist $(EXAMPLE_SRC)/Example/Mnist.idr
	cp $(LIB) build/exec/mnist_app/
	$(STDBUF) ./build/exec/mnist $(SEED_FLAG) $(MNIST_ARGS)

example-seq-classify: install
	idris2 $(IDRIS_FLAGS) -o seq-classify $(EXAMPLE_SRC)/Example/SeqClassify.idr
	cp $(LIB) build/exec/seq-classify_app/
	$(STDBUF) ./build/exec/seq-classify $(SEED_FLAG) $(SEQ_ARGS)

example-reinforce: install
	idris2 $(IDRIS_FLAGS) -o reinforce $(EXAMPLE_SRC)/Example/Reinforce.idr
	cp $(LIB) build/exec/reinforce_app/
	./build/exec/reinforce $(SEED_FLAG) $(REINFORCE_ARGS)

example-q-learning: install
	idris2 $(IDRIS_FLAGS) -o q-learning $(EXAMPLE_SRC)/Example/QLearning.idr
	cp $(LIB) build/exec/q-learning_app/
	./build/exec/q-learning $(SEED_FLAG) $(Q_LEARNING_ARGS)

example-sarsa: install
	idris2 $(IDRIS_FLAGS) -o sarsa $(EXAMPLE_SRC)/Example/Sarsa.idr
	cp $(LIB) build/exec/sarsa_app/
	./build/exec/sarsa $(SEED_FLAG) $(SARSA_ARGS)

example-monte-carlo: install
	idris2 $(IDRIS_FLAGS) -o monte-carlo $(EXAMPLE_SRC)/Example/MonteCarlo.idr
	cp $(LIB) build/exec/monte-carlo_app/
	./build/exec/monte-carlo $(SEED_FLAG) $(MONTE_CARLO_ARGS)

example-frozen-lake: install
	idris2 $(IDRIS_FLAGS) -o frozen-lake $(EXAMPLE_SRC)/Example/FrozenLake.idr
	cp $(LIB) build/exec/frozen-lake_app/
	./build/exec/frozen-lake $(SEED_FLAG) $(FROZEN_LAKE_ARGS)

example-taxi: install
	idris2 $(IDRIS_FLAGS) -o taxi $(EXAMPLE_SRC)/Example/Taxi.idr
	cp $(LIB) build/exec/taxi_app/
	./build/exec/taxi $(SEED_FLAG) $(TAXI_ARGS)

example-dqn: install
	idris2 $(IDRIS_FLAGS) -o dqn $(EXAMPLE_SRC)/Example/Dqn.idr
	cp $(LIB) build/exec/dqn_app/
	$(STDBUF) ./build/exec/dqn $(SEED_FLAG) $(DQN_ARGS)

example-mountain-car: install
	idris2 $(IDRIS_FLAGS) -o mountain-car $(EXAMPLE_SRC)/Example/MountainCar.idr
	cp $(LIB) build/exec/mountain-car_app/
	$(STDBUF) ./build/exec/mountain-car $(SEED_FLAG) $(MOUNTAIN_CAR_ARGS)

example-mountain-car-cont: install
	idris2 $(IDRIS_FLAGS) -o mountain-car-cont $(EXAMPLE_SRC)/Example/MountainCarCont.idr
	cp $(LIB) build/exec/mountain-car-cont_app/
	$(STDBUF) ./build/exec/mountain-car-cont $(SEED_FLAG) $(MOUNTAIN_CAR_CONT_ARGS)

example-a2c: install
	idris2 $(IDRIS_FLAGS) -o a2c $(EXAMPLE_SRC)/Example/A2c.idr
	cp $(LIB) build/exec/a2c_app/
	$(STDBUF) ./build/exec/a2c $(SEED_FLAG) $(A2C_ARGS)

example-ppo: install
	idris2 $(IDRIS_FLAGS) -o ppo $(EXAMPLE_SRC)/Example/Ppo.idr
	cp $(LIB) build/exec/ppo_app/
	$(STDBUF) ./build/exec/ppo $(SEED_FLAG) $(PPO_ARGS)

example-sac: install
	idris2 $(IDRIS_FLAGS) -o sac $(EXAMPLE_SRC)/Example/Sac.idr
	cp $(LIB) build/exec/sac_app/
	$(STDBUF) ./build/exec/sac $(SEED_FLAG) $(SAC_ARGS)

example-transfer: install
	idris2 $(IDRIS_FLAGS) -o transfer $(EXAMPLE_SRC)/Example/Transfer.idr
	cp $(LIB) build/exec/transfer_app/
	./build/exec/transfer $(SEED_FLAG) $(TRANSFER_ARGS)

example-transfer-demo:
	@echo "=== Phase 1: Train on tape ==="
	$(MAKE) BACKEND=tape example-transfer TRANSFER_ARGS="--mode train --epochs 500 --save /tmp/transfer.safetensors"
	@echo ""
	@echo "=== Phase 2: Continue on mlx ==="
	$(MAKE) BACKEND=mlx example-transfer TRANSFER_ARGS="--mode continue --load /tmp/transfer.safetensors --epochs 500 --save /tmp/transfer2.safetensors"
	@echo ""
	@echo "=== Phase 3: Infer on torch ==="
	$(MAKE) BACKEND=torch example-transfer TRANSFER_ARGS="--mode infer --load /tmp/transfer2.safetensors"

example-matmul-bench: install
	idris2 $(IDRIS_FLAGS) -o matmul-bench $(EXAMPLE_SRC)/Example/MatmulBench.idr
	cp $(LIB) build/exec/matmul-bench_app/
	$(STDBUF) ./build/exec/matmul-bench $(MATMUL_BENCH_ARGS)

example-bench: install
	idris2 $(IDRIS_FLAGS) -o bench $(EXAMPLE_SRC)/Example/Bench.idr
	cp $(LIB) build/exec/bench_app/
	@# Each benchmark runs in its own process. Sharing one process across
	@# all six accumulates allocator state that nondeterministically trips
	@# the unresolved tape stale-reader bug (see TODO.md High Priority).
	@for b in supervised rnn ntm ntm-copy ntm-copy-1k ntm-recall; do \
	    ./build/exec/bench $$b || exit $$?; \
	done

$(BUILD):
	mkdir -p $(BUILD)

example-profile: install
	idris2 $(IDRIS_FLAGS) -o profile $(EXAMPLE_SRC)/Example/Profile.idr
	cp $(LIB) build/exec/profile_app/
	./build/exec/profile

example-profile-micro: install
	idris2 $(IDRIS_FLAGS) -o profile-micro $(EXAMPLE_SRC)/Example/ProfileMicro.idr
	cp $(LIB) build/exec/profile-micro_app/
	./build/exec/profile-micro

sweep: backend
	bash scripts/sweep.sh --parallel 4

sweep-quick: backend
	bash scripts/sweep.sh --parallel 4 --quick

# PyTorch reference implementation (uv manages Python)
ref-setup:
	cd packages/pytorch && uv sync --dev

bench-py:
	cd packages/pytorch && uv run python -m torch_ref.benchmark $(BENCH)

bench-compare: example-bench
	cd packages/pytorch && uv run python -m torch_ref.compare

# Build bench_ops linked against the active backend
$(BUILD)/bench_ops: $(BACKENDS_DIR)/bench_ops.c backend | $(BUILD)
	cc -o $(BUILD)/bench_ops $(BACKENDS_DIR)/bench_ops.c -L$(BUILD) -lidrisml -Wl,-rpath,$(CURDIR)/$(BUILD) -lm

# Build bench_ops for a specific backend (e.g., make bench-ops-build-tape)
bench-ops-build-%: $(BACKENDS_DIR)/bench_ops.c | $(BUILD)
	@$(MAKE) --no-print-directory BACKEND=$* backend 2>/dev/null
	cc -o $(BUILD)/bench_ops_$* $(BACKENDS_DIR)/bench_ops.c -L$(BUILD) -lidrisml -Wl,-rpath,$(CURDIR)/$(BUILD) -lm

bench-ops: $(BUILD)/bench_ops
	./$(BUILD)/bench_ops

bench-ops-py:
	cd packages/pytorch && uv run python -m torch_ref.bench_ops

# Compare all available backends vs PyTorch.
# Each iteration rebuilds libidrisml.dylib with only one backend as
# primary (BACKEND=$$b → single-element list), then copies it to a
# backend-named filename for the bench_ops binary to link against.
# Under multi-link this is a real rebuild per backend, but bench_ops is
# operator-level so we want isolated per-backend timings anyway.
bench-ops-compare:
	@for b in tape mlx torch; do \
		$(MAKE) --no-print-directory BACKEND=$$b backend 2>/dev/null || continue; \
		cp $(BUILD)/libidrisml.$(LIB_EXT) $(BUILD)/libidrisml_$$b.$(LIB_EXT); \
		cc -o $(BUILD)/bench_ops_$$b $(BACKENDS_DIR)/bench_ops.c \
			$(BUILD)/libidrisml_$$b.$(LIB_EXT) -Wl,-rpath,$(CURDIR)/$(BUILD) -lm -lc++ 2>/dev/null \
		|| true; \
	done
	cd packages/pytorch && uv run python -m torch_ref.compare_ops

ref-supervised:
	cd packages/pytorch && uv run python -m torch_ref.scripts.supervised

ref-rnn:
	cd packages/pytorch && uv run python -m torch_ref.scripts.rnn

ref-lstm:
	cd packages/pytorch && uv run python -m torch_ref.scripts.lstm

ref-gru:
	cd packages/pytorch && uv run python -m torch_ref.scripts.gru

ref-ntm-copy:
	cd packages/pytorch && uv run python -m torch_ref.scripts.ntm_copy

ref-ntm-recall:
	cd packages/pytorch && uv run python -m torch_ref.scripts.ntm_recall

ref-dnc-copy:
	cd packages/pytorch && uv run python -m torch_ref.scripts.dnc_copy

ref-dnc-recall:
	cd packages/pytorch && uv run python -m torch_ref.scripts.dnc_recall

ref-transformer:
	cd packages/pytorch && uv run python -m torch_ref.scripts.transformer

ref-gpt:
	cd packages/pytorch && uv run python -m torch_ref.scripts.gpt

ref-reinforce:
	cd packages/pytorch && uv run python -m torch_ref.scripts.reinforce

ref-a2c:
	cd packages/pytorch && uv run python -m torch_ref.scripts.a2c

ref-ppo:
	cd packages/pytorch && uv run python -m torch_ref.scripts.ppo

ref-dqn:
	cd packages/pytorch && uv run python -m torch_ref.scripts.dqn

ref-mountain-car:
	cd packages/pytorch && uv run python -m torch_ref.scripts.mountain_car

ref-mountain-car-cont:
	cd packages/pytorch && uv run python -m torch_ref.scripts.mountain_car_cont

# SAC, tabular RL, and Monte Carlo have no scripts/ wrapper — invoke
# models/*.py:__main__ directly (paired-side entry point in both cases).
ref-sac:
	cd packages/pytorch && uv run python -m torch_ref.models.sac

ref-q-learning:
	cd packages/pytorch && uv run python -m torch_ref.models.q_learning

ref-sarsa:
	cd packages/pytorch && uv run python -m torch_ref.models.sarsa

ref-frozen-lake:
	cd packages/pytorch && uv run python -m torch_ref.models.frozen_lake

ref-taxi:
	cd packages/pytorch && uv run python -m torch_ref.models.taxi

ref-monte-carlo:
	cd packages/pytorch && uv run python -m torch_ref.models.monte_carlo

test-ref ref-test:
	cd packages/pytorch && uv run pytest torch_ref/correctness/ -v

ref-lint:
	cd packages/pytorch && uv run ruff check torch_ref/ && uv run ruff format --check torch_ref/

ref-typecheck:
	cd packages/pytorch && uv run pyright torch_ref/

ref-convergence:
	cd packages/pytorch && uv run python -u -m torch_ref.scripts.convergence --task both

ref-convergence-copy:
	cd packages/pytorch && uv run python -u -m torch_ref.scripts.convergence --task copy

ref-convergence-recall:
	cd packages/pytorch && uv run python -u -m torch_ref.scripts.convergence --task recall

# CUDA test (run on Colab or Linux with CUDA GPU)
test-cuda:
	bash scripts/test_cuda_colab.sh

# Jupyter kernel (venv in packages/jupyter/.venv)
# Apple-bundled /usr/bin/python3 is 3.9 (too old for our deps), so we
# prefer a managed 3.12+. Resolution order:
#   1. uv-managed python, if uv is installed (most projects on this
#      codebase already have it via the pyproject/uv.lock pattern)
#   2. system `python3` on $PATH — falls back loudly if it's < 3.12
#
# Deliberately not falling back to `nix build nixpkgs#python3`: that
# materialises python3 in the nix store on every build regardless of
# user config (same pattern as the removed mlx fallback above).
UV_PYTHON := $(shell uv python find 2>/dev/null)
VENV_PYTHON := $(shell [ -x "$(UV_PYTHON)" ] && echo "$(UV_PYTHON)" || echo python3)
JUPYTER_VENV := packages/jupyter/.venv
JUPYTER_PIP := $(JUPYTER_VENV)/bin/pip
JUPYTER_PYTHON := $(JUPYTER_VENV)/bin/python3
JUPYTER_PYTEST := $(JUPYTER_VENV)/bin/pytest

$(JUPYTER_VENV)/bin/activate:
	$(VENV_PYTHON) -m venv $(JUPYTER_VENV)
	$(JUPYTER_PIP) install --upgrade pip setuptools >/dev/null

jupyter-install: backend check $(JUPYTER_VENV)/bin/activate
	$(JUPYTER_PIP) install -e packages/jupyter/.[dev]
	$(JUPYTER_PYTHON) -m idris_ml_kernel.install

jupyter-lab: jupyter-install
	$(JUPYTER_PIP) install -q jupyterlab
	$(JUPYTER_VENV)/bin/jupyter lab --notebook-dir=packages/jupyter/notebooks

# Jupyter kernel tests (requires backend + idris2)
test-jupyter: backend check $(JUPYTER_VENV)/bin/activate
	$(JUPYTER_PIP) install -q -e packages/jupyter/.[dev]
	cd packages/jupyter && ../../$(JUPYTER_PYTEST) tests/ -v

# Quick: just cell parser (no REPL, no backend needed)
test-jupyter-unit: $(JUPYTER_VENV)/bin/activate
	$(JUPYTER_PIP) install -q -e packages/jupyter/.[dev]
	cd packages/jupyter && ../../$(JUPYTER_PYTEST) tests/test_cell_parser.py -v

# Run all notebooks headless to check for API breakage
test-notebooks: jupyter-install
	@fail=0; \
	for nb in packages/jupyter/notebooks/tutorials/*.ipynb packages/jupyter/notebooks/models/*.ipynb; do \
		echo "--- $$nb ---"; \
		$(JUPYTER_VENV)/bin/jupyter nbconvert --execute --to notebook \
			--ExecutePreprocessor.timeout=120 "$$nb" \
			--output /tmp/test_nb_out.ipynb 2>&1 || { echo "FAIL: $$nb"; fail=1; continue; }; \
		echo "ok"; \
	done; \
	rm -f /tmp/test_nb_out.ipynb; \
	[ $$fail -eq 0 ] && echo "All notebooks passed" || { echo "Some notebooks failed"; exit 1; }

clean:
	rm -rf $(BUILD)/ttc $(BUILD)/exec
	rm -f $(BUILD)/.library-cache-stamp $(BUILD)/.backend_stamp
	rm -f $(BUILD)/libidrisml*.dylib $(BUILD)/libidrisml*.so
	rm -rf $(BUILD)/libidrisml*.dylib.dSYM
	rm -f $(BUILD)/*.o
	rm -f $(BUILD)/test_backend $(BUILD)/test_backend_debug $(BUILD)/test_safetensors \
	      $(BUILD)/test_ntm_grad $(BUILD)/test_ntm_timestep $(BUILD)/test_tape \
	      $(BUILD)/test_tensor $(BUILD)/bench_ops $(BUILD)/bench_ops_*

# Examples run on every built backend. Keep in sync with packages/idris-ml-examples/src/Example/.
# Excluded intentionally:
#   Bench, Profile — no RESULT lines (covered by bench-compare / example-profile).
EXAMPLES := example-supervised example-rnn example-lstm example-gru example-transformer example-gpt example-matmul-bench example-mnist example-seq-classify example-ntm-copy example-ntm-associative-recall example-dnc-copy example-dnc-recall example-reinforce example-q-learning example-sarsa example-monte-carlo example-frozen-lake example-taxi example-dqn example-mountain-car example-mountain-car-cont example-a2c example-ppo example-sac example-transfer
# 4-lane matrix. `mlx-gpu` is a virtual lane that builds with
# BACKEND=mlx MLX_DEVICE=gpu, exercising the F32-on-MlxGpu code path
# (per BuildConfig.idr). All other lanes build at F64.
BACKENDS := tape mlx mlx-gpu torch

# Crash-only smoke gate: every example × 3 backends, 3-10 epochs each,
# safety-net thresholds in test-examples.expect. Catches crashes / NaN /
# divergence / missing RESULT keys; does NOT require any model to learn.
# See docs/develop/testing.md for the full testing-layer overview.
test-examples:
	@fail=0; skip=""; \
	if command -v timeout >/dev/null 2>&1; then TIMEOUT_PREFIX="timeout $(EXAMPLE_TIMEOUT)"; \
	elif command -v gtimeout >/dev/null 2>&1; then TIMEOUT_PREFIX="gtimeout $(EXAMPLE_TIMEOUT)"; \
	else echo "WARNING: no timeout/gtimeout binary; examples will not be time-bounded"; TIMEOUT_PREFIX=""; fi; \
	for lane in $(BACKENDS); do \
		case "$$lane" in \
			mlx-gpu) b=mlx; lane_env="MLX_DEVICE=gpu"; expect_suffix=.mlx-gpu ;; \
			*)       b=$$lane; lane_env=""; expect_suffix="" ;; \
		esac; \
		backend_output=$$(env $$lane_env $(MAKE) --no-print-directory BACKEND=$$b backend 2>&1) || { \
			echo "--- backend $$lane: build failed, skipping its examples ---"; \
			echo "$$backend_output" | tail -20 | sed 's/^/  | /'; \
			skip="$$skip $$lane"; continue; \
		}; \
		for e in $(EXAMPLES); do \
			case " $(SKIP_EXAMPLES) " in *" $$lane:$$e "*|*" $$b:$$e "*) \
				echo "skip: $$e [$$lane] (in SKIP_EXAMPLES)"; continue ;; \
			esac; \
			echo "--- $$e [$$lane] ---"; \
			extra_args=""; \
			case "$$e" in \
				example-supervised)  extra_args="SUPERVISED_ARGS=--epochs 5" ;; \
				example-rnn)         extra_args="RNN_ARGS=--epochs 5" ;; \
				example-lstm)        extra_args="LSTM_ARGS=--epochs 5" ;; \
				example-gru)         extra_args="GRU_ARGS=--epochs 5" ;; \
				example-transformer) extra_args="TRANSFORMER_ARGS=--epochs 5" ;; \
				example-reinforce)   extra_args="REINFORCE_ARGS=--epochs 10" ;; \
				example-gpt)         extra_args="GPT_ARGS=--epochs 3" ;; \
				example-matmul-bench) extra_args="MATMUL_BENCH_ARGS=--size 1024 --iters 3" ;; \
				example-mnist)       extra_args="MNIST_ARGS=--epochs 1 --train-count 6000" ;; \
				example-seq-classify) extra_args="SEQ_ARGS=--epochs 5" ;; \
				example-dqn)         extra_args="DQN_ARGS=--epochs 10" ;; \
				example-mountain-car) extra_args="MOUNTAIN_CAR_ARGS=--epochs 5" ;; \
				example-mountain-car-cont) extra_args="MOUNTAIN_CAR_CONT_ARGS=--epochs 5" ;; \
				example-a2c)         extra_args="A2C_ARGS=--epochs 50" ;; \
				example-ppo)         extra_args="PPO_ARGS=--epochs 5" ;; \
				example-sac)         extra_args="SAC_ARGS=--epochs 100" ;; \
				example-ntm-copy)    extra_args="NTM_COPY_ARGS=--epochs 5" ;; \
				example-ntm-associative-recall) extra_args="NTM_RECALL_ARGS=--epochs 5" ;; \
				example-dnc-copy)    extra_args="DNC_COPY_ARGS=--epochs 5 --max-len 3 --batch 1" ;; \
				example-dnc-recall)  extra_args="DNC_RECALL_ARGS=--epochs 5 --max-items 2 --batch 1" ;; \
			esac; \
			t_start=$$(date +%s); \
			if [ -n "$$extra_args" ]; then \
				output=$$(env $$lane_env $$TIMEOUT_PREFIX $(MAKE) --no-print-directory BACKEND=$$b $$e "$$extra_args" 2>&1); rc=$$?; \
			else \
				output=$$(env $$lane_env $$TIMEOUT_PREFIX $(MAKE) --no-print-directory BACKEND=$$b $$e 2>&1); rc=$$?; \
			fi; \
			t_end=$$(date +%s); elapsed=$$((t_end - t_start)); \
			if [ $$elapsed -lt 60 ]; then elapsed_fmt="$${elapsed}s"; \
			elif [ $$elapsed -lt 3600 ]; then elapsed_fmt="$$((elapsed/60))m$$((elapsed%60))s"; \
			else elapsed_fmt="$$((elapsed/3600))h$$(((elapsed%3600)/60))m"; fi; \
			if [ $$rc -ne 0 ]; then \
				if [ $$rc -eq 124 ]; then \
					echo "FAIL: $$e [$$lane] timed out (>$(EXAMPLE_TIMEOUT)s) ($$elapsed_fmt)"; \
				else \
					echo "FAIL: $$e [$$lane] crashed (rc=$$rc) ($$elapsed_fmt)"; \
				fi; \
				echo "$$output" | tail -40 | sed 's/^/  | /'; \
				fail=1; continue; \
			fi; \
			result_line=$$(echo "$$output" | grep '^RESULT' | head -1); \
			if [ -z "$$result_line" ]; then \
				echo "FAIL: $$e [$$lane] -- no RESULT line ($$elapsed_fmt)"; \
				echo "$$output" | tail -40 | sed 's/^/  | /'; \
				fail=1; \
			else \
				expect_path="$$(dirname scripts/check-result.sh)/../test-examples.expect$$expect_suffix"; \
				if [ -f "test-examples.expect$$expect_suffix" ]; then \
					scripts/check-result.sh "$$e" "$$result_line" "test-examples.expect$$expect_suffix" || fail=1; \
				else \
					scripts/check-result.sh "$$e" "$$result_line" || fail=1; \
				fi; \
				echo "  ($$elapsed_fmt)"; \
			fi; \
		done; \
	done; \
	if [ -z "$$skip" ]; then \
		echo "--- example-transfer-demo (tape->mlx->torch round-trip) ---"; \
		t_start=$$(date +%s); \
		demo_out=$$($$TIMEOUT_PREFIX $(MAKE) --no-print-directory example-transfer-demo 2>&1); demo_rc=$$?; \
		t_end=$$(date +%s); elapsed=$$((t_end - t_start)); \
		if [ $$elapsed -lt 60 ]; then elapsed_fmt="$${elapsed}s"; \
		elif [ $$elapsed -lt 3600 ]; then elapsed_fmt="$$((elapsed/60))m$$((elapsed%60))s"; \
		else elapsed_fmt="$$((elapsed/3600))h$$(((elapsed%3600)/60))m"; fi; \
		if [ $$demo_rc -ne 0 ]; then \
			if [ $$demo_rc -eq 124 ]; then echo "FAIL: example-transfer-demo timed out (>$(EXAMPLE_TIMEOUT)s) ($$elapsed_fmt)"; \
			else echo "FAIL: example-transfer-demo crashed (rc=$$demo_rc) ($$elapsed_fmt)"; fi; \
			echo "$$demo_out" | tail -40 | sed 's/^/  | /'; \
			fail=1; \
		else \
			result_line=$$(echo "$$demo_out" | grep '^RESULT' | tail -1); \
			if [ -z "$$result_line" ]; then \
				echo "FAIL: example-transfer-demo -- no RESULT line ($$elapsed_fmt)"; \
				echo "$$demo_out" | tail -40 | sed 's/^/  | /'; \
				fail=1; \
			else \
				scripts/check-result.sh "example-transfer-demo" "$$result_line" || fail=1; \
				echo "  ($$elapsed_fmt)"; \
			fi; \
		fi; \
	else \
		echo "--- example-transfer-demo: skipped (requires tape+mlx+torch; skipped:$$skip) ---"; \
	fi; \
	if [ -n "$$skip" ]; then echo "Skipped backends (not installed or build failed):$$skip"; fi; \
	if [ $$fail -ne 0 ]; then echo "Some integration tests FAILED"; exit 1; fi; \
	echo "All integration tests passed."

all-backends: test-examples

# Run every example to convergence at full default epochs, single seed=42,
# tape backend, with tight thresholds from test-examples-convergence.expect.
# Hours of wall time (NTM/DNC dominate). Intended for release validation,
# not CI. See docs/develop/testing.md for the testing-layer overview.
# 4h per-example cap. DNC-copy at default 50K epochs now runs in ~1.7h on
# tape (~130ms/epoch post the 2026-05-02 tensor-handle rewrite — see
# `dnc-perf-baseline.md`). Other examples are well under this cap.
CONVERGENCE_TIMEOUT ?= 14400
CONVERGENCE_EXPECT := test-examples-convergence.expect

test-examples-convergence:
	@echo "WARNING: full-convergence runs take several hours on tape."
	@echo "         Press Ctrl-C in the next 5s to abort." && sleep 5
	@fail=0; \
	if command -v timeout >/dev/null 2>&1; then TIMEOUT_PREFIX="timeout $(CONVERGENCE_TIMEOUT)"; \
	elif command -v gtimeout >/dev/null 2>&1; then TIMEOUT_PREFIX="gtimeout $(CONVERGENCE_TIMEOUT)"; \
	else TIMEOUT_PREFIX=""; fi; \
	for e in $(EXAMPLES); do \
		echo "=== $$e ==="; \
		t_start=$$(date +%s); \
		output=$$($$TIMEOUT_PREFIX $(MAKE) --no-print-directory BACKEND=tape $$e 2>&1); rc=$$?; \
		t_end=$$(date +%s); elapsed=$$((t_end - t_start)); \
		if [ $$elapsed -lt 60 ]; then elapsed_fmt="$${elapsed}s"; \
		elif [ $$elapsed -lt 3600 ]; then elapsed_fmt="$$((elapsed/60))m$$((elapsed%60))s"; \
		else elapsed_fmt="$$((elapsed/3600))h$$(((elapsed%3600)/60))m"; fi; \
		if [ $$rc -ne 0 ]; then \
			if [ $$rc -eq 124 ]; then \
				echo "FAIL: $$e timed out (>$(CONVERGENCE_TIMEOUT)s) ($$elapsed_fmt)"; \
			else \
				echo "FAIL: $$e crashed (rc=$$rc) ($$elapsed_fmt)"; \
			fi; \
			echo "$$output" | tail -30 | sed 's/^/  | /'; \
			fail=1; continue; \
		fi; \
		result_line=$$(echo "$$output" | grep '^RESULT' | head -1); \
		if [ -z "$$result_line" ]; then \
			echo "FAIL: $$e -- no RESULT line ($$elapsed_fmt)"; \
			echo "$$output" | tail -30 | sed 's/^/  | /'; \
			fail=1; continue; \
		fi; \
		scripts/check-result.sh "$$e" "$$result_line" "$(CONVERGENCE_EXPECT)" || fail=1; \
		echo "  ($$elapsed_fmt)"; \
	done; \
	if [ $$fail -ne 0 ]; then echo "Some convergence runs FAILED"; exit 1; fi; \
	echo "All convergence runs passed."

# Run everything: Idris unit tests, C backend tests, specialized tests,
# integration tests, PyTorch reference tests (if available)
test-all:
	@echo "=== Idris unit tests ==="
	$(MAKE) test
	@echo ""
	@echo "=== Gym unit tests ==="
	$(MAKE) test-gym
	@echo ""
	@echo "=== Examples unit tests ==="
	$(MAKE) test-examples-unit
	@echo ""
	@echo "=== C backend tests ==="
	@for b in tape mlx torch; do \
		echo "--- test-backend [$$b] ---"; \
		$(MAKE) BACKEND=$$b test-backend 2>&1 && echo "" || echo "FAILED or SKIPPED: $$b"; \
	done
	@echo "=== Specialized C tests ==="
	$(MAKE) BACKEND=tape test-safetensors
	$(MAKE) BACKEND=tape test-ntm-grad
	$(MAKE) BACKEND=tape test-ntm-timestep
	@echo ""
	@echo "=== Integration tests (examples on all backends) ==="
	$(MAKE) test-examples
	@echo ""
	@if command -v uv >/dev/null 2>&1 && [ -f packages/pytorch/pyproject.toml ]; then \
		echo "=== PyTorch reference tests ==="; \
		$(MAKE) test-ref; \
	else \
		echo "=== PyTorch reference tests SKIPPED (uv not found) ==="; \
	fi
	@echo ""
	@if command -v pytest >/dev/null 2>&1 && [ -f packages/jupyter/pyproject.toml ]; then \
		echo "=== Jupyter kernel tests ==="; \
		$(MAKE) test-jupyter; \
	else \
		echo "=== Jupyter kernel tests SKIPPED (pytest or jupyter not found) ==="; \
	fi
	@echo ""
	@if [ -d packages/jupyter/.venv ] && $(JUPYTER_VENV)/bin/jupyter --version >/dev/null 2>&1; then \
		echo "=== Notebook execution tests ==="; \
		$(MAKE) test-notebooks; \
	else \
		echo "=== Notebook execution tests SKIPPED (jupyter not installed) ==="; \
	fi
	@echo ""
	@echo "=== All tests complete ==="

# Type-check notebook prelude package
check-notebook: install-core
	cd packages/idris-ml-notebook && idris2 --build idris-ml-notebook.ipkg

# Build backend + type-check all packages (default target)
check-all: check check-gym check-notebook check-examples

# Verify everything: check-all + run all tests
all: check-all test-all

.PHONY: all check-all all-backends test test-gym test-examples-unit test-all dataset-mnist dataset-tinyshakespeare test-backend test-backend-tape test-backend-mlx \
        test-backend-torch test-safetensors test-ntm-grad test-ntm-timestep \
        test-examples test-examples-convergence \
        check check-gym check-notebook check-examples install install-core install-gym install-notebook install-examples \
        example-supervised example-rnn example-lstm example-gru \
        example-ntm-copy example-ntm-associative-recall example-dnc-copy example-dnc-recall \
        example-reinforce example-q-learning example-sarsa example-monte-carlo example-frozen-lake example-taxi \
        example-dqn example-mountain-car example-mountain-car-cont example-a2c example-ppo example-sac \
        example-gpt example-gpt-full example-matmul-bench example-mnist example-seq-classify example-transformer \
        ref-gpt \
        example-transfer example-transfer-demo \
        example-bench example-profile sweep sweep-quick clean \
        backend print-torch ref-setup ref-supervised ref-rnn ref-lstm ref-gru ref-ntm-copy \
        ref-ntm-recall ref-dnc-copy ref-dnc-recall \
        ref-transformer bench-py bench-compare bench-ops bench-ops-py bench-ops-compare test-ref ref-test ref-lint \
        ref-typecheck ref-convergence ref-convergence-copy ref-convergence-recall \
        jupyter-install jupyter-lab test-jupyter test-jupyter-unit test-notebooks
