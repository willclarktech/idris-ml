UNAME := $(shell uname)
BACKEND ?= tape

# CPU core count for parallel builds. Used by recursive $(MAKE) calls
# (notably test-coverage-backend) that need to force -j even when the outer
# make was invoked without -j. macOS: hw.ncpu; Linux: nproc; fallback 4.
NPROC ?= $(shell sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)

# Per-build flag injection (Phase 0.4 coverage; future use for sanitizers,
# debug builds, etc.). Threaded through every compile + link site below.
EXTRA_CFLAGS  ?=
EXTRA_LDFLAGS ?=

# MLX stream selection at runtime, also consumed by the BuildConfig
# generation rule below — when PRIMARY=mlx and MLX_DEVICE=gpu, examples
# spell `Tensor [..] (MlxDev MGpu) F32 WithGrad` so the type-level
# claim matches what mlx actually runs (Metal GPU is float32-only per
# the f32 rewrite).
MLX_DEVICE ?= cpu

# Torch hardware selection for the BuildConfig rule. When PRIMARY=torch
# the example types resolve to `TorchDev TCpu`/`TorchDev TMps`/
# `TorchDev (TCuda 0)` based on this env var. TMps forces F32 (libtorch
# rejects F64 at MPS tensor construction); TCpu and TCuda stay at F64.
TORCH_DEVICE ?= cpu

# Per-backend dtype overrides for the BuildConfig rule. Empty (default)
# uses the per-device default in the case below.
#
# TORCH_DTYPE=BF16 overrides the torch cell to BFloat16 — halves memory
# for HF model inference (Llama-3.2-1B drops 5 GB → 2.5 GB) on a libtorch
# that ships BF16 kernel coverage on MPS. F32 stays the default until
# BF16 coverage is proven across more examples.
#
# MLX_DTYPE / TAPE_DTYPE are the same shape for the other two backends.
# Each is honored only when its PRIMARY matches (mixing knobs across
# unrelated PRIMARYs is a no-op so multi-link builds don't accidentally
# pick up another backend's setting). Use `MLX_DTYPE=F32` for mlx-cpu
# Llama inference (real 4-byte storage; the default F64 is 10 GB and
# OOMs a 16 GB VM); `TAPE_DTYPE=F32` for tape Llama (same memory math —
# tape's lingua-franca BF16/F16 are still doubles internally, so F32 is
# the only sub-F64 dtype that actually halves storage).
TORCH_DTYPE ?=
MLX_DTYPE   ?=
TAPE_DTYPE  ?=

# --- Backend selection + per-backend-set build key ---
# Defined early so downstream variables (IDRIS2_LOCAL, LIB, BACKEND_OBJS,
# stamps, …) can reference $(BUILD) / $(PRIMARY) / $(BACKEND_LIST) at
# `:=` expansion time.
#
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

# Per-backend-set build key. Distinct values of `(BACKEND, MLX_DEVICE,
# TORCH_DEVICE)` get their own `build/<KEY>/` tree (ttc cache, installed
# library prefix, dylib, example executables, stamps). Each set's warm
# cache survives backend-set switches indefinitely; switching between
# `BACKEND=tape make test` and `BACKEND=torch TORCH_DEVICE=mps make
# example-hf-llama-inference` no longer triggers full re-elaboration.
#
# The key includes the backend-list ordering because PRIMARY decides
# multi-link symbol resolution: `tape,torch` and `torch,tape` produce
# different dylibs and would clobber each other under a PRIMARY-only key.
# It includes MLX_DEVICE / TORCH_DEVICE because `BuildConfig.idr` and
# `TestConfig.idr` content depend on them (F32 on mlx-gpu / torch-mps,
# F64 elsewhere — see those files' .in templates for the matrix).
#
# See `docs/develop/design-decisions.md` "Per-backend-set build cache".
#
# Auto-F32 for HF inference targets: the 1.24B-param Llama at F64
# (~10 GB) + per-forward intermediates don't fit comfortably on a
# 16 GB VM. The other Hf models (BERT-tiny / GPT-2-small) are
# small enough that F64 is fine; the heavy-LM examples + their
# roundtrip gates are the only ones that need the override. Set
# TORCH_DTYPE/MLX_DTYPE/TAPE_DTYPE to F32 ONLY IF the user hasn't
# already specified them on the command line (`?=` semantics
# inlined since this is `:=` parse-time). User-side override:
# `TORCH_DTYPE=F64 make test-e2e-hf-llama-roundtrip` keeps F64 (e.g.
# for numerical bisection vs the F64 oracle path).
# Every HF model target — Llama / BitNet need F32 for memory
# (1.24B / 2B params at F64 don't fit on a 16 GB VM); BERT-tiny /
# GPT-2-small don't NEED F32 for memory but the convention is "no
# HF model runs at F64". The HF on-disk reference weights are
# BF16, oracle generators cast to F32 — running Idris at F64
# means we're MORE precise than the comparison oracle, which is
# pure waste. F32 is the canonical HF inference dtype.
HF_GOALS := example-hf-bert-inference \
                  example-hf-bitnet-inference \
                  example-hf-gpt2-inference \
                  example-hf-llama-inference \
                  test-e2e-hf-bert-roundtrip \
                  test-e2e-hf-bitnet-roundtrip \
                  test-e2e-hf-gpt2-roundtrip \
                  test-e2e-hf-llama-roundtrip \
                  test-e2e-hf-llama-generate-roundtrip \
                  test-e2e-transformers-oracle-bert \
                  test-e2e-transformers-oracle-gpt2 \
                  test-e2e-transformers-oracle-llama \
                  test-e2e-transformers-oracle-llama-generate
ifneq ($(filter $(HF_GOALS),$(MAKECMDGOALS)),)
  # `?=` not used here — Make's `?=` treats an exported-empty env var
  # ("" from the shell) as already-set and skips the default. The HF
  # heavy targets need F32 unless the user EXPLICITLY set a non-empty
  # value at the command line. Detect "no value" via filter against
  # empty string. Set only the dtype for the active PRIMARY (BACKEND's
  # first item) — setting all three would balloon the BUILD_KEY with
  # `-tdtF32-mdtF32-tpdtF32` suffixes when only one matters.
  HF_PRIMARY := $(firstword $(subst $(comma), ,$(BACKEND)))
  ifeq ($(HF_PRIMARY),torch)
    ifeq ($(strip $(TORCH_DTYPE)),)
      TORCH_DTYPE := F32
    endif
  else ifeq ($(HF_PRIMARY),mlx)
    ifeq ($(strip $(MLX_DTYPE)),)
      MLX_DTYPE := F32
    endif
  else ifeq ($(HF_PRIMARY),tape)
    ifeq ($(strip $(TAPE_DTYPE)),)
      TAPE_DTYPE := F32
    endif
  endif
endif

BUILD_KEY := $(subst $(comma),-,$(strip $(BACKEND)))-mlx$(MLX_DEVICE)-torch$(TORCH_DEVICE)$(if $(TORCH_DTYPE),-tdt$(TORCH_DTYPE),)$(if $(MLX_DTYPE),-mdt$(MLX_DTYPE),)$(if $(TAPE_DTYPE),-tpdt$(TAPE_DTYPE),)
BUILD := build/$(BUILD_KEY)

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
TEST_SRC := packages/idris-ml/src
BACKENDS_DIR := packages/backends

# Local package install prefix (writable, avoids polluting system Idris2).
# Per-backend-set (under `$(BUILD)`) so each set has its own installed
# library tree — `idris-ml-0`'s installed `.ttc` interface hashes differ
# across backend sets (they embed the `HwConfig.idr` / `HwDevices.idr`
# linkage instances), so they cannot share a prefix.
IDRIS2_LOCAL := $(CURDIR)/$(BUILD)/idris2-prefix

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

# Idris flags for example/test builds (use installed packages). `--build-dir`
# routes ttc + exec output under `$(BUILD)` so each backend set has its own
# warm cache for example/test compilation, mirroring the per-set install tree.
IDRIS_FLAGS := --build-dir $(BUILD) --source-dir $(EXAMPLE_SRC) -p contrib -p idris-ml -p idris-gym -p idris-transformers

# Library source files — any change invalidates the per-set ttc caches.
# Idris 2's interface-hash dependency tracking doesn't invalidate downstream
# TTCs when a module's public interface is unchanged but a where-clause body
# (or other inlined internal) changed. Single-file `idris2 -o <name>` example
# builds then reuse stale `$(BUILD)/ttc-*/.../Example/*.ttc` with old inlined
# code baked in. Wiping the per-set ttc when any library source is newer than
# this stamp forces a clean rebuild. See docs/develop/gotchas.md.
#
# The generated `.idr` files (HwConfig, HwDevices) get *rewritten on backend-set
# switch* — their mtime bumps even when their per-set content is stable. Including
# them here would defeat the per-set ttc cache: `tape → torch → tape` would
# rewrite HwConfig.idr (set-A → set-B), then rewrite back (set-B → set-A), then
# the next tape install would see the stamp older than HwConfig.idr and wipe
# `build/tape-…/ttc-*`. Their own staleness tracking via `--build-dir`-keyed
# ttc + interface-hash check is sufficient.
LIBRARY_SRCS := $(filter-out packages/idris-ml/src/HwConfig.idr packages/idris-ml/src/HwDevices.idr, \
                  $(shell find packages/idris-ml/src packages/idris-gym/src packages/idris-transformers/src -name '*.idr' 2>/dev/null)) \
                packages/idris-ml-examples/src/Generate.idr

$(BUILD)/.library-cache-stamp: $(LIBRARY_SRCS)
	@echo "[$(BUILD_KEY)] Library source changed — invalidating ttc caches"
	@rm -rf $(BUILD)/ttc-*
	@mkdir -p $(BUILD)
	@touch $@

# Per-backend property tables. Common compile flags (`-O2 -fPIC
# -include rename_<b>.h`) are applied by the per-backend rule below;
# `<b>_CFLAGS` adds whatever else that backend's compile needs
# (include paths, C++ std). `<b>_LDFLAGS_<UNAME>` is per-platform.

# Tape has no monolithic backend_tape.{c,cpp} — every TU lives under
# backend_tape/. The per-backend compile rule's foreach skips tape;
# its .o objects come from BACKEND_TAPE_OBJS instead.
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
  # CWD at runtime (the matmul-bench wrapper chdir's into $(BUILD)/exec/...).
  MLX_INC := $(abspath $(MLX_SITE)/include)
  MLX_LIB := $(abspath $(MLX_SITE)/lib)
endif

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

# Primary backend's rename header. Drives the shared-source rename:
# `safetensors.c` and `mnist.c` are compiled once with this header so
# their internal tensor calls resolve to the primary's suffixed defs
# (and they export the primary's suffixed `param_save_<p>` /
# `mnist_get_image_<p>` symbols, which the corresponding device
# instance methods call). cJSON.c and shared_utils.c are pure-C with
# no tensor surface, so they compile without it.
#
# The former link-time unified-name alias machinery
# (`-Wl,-alias_list` on macOS / `-Wl,--defsym=` on Linux) was deleted
# once every Idris `%foreign` migrated off unified names into
# per-instance `UserDevice*` methods bound to the suffixed symbols
# directly. A repo-wide scan now finds zero unified-name references to
# per-backend-renamed C symbols, so nothing needs the alias.
BACKEND_RENAME_H := $(BACKENDS_DIR)/rename_$(PRIMARY).h

# backend_tape/** modular sources. Each .c compiles to its own .o
# via the per-TU rule below; the dylib link picks them all up
# directly (tape has no monolithic backend_tape.{c,cpp} TU).
BACKEND_TAPE_HEADERS := $(shell find $(BACKENDS_DIR)/backend_tape -name '*.h' 2>/dev/null)
# Exclude colocated test_*.c — they ride the Criterion test build (see
# CRITERION_BACKEND_TEST_SRCS below), not the dylib build.
BACKEND_TAPE_SRCS    := $(shell find $(BACKENDS_DIR)/backend_tape -name '*.c' ! -name 'test_*.c' 2>/dev/null)
BACKEND_TAPE_OBJS    := $(patsubst $(BACKENDS_DIR)/backend_tape/%.c,$(BUILD)/backend_tape/%.o,$(BACKEND_TAPE_SRCS))

# backend_torch/** and backend_mlx/** modular sources. Each .cpp compiles
# to its own .o via the per-TU rule below; the monolithic
# backend_<b>.cpp keeps shrinking as ops migrate into the tree. Both
# trees grow incrementally during Phase 6 — empty tree on day zero is
# fine (find returns nothing, OBJS is empty, link sees only the monolith).
BACKEND_TORCH_HEADERS := $(shell find $(BACKENDS_DIR)/backend_torch -name '*.h' 2>/dev/null)
BACKEND_TORCH_SRCS    := $(shell find $(BACKENDS_DIR)/backend_torch -name '*.cpp' 2>/dev/null)
BACKEND_TORCH_OBJS    := $(patsubst $(BACKENDS_DIR)/backend_torch/%.cpp,$(BUILD)/backend_torch/%.o,$(BACKEND_TORCH_SRCS))

BACKEND_MLX_HEADERS := $(shell find $(BACKENDS_DIR)/backend_mlx -name '*.h' 2>/dev/null)
BACKEND_MLX_SRCS    := $(shell find $(BACKENDS_DIR)/backend_mlx -name '*.cpp' 2>/dev/null)
BACKEND_MLX_OBJS    := $(patsubst $(BACKENDS_DIR)/backend_mlx/%.cpp,$(BUILD)/backend_mlx/%.o,$(BACKEND_MLX_SRCS))

# shared/training/** sources + headers (the shared-port lift). Backend
# adapters under backend_<b>/training/adapter.<c|cpp> #include the
# shared header via relative path; per-TU compile picks the dependency
# up via the implicit include scan, but list the headers explicitly so
# changes to the port surface re-build dependent .o files.
SHARED_TRAINING_HEADERS := $(shell find $(BACKENDS_DIR)/shared -name '*.h' 2>/dev/null)
SHARED_TRAINING_SRCS    := $(shell find $(BACKENDS_DIR)/shared -name '*.c' 2>/dev/null)

# Per-TU opt-in lists: which backends compile + link each shared/training/
# TU. Backends NOT in a list keep their own monolithic in-file impl
# (delete it before opting in or the link sees duplicate symbols). The
# four shared TUs adopt independently so a backend can incrementally
# migrate — e.g. torch joins param_registry first, then ffi_shims, etc.
# Each list intersects with BACKEND_LIST so the build only considers
# backends actually being linked.
SHARED_BACKENDS_param_registry := tape torch mlx
SHARED_BACKENDS_optimizer      := tape
SHARED_BACKENDS_ffi_shims      := tape torch mlx
SHARED_BACKENDS_dtype_streamed := tape torch
SHARED_BACKENDS_profiler       :=
# Union (used to gate the adapter compile only — not the per-TU compile).
TRAINING_ADAPTER_BACKENDS := $(sort $(SHARED_BACKENDS_param_registry) \
                                    $(SHARED_BACKENDS_optimizer) \
                                    $(SHARED_BACKENDS_ffi_shims) \
                                    $(SHARED_BACKENDS_dtype_streamed))

# Per-backend object outputs. All three backends now live entirely under
# their backend_<b>/ trees — no `backend_<b>.{c,cpp}` monoliths to compile.
# Backends that haven't been modularized yet would fall through this
# filter-out and pick up a monolithic `backend_<b>.o` via the compile
# rule below; today the list is empty.
BACKEND_OBJS := $(foreach b,$(filter-out tape torch mlx,$(BACKEND_LIST)),$(BUILD)/backend_$(b).o)

# Pull in each backend's modular .o set when that backend is in BACKEND_LIST.
ifneq ($(filter tape,$(BACKEND_LIST)),)
  BACKEND_OBJS += $(BACKEND_TAPE_OBJS)
endif
ifneq ($(filter torch,$(BACKEND_LIST)),)
  BACKEND_OBJS += $(BACKEND_TORCH_OBJS)
endif
ifneq ($(filter mlx,$(BACKEND_LIST)),)
  BACKEND_OBJS += $(BACKEND_MLX_OBJS)
endif

# Per-backend compile template — generates a rule for each backend's
# backend_<b>.o file. Each backend uses its own CC + CFLAGS + rename
# header; the union of all per-backend LDFLAGS gets passed to the
# final link.
# `core/elementwise/_kernels.inc` (the X-macro stamped elementwise kernel
# bodies) is included twice by `core/elementwise/_dispatch.c` (once for
# F64, once for F32). The per-TU compile rule below picks it up via the
# implicit include scan; no explicit dependency needed since BACKEND_TAPE_HEADERS
# covers the `backend_tape/**` headers and the per-TU rule re-walks the
# include graph.

# Per-TU compile for backend_tape/**/*.c. Force-includes the rename
# header so every symbol gets the tape suffix at link time. Compile
# only when tape is in BACKEND_LIST — torch / mlx don't need tape's
# internals, and their builds would needlessly compile + link the
# tape TUs otherwise.
$(BUILD)/backend_tape/%.o: $(BACKENDS_DIR)/backend_tape/%.c $(BACKEND_TAPE_HEADERS) $(SHARED_TRAINING_HEADERS) $(BACKENDS_DIR)/rename_tape.h | $(BUILD)
	@mkdir -p $(dir $@)
	cc -O2 -fPIC $(EXTRA_CFLAGS) $(tape_CFLAGS) -include $(BACKENDS_DIR)/rename_tape.h -c -o $@ $<

# Per-TU compile for backend_torch/**/*.cpp and backend_mlx/**/*.cpp.
# Mirrors tape's pattern but uses each backend's C++ compiler + CFLAGS
# (incl. libtorch / mlx include paths). Force-includes the rename
# header so every symbol gets the backend suffix at link time. Rules
# defined unconditionally (only fire if BACKEND_<b>_OBJS pulls them in).
# Precompiled header for torch — `<torch/torch.h>` is ~30K lines of
# templates and parsing it 90× per cold build dominates the wall.
# Build the PCH once into $(BUILD)/torch_pch.gch with the same flags
# as the per-TU compile, then `-include-pch` it from every TU below.
# PCH lives in $(BUILD)/ so coverage and normal builds get their own
# (clang rejects PCHs whose flags don't match the consuming TU).
$(BUILD)/torch_pch.gch: $(BACKENDS_DIR)/backend_torch/torch_pch.h | $(BUILD)
	$(torch_CC) -O2 -fPIC $(EXTRA_CFLAGS) $(torch_CFLAGS) -x c++-header -c -o $@ $<

$(BUILD)/backend_torch/%.o: $(BACKENDS_DIR)/backend_torch/%.cpp $(BACKEND_TORCH_HEADERS) $(SHARED_TRAINING_HEADERS) $(BACKENDS_DIR)/rename_torch.h $(BUILD)/torch_pch.gch | $(BUILD)
	@mkdir -p $(dir $@)
	$(torch_CC) -O2 -fPIC $(EXTRA_CFLAGS) $(torch_CFLAGS) -include-pch $(BUILD)/torch_pch.gch -include $(BACKENDS_DIR)/rename_torch.h -c -o $@ $<

$(BUILD)/backend_mlx/%.o: $(BACKENDS_DIR)/backend_mlx/%.cpp $(BACKEND_MLX_HEADERS) $(SHARED_TRAINING_HEADERS) $(BACKENDS_DIR)/rename_mlx.h | $(BUILD)
	@mkdir -p $(dir $@)
	$(mlx_CC) -O2 -fPIC $(EXTRA_CFLAGS) $(mlx_CFLAGS) -include $(BACKENDS_DIR)/rename_mlx.h -c -o $@ $<

# Per-backend compile rule for shared/training/*.c. One .o per (backend,
# TU) pair with that backend's rename header (so `param_register`
# becomes `param_register_<b>` etc. and multi-link doesn't collide).
# Output lives at build/shared_training_<b>/<file>.o. The compile rule
# is generated for every backend in BACKEND_LIST (so the .o paths exist
# regardless of which TUs each backend actually compiles); BACKEND_OBJS
# only links the specific (backend, TU) pairs each SHARED_BACKENDS_<tu>
# list selects.
define shared_training_compile_rule
$(BUILD)/shared_training_$(1)/%.o: $(BACKENDS_DIR)/shared/training/%.c $(SHARED_TRAINING_HEADERS) $(BACKENDS_DIR)/backend.h $(BACKENDS_DIR)/rename_$(1).h | $(BUILD)
	@mkdir -p $$(dir $$@)
	$($(1)_CC) -O2 -fPIC $$(EXTRA_CFLAGS) $($(1)_CFLAGS) -include $(BACKENDS_DIR)/rename_$(1).h -c -o $$@ $$<
endef

$(foreach b,$(BACKEND_LIST),$(eval $(call shared_training_compile_rule,$(b))))

# Per-TU object selection. For each shared/training/<TU>.c, link the .o
# variant for every backend in its SHARED_BACKENDS_<tu> list (intersected
# with BACKEND_LIST so non-active backends don't pull objects).
define add_shared_training_obj
ifneq ($$(filter $(2),$(BACKEND_LIST)),)
BACKEND_OBJS += $(BUILD)/shared_training_$(2)/$(1).o
endif
endef

$(foreach b,$(SHARED_BACKENDS_param_registry),$(eval $(call add_shared_training_obj,param_registry,$(b))))
$(foreach b,$(SHARED_BACKENDS_optimizer),$(eval $(call add_shared_training_obj,optimizer,$(b))))
$(foreach b,$(SHARED_BACKENDS_ffi_shims),$(eval $(call add_shared_training_obj,ffi_shims,$(b))))
$(foreach b,$(SHARED_BACKENDS_dtype_streamed),$(eval $(call add_shared_training_obj,dtype_streamed,$(b))))

define backend_compile_rule
$(BUILD)/backend_$(1).o: $($(1)_SRC) $(BACKENDS_DIR)/backend.h $(BACKENDS_DIR)/rename_$(1).h $(BACKEND_TAPE_HEADERS) $(BACKEND_TORCH_HEADERS) $(BACKEND_MLX_HEADERS) $(SHARED_TRAINING_HEADERS) | $(BUILD)
	$($(1)_CC) -O2 -fPIC $(EXTRA_CFLAGS) $($(1)_CFLAGS) -include $(BACKENDS_DIR)/rename_$(1).h -c -o $$@ $$<
endef

$(foreach b,$(filter-out tape torch mlx,$(BACKEND_LIST)),$(eval $(call backend_compile_rule,$(b))))

# ---------------------------------------------------------------------
# Vendored deps — fetched on demand into a gitignored directory.
#
# cJSON (Dave Gamble, MIT) is used only by safetensors.c. The pinned
# v1.7.18 source previously sat in-tree at packages/backends/cJSON.{c,h}
# (~3.1k lines committed); now fetched into vendored/cJSON/ on first
# build with SHA256 verify. The file rules below only run when the
# files are missing, so offline rebuilds reuse the cached copy.
# ---------------------------------------------------------------------
VENDORED_DIR    := vendored
CJSON_VERSION   := v1.7.18
CJSON_DIR       := $(VENDORED_DIR)/cJSON
CJSON_C         := $(CJSON_DIR)/cJSON.c
CJSON_H         := $(CJSON_DIR)/cJSON.h
CJSON_URL_BASE  := https://raw.githubusercontent.com/DaveGamble/cJSON/$(CJSON_VERSION)
CJSON_C_SHA256  := 75c51de8fa40ac9d7a99319c6330719bd692eb81c0a869265f3d4c682533f9b9
CJSON_H_SHA256  := 0578cc29132912edbc88f83207a8fc76e5db3db0605497e909a9384ef3cc474b

$(CJSON_DIR):
	mkdir -p $@

$(CJSON_C): | $(CJSON_DIR)
	@echo "[vendor] fetching cJSON $(CJSON_VERSION) cJSON.c"
	@curl -fsSL -o $@ $(CJSON_URL_BASE)/cJSON.c
	@echo "$(CJSON_C_SHA256)  $@" | shasum -a 256 -c -

$(CJSON_H): | $(CJSON_DIR)
	@echo "[vendor] fetching cJSON $(CJSON_VERSION) cJSON.h"
	@curl -fsSL -o $@ $(CJSON_URL_BASE)/cJSON.h
	@echo "$(CJSON_H_SHA256)  $@" | shasum -a 256 -c -

vendor-deps: $(CJSON_C) $(CJSON_H)
.PHONY: vendor-deps

# Shared C sources (serialization, JSON, MNIST data) compiled with the
# PRIMARY backend's rename header so their cross-TU references match
# the primary's suffixed defs (other backends' suffixed defs are
# reachable but not by these shared TUs). cJSON and shared_utils.c
# are pure-C (no tensor surface) so they stay backend-agnostic — no
# rename header. shared_utils.c hosts the `index_array_*` +
# `get_rss_mb` symbols (intentionally unified — they don't dispatch
# per backend).
SHARED_OBJ := $(BUILD)/safetensors_$(PRIMARY).o $(BUILD)/cJSON.o $(BUILD)/mnist_$(PRIMARY).o $(BUILD)/shared_utils.o

$(BUILD)/safetensors_$(PRIMARY).o: $(BACKENDS_DIR)/safetensors.c $(BACKENDS_DIR)/backend.h $(CJSON_H) $(BACKEND_RENAME_H) | $(BUILD)
	cc -O2 -fPIC $(EXTRA_CFLAGS) -include $(BACKEND_RENAME_H) -I$(CJSON_DIR) -c -o $@ $<

$(BUILD)/cJSON.o: $(CJSON_C) $(CJSON_H) | $(BUILD)
	cc -O2 -fPIC $(EXTRA_CFLAGS) -c -o $@ $<

$(BUILD)/mnist_$(PRIMARY).o: $(BACKENDS_DIR)/mnist.c $(BACKENDS_DIR)/backend.h $(BACKEND_RENAME_H) | $(BUILD)
	cc -O2 -fPIC $(EXTRA_CFLAGS) -include $(BACKEND_RENAME_H) -c -o $@ $<

$(BUILD)/shared_utils.o: $(BACKENDS_DIR)/shared_utils.c $(BACKENDS_DIR)/shared_utils.h | $(BUILD)
	cc -O2 -fPIC $(EXTRA_CFLAGS) -c -o $@ $<

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

# Stamp + generated source for the example device/dtype selection.
# The Selection matrix lives in BuildConfig.idr.in's module docstring;
# this rule observes PRIMARY + MLX_DEVICE + TORCH_DEVICE and emits the
# right (ExampleDevice, ExampleDType) into BuildConfig.idr via sed.
# The stamp records the active tuple so the generation step only
# writes the source file when the config actually changes (avoiding
# TTC churn on no-op rebuilds), mirror of the .backend-stamp pattern.
BUILDCONFIG_KEY := $(PRIMARY):$(MLX_DEVICE):$(TORCH_DEVICE)
BUILDCONFIG_IDR := packages/idris-ml-examples/src/BuildConfig.idr
BUILDCONFIG_IN  := packages/idris-ml-examples/src/BuildConfig.idr.in

# Generated `Linked` instances for the compiled-in backends. Unlike
# BuildConfig (one example device/dtype cell), HwConfig emits a variable
# number of instance blocks — one per backend in BACKEND_LIST — so the
# recipe appends them to the .in header rather than sed-substituting.
# Keyed on the whole BACKEND list. Lives in the core library (the Device
# barrel re-exports it); git-ignored, regenerated each build.
HWCONFIG_KEY := $(BACKEND)
HWCONFIG_IDR := packages/idris-ml/src/HwConfig.idr
HWCONFIG_IN  := packages/idris-ml/src/HwConfig.idr.in

# Generated `builtinDevices : List SomeDevice` — the value-level mirror of
# HwConfig's `Linked` instances (one `someDevice` candidate per linked
# backend's admissible (device, dtype) cells). Lives downstream of `Tensor`
# (where `someDevice` is defined), unlike HwConfig which the Device barrel
# re-exports upstream. Keyed on the BACKEND list; git-ignored, regenerated.
HWDEVICES_IDR := packages/idris-ml/src/HwDevices.idr
HWDEVICES_IN  := packages/idris-ml/src/HwDevices.idr.in

# Generated `TestDevice` / `TestDType` for the Idris unit test suite. Same
# template trick as BuildConfig (one cell, sed-substituted from the active
# PRIMARY × hw-device envs); lives in the test sourcedir (now colocated
# under src/Test/ alongside the rest of the test files — dual-ipkg pattern,
# see docs/develop/testing.md). Keyed on the same tuple.
TESTCONFIG_IDR := packages/idris-ml/src/Test/Config.idr
TESTCONFIG_IN  := packages/idris-ml/src/Test/Config.idr.in

# Always-touch the stamp so it's at least as fresh as the current `make`
# invocation, even when content matches. Without the trailing touch, a
# *different* backend set rewriting BuildConfig.idr (mtime bumps) would
# leave this set's stamp older than the (now-other-content) BuildConfig.idr,
# Make would consider BuildConfig.idr "fresh" relative to its dep, and skip
# the recipe — leaving torch content on disk during a tape build. The
# recipe for the generated `.idr` is cmp-then-mv, so always-running is
# essentially free when content matches.
$(BUILD)/.buildconfig-stamp: FORCE | $(BUILD)
	@if [ "$$(cat $@ 2>/dev/null)" != "$(BUILDCONFIG_KEY)" ]; then echo "$(BUILDCONFIG_KEY)" > $@; else touch $@; fi

# Write-if-different: render to `.tmp`, then `cmp` against the existing
# generated file and `mv` only when content actually differs. Within a
# single set's reruns this keeps the .idr's mtime stable across re-makes;
# across sets it correctly bumps mtime when the (device, dtype) changes.
# The per-set ttc cache then absorbs the cascade — the .idr re-elabs but
# downstream modules with matching interface hashes don't.
$(BUILDCONFIG_IDR): $(BUILDCONFIG_IN) $(BUILD)/.buildconfig-stamp
	@case "$(PRIMARY)/$(MLX_DEVICE)/$(TORCH_DEVICE)" in \
		mlx/gpu/*)    DEVICE="MlxDev MGpu";       DTYPE="F32" ;; \
		mlx/cpu/*)    DEVICE="MlxDev MCpu";       DTYPE="F64" ;; \
		torch/*/mps)  DEVICE="TorchDev TMps";     DTYPE="F32" ;; \
		torch/*/cuda) DEVICE="TorchDev (TCuda 0)"; DTYPE="F64" ;; \
		torch/*/*)    DEVICE="TorchDev TCpu";     DTYPE="F64" ;; \
		tape/*/*)     DEVICE="TapeDev";           DTYPE="F64" ;; \
		*)            DEVICE="TapeDev";           DTYPE="F64" ;; \
	esac; \
	if [ -n "$(TORCH_DTYPE)" ] && [ "$(PRIMARY)" = "torch" ]; then DTYPE="$(TORCH_DTYPE)"; fi; \
	if [ -n "$(MLX_DTYPE)"   ] && [ "$(PRIMARY)" = "mlx"   ]; then DTYPE="$(MLX_DTYPE)";   fi; \
	if [ -n "$(TAPE_DTYPE)"  ] && [ "$(PRIMARY)" = "tape"  ]; then DTYPE="$(TAPE_DTYPE)";  fi; \
	sed "s|@DEVICE@|$$DEVICE|g; s|@DTYPE@|$$DTYPE|g" $< > $@.tmp; \
	if cmp -s $@.tmp $@ 2>/dev/null; then rm $@.tmp; else mv $@.tmp $@; fi
	@echo "[BuildConfig] PRIMARY=$(PRIMARY) MLX_DEVICE=$(MLX_DEVICE) TORCH_DEVICE=$(TORCH_DEVICE) TORCH_DTYPE=$(TORCH_DTYPE) MLX_DTYPE=$(MLX_DTYPE) TAPE_DTYPE=$(TAPE_DTYPE) → $$(awk -F' = ' '/^ExampleDevice = / { print $$2; exit }' $@) / $$(awk -F' = ' '/^ExampleDType = / { print $$2; exit }' $@)"

$(TESTCONFIG_IDR): $(TESTCONFIG_IN) $(BUILD)/.buildconfig-stamp
	@case "$(PRIMARY)/$(MLX_DEVICE)/$(TORCH_DEVICE)" in \
		mlx/gpu/*)    DEVICE="MlxDev MGpu";       DTYPE="F32" ;; \
		mlx/cpu/*)    DEVICE="MlxDev MCpu";       DTYPE="F64" ;; \
		torch/*/mps)  DEVICE="TorchDev TMps";     DTYPE="F32" ;; \
		torch/*/cuda) DEVICE="TorchDev (TCuda 0)"; DTYPE="F64" ;; \
		torch/*/*)    DEVICE="TorchDev TCpu";     DTYPE="F64" ;; \
		tape/*/*)     DEVICE="TapeDev";           DTYPE="F64" ;; \
		*)            DEVICE="TapeDev";           DTYPE="F64" ;; \
	esac; \
	if [ -n "$(TORCH_DTYPE)" ] && [ "$(PRIMARY)" = "torch" ]; then DTYPE="$(TORCH_DTYPE)"; fi; \
	if [ -n "$(MLX_DTYPE)"   ] && [ "$(PRIMARY)" = "mlx"   ]; then DTYPE="$(MLX_DTYPE)";   fi; \
	if [ -n "$(TAPE_DTYPE)"  ] && [ "$(PRIMARY)" = "tape"  ]; then DTYPE="$(TAPE_DTYPE)";  fi; \
	sed "s|@DEVICE@|$$DEVICE|g; s|@DTYPE@|$$DTYPE|g" $< > $@.tmp; \
	if cmp -s $@.tmp $@ 2>/dev/null; then rm $@.tmp; else mv $@.tmp $@; fi
	@echo "[TestConfig] PRIMARY=$(PRIMARY) MLX_DEVICE=$(MLX_DEVICE) TORCH_DEVICE=$(TORCH_DEVICE) TORCH_DTYPE=$(TORCH_DTYPE) MLX_DTYPE=$(MLX_DTYPE) TAPE_DTYPE=$(TAPE_DTYPE) → $$(awk -F' = ' '/^TestDevice = / { print $$2; exit }' $@) / $$(awk -F' = ' '/^TestDType = / { print $$2; exit }' $@)"

# Same always-touch logic as the .buildconfig-stamp recipe above; see
# the comment there for why a content-equal stamp still needs an mtime
# bump when other backend sets share the generated `.idr` file paths.
$(BUILD)/.hwconfig-stamp: FORCE | $(BUILD)
	@if [ "$$(cat $@ 2>/dev/null)" != "$(HWCONFIG_KEY)" ]; then echo "$(HWCONFIG_KEY)" > $@; else touch $@; fi

# Emit one `Linked` instance per backend in BACKEND_LIST, appended to the
# .in header. Per-backend linkage admits every hardware variant of that
# backend (runtime presence is the EAFP concern, not this gate).
# Write-if-different (see BuildConfig rule above for rationale).
$(HWCONFIG_IDR): $(HWCONFIG_IN) $(BUILD)/.hwconfig-stamp
	@{ cat $(HWCONFIG_IN); \
	   for b in $(BACKEND_LIST); do \
	     case $$b in \
	       tape)  printf 'public export\nLinked TapeDev where\n\n' ;; \
	       torch) printf 'public export\n{hw : TorchHwDev} -> Linked (TorchDev hw) where\n\n' ;; \
	       mlx)   printf 'public export\n{s : MlxStream} -> Linked (MlxDev s) where\n\n' ;; \
	     esac; \
	   done; \
	 } > $@.tmp
	@if cmp -s $@.tmp $@ 2>/dev/null; then rm $@.tmp; else mv $@.tmp $@; fi
	@echo "[HwConfig] BACKEND=$(BACKEND) → Linked instances for: $(BACKEND_LIST)"

# Emit `builtinDevices` as `[] ++ <per-backend candidate lists>`. Seeding
# with `[]` keeps every backend fragment a uniform `++ [...]`, so a
# tape-only build is `[] ++ [TapeDev/F64]` and the empty BACKEND case is a
# well-typed `[]`. Each `someDevice {d} {dt}` resolves its Linked /
# Compatible / HardwareClassed / UserDeviceTape constraints from the
# instances brought in via `import Device` / `import Tensor`. torch lists
# all three hw variants (TCpu/TMps/TCuda 0) — EAFP filters to what's
# present (multi-GPU `TCuda n` enumeration via cuda_device_count is a
# separate follow-up).
$(HWDEVICES_IDR): $(HWDEVICES_IN) $(BUILD)/.hwconfig-stamp
	@{ cat $(HWDEVICES_IN); \
	   printf 'public export\nbuiltinDevices : List SomeDevice\nbuiltinDevices = []\n'; \
	   for b in $(BACKEND_LIST); do \
	     case $$b in \
	       tape)  printf '  ++ [someDevice {d = TapeDev} {dt = F64}]\n' ;; \
	       torch) printf '  ++ [ someDevice {d = TorchDev TCpu} {dt = F64}\n     , someDevice {d = TorchDev TMps} {dt = F32}\n     , someDevice {d = TorchDev (TCuda 0)} {dt = F64} ]\n' ;; \
	       mlx)   printf '  ++ [ someDevice {d = MlxDev MCpu} {dt = F64}\n     , someDevice {d = MlxDev MGpu} {dt = F32} ]\n' ;; \
	     esac; \
	   done; \
	 } > $@.tmp
	@if cmp -s $@.tmp $@ 2>/dev/null; then rm $@.tmp; else mv $@.tmp $@; fi
	@echo "[HwDevices] BACKEND=$(BACKEND) → builtinDevices for: $(BACKEND_LIST)"

# Final link: all listed backends' .o + shared objects (primary's
# suffix). One dylib, no symlink. Every symbol is reached by its
# suffixed name from the per-instance UserDevice methods — no aliases.
$(LIB): $(BACKEND_OBJS) $(SHARED_OBJ) $(BUILD)/.backend-stamp | $(BUILD)
	$(LINK_CC) -O2 -shared $(EXTRA_LDFLAGS) -o $@ $(BACKEND_OBJS) $(SHARED_OBJ) $(BACKEND_LDFLAGS)

# Datasets: file-as-target so Make skips the fetch when the data is
# already on disk (same pattern as HF_MODELS_DIR's HF safetensors).
# Sentinel files anchor the recipe — `dataset_mnist.sh` writes 4
# files in one shot; using the first as the Make target is enough
# to gate the recipe.
TINYSHAKESPEARE_FILE := data/tinyshakespeare/input.txt
MNIST_SENTINEL       := data/mnist/train-images-idx3-ubyte

$(TINYSHAKESPEARE_FILE):
	bash scripts/dataset_tinyshakespeare.sh

$(MNIST_SENTINEL):
	bash scripts/dataset_mnist.sh

# Convenience phony aliases preserving the public `make dataset-*`
# interface. Existing CI / docs / users referencing these names keep
# working; they just no-op when the data is already on disk.
dataset-mnist: $(MNIST_SENTINEL)

# Download tinyshakespeare corpus (~1 MB, 65-char vocab) for the GPT
# convergence run. Smoke gate uses the small embedded corpus and does
# not need this file.
dataset-tinyshakespeare: $(TINYSHAKESPEARE_FILE)

# Multi-link: one libidrisml.{so,dylib} with all listed BACKENDs in it.
# Primary backend's symbols are exported under both unified
# (`tensor_add`) and suffixed (`tensor_add_<primary>`) names; other
# backends' symbols are reachable only via their suffixed names.
backend: $(LIB)

# Regenerate the per-backend rename headers from backend.h. The
# generated files are checked in; `make test-integration-lint-rename-headers`
# (in CI) gates that they stay in sync with backend.h.
rename-headers:
	@python3 scripts/gen-rename-headers.py

test-integration-lint-rename-headers:
	@python3 scripts/gen-rename-headers.py --check

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
# scripts/lifecycle/ffi_manifest.py — both the converter and the linter
# read from it.
test-integration-lint-ffi-wrap-template:
	@python3 scripts/lifecycle/check-ffi-wrap-template.py

# Lint: flag %foreign declarations whose Idris type is non-IO but whose
# C body has side effects (allocate, mutate, log, append to tape).
# Catches the bug class fixed by the IO refactor (commits leading up to
# e337512) — see the audit doc + `feedback_typeclass_zero_arg_method_eval.md`
# for the underlying mechanism. Known dead surfaces are skip-listed in
# the script until the dead-code cleanup row lands.
test-integration-lint-non-io-side-effects:
	@python3 scripts/lifecycle/check-non-io-side-effects.py

# Criterion-driven test suite (per-test process isolation + JUnit XML).
# Today only ships a smoke test (test_criterion_smoke.c) verifying the
# framework links and runs. Phase 1 (per /Users/admin/.claude/plans/modular-petting-minsky.md)
# migrates the per-op suites into packages/backends/test/<backend>/...,
# which this target will discover and link.
#
# Criterion is provided by nix (nixpkgs `criterion` + `criterion.dev`).
# Include / lib paths derived from the user nix-profile; an explicit
# CRITERION_PREFIX= overrides for non-nix environments.
CRITERION_PREFIX ?= $(HOME)/.nix-profile
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

test-unit-backend: $(CRITERION_TEST_SRCS) $(BACKEND_RENAME_H) backend | $(BUILD)
	cc -o $(BUILD)/test_criterion_smoke $(EXTRA_CFLAGS) -include $(BACKEND_RENAME_H) $(TEST_C_INCLUDES) $(CRITERION_TEST_SRCS) -DBACKEND_$(shell echo $(PRIMARY) | tr a-z A-Z) $(CRITERION_CFLAGS) -L$(BUILD) -lidrisml -Wl,-rpath,$(BUILD) $(EXTRA_LDFLAGS) $(CRITERION_LDFLAGS) -lm
	./$(BUILD)/test_criterion_smoke $(CRITERION_FLAGS) --xml=$(BUILD)/test-criterion-$(PRIMARY).xml

test-unit-backend-tape:
	$(MAKE) BACKEND=tape test-unit-backend

test-unit-backend-mlx:
	$(MAKE) BACKEND=mlx test-unit-backend

test-unit-backend-torch:
	$(MAKE) BACKEND=torch test-unit-backend

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

test-coverage-backend:
	$(MAKE) -j$(NPROC) BUILD=$(COV_BUILD) \
	  EXTRA_CFLAGS="$(COV_CFLAGS)" \
	  EXTRA_LDFLAGS="$(COV_LDFLAGS)" \
	  BACKEND=$(BACKEND) \
	  $(COV_BUILD)/test_criterion_smoke
	@mkdir -p $(COV_BUILD)/profraw
	@rm -f $(COV_BUILD)/profraw/*.profraw
	LLVM_PROFILE_FILE='$(COV_BUILD)/profraw/test_criterion_%p_%m.profraw' \
	  ./$(COV_BUILD)/test_criterion_smoke --xml=$(COV_BUILD)/test-criterion-$(PRIMARY).xml > /dev/null
	xcrun llvm-profdata merge -sparse $(COV_BUILD)/profraw/*.profraw -o $(COV_BUILD)/$(PRIMARY).profdata
	@echo ""
	@echo "=== Coverage report ($(PRIMARY)) ==="
	xcrun llvm-cov report $(COV_BUILD)/libidrisml.$(LIB_EXT) -instr-profile=$(COV_BUILD)/$(PRIMARY).profdata -ignore-filename-regex='($(BACKENDS_DIR)/(cJSON|safetensors|shared_utils|mnist))|(/(usr|nix|opt|Library|System|\.venv)/)|(\.cache/)'
	@rm -rf $(COV_BUILD)/html-$(PRIMARY)
	xcrun llvm-cov show $(COV_BUILD)/libidrisml.$(LIB_EXT) -instr-profile=$(COV_BUILD)/$(PRIMARY).profdata -format=html -output-dir=$(COV_BUILD)/html-$(PRIMARY) -ignore-filename-regex='($(BACKENDS_DIR)/(cJSON|safetensors|shared_utils|mnist))|(/(usr|nix|opt|Library|System|\.venv)/)|(\.cache/)'
	@echo ""
	@echo "Coverage HTML: file://$(PWD)/$(COV_BUILD)/html-$(PRIMARY)/index.html"

# Build-only the criterion suite with coverage flags so the
# test-coverage-backend recipe can set LLVM_PROFILE_FILE before running.
# Matches the test-unit-backend build recipe — link the full
# discovered suite, not just the smoke shell.
$(COV_BUILD)/test_criterion_smoke: $(CRITERION_TEST_SRCS) $(BACKEND_RENAME_H) $(LIB) | $(COV_BUILD)
	cc -o $@ $(EXTRA_CFLAGS) -include $(BACKEND_RENAME_H) $(TEST_C_INCLUDES) $(CRITERION_TEST_SRCS) -DBACKEND_$(shell echo $(PRIMARY) | tr a-z A-Z) $(CRITERION_CFLAGS) -L$(BUILD) -lidrisml -Wl,-rpath,$(PWD)/$(BUILD) $(EXTRA_LDFLAGS) $(CRITERION_LDFLAGS) -lm

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
	@bash scripts/coverage-gap-probe.sh $(BUILD)

# Specialized C test suites. The NTM + mlx-compile tests live under
# packages/idris-test-c/src/ (cross-cutting integration; no 1:1 source
# pair). They're standalone main()s (NOT Criterion) so they get their
# own recipes rather than folding into test-unit-backend.
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

# Install core library to local prefix (needed before building examples/tests).
# `--build-dir` keys the per-package ttc cache on the active BUILD_KEY so
# `BACKEND=tape` and `BACKEND=torch` (etc.) each have their own warm cache.
install-core: backend $(HWCONFIG_IDR) $(HWDEVICES_IDR)
	@cd packages/idris-ml && IDRIS2_PREFIX=$(IDRIS2_LOCAL) idris2 --build-dir $(CURDIR)/$(BUILD)/ttc-idris-ml --install idris-ml.ipkg >/dev/null

# Install gym to local prefix
install-gym:
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
install: install-core install-gym install-transformers install-notebook install-examples $(BUILD)/.library-cache-stamp

# Idris build (type-check core library)
check: backend $(HWCONFIG_IDR) $(HWDEVICES_IDR)
	cd packages/idris-ml && idris2 --build-dir $(CURDIR)/$(BUILD)/ttc-idris-ml --build idris-ml.ipkg

# Type-check gym package
check-gym:
	cd packages/idris-gym && idris2 --build-dir $(CURDIR)/$(BUILD)/ttc-idris-gym --build idris-gym.ipkg

# Type-check idris-transformers package (depends on idris-ml being installed).
check-transformers: install-core
	cd packages/idris-transformers && IDRIS2_PREFIX=$(IDRIS2_LOCAL) idris2 --build-dir $(CURDIR)/$(BUILD)/ttc-idris-transformers --build idris-transformers.ipkg

# Verify Idris example defaults match the paired torch_ref/scripts/*.py defaults.
# Catches the "I changed Idris's default but forgot the matching ref" drift class.
test-integration-lint-paired-defaults:
	@python3 scripts/check-paired-defaults.py

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

# Type-check examples (builds each as executable, which is the real check)
check-examples: install
	@for f in $(EXAMPLE_SRC)/Example/*.idr; do \
		mod=$$(basename "$$f" .idr); \
		case "$$mod" in \
			Transfer|MlxStreamDemo) \
				echo "Skipping Example.$$mod (cross-backend: names non-linked devices, so it only compiles under a multi-backend BACKEND — checked via its own target)"; \
				continue ;; \
			DTypeSerialize) \
				echo "Skipping Example.$$mod (torch-only: constructs bf16/f16/i32, no Compatible instance on tape/mlx — checked via example-dtype-serialize)"; \
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
# Canonical aggregator: every unit-layer leaf, across active backend.
# Run locally pre-commit: `make test-unit` (~2 min on tape). Adding
# a new unit-layer test means adding the target name to this list;
# the CI workflow consumes `make test-unit` (post-Phase-4) and so
# auto-includes any new leaf without a workflow edit.
test-unit: test-unit-idris-ml test-unit-backend

# Integration test layer — see docs/develop/testing-taxonomy.md.
#
# Canonical aggregator: every integration-layer leaf (negative type-check
# gates, source-code linters, multi-module integration probes that don't
# run a full training loop). Run locally when you touched a type-level
# guarantee or the FFI wrap convention. Adding a new integration-layer
# test means adding the target name to this list.
test-integration: \
		test-integration-lint-rename-headers \
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
# resolve through `{d=TestDevice}` which the Makefile-generated
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
# `toDevice` smoke + roundtrip) to the unit-test list. Forces
# BACKEND=torch,tape,mlx so tape / torch / mlx C symbols are all
# linked into one dylib — Test.Transfer references all three by
# name through `UserDeviceTransfer` instance dispatch and would
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
	$(MAKE) BACKEND=tape test-unit-backend
	$(MAKE) BACKEND=torch test-unit-backend
	$(MAKE) BACKEND=mlx test-unit-backend

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

# Idris tests for idris-transformers package. Pure-Idris suite for
# bertParamNames catalogue + an FFI suite that constructs a real
# HfBert and asserts the C-side param registry matches the catalogue
# exactly. The dylib gets copied alongside the test executable so the
# FFI registry calls land on the active backend's symbols (mirrors
# the test-unit-idris-ml recipe).
test-unit-idris-transformers: backend $(HWCONFIG_IDR) $(HWDEVICES_IDR)
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

# Build and run examples (require: make install)
example-supervised: install
	idris2 $(IDRIS_FLAGS) -o supervised $(EXAMPLE_SRC)/Example/Supervised.idr
	cp $(LIB) $(BUILD)/exec/supervised_app/
	./$(BUILD)/exec/supervised $(SEED_FLAG) $(SUPERVISED_ARGS)

# HuggingFace BERT inference example. Loads google/bert_uncased_L-2_H-128_A-2
# weights via the HF-aligned HfBert layer module (from idris-transformers)
# and dumps the 128-dim pooled [CLS] output to stdout, one value per line.
# Pre-requisite: bash packages/idris-transformers/scripts/hf-download.sh
# google/bert_uncased_L-2_H-128_A-2 must have run at least once to populate
# packages/idris-transformers/models/.
# Pattern rule for any HuggingFace single-file checkpoint. Make-native
# dep tracking: each example/gate declares the safetensors path as a
# prerequisite; Make skips the recipe when the file is already on disk.
# Replaces the older shape (unconditional `bash hf-download.sh …` in
# every recipe + an internal cache check inside the script).
#
# `%` matches the HF repo path (e.g. `meta-llama/Llama-3.2-1B`). HF_TOKEN
# is checked here (the one place that actually fetches) rather than in
# every consumer recipe. Gated models that need the token surface a
# clear error; ungated models (BERT-tiny, distilgpt2) ignore the check.
HF_MODELS_DIR := models

$(HF_MODELS_DIR)/%/config.json:
	@if echo "$*" | grep -q '^meta-llama/' && [ -z "$$HF_TOKEN" ]; then \
		echo "ERR: HF_TOKEN must be set ($* is gated)."; \
		echo "     1. Accept the license at https://huggingface.co/$*"; \
		echo "     2. Get a token at https://huggingface.co/settings/tokens"; \
		echo "     3. export HF_TOKEN=hf_..."; \
		exit 1; \
	fi
	bash packages/idris-transformers/scripts/hf-download.sh $*

example-hf-bert-inference: install $(HF_MODELS_DIR)/google/bert_uncased_L-2_H-128_A-2/config.json
	idris2 $(IDRIS_FLAGS) -o hf-bert-inference $(EXAMPLE_SRC)/Example/HfBertInference.idr
	cp $(LIB) $(BUILD)/exec/hf-bert-inference_app/
	./$(BUILD)/exec/hf-bert-inference

# Cross-language correctness gate for HfBert: regenerates the Python
# oracle via save_oracle.py, then runs the Idris example and compares
# stdout against the oracle within F32 tolerance.
test-e2e-hf-bert-roundtrip: install $(HF_MODELS_DIR)/google/bert_uncased_L-2_H-128_A-2/config.json
	cd packages/pytorch && uv run pytest \
		../idris-transformers/scripts/test_save_oracle.py -v
	idris2 $(IDRIS_FLAGS) -o hf-bert-inference $(EXAMPLE_SRC)/Example/HfBertInference.idr
	cp $(LIB) $(BUILD)/exec/hf-bert-inference_app/
	./$(BUILD)/exec/hf-bert-inference --dump-pooled > $(BUILD)/hf-bert-idris-out.txt
	cd packages/pytorch && uv run python \
		../idris-transformers/scripts/compare_inference.py \
		../../$(BUILD)/hf-bert-idris-out.txt \
		../../models/bert-tiny-oracle.safetensors \
		1e-3

# Build + run Example/HfGpt2Inference. Fetches distilgpt2 once via the
# pattern rule above.
example-hf-gpt2-inference: install $(HF_MODELS_DIR)/distilgpt2/config.json
	idris2 $(IDRIS_FLAGS) -o hf-gpt2-inference $(EXAMPLE_SRC)/Example/HfGpt2Inference.idr
	cp $(LIB) $(BUILD)/exec/hf-gpt2-inference_app/
	./$(BUILD)/exec/hf-gpt2-inference

# Cross-language correctness gate for HfGpt2: regenerate the Python
# oracle from distilgpt2 + run the Idris example + compare
# stdout against the oracle within F32 tolerance. The Idris example
# prints the final-position hidden state (the `last_hidden_state[-1]`
# row) which the comparator diffs elementwise.
# Build + run the Llama 3.2 1B inference example. Requires HF_TOKEN
# with Llama 3.2 license accepted on huggingface.co. The first
# invocation fetches the ~2.5 GB safetensors; subsequent runs reuse
# the cached file (Make's existence check handles it — the pattern
# rule's recipe doesn't fire).
#
# Tape lane (F64) doesn't fit in 16 GB; build with
# `BACKEND=torch TORCH_DEVICE=mps make example-hf-llama-inference`
# or `BACKEND=mlx MLX_DEVICE=gpu make example-hf-llama-inference` for
# the F32 / GPU paths.
#
# HF inference targets auto-set TORCH_DTYPE/MLX_DTYPE/TAPE_DTYPE
# to F32 (see the MAKECMDGOALS conditional near BUILD_KEY); the
# 1.24B-param Llama at F64 is ~10 GB which doesn't fit comfortably
# on a 16 GB VM. Override by setting TORCH_DTYPE=F64 (etc) on the
# command line if you genuinely want F64 (e.g. for numerical
# bisection vs the F64 oracle in `save_oracle_llama.py`).
example-hf-llama-inference: install $(HF_MODELS_DIR)/meta-llama/Llama-3.2-1B/config.json
	idris2 $(IDRIS_FLAGS) -o hf-llama-inference $(EXAMPLE_SRC)/Example/HfLlamaInference.idr
	cp $(LIB) $(BUILD)/exec/hf-llama-inference_app/
	./$(BUILD)/exec/hf-llama-inference

# Fast feedback loop for HfLlamaInference: type-check only (`--check`),
# skip Scheme codegen + linking. Turns around in tens of seconds vs the
# multi-minute `example-hf-llama-inference` build. Useful when iterating
# on the typed surface (signatures, implicit-resolution, totality)
# without caring about an executable binary yet.
#
# Same install dep as the full build so dependent libraries (idris-ml,
# idris-transformers) are present; the difference is that the example
# file itself is `--check`ed rather than `-o`'d.
test-integration-lint-hf-llama-inference: install
	IDRIS2_PREFIX=$(IDRIS2_LOCAL) idris2 -p contrib -p idris-ml -p idris-gym -p idris-transformers \
		--build-dir $(BUILD)/check-hf-llama-inference --source-dir $(EXAMPLE_SRC) \
		--check $(EXAMPLE_SRC)/Example/HfLlamaInference.idr

# Build + run Example/HfBitNetInference. Fetches microsoft/bitnet-b1.58-2B-4T
# once via the pattern rule (1.18 GB, not gated). Default mode runs the
# fixed-prompt forward and prints the top 5 logits; `--dump-logits` mode
# prints all 128256 logits for the roundtrip gate.
#
# Tape lane (F64) won't fit in 16 GB; build with
# `BACKEND=torch TORCH_DEVICE=mps make example-hf-bitnet-inference` or
# `BACKEND=mlx MLX_DEVICE=gpu make example-hf-bitnet-inference`.
example-hf-bitnet-inference: install $(HF_MODELS_DIR)/microsoft/bitnet-b1.58-2B-4T/config.json
	idris2 $(IDRIS_FLAGS) -o hf-bitnet-inference $(EXAMPLE_SRC)/Example/HfBitNetInference.idr
	cp $(LIB) $(BUILD)/exec/hf-bitnet-inference_app/
	./$(BUILD)/exec/hf-bitnet-inference

# Cross-language correctness gate for HfBitNet: regenerate the Python
# oracle from microsoft/bitnet-b1.58-2B-4T, run the Idris example
# in --dump-logits mode, compare stdout against the oracle.
# Tolerance is 1.0 max-abs-diff + an argmax-match assertion. The
# tolerance is loose because BitNet's BF16-storage + ternary-weight
# noise compounds across 30 decoder blocks: per-element diff at the
# logits layer settles at ~0.7 even with the kernel math correct.
# The argmax-match assertion catches the meaningful regression class
# (the model picking a different next token) without burdening the
# gate with the per-element noise floor.
test-e2e-hf-bitnet-roundtrip: install $(HF_MODELS_DIR)/microsoft/bitnet-b1.58-2B-4T/config.json
	cd packages/pytorch && uv run python \
		../idris-transformers/scripts/save_oracle_bitnet.py
	idris2 $(IDRIS_FLAGS) -o hf-bitnet-inference $(EXAMPLE_SRC)/Example/HfBitNetInference.idr
	cp $(LIB) $(BUILD)/exec/hf-bitnet-inference_app/
	./$(BUILD)/exec/hf-bitnet-inference --dump-logits > $(BUILD)/hf-bitnet-idris-out.txt
	cd packages/pytorch && uv run python \
		../idris-transformers/scripts/compare_inference.py \
		../../$(BUILD)/hf-bitnet-idris-out.txt \
		../../models/bitnet-2b-4t-oracle.safetensors \
		1.0 --argmax-match

test-e2e-hf-gpt2-roundtrip: install $(HF_MODELS_DIR)/distilgpt2/config.json
	cd packages/pytorch && uv run pytest \
		../idris-transformers/scripts/test_save_oracle_gpt2.py -v
	idris2 $(IDRIS_FLAGS) -o hf-gpt2-inference $(EXAMPLE_SRC)/Example/HfGpt2Inference.idr
	cp $(LIB) $(BUILD)/exec/hf-gpt2-inference_app/
	./$(BUILD)/exec/hf-gpt2-inference --dump-final-hidden > $(BUILD)/hf-gpt2-idris-out.txt
	cd packages/pytorch && uv run python \
		../idris-transformers/scripts/compare_inference.py \
		../../$(BUILD)/hf-gpt2-idris-out.txt \
		../../models/distilgpt2-oracle.safetensors \
		1e-3

# Cross-language correctness gate for HfLlama: regenerate the Python
# oracle from meta-llama/Llama-3.2-1B (gated by HF_TOKEN + license),
# run the Idris example in --dump-final-hidden mode, compare stdout
# against the oracle's last-position hidden state. Tolerance is 1.0
# max-abs-diff — Llama 3.2 1B is 16 layers × hidden=2048 with on-disk
# BF16 cast to F32, so per-element drift accumulates; the gate's job
# is catching macro regressions (broken forward, broken param load,
# bad RoPE), not pinning numerics to BF16-noise-floor precision.
# Tighten if measurements show consistent tighter alignment.
test-e2e-hf-llama-roundtrip: install $(HF_MODELS_DIR)/meta-llama/Llama-3.2-1B/config.json
	cd packages/pytorch && uv run pytest \
		../idris-transformers/scripts/test_save_oracle_llama.py -v
	idris2 $(IDRIS_FLAGS) -o hf-llama-inference $(EXAMPLE_SRC)/Example/HfLlamaInference.idr
	cp $(LIB) $(BUILD)/exec/hf-llama-inference_app/
	./$(BUILD)/exec/hf-llama-inference --dump-final-hidden > $(BUILD)/hf-llama-idris-out.txt
	cd packages/pytorch && uv run python \
		../idris-transformers/scripts/compare_inference.py \
		../../$(BUILD)/hf-llama-idris-out.txt \
		../../models/llama-3.2-1b-oracle.safetensors \
		1.0

# Multi-step generation gate for HfLlama. Regenerates the Python
# oracle by greedy-decoding 8 tokens from `model.generate(do_sample=
# False, use_cache=True)` on the same prompt the user-facing demo
# uses ("The capital of France is"), runs the Idris example in
# --dump-tokens mode for the same prompt + budget, and asserts the
# resulting token-ID sequences match element-wise. Catches
# generation-path drift the single-forward
# `test-e2e-hf-llama-roundtrip` can't see.
#
# Budget bumped 2026-06-04 from 4 to 8 after the KV cache landed
# (commits `b5443135` ... `3b87291f`): with cached decode each step
# is constant-cost in Q/K/V projection (vs the no-cache path's
# growing prefix), so 8 tokens is cheap.
#
# Tape lane (F64) doesn't fit in 16 GB; build with
# `BACKEND=torch TORCH_DEVICE=cpu` for CI or
# `BACKEND=torch TORCH_DEVICE=mps` / `BACKEND=mlx MLX_DEVICE=gpu`
# for paired-lane dev verification.
test-e2e-hf-llama-generate-roundtrip: install $(HF_MODELS_DIR)/meta-llama/Llama-3.2-1B/config.json
	cd packages/pytorch && uv run pytest \
		../idris-transformers/scripts/test_save_oracle_llama_generate.py -v
	idris2 $(IDRIS_FLAGS) -o hf-llama-inference $(EXAMPLE_SRC)/Example/HfLlamaInference.idr
	cp $(LIB) $(BUILD)/exec/hf-llama-inference_app/
	./$(BUILD)/exec/hf-llama-inference --dump-tokens --num-tokens 8 > $(BUILD)/hf-llama-tokens-out.txt
	cd packages/pytorch && uv run python \
		../idris-transformers/scripts/compare_inference.py \
		../../$(BUILD)/hf-llama-tokens-out.txt \
		../../models/llama-3.2-1b-generate-oracle.safetensors \
		--token-sequence

# Manual oracle-regen entry point (pytest harness pairs with
# `test-e2e-hf-llama-generate-roundtrip` above). Useful when bumping
# the budget after KV cache lands.
test-e2e-transformers-oracle-llama-generate:
	cd packages/pytorch && uv run pytest \
		../idris-transformers/scripts/test_save_oracle_llama_generate.py -v

example-rnn: install
	idris2 $(IDRIS_FLAGS) -o rnn $(EXAMPLE_SRC)/Example/Rnn.idr
	cp $(LIB) $(BUILD)/exec/rnn_app/
	./$(BUILD)/exec/rnn $(SEED_FLAG) $(RNN_ARGS)

example-lstm: install
	idris2 $(IDRIS_FLAGS) -o lstm $(EXAMPLE_SRC)/Example/Lstm.idr
	cp $(LIB) $(BUILD)/exec/lstm_app/
	./$(BUILD)/exec/lstm $(SEED_FLAG) $(LSTM_ARGS)

example-gru: install
	idris2 $(IDRIS_FLAGS) -o gru $(EXAMPLE_SRC)/Example/Gru.idr
	cp $(LIB) $(BUILD)/exec/gru_app/
	./$(BUILD)/exec/gru $(SEED_FLAG) $(GRU_ARGS)

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
	cp $(LIB) $(BUILD)/libbyo.$(LIB_EXT) $(BUILD)/exec/bring-your-own_app/
	./$(BUILD)/exec/bring-your-own

example-ntm-copy: install
	idris2 $(IDRIS_FLAGS) -o ntm-copy $(EXAMPLE_SRC)/Example/NtmCopy.idr
	cp $(LIB) $(BUILD)/exec/ntm-copy_app/
	$(STDBUF) ./$(BUILD)/exec/ntm-copy $(SEED_FLAG) $(NTM_COPY_ARGS)

example-ntm-associative-recall: install
	idris2 $(IDRIS_FLAGS) -o ntm-associative-recall $(EXAMPLE_SRC)/Example/NtmAssociativeRecall.idr
	cp $(LIB) $(BUILD)/exec/ntm-associative-recall_app/
	$(STDBUF) ./$(BUILD)/exec/ntm-associative-recall $(SEED_FLAG) $(NTM_RECALL_ARGS)

example-dnc-copy: install
	idris2 $(IDRIS_FLAGS) -o dnc-copy $(EXAMPLE_SRC)/Example/DncCopy.idr
	cp $(LIB) $(BUILD)/exec/dnc-copy_app/
	$(STDBUF) ./$(BUILD)/exec/dnc-copy $(SEED_FLAG) $(DNC_COPY_ARGS)

example-dnc-recall: install
	idris2 $(IDRIS_FLAGS) -o dnc-recall $(EXAMPLE_SRC)/Example/DncAssociativeRecall.idr
	cp $(LIB) $(BUILD)/exec/dnc-recall_app/
	$(STDBUF) ./$(BUILD)/exec/dnc-recall $(SEED_FLAG) $(DNC_RECALL_ARGS)

example-transformer: install
	idris2 $(IDRIS_FLAGS) -o transformer $(EXAMPLE_SRC)/Example/Transformer.idr
	cp $(LIB) $(BUILD)/exec/transformer_app/
	./$(BUILD)/exec/transformer $(SEED_FLAG) $(TRANSFORMER_ARGS)

example-tcast-demo: install
	idris2 $(IDRIS_FLAGS) -o tcast-demo $(EXAMPLE_SRC)/Example/TCastDemo.idr
	cp $(LIB) $(BUILD)/exec/tcast-demo_app/
	./$(BUILD)/exec/tcast-demo $(TCAST_DEMO_ARGS)

# Cross-language dtype serialization demo. Forces BACKEND=torch (bf16/f16/
# int are Compatible only on torch), writes a multi-dtype .safetensors from
# Idris, then verifies the byte layout via the reference safetensors.torch
# reader (Python). Verifier is skipped if the pytorch venv is absent.
example-dtype-serialize:
	$(MAKE) BACKEND=torch install >/dev/null
	idris2 $(IDRIS_FLAGS) -o dtype-serialize $(EXAMPLE_SRC)/Example/DTypeSerialize.idr
	cp $(LIB) $(BUILD)/exec/dtype-serialize_app/
	./$(BUILD)/exec/dtype-serialize /tmp/idrisml-dtypes.safetensors
	@if [ -x packages/pytorch/.venv/bin/python3 ]; then \
		echo "=== cross-language verify (safetensors.torch) ==="; \
		packages/pytorch/.venv/bin/python3 packages/idris-ml-examples/scripts/verify_dtypes.py /tmp/idrisml-dtypes.safetensors; \
	else \
		echo "=== cross-language verify SKIPPED (pytorch venv not found) ==="; \
	fi

# Type-safe integral index API demo. Forces BACKEND=torch (an I64 index
# tensor is Compatible only on torch-cpu/cuda), then runs the typed
# targsort/tgather/tscatterAdd round-trip with order-sensitive readouts.
example-index-ops:
	$(MAKE) BACKEND=torch install >/dev/null
	idris2 $(IDRIS_FLAGS) -o index-ops $(EXAMPLE_SRC)/Example/IndexOps.idr
	cp $(LIB) $(BUILD)/exec/index-ops_app/
	./$(BUILD)/exec/index-ops

# Compile-time (device, dtype) Compatible gate demo. The example's `ok*`
# witnesses typecheck against the real constructor across all backends;
# main constructs on the build-selected cell, so it runs on any BACKEND.
example-dtype-pitch: install
	idris2 $(IDRIS_FLAGS) -o dtype-pitch $(EXAMPLE_SRC)/Example/DTypePitch.idr
	cp $(LIB) $(BUILD)/exec/dtype-pitch_app/
	./$(BUILD)/exec/dtype-pitch

# Cross-dtype SafeTensors round-trip smoke test for L63.
#   1. Save F32 (BACKEND=mlx MLX_DEVICE=gpu): writes a checkpoint with
#      "dtype":"F32" headers and 4-byte-per-element data.
#   2. Load-strict in F64 (BACKEND=mlx): expects to FAIL with a dtype
#      mismatch — `loadModel` returns False, the example exits nonzero.
#   3. Load-cast in F64 (BACKEND=mlx): expects to PASS — bytes widened
#      f32 -> f64 at load time, eval loss reproduces the trained loss.
example-precision-checkpoint:
	@rm -f /tmp/precision-checkpoint.safetensors
	@echo "=== Step 1: save F32 (BACKEND=mlx MLX_DEVICE=gpu) ==="
	$(MAKE) BACKEND=mlx MLX_DEVICE=gpu install >/dev/null
	idris2 $(IDRIS_FLAGS) -o precision-checkpoint $(EXAMPLE_SRC)/Example/PrecisionCheckpoint.idr
	cp $(LIB) $(BUILD)/exec/precision-checkpoint_app/
	./$(BUILD)/exec/precision-checkpoint --mode save --path /tmp/precision-checkpoint.safetensors --expect pass
	@echo ""
	@echo "=== Step 2: load-strict into F64 (BACKEND=mlx), expect FAIL ==="
	$(MAKE) BACKEND=mlx install >/dev/null
	idris2 $(IDRIS_FLAGS) -o precision-checkpoint $(EXAMPLE_SRC)/Example/PrecisionCheckpoint.idr
	cp $(LIB) $(BUILD)/exec/precision-checkpoint_app/
	./$(BUILD)/exec/precision-checkpoint --mode load-strict --path /tmp/precision-checkpoint.safetensors --expect fail
	@echo ""
	@echo "=== Step 3: load-cast into F64 (BACKEND=mlx), expect PASS ==="
	./$(BUILD)/exec/precision-checkpoint --mode load-cast --path /tmp/precision-checkpoint.safetensors --expect pass
	@echo ""
	@echo "All three steps passed (PrecisionCheckpoint L63 round-trip)."

# Training-loop checkpoint/resume smoke test (tape backend, fast).
# Trains gpt 10 epochs to a checkpoint dir, resumes to 20, asserts the
# sidecar epoch + resume log + completion. Gates the Train/Checkpoint
# integration. See scripts/test-checkpoint-resume.sh.
test-integration-checkpoint-resume: install
	idris2 $(IDRIS_FLAGS) -o gpt $(EXAMPLE_SRC)/Example/Gpt.idr
	cp $(LIB) $(BUILD)/exec/gpt_app/
	bash scripts/test-checkpoint-resume.sh ./$(BUILD)/exec/gpt

# Mlx-only: cross-stream MlxCpu F64 / MlxGpu F32 smoke test. Builds
# under any BACKEND list that includes mlx; references MlxCpu / MlxGpu
# directly, so won't link under tape-only or torch-only builds.
example-mlx-stream-demo: install
	idris2 $(IDRIS_FLAGS) -o mlx-stream-demo $(EXAMPLE_SRC)/Example/MlxStreamDemo.idr
	cp $(LIB) $(BUILD)/exec/mlx-stream-demo_app/
	./$(BUILD)/exec/mlx-stream-demo $(MLX_STREAM_DEMO_ARGS)

example-gpt: install
	idris2 $(IDRIS_FLAGS) -o gpt $(EXAMPLE_SRC)/Example/Gpt.idr
	cp $(LIB) $(BUILD)/exec/gpt_app/
	$(STDBUF) ./$(BUILD)/exec/gpt $(SEED_FLAG) $(GPT_ARGS)

# Full-corpus convergence run (~hours on tape). Default `make example-gpt`
# is a ~30s embedded-corpus demo; this target is the real char-LM
# convergence target (matching nanoGPT/train_shakespeare_char.py).
example-gpt-full: install $(TINYSHAKESPEARE_FILE)
	idris2 $(IDRIS_FLAGS) -o gpt $(EXAMPLE_SRC)/Example/Gpt.idr
	cp $(LIB) $(BUILD)/exec/gpt_app/
	$(STDBUF) ./$(BUILD)/exec/gpt $(SEED_FLAG) --corpus tinyshakespeare --epochs 1000 $(GPT_ARGS)

example-mnist: install $(MNIST_SENTINEL)
	idris2 $(IDRIS_FLAGS) -o mnist $(EXAMPLE_SRC)/Example/Mnist.idr
	cp $(LIB) $(BUILD)/exec/mnist_app/
	$(STDBUF) ./$(BUILD)/exec/mnist $(SEED_FLAG) $(MNIST_ARGS)

example-seq-classify: install
	idris2 $(IDRIS_FLAGS) -o seq-classify $(EXAMPLE_SRC)/Example/SeqClassify.idr
	cp $(LIB) $(BUILD)/exec/seq-classify_app/
	$(STDBUF) ./$(BUILD)/exec/seq-classify $(SEED_FLAG) $(SEQ_ARGS)

example-reinforce: install
	idris2 $(IDRIS_FLAGS) -o reinforce $(EXAMPLE_SRC)/Example/Reinforce.idr
	cp $(LIB) $(BUILD)/exec/reinforce_app/
	./$(BUILD)/exec/reinforce $(SEED_FLAG) $(REINFORCE_ARGS)

example-q-learning: install
	idris2 $(IDRIS_FLAGS) -o q-learning $(EXAMPLE_SRC)/Example/QLearning.idr
	cp $(LIB) $(BUILD)/exec/q-learning_app/
	./$(BUILD)/exec/q-learning $(SEED_FLAG) $(Q_LEARNING_ARGS)

example-sarsa: install
	idris2 $(IDRIS_FLAGS) -o sarsa $(EXAMPLE_SRC)/Example/Sarsa.idr
	cp $(LIB) $(BUILD)/exec/sarsa_app/
	./$(BUILD)/exec/sarsa $(SEED_FLAG) $(SARSA_ARGS)

example-monte-carlo: install
	idris2 $(IDRIS_FLAGS) -o monte-carlo $(EXAMPLE_SRC)/Example/MonteCarlo.idr
	cp $(LIB) $(BUILD)/exec/monte-carlo_app/
	./$(BUILD)/exec/monte-carlo $(SEED_FLAG) $(MONTE_CARLO_ARGS)

example-frozen-lake: install
	idris2 $(IDRIS_FLAGS) -o frozen-lake $(EXAMPLE_SRC)/Example/FrozenLake.idr
	cp $(LIB) $(BUILD)/exec/frozen-lake_app/
	./$(BUILD)/exec/frozen-lake $(SEED_FLAG) $(FROZEN_LAKE_ARGS)

example-taxi: install
	idris2 $(IDRIS_FLAGS) -o taxi $(EXAMPLE_SRC)/Example/Taxi.idr
	cp $(LIB) $(BUILD)/exec/taxi_app/
	./$(BUILD)/exec/taxi $(SEED_FLAG) $(TAXI_ARGS)

example-dqn: install
	idris2 $(IDRIS_FLAGS) -o dqn $(EXAMPLE_SRC)/Example/Dqn.idr
	cp $(LIB) $(BUILD)/exec/dqn_app/
	$(STDBUF) ./$(BUILD)/exec/dqn $(SEED_FLAG) $(DQN_ARGS)

example-mountain-car: install
	idris2 $(IDRIS_FLAGS) -o mountain-car $(EXAMPLE_SRC)/Example/MountainCar.idr
	cp $(LIB) $(BUILD)/exec/mountain-car_app/
	$(STDBUF) ./$(BUILD)/exec/mountain-car $(SEED_FLAG) $(MOUNTAIN_CAR_ARGS)

example-mountain-car-cont: install
	idris2 $(IDRIS_FLAGS) -o mountain-car-cont $(EXAMPLE_SRC)/Example/MountainCarCont.idr
	cp $(LIB) $(BUILD)/exec/mountain-car-cont_app/
	$(STDBUF) ./$(BUILD)/exec/mountain-car-cont $(SEED_FLAG) $(MOUNTAIN_CAR_CONT_ARGS)

example-a2c: install
	idris2 $(IDRIS_FLAGS) -o a2c $(EXAMPLE_SRC)/Example/A2c.idr
	cp $(LIB) $(BUILD)/exec/a2c_app/
	$(STDBUF) ./$(BUILD)/exec/a2c $(SEED_FLAG) $(A2C_ARGS)

example-ppo: install
	idris2 $(IDRIS_FLAGS) -o ppo $(EXAMPLE_SRC)/Example/Ppo.idr
	cp $(LIB) $(BUILD)/exec/ppo_app/
	$(STDBUF) ./$(BUILD)/exec/ppo $(SEED_FLAG) $(PPO_ARGS)

example-sac: install
	idris2 $(IDRIS_FLAGS) -o sac $(EXAMPLE_SRC)/Example/Sac.idr
	cp $(LIB) $(BUILD)/exec/sac_app/
	$(STDBUF) ./$(BUILD)/exec/sac $(SEED_FLAG) $(SAC_ARGS)

# Live cross-backend Tensor transfer demo. Builds with all three
# backends linked so the example can call tape / torch / mlx C
# symbols in a single process. Exits 0 with RESULT line on success;
# crashes at FFI resolution if any backend's symbols are missing.
#
# Torch is the primary because the F32 hop uses `tcastUnsafe` (a
# RuntimeDType operation), which routes via unified C names; only
# the primary backend's `tensor_cast_dtype_*` survives the link-time
# aliasing. Tape's `tensor_cast_dtype_f32` aborts at runtime (no
# F32 arena); mlx and torch implement it for real. Torch-primary is
# also necessary for the *Torch* cells' creation path —
# `prim__createTorch` is hardcoded F64 today, so the F32-typed
# starting tensor lands F64 and gets narrowed to F32 by
# `tcastUnsafe`, which needs the cast op to land on a backend that
# supports it.
example-transfer:
	$(MAKE) BACKEND=torch,tape,mlx install
	idris2 $(IDRIS_FLAGS) -o transfer $(EXAMPLE_SRC)/Example/Transfer.idr
	cp $(LIB) $(BUILD)/exec/transfer_app/
	./$(BUILD)/exec/transfer $(TRANSFER_ARGS)

# F32/F64 precision artifact + cross-backend hop demo. References
# TapeDev/TorchDev/MlxDev directly, so it needs all three backends
# linked (same as `example-transfer`). Unblocked by tape's F32 storage
# + kernel coverage — every cell is first-class for both precisions.
example-precision-demo:
	$(MAKE) BACKEND=tape,torch,mlx install
	idris2 $(IDRIS_FLAGS) -o precision-demo $(EXAMPLE_SRC)/Example/PrecisionDemo.idr
	cp $(LIB) $(BUILD)/exec/precision-demo_app/
	./$(BUILD)/exec/precision-demo $(PRECISION_DEMO_ARGS)

# SafeTensors checkpoint demo (formerly the Example/Transfer.idr
# content). Per-phase BACKEND= invocation; `example-checkpoint-demo`
# drives the tape→mlx→torch on-disk round-trip via three calls.
example-checkpoint: install
	idris2 $(IDRIS_FLAGS) -o checkpoint $(EXAMPLE_SRC)/Example/Checkpoint.idr
	cp $(LIB) $(BUILD)/exec/checkpoint_app/
	./$(BUILD)/exec/checkpoint $(SEED_FLAG) $(CHECKPOINT_ARGS)

example-checkpoint-demo:
	@echo "=== Phase 1: Train on tape ==="
	$(MAKE) BACKEND=tape example-checkpoint CHECKPOINT_ARGS="--mode train --epochs 500 --save /tmp/checkpoint.safetensors"
	@echo ""
	@echo "=== Phase 2: Continue on mlx ==="
	$(MAKE) BACKEND=mlx example-checkpoint CHECKPOINT_ARGS="--mode continue --load /tmp/checkpoint.safetensors --epochs 500 --save /tmp/checkpoint2.safetensors"
	@echo ""
	@echo "=== Phase 3: Infer on torch ==="
	$(MAKE) BACKEND=torch example-checkpoint CHECKPOINT_ARGS="--mode infer --load /tmp/checkpoint2.safetensors"

example-matmul-bench: install
	idris2 $(IDRIS_FLAGS) -o matmul-bench $(EXAMPLE_SRC)/Example/MatmulBench.idr
	cp $(LIB) $(BUILD)/exec/matmul-bench_app/
	$(STDBUF) ./$(BUILD)/exec/matmul-bench $(MATMUL_BENCH_ARGS)

# #402 Idris-level rank-3 broadcast microbench. Counterpart to the
# `bench-rank3-broadcast{,-wrapped}` C harnesses; calls `primMul` in a
# tight loop on `[6, 32, 32] x [6, 1, 32]` — same shape and iteration
# counts. The delta vs the wrapped C bench is the Scheme wrap layer
# (cached foreign-procedure dispatch + tensor-handle-v2 unwrap/wrap +
# guardian register). Identical wrap structure across all three
# backends — any wrap-layer overhead measured here applies symmetrically.
example-rank-broadcast-bench: install
	idris2 $(IDRIS_FLAGS) -o rank-broadcast-bench $(EXAMPLE_SRC)/Example/RankBroadcastBench.idr
	cp $(LIB) $(BUILD)/exec/rank-broadcast-bench_app/
	$(STDBUF) ./$(BUILD)/exec/rank-broadcast-bench $(RANK_BROADCAST_BENCH_ARGS)

example-bench: install
	idris2 $(IDRIS_FLAGS) -o bench $(EXAMPLE_SRC)/Example/Bench.idr
	cp $(LIB) $(BUILD)/exec/bench_app/
	@# Each benchmark runs in its own process. Sharing one process across
	@# all six accumulates allocator state that nondeterministically trips
	@# the unresolved tape stale-reader bug (see TODO.md High Priority).
	@for b in supervised rnn ntm ntm-copy ntm-copy-1k ntm-recall; do \
	    ./$(BUILD)/exec/bench $$b || exit $$?; \
	done

$(BUILD):
	mkdir -p $(BUILD)

example-profile: install
	idris2 $(IDRIS_FLAGS) -o profile $(EXAMPLE_SRC)/Example/Profile.idr
	cp $(LIB) $(BUILD)/exec/profile_app/
	./$(BUILD)/exec/profile

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
#   test-perf-fast    — Tier 1, CI, <= 5 min (Axis A op kernels today).
#   test-perf-nightly — Tier 2, nightly, <= 20 min (will fold in B/C/D).
#   test-perf-full    — Tier 3, manual / pre-tag (the 80-cell sweep).
# All three append to docs/develop/perf-log.jsonl and regenerate
# BENCHMARKS.md via scripts/render-benchmarks.py. Framework details:
# docs/develop/testing-taxonomy.md (Axis A/B/C/D + selection rule).
####################################################################

test-perf-fast:
	bash scripts/perf-fast.sh

test-perf-nightly:
	bash scripts/perf-nightly.sh

test-perf-full:
	bash scripts/perf-sweep.sh

# CI preflight: BENCHMARKS.md must agree with perf-log.jsonl.
test-integration-lint-benchmarks:
	python3 scripts/render-benchmarks.py --check

# CI preflight: perf-regression advisory gate. Reads perf-log.jsonl,
# computes a median-of-last-5 baseline per (axis, label, runtime),
# classifies the latest as OK / WARN (>15%) / FAIL (>40%) vs that
# baseline. Always exits 0 today (Phase 5a — advisory); a later
# commit will flip the FAIL threshold to exit 1.
test-integration-lint-perf-regression:
	python3 scripts/check-perf-regression.py

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

# PyTorch reference inference for the HF-aligned models. Each invokes the
# canonical HF transformers forward pass for the same model the matching
# Idris example runs, so users can eyeball PyTorch's output (or wall
# time) for direct comparison with `make example-hf-{bert,gpt2,llama}-inference`.
#
# bert + gpt2 reuse the oracle scripts (load via HF, run forward, save
# the comparison-target tensor) — re-running them refreshes the oracle
# files used by `test-hf-{bert,gpt2}-roundtrip`. llama uses
# `time_inference_llama.py` (PyTorch greedy decode, stage timers
# mirroring the Idris example).
ref-hf-bert:
	cd packages/pytorch && uv run python \
		../idris-transformers/scripts/save_oracle.py

ref-hf-gpt2:
	cd packages/pytorch && uv run python \
		../idris-transformers/scripts/save_oracle_gpt2.py

ref-hf-llama:
	cd packages/pytorch && uv run python \
		../idris-transformers/scripts/time_inference_llama.py

test-e2e-pytorch-ref:
	cd packages/pytorch && uv run pytest torch_ref/correctness/ -v

ref-lint:
	cd packages/pytorch && uv run ruff check torch_ref/ && uv run ruff format --check torch_ref/

ref-typecheck:
	cd packages/pytorch && uv run pyright torch_ref/

# Regenerate + validate the HfBert forward-pass oracle. Runs
# packages/idris-transformers/scripts/save_oracle.py through pytest
# under the pytorch package's uv-managed venv (which carries the
# `transformers` dep). The pytest is colocated with the script per
# feedback_paired_side_alignment. Wire into CI alongside test-transformers.
#
# This target only runs the generator + asserts the fixture is
# well-formed (shape, dtype, finite, nontrivial). The cross-language
# Idris-vs-Python comparison gate lands in Phase 6 as
# test-e2e-hf-bert-roundtrip.
test-e2e-transformers-oracle-bert:
	cd packages/pytorch && uv run pytest \
		../idris-transformers/scripts/test_save_oracle.py -v

# Produce the Llama-3 RoPE table oracle (inv_freq + a slice of
# cos/sin tables). Pinned by Test.RoPE in the idris-ml unit suite;
# this target lets you regenerate the oracle if the upstream Llama-3
# rope_scaling formula changes.
test-e2e-rope-oracle:
	cd packages/pytorch && uv run python \
		../idris-transformers/scripts/save_rope_oracle.py

# Same shape as test-e2e-transformers-oracle-bert, paired with HfGpt2.idr:
# generates `models/tiny-gpt2-oracle.safetensors` from
# `distilgpt2`'s last-hidden-state for [15496, 995] and
# asserts the fixture is well-formed. The cross-language gate lands
# as test-e2e-hf-gpt2-roundtrip alongside the Idris example.
test-e2e-transformers-oracle-gpt2:
	cd packages/pytorch && uv run pytest \
		../idris-transformers/scripts/test_save_oracle_gpt2.py -v

# Same shape, paired with HfLlama.idr: generates
# `models/llama-3.2-1b-oracle.safetensors` from `meta-llama/Llama-3.2-1B`'s
# last-hidden-state for [9906] ("Hello") and asserts the fixture is
# well-formed. The cross-language gate lands as test-e2e-hf-llama-roundtrip
# alongside the Idris example.
test-e2e-transformers-oracle-llama:
	cd packages/pytorch && uv run pytest \
		../idris-transformers/scripts/test_save_oracle_llama.py -v

ref-convergence:
	cd packages/pytorch && uv run python -u -m torch_ref.scripts.convergence --task both

ref-convergence-copy:
	cd packages/pytorch && uv run python -u -m torch_ref.scripts.convergence --task copy

ref-convergence-recall:
	cd packages/pytorch && uv run python -u -m torch_ref.scripts.convergence --task recall

# CUDA test (run on Colab or Linux with CUDA GPU)
test-e2e-cuda:
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
test-e2e-jupyter: backend check $(JUPYTER_VENV)/bin/activate
	$(JUPYTER_PIP) install -q -e packages/jupyter/.[dev]
	cd packages/jupyter && ../../$(JUPYTER_PYTEST) tests/ -v

# Quick: just cell parser (no REPL, no backend needed)
test-integration-jupyter-cellparser: $(JUPYTER_VENV)/bin/activate
	$(JUPYTER_PIP) install -q -e packages/jupyter/.[dev]
	cd packages/jupyter && ../../$(JUPYTER_PYTEST) tests/test_cell_parser.py -v

# Run all notebooks headless to check for API breakage
test-e2e-notebooks: jupyter-install
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

# `clean` removes everything `make install` / `make backend` can regenerate
# from source: every backend set's tree under `build/`, the coverage tree
# under `build-cov/`, and the legacy pre-per-set `.idris2/` install prefix
# (orphan from before commit-XXX). Does NOT touch downloaded deps:
# `vendored/` (third-party C source), `data/` (datasets), `models/` (HF
# checkpoints) — those are network-expensive and out of scope for clean.
# Use `clean-all` to nuke those too.
clean:
	rm -rf build/
	rm -rf build-cov/
	rm -rf .idris2/

# Active backend set's tree only — `BACKEND=tape make clean-set` removes
# `build/tape-mlxcpu-torchcpu/` but leaves other set caches alone. Use
# when a single set is in a weird state and a full `clean` would discard
# other sets' warm caches unnecessarily.
clean-set:
	rm -rf $(BUILD)

# Everything that's gitignored: build artifacts + vendored third-party
# source + downloaded datasets + downloaded HF model checkpoints.
# Network-expensive (re-running `make backend` will re-clone vendored/,
# re-running examples will re-download datasets, and HF models are
# gigabytes). Reach for this when freeing disk space or before a deep
# refactor; otherwise plain `clean` is enough.
clean-all: clean clean-models
	rm -rf vendored/
	rm -rf data/

# Downloaded HuggingFace checkpoints, tokenizer vocab files, and the
# generated test oracle. Kept out of plain `clean` because re-downloading
# is slow; run this explicitly when you need to free disk space or force
# a fresh fetch.
clean-models:
	rm -rf models/
	# Legacy location (pre-2026-05-27 refactor); remove if leftover.
	rm -rf packages/idris-transformers/models/

# Examples run on every built backend. Keep in sync with packages/idris-ml-examples/src/Example/.
# Excluded intentionally:
#   Bench, Profile — no RESULT lines (covered by bench-compare / example-profile).
EXAMPLES := example-supervised example-rnn example-lstm example-gru example-transformer example-gpt example-matmul-bench example-mnist example-seq-classify example-ntm-copy example-ntm-associative-recall example-dnc-copy example-dnc-recall example-reinforce example-q-learning example-sarsa example-monte-carlo example-frozen-lake example-taxi example-dqn example-mountain-car example-mountain-car-cont example-a2c example-ppo example-sac example-checkpoint
# 5-lane matrix. `mlx-gpu` (BACKEND=mlx MLX_DEVICE=gpu) and `torch-mps`
# (BACKEND=torch TORCH_DEVICE=mps) are virtual lanes that exercise the
# F32 code paths (per BuildConfig.idr); tape / mlx / torch build at F64.
BACKENDS := tape mlx mlx-gpu torch torch-mps

# Crash-only smoke gate: every example × lane, 3-10 epochs each,
# safety-net thresholds in test-examples.expect. Catches crashes / NaN /
# divergence / missing RESULT keys; does NOT require any model to learn.
# See docs/develop/testing.md for the full testing-layer overview.
#
# FAIL_FAST=1 bails on the first failure (handy for the iteration loop);
# the default empty value runs the whole matrix so a final confirmation
# surfaces every failure at once.
FAIL_FAST ?=

# Readiness gate for the example-precision-demo post-matrix step.
# Defaults on; flip to 0 only when temporarily skipping the demo
# (e.g. while debugging the multi-backend hop). Folds away once
# the demo has lived through a few stable CI runs.
PRECISION_DEMO_READY ?= 1
test-e2e-examples:
	@fail=0; skip=""; \
	if command -v timeout >/dev/null 2>&1; then TIMEOUT_PREFIX="timeout $(EXAMPLE_TIMEOUT)"; \
	elif command -v gtimeout >/dev/null 2>&1; then TIMEOUT_PREFIX="gtimeout $(EXAMPLE_TIMEOUT)"; \
	else echo "WARNING: no timeout/gtimeout binary; examples will not be time-bounded"; TIMEOUT_PREFIX=""; fi; \
	for lane in $(BACKENDS); do \
		case "$$lane" in \
			mlx-gpu)   b=mlx;   lane_env="MLX_DEVICE=gpu";   expect_suffix=.mlx-gpu ;; \
			torch-mps) b=torch; lane_env="TORCH_DEVICE=mps"; expect_suffix=.torch-mps ;; \
			*)         b=$$lane; lane_env="";                expect_suffix="" ;; \
		esac; \
		backend_output=$$(env $$lane_env $(MAKE) --no-print-directory BACKEND=$$b backend 2>&1) || { \
			echo "--- backend $$lane: build failed, skipping its examples ---"; \
			echo "$$backend_output" | tail -20 | sed 's/^/  | /'; \
			skip="$$skip $$lane"; continue; \
		}; \
		for e in $(EXAMPLES); do \
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
				fail=1; [ -n "$$FAIL_FAST" ] && { echo "FAIL_FAST: bail on first failure ($$e [$$lane])"; exit 1; }; continue; \
			fi; \
			result_line=$$(echo "$$output" | grep '^RESULT' | head -1); \
			if [ -z "$$result_line" ]; then \
				echo "FAIL: $$e [$$lane] -- no RESULT line ($$elapsed_fmt)"; \
				echo "$$output" | tail -40 | sed 's/^/  | /'; \
				fail=1; [ -n "$$FAIL_FAST" ] && { echo "FAIL_FAST: bail on first failure ($$e [$$lane])"; exit 1; }; \
			else \
				expect_path="$$(dirname scripts/check-result.sh)/../test-examples.expect$$expect_suffix"; \
				if [ -f "test-examples.expect$$expect_suffix" ]; then \
					scripts/check-result.sh "$$e" "$$result_line" "test-examples.expect$$expect_suffix" || { fail=1; [ -n "$$FAIL_FAST" ] && { echo "FAIL_FAST: bail ($$e [$$lane])"; exit 1; }; }; \
				else \
					scripts/check-result.sh "$$e" "$$result_line" || { fail=1; [ -n "$$FAIL_FAST" ] && { echo "FAIL_FAST: bail ($$e [$$lane])"; exit 1; }; }; \
				fi; \
				echo "  ($$elapsed_fmt)"; \
			fi; \
		done; \
	done; \
	if [ -z "$$skip" ]; then \
		echo "--- example-checkpoint-demo (tape->mlx->torch round-trip) ---"; \
		t_start=$$(date +%s); \
		demo_out=$$($$TIMEOUT_PREFIX $(MAKE) --no-print-directory example-checkpoint-demo 2>&1); demo_rc=$$?; \
		t_end=$$(date +%s); elapsed=$$((t_end - t_start)); \
		if [ $$elapsed -lt 60 ]; then elapsed_fmt="$${elapsed}s"; \
		elif [ $$elapsed -lt 3600 ]; then elapsed_fmt="$$((elapsed/60))m$$((elapsed%60))s"; \
		else elapsed_fmt="$$((elapsed/3600))h$$(((elapsed%3600)/60))m"; fi; \
		if [ $$demo_rc -ne 0 ]; then \
			if [ $$demo_rc -eq 124 ]; then echo "FAIL: example-checkpoint-demo timed out (>$(EXAMPLE_TIMEOUT)s) ($$elapsed_fmt)"; \
			else echo "FAIL: example-checkpoint-demo crashed (rc=$$demo_rc) ($$elapsed_fmt)"; fi; \
			echo "$$demo_out" | tail -40 | sed 's/^/  | /'; \
			fail=1; \
		else \
			result_line=$$(echo "$$demo_out" | grep '^RESULT' | tail -1); \
			if [ -z "$$result_line" ]; then \
				echo "FAIL: example-checkpoint-demo -- no RESULT line ($$elapsed_fmt)"; \
				echo "$$demo_out" | tail -40 | sed 's/^/  | /'; \
				fail=1; \
			else \
				scripts/check-result.sh "example-checkpoint-demo" "$$result_line" || fail=1; \
				echo "  ($$elapsed_fmt)"; \
			fi; \
		fi; \
	else \
		echo "--- example-checkpoint-demo: skipped (requires tape+mlx+torch; skipped:$$skip) ---"; \
	fi; \
	if [ "$(PRECISION_DEMO_READY)" = "1" ] && [ -z "$$skip" ]; then \
		echo "--- example-precision-demo (F32/F64 cast + cross-backend hop) ---"; \
		t_start=$$(date +%s); \
		pdemo_out=$$($$TIMEOUT_PREFIX $(MAKE) --no-print-directory example-precision-demo 2>&1); pdemo_rc=$$?; \
		t_end=$$(date +%s); elapsed=$$((t_end - t_start)); \
		if [ $$elapsed -lt 60 ]; then elapsed_fmt="$${elapsed}s"; \
		elif [ $$elapsed -lt 3600 ]; then elapsed_fmt="$$((elapsed/60))m$$((elapsed%60))s"; \
		else elapsed_fmt="$$((elapsed/3600))h$$(((elapsed%3600)/60))m"; fi; \
		if [ $$pdemo_rc -ne 0 ]; then \
			if [ $$pdemo_rc -eq 124 ]; then echo "FAIL: example-precision-demo timed out (>$(EXAMPLE_TIMEOUT)s) ($$elapsed_fmt)"; \
			else echo "FAIL: example-precision-demo crashed (rc=$$pdemo_rc) ($$elapsed_fmt)"; fi; \
			echo "$$pdemo_out" | tail -40 | sed 's/^/  | /'; \
			fail=1; \
		else \
			result_line=$$(echo "$$pdemo_out" | grep '^RESULT' | tail -1); \
			if [ -z "$$result_line" ]; then \
				echo "FAIL: example-precision-demo -- no RESULT line ($$elapsed_fmt)"; \
				echo "$$pdemo_out" | tail -40 | sed 's/^/  | /'; \
				fail=1; \
			else \
				scripts/check-result.sh "example-precision-demo" "$$result_line" || fail=1; \
				echo "  ($$elapsed_fmt)"; \
			fi; \
		fi; \
	elif [ "$(PRECISION_DEMO_READY)" != "1" ]; then \
		echo "--- example-precision-demo: skipped (PRECISION_DEMO_READY=0; example not yet landed) ---"; \
	else \
		echo "--- example-precision-demo: skipped (requires tape+mlx+torch; skipped:$$skip) ---"; \
	fi; \
	if [ -n "$$skip" ]; then echo "Skipped backends (not installed or build failed):$$skip"; fi; \
	if [ $$fail -ne 0 ]; then echo "Some integration tests FAILED"; exit 1; fi; \
	echo "All integration tests passed."

all-backends: test-e2e-examples

# Run every example to convergence at full default epochs, single seed=42,
# tape backend, with tight thresholds from test-examples-convergence.expect.
# Hours of wall time (NTM/DNC dominate). Intended for release validation,
# not CI. See docs/develop/testing.md for the testing-layer overview.
# 4h per-example cap. DNC-copy at default 50K epochs now runs in ~1.7h on
# tape (~130ms/epoch post the 2026-05-02 tensor-handle rewrite — see
# `dnc-perf-baseline.md`). Other examples are well under this cap.
CONVERGENCE_TIMEOUT ?= 14400
CONVERGENCE_EXPECT := test-examples-convergence.expect

test-convergence:
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

# Run everything: unit + gym + examples-unit + multi-backend criterion +
# specialized + e2e examples + PyTorch ref + jupyter. Multi-hour aggregate;
# not a CI gate. Subsequent phases will collapse this into per-layer
# aggregators (test-unit / test-integration / test-e2e) — for now it
# chains the layer aggregators directly.
test-all:
	@echo "=== Unit layer (Idris + Criterion + safetensors + NTM unit) ==="
	$(MAKE) test-unit
	@echo ""
	@echo "=== Gym unit tests ==="
	$(MAKE) test-unit-gym
	@echo ""
	@echo "=== Examples unit tests ==="
	$(MAKE) test-unit-examples
	@echo ""
	@echo "=== C backend tests (all available backends) ==="
	@for b in tape mlx torch; do \
		echo "--- test-unit-backend [$$b] ---"; \
		$(MAKE) BACKEND=$$b test-unit-backend 2>&1 && echo "" || echo "FAILED or SKIPPED: $$b"; \
	done
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

# Build backend + type-check all packages (default target)
check-all: check check-gym check-notebook check-examples

# Verify everything: check-all + run all tests
all: check-all test-all

.PHONY: all check-all all-backends test-unit test-unit-idris-ml test-unit-idris-transformers \
        test-unit-gym test-unit-examples test-unit-multi-backend test-all dataset-mnist dataset-tinyshakespeare \
        test-unit-backend test-unit-backend-tape test-unit-backend-mlx test-unit-backend-torch \
        test-integration test-integration-lint-rename-headers test-integration-lint-ffi-wrap-template \
        test-integration-lint-non-io-side-effects test-integration-lint-paired-defaults \
        test-integration-lint-hf-llama-inference test-integration-lint-ci-workflow \
        test-integration-lint-benchmarks test-perf-fast test-perf-nightly test-perf-full \
        test-integration-typegate-gradmode \
        test-integration-typegate-gradmode-aliasing test-integration-typegate-lossy-cast \
        test-integration-typegate-int-overflow-cast test-integration-checkpoint-resume \
        test-integration-jupyter-cellparser \
        test-coverage test-coverage-backend test-coverage-backend-tape test-coverage-backend-mlx \
        test-coverage-backend-torch test-coverage-gap-probe \
        test-e2e test-e2e-examples test-e2e-pytorch-ref test-e2e-jupyter test-e2e-notebooks test-e2e-cuda \
        test-e2e-hf-bert-roundtrip test-e2e-hf-gpt2-roundtrip test-e2e-hf-bitnet-roundtrip \
        test-e2e-hf-llama-roundtrip test-e2e-hf-llama-generate-roundtrip \
        test-e2e-transformers-oracle-bert test-e2e-transformers-oracle-gpt2 \
        test-e2e-transformers-oracle-llama test-e2e-transformers-oracle-llama-generate \
        test-e2e-rope-oracle \
        test-convergence \
        check check-gym check-notebook check-examples install install-core install-gym install-notebook install-examples \
        example-supervised example-rnn example-lstm example-gru \
        example-ntm-copy example-ntm-associative-recall example-dnc-copy example-dnc-recall \
        example-reinforce example-q-learning example-sarsa example-monte-carlo example-frozen-lake example-taxi \
        example-dqn example-mountain-car example-mountain-car-cont example-a2c example-ppo example-sac \
        example-gpt example-gpt-full example-matmul-bench example-mnist example-seq-classify example-transformer \
        ref-gpt \
        example-transfer example-checkpoint example-checkpoint-demo \
        example-bench example-profile sweep sweep-quick clean \
        backend print-torch ref-setup ref-supervised ref-rnn ref-lstm ref-gru ref-ntm-copy \
        ref-ntm-recall ref-dnc-copy ref-dnc-recall \
        ref-transformer ref-hf-bert ref-hf-gpt2 ref-hf-llama \
        bench-py bench-compare bench-ops bench-ops-py bench-ops-compare \
        bench-layers bench-layers-py ref-lint \
        ref-typecheck ref-convergence ref-convergence-copy ref-convergence-recall \
        jupyter-install jupyter-lab




