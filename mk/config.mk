# mk/config.mk — knobs + paths. BACKEND/HARDWARE/MACHINE selection,
# device strings, BUILD_KEY/BUILD, seeds, IDRIS_FLAGS, library-cache
# stamp. MUST be included first: every fragment consumes these (and
# the HF_GOALS MAKECMDGOALS sniff has to run before any rule is read).

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

# HF checkpoint store (file-as-make-target fetch; pattern rule in
# mk/examples.mk). Defined here, not in examples.mk, because tests.mk
# is included earlier and its targets (test-unit-idris-transformers)
# declare fixture prerequisites under this dir — prerequisites expand
# at parse time.
HF_MODELS_DIR := models

# MLX stream selection at runtime, also consumed by the BuildConfig
# generation rule below — when PRIMARY=mlx and MLX_DEVICE=gpu, examples
# spell `Tensor [..] (MlxExecutor MGpu) F32 WithGrad` so the type-level
# claim matches what mlx actually runs (Metal GPU is float32-only per
# the f32 rewrite).
MLX_DEVICE ?= cpu

# Torch hardware selection for the BuildConfig rule. When PRIMARY=torch
# the example types resolve to `TorchExecutor TCpu`/`TorchExecutor TMps`/
# `TorchExecutor (TCuda 0)` based on this env var. TMps forces F32 (libtorch
# rejects F64 at MPS tensor construction); TCpu and TCuda stay at F64.
TORCH_DEVICE ?= cpu

# --- HARDWARE / MACHINE: unified knobs above per-backend device strings ---
# HARDWARE picks the example default Hardware tag (Cpu | AppleGpu | Cuda 0);
# MACHINE names the physical compute environment (mac-m-series | mac-intel
# | intel-cuda-N | linux-cpu | linux-cuda-N). When unset (auto), they're
# derived from MLX_DEVICE/TORCH_DEVICE/uname. Either knob, once set,
# overrides its auto-derivation. The per-backend device strings
# (MLX_DEVICE, TORCH_DEVICE) remain available as fine-grained overrides.
HARDWARE ?= auto
MACHINE  ?= auto

# Resolve HARDWARE=auto from MLX_DEVICE / TORCH_DEVICE / uname.
ifeq ($(HARDWARE),auto)
  ifeq ($(MLX_DEVICE),gpu)
    HARDWARE_RESOLVED := metal
  else ifeq ($(TORCH_DEVICE),mps)
    HARDWARE_RESOLVED := metal
  else ifeq ($(TORCH_DEVICE),cuda)
    HARDWARE_RESOLVED := cuda
  else
    HARDWARE_RESOLVED := cpu
  endif
else
  HARDWARE_RESOLVED := $(HARDWARE)
endif

# Resolve MACHINE=auto from uname.
ifeq ($(MACHINE),auto)
  ifeq ($(UNAME),Darwin)
    UNAME_M := $(shell uname -m)
    ifeq ($(UNAME_M),arm64)
      MACHINE_RESOLVED := mac-m-series
    else
      MACHINE_RESOLVED := mac-intel
    endif
  else
    ifneq ($(HARDWARE_RESOLVED),cuda)
      MACHINE_RESOLVED := linux-cpu
    else
      MACHINE_RESOLVED := linux-cuda-1
    endif
  endif
else
  MACHINE_RESOLVED := $(MACHINE)
endif

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
# which Phase 2.x UserExecutor instance methods will target directly.
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

# Multi-backend examples: those whose source references more than one
# backend's `Executor` directly (cross-backend `toExecutor` hops), so they
# need every named backend both linked into the dylib AND present in the
# build's generated `HwConfig` (`Linked` instances). On a build whose
# `BACKEND` list is missing one, they are a CLEAN SKIP — not a crash (the
# old force-build `$(MAKE) BACKEND=tape,torch,mlx install` overwrote the
# shared HwConfig.idr with a config that *claimed* all three were linked
# while the dylib it linked had only the selected backend's symbols →
# compile-passes-against-a-lying-config, runtime FFI-resolution crash;
# campaign 2026-06-17). Dropping the force-build makes a stray
# single-backend cross-backend hop a compile-time `Linked` error instead.
# To actually run one, opt in: `make BACKEND=tape,torch,mlx example-transfer`.
MULTI_BACKEND_EXAMPLES := example-transfer example-precision-demo
MULTI_BACKEND_REQUIRED := tape torch mlx
# Non-empty ("yes") iff every required backend is in BACKEND_LIST.
HAVE_ALL_MULTI_BACKENDS := $(if $(filter-out $(BACKEND_LIST),$(MULTI_BACKEND_REQUIRED)),,yes)

# Per-backend-set build key. Distinct values of `(BACKEND, MLX_DEVICE,
# TORCH_DEVICE)` get their own `build/<KEY>/` tree (ttc cache, installed
# library prefix, dylib, example executables, stamps). Each set's warm
# cache survives backend-set switches indefinitely; switching between
# `BACKEND=tape make test` and `BACKEND=torch TORCH_DEVICE=mps make
# example-llama-inference` no longer triggers full re-elaboration.
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
# `TORCH_DTYPE=F64 make test-e2e-llama-roundtrip` keeps F64 (e.g.
# for numerical bisection vs the F64 oracle path).
# Every HF model target — Llama / BitNet need F32 for memory
# (1.24B / 2B params at F64 don't fit on a 16 GB VM); BERT-tiny /
# GPT-2-small don't NEED F32 for memory but the convention is "no
# HF model runs at F64". The HF on-disk reference weights are
# BF16, oracle generators cast to F32 — running Idris at F64
# means we're MORE precise than the comparison oracle, which is
# pure waste. F32 is the canonical HF inference dtype.
HF_GOALS := example-bert-inference \
                  example-bitnet-inference \
                  example-gpt2-inference \
                  example-llama-inference \
                  test-e2e-bert-roundtrip \
                  test-e2e-bitnet-roundtrip \
                  test-e2e-gpt2-roundtrip \
                  test-e2e-llama-roundtrip \
                  test-e2e-llama-generate-roundtrip \
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

BUILD_KEY := $(subst $(comma),-,$(strip $(BACKEND)))-mlx$(MLX_DEVICE)-torch$(TORCH_DEVICE)-mach$(MACHINE_RESOLVED)-hw$(HARDWARE_RESOLVED)$(if $(TORCH_DTYPE),-tdt$(TORCH_DTYPE),)$(if $(MLX_DTYPE),-mdt$(MLX_DTYPE),)$(if $(TAPE_DTYPE),-tpdt$(TAPE_DTYPE),)$(if $(ASAN),-asan,)
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

# Apple's `cc` finds the macOS SDK automatically via xcrun. The
# nix-installed clang-tidy doesn't — it ships with its own libc++
# headers but no system C SDK. Pass -isysroot at lint time so
# clang-tidy resolves <stdio.h> etc.; harmless on Linux (variable
# resolves to empty there).
#
# Prefer nix's apple-sdk (IDRISML_MACOS_SDKROOT, exported by the default
# devShell) over the host CLT sdk: nix clang-tidy's libc++ skews against
# the host sdk for C++ (host <sys/resource.h> uint8_t / <math.h> FP_*
# fail to resolve), which blocked the torch/mlx C++ lint on macOS. nix
# sdk + nix libc++ are self-consistent → C++ parses. Falls back to xcrun
# for a bare-macOS checkout (Apple clang-tidy, which wants the host sdk).
ifeq ($(UNAME), Darwin)
  ifdef IDRISML_MACOS_SDKROOT
    CLANG_TIDY_EXTRA_CFLAGS := -isysroot $(IDRISML_MACOS_SDKROOT)
  else
    CLANG_TIDY_EXTRA_CFLAGS := -isysroot $(shell xcrun --show-sdk-path 2>/dev/null)
  endif
else
  CLANG_TIDY_EXTRA_CFLAGS :=
endif

# Per-example wall-clock cap for test-examples. Examples exceeding this are
# killed and reported as timeouts. Override with `EXAMPLE_TIMEOUT=900 make ...`.
EXAMPLE_TIMEOUT ?= 600

# Line-buffer Chez output. Without this, stdout fully-buffers when piped or
# redirected and progress logs only appear at process exit. We use stdbuf
# unless its libstdbuf.so is incompatible with the system's dyld (e.g. nix /
# brew coreutils stdbuf on Apple-Silicon GH runners is arm64 but the
# inserted-library loader refuses to inject it into the arm64e binaries idris2
# produces — Abort trap: 6). Probe against /usr/bin/true, a SYSTEM arm64e
# binary, NOT a bare `true` (which resolves to nix/brew coreutils' own arm64
# `true`, so the inject always "succeeds" against a matching-arch target and
# masks the incompatibility with our arm64e executables). Strict CI dyld
# fails the inject → STDBUF empty → runs unbuffered but don't abort;
# permissive local dyld passes → line-buffering kept.
STDBUF := $(shell stdbuf -oL /usr/bin/true >/dev/null 2>&1 && echo "stdbuf -oL")


# --- Package paths ---
CORE_SRC := packages/idris-ml/src
GYM_SRC := packages/idris-gym/src
EXAMPLE_SRC := packages/idris-ml-examples/src
TEST_SRC := packages/idris-ml/src
BACKENDS_DIR := packages/backends

# Local package install prefix (writable, avoids polluting system Idris2).
# Per-backend-set (under `$(BUILD)`) so each set has its own installed
# library tree — `idris-ml-0`'s installed `.ttc` interface hashes differ
# across backend sets (they embed the `HwConfig.idr` / `HwExecutors.idr`
# linkage instances), so they cannot share a prefix.
IDRIS2_LOCAL := $(CURDIR)/$(BUILD)/idris2-prefix

# Single compiler: pack's idris2 — the exact commit the pinned collection
# (pack.toml) was built against. Using it everywhere (library install,
# example builds, and the pack-driven test builds) means one compiler
# produces every `.ttc`, so interface hashes always match, and the
# collection libs the library now depends on (elab-util, linear, contrib —
# plus hedgehog / Test.Golden for tests) all resolve. The previous setup
# ran nixpkgs' idris2 for the library/example path while `pack build` used
# pack's: two different commits whose `.ttc` only interoperated by luck,
# which is why a manual `cp` of elab-util into the prefix was needed.
#
# Falls back to a bare PATH idris2 when pack is absent (C-only `make
# backend`, which needs no Idris compiler at all).
IDRIS2 := $(shell pack app-path idris2 2>/dev/null || command -v idris2)
# Exported so child scripts (scripts/check-*-gate.sh, perf tooling) compile
# against the SAME compiler the build installed — else they'd load pack-built
# .ttc with a different idris2 and hit interface-hash mismatches.
export IDRIS2

# `pack package-path` is a colon-separated list of every collection package
# dir — its first entry is the compiler's bundled stdlib (base/contrib/
# linear/prelude/...), the rest are the externals (elab-util, hedgehog, ...).
# We need it because overriding IDRIS2_PREFIX to $(IDRIS2_LOCAL) below points
# idris2 away from its own prefix; the package path puts the stdlib + externals
# back in view. $(IDRIS2_LOCAL) is listed FIRST so a freshly `--install`ed
# idris-ml/idris-gym shadows any stale copy in pack's store.
PACK_PKG_PATH := $(shell pack package-path 2>/dev/null)
export IDRIS2_PACKAGE_PATH := $(IDRIS2_LOCAL)/idris2-0.8.0$(if $(PACK_PKG_PATH),:$(PACK_PKG_PATH),)

# Idris flags for example/test builds (use installed packages). `--build-dir`
# routes ttc + exec output under `$(BUILD)` so each backend set has its own
# warm cache for example/test compilation, mirroring the per-set install tree.
IDRIS_FLAGS := --build-dir $(BUILD) --source-dir $(EXAMPLE_SRC) -p contrib -p linear -p elab-util -p idris-ml -p idris-gym -p idris-transformers

# Variable introspection: `make -s print-BUILD` / `print-EXAMPLE_SRC` etc.
# echoes the resolved value of any make variable. Used by perf-compile.sh to
# reuse the build's flag/prefix resolution as the single source of truth
# instead of duplicating it. A pattern rule, so it never becomes the
# default goal; the explicit `print-torch` (mk/tests.mk) still wins for
# that name.
print-%: ; @echo '$($*)'
# Idris 2's interface-hash dependency tracking doesn't invalidate downstream
# TTCs when a module's public interface is unchanged but a where-clause body
# (or other inlined internal) changed. Single-file `idris2 -o <name>` example
# builds then reuse stale `$(BUILD)/ttc-*/.../Example/*.ttc` with old inlined
# code baked in. Wiping the per-set ttc when any library source is newer than
# this stamp forces a clean rebuild. See docs/develop/gotchas.md.
#
# The generated `.idr` files (HwConfig, HwExecutors) get *rewritten on backend-set
# switch* — their mtime bumps even when their per-set content is stable. Including
# them here would defeat the per-set ttc cache: `tape → torch → tape` would
# rewrite HwConfig.idr (set-A → set-B), then rewrite back (set-B → set-A), then
# the next tape install would see the stamp older than HwConfig.idr and wipe
# `build/tape-…/ttc-*`. Their own staleness tracking via `--build-dir`-keyed
# ttc + interface-hash check is sufficient.
LIBRARY_SRCS := $(filter-out packages/idris-ml/src/HwConfig.idr packages/idris-ml/src/HwExecutors.idr, \
                  $(shell find packages/idris-ml/src packages/idris-gym/src packages/idris-transformers/src -name '*.idr' 2>/dev/null)) \
                packages/idris-ml-examples/src/Generate.idr

# Content guard for the stamp rule below. Pure mtime invalidation is
# sound locally but not under CI's cross-commit ttc cache restore +
# git restore-mtime: a tree saved LATER than the current commit's
# file dates makes both make and idris2 see everything as fresh while
# module content/DAG changed — the 2026-06-12 macOS failure (run
# 27434248717) loaded a stale Init.ttc and died with "Undefined name
# InitSpec" inside Tensor.idr. The sha sidecar is rewritten (mtime
# bump) only when the concatenated lib-source content changes, so a
# restored foreign tree triggers the nuke regardless of timestamps,
# and a same-content restore stays warm. (GNU make re-checks mtimes
# after remaking an always-run prerequisite: unchanged sha file =
# stamp not stale.)
$(BUILD)/.library-src-sha: FORCE
	@mkdir -p $(BUILD)
	@cat $(LIBRARY_SRCS) | shasum -a 256 | cut -d' ' -f1 | cmp -s - $@ 2>/dev/null || \
		cat $(LIBRARY_SRCS) | shasum -a 256 | cut -d' ' -f1 > $@

$(BUILD)/.library-cache-stamp: $(LIBRARY_SRCS) $(BUILD)/.library-src-sha
	@echo "[$(BUILD_KEY)] Library source changed — invalidating ttc caches"
	@rm -rf $(BUILD)/ttc-* $(BUILD)/ttc
	@mkdir -p $(BUILD)
	@touch $@
