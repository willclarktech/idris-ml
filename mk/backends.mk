# mk/backends.mk — C/C++ backend build. Per-backend compiler/flag
# detection (tape/torch/mlx), modular source lists, compile rules,
# cJSON vendoring, shared objects, executor-drift gates, final dylib
# link, the `backend` entry point.

# Per-backend property tables. Common compile flags (`-O2 -fPIC
# -include rename_<b>.h`) are applied by the per-backend rule below;
# `<b>_CFLAGS` adds whatever else that backend's compile needs
# (include paths, C++ std). `<b>_LDFLAGS_<UNAME>` is per-platform.

# Tape has no monolithic backend_tape.{c,cpp} — every TU lives under
# backend_tape/. The per-backend compile rule's foreach skips tape;
# its .o objects come from BACKEND_TAPE_OBJS instead.
.PHONY: backend

tape_CC := cc
# ACCELERATE_NEW_LAPACK is a compile-time #define (gates BLAS API
# version); the framework flag is link-time.
tape_CFLAGS := -DACCELERATE_NEW_LAPACK
tape_LDFLAGS_Darwin := -framework Accelerate
# Linux BLAS: prefer pkg-config (the nix dev shell ships openblas.pc) so
# the include + lib paths are explicit — the coverage lane's raw clang
# bypasses the nix cc wrapper's NIX_CFLAGS_COMPILE / NIX_LDFLAGS and so
# can't find cblas.h or libopenblas implicitly. Fall back to -lblas for
# plain apt environments without pkg-config/openblas.pc. Empty on macOS
# (no openblas.pc — Darwin uses the Accelerate framework above), so
# tape_CFLAGS is unchanged there.
TAPE_BLAS_PC_LIBS := $(shell pkg-config --libs openblas 2>/dev/null)
ifneq ($(strip $(TAPE_BLAS_PC_LIBS)),)
tape_CFLAGS += $(shell pkg-config --cflags openblas 2>/dev/null)
tape_LDFLAGS_Linux := -lm $(TAPE_BLAS_PC_LIBS)
else
tape_LDFLAGS_Linux := -lm -lblas
endif

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
  # The top header `<torch/torch.h>` whose mtime clang bakes into the PCH.
  # Listed as a PCH prerequisite below so a cache restore that lands a newer
  # torch.h than the cached .gch rebuilds the PCH instead of failing the
  # consuming TUs with "torch.h has been modified since the precompiled
  # header was built".
  TORCH_MAIN_HEADER := $(TORCH_INC_API)/torch/torch.h
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
# `safetensors.c` is compiled once with this header so its internal
# tensor calls resolve to the primary's suffixed defs (and it exports
# the primary's suffixed `param_save_<p>` symbols, which the
# corresponding device instance methods call). cJSON.c, shared_utils.c,
# and idx.c are pure-C with no tensor surface, so they compile without
# it.
#
# The former link-time unified-name alias machinery
# (`-Wl,-alias_list` on macOS / `-Wl,--defsym=` on Linux) was deleted
# once every Idris `%foreign` migrated off unified names into
# per-instance `UserExecutor*` methods bound to the suffixed symbols
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
$(BUILD)/backend_tape/%.o: $(BACKENDS_DIR)/backend_tape/%.c $(BACKEND_TAPE_HEADERS) $(SHARED_TRAINING_HEADERS) $(BACKENDS_DIR)/rename_tape.h $(BUILD)/.backends-cache-stamp | $(BUILD)
	@mkdir -p $(dir $@)
	$(tape_CC) -O2 -fPIC $(EXTRA_CFLAGS) $(tape_CFLAGS) -include $(BACKENDS_DIR)/rename_tape.h -c -o $@ $<

# Per-TU compile for backend_torch/**/*.cpp and backend_mlx/**/*.cpp.
# Mirrors tape's pattern but uses each backend's C++ compiler + CFLAGS
# (incl. libtorch / mlx include paths). Force-includes the rename
# header so every symbol gets the backend suffix at link time. Rules
# defined unconditionally (only fire if BACKEND_<b>_OBJS pulls them in).
# Precompiled header for torch — `<torch/torch.h>` is ~30K lines of
# templates and parsing it 90× per cold build dominates the wall.
# Build the PCH once with the same flags as the per-TU compile, then
# pull it into every TU below. PCH lives in $(BUILD)/ so coverage and
# normal builds get their own (clang rejects PCHs whose flags don't
# match the consuming TU).
# The two compilers spell PCH consumption differently: clang takes the
# PCH file explicitly via `-include-pch <f>`; gcc has no such flag (it
# parses it as `-include` + a file named `-pch`) and instead auto-loads
# `<hdr>.h.gch` sitting beside the `-include`d header. So the gcc shape
# copies the header into $(BUILD) and builds the .gch next to the copy.
# -Winvalid-pch surfaces gcc's otherwise-silent fallback to textual
# inclusion when the .gch doesn't match the consuming TU's flags.
ifneq ($(findstring clang,$(shell $(torch_CC) --version 2>/dev/null)),)
TORCH_PCH := $(BUILD)/torch_pch.gch
TORCH_PCH_USE := -include-pch $(TORCH_PCH)
else
TORCH_PCH := $(BUILD)/torch_pch.h.gch
TORCH_PCH_USE := -Winvalid-pch -include $(BUILD)/torch_pch.h
endif

$(BUILD)/torch_pch.gch: $(BACKENDS_DIR)/backend_torch/torch_pch.h $(TORCH_MAIN_HEADER) $(BUILD)/.backends-cache-stamp | $(BUILD)
	$(torch_CC) -O2 -fPIC $(EXTRA_CFLAGS) $(torch_CFLAGS) -x c++-header -c -o $@ $<

$(BUILD)/torch_pch.h.gch: $(BACKENDS_DIR)/backend_torch/torch_pch.h $(TORCH_MAIN_HEADER) $(BUILD)/.backends-cache-stamp | $(BUILD)
	cp $< $(BUILD)/torch_pch.h
	$(torch_CC) -O2 -fPIC $(EXTRA_CFLAGS) $(torch_CFLAGS) -x c++-header -c -o $@ $(BUILD)/torch_pch.h

$(BUILD)/backend_torch/%.o: $(BACKENDS_DIR)/backend_torch/%.cpp $(BACKEND_TORCH_HEADERS) $(SHARED_TRAINING_HEADERS) $(BACKENDS_DIR)/rename_torch.h $(TORCH_PCH) $(BUILD)/.backends-cache-stamp | $(BUILD)
	@mkdir -p $(dir $@)
	$(torch_CC) -O2 -fPIC $(EXTRA_CFLAGS) $(torch_CFLAGS) $(TORCH_PCH_USE) -include $(BACKENDS_DIR)/rename_torch.h -c -o $@ $<

$(BUILD)/backend_mlx/%.o: $(BACKENDS_DIR)/backend_mlx/%.cpp $(BACKEND_MLX_HEADERS) $(SHARED_TRAINING_HEADERS) $(BACKENDS_DIR)/rename_mlx.h $(BUILD)/.backends-cache-stamp | $(BUILD)
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
$(BUILD)/shared_training_$(1)/%.o: $(BACKENDS_DIR)/shared/training/%.c $(SHARED_TRAINING_HEADERS) $(BACKENDS_DIR)/backend.h $(BACKENDS_DIR)/rename_$(1).h $(BUILD)/.backends-cache-stamp | $(BUILD)
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
$(BUILD)/backend_$(1).o: $($(1)_SRC) $(BACKENDS_DIR)/backend.h $(BACKENDS_DIR)/rename_$(1).h $(BACKEND_TAPE_HEADERS) $(BACKEND_TORCH_HEADERS) $(BACKEND_MLX_HEADERS) $(SHARED_TRAINING_HEADERS) $(BUILD)/.backends-cache-stamp | $(BUILD)
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
# IDRISML_LOG: build-time level ceiling for the C/Idris log scheme.
# Accepted: silent | error | warn | info | debug | trace. Default info.
# Levels above this are #if-elided at compile time; the runtime env var
# IDRISML_LOG_LEVEL can lower (but not raise) the active level.
IDRISML_LOG ?= info
ifeq ($(IDRISML_LOG),silent)
  IDRISML_LOG_CFLAG := -DIDRISML_LOG_LEVEL=IDRISML_LEVEL_SILENT
else ifeq ($(IDRISML_LOG),error)
  IDRISML_LOG_CFLAG := -DIDRISML_LOG_LEVEL=IDRISML_LEVEL_ERROR
else ifeq ($(IDRISML_LOG),warn)
  IDRISML_LOG_CFLAG := -DIDRISML_LOG_LEVEL=IDRISML_LEVEL_WARN
else ifeq ($(IDRISML_LOG),info)
  IDRISML_LOG_CFLAG := -DIDRISML_LOG_LEVEL=IDRISML_LEVEL_INFO
else ifeq ($(IDRISML_LOG),debug)
  IDRISML_LOG_CFLAG := -DIDRISML_LOG_LEVEL=IDRISML_LEVEL_DEBUG
else ifeq ($(IDRISML_LOG),trace)
  IDRISML_LOG_CFLAG := -DIDRISML_LOG_LEVEL=IDRISML_LEVEL_TRACE
else
  IDRISML_LOG_CFLAG := -DIDRISML_LOG_LEVEL=IDRISML_LEVEL_INFO
endif

SHARED_OBJ := $(BUILD)/safetensors_$(PRIMARY).o $(BUILD)/cJSON.o $(BUILD)/shared_utils.o $(BUILD)/idx.o $(BUILD)/log.o $(BUILD)/probes.o

# Overridable so the coverage path can force clang on Linux (EXTRA_CFLAGS
# carries clang-only instrumentation flags there) — same contract as the
# per-backend tape_CC/torch_CC/mlx_CC and TEST_CC in mk/tests.mk.
SHARED_CC := cc

$(BUILD)/safetensors_$(PRIMARY).o: $(BACKENDS_DIR)/safetensors.c $(BACKENDS_DIR)/backend.h $(CJSON_H) $(BACKEND_RENAME_H) $(BUILD)/.backends-cache-stamp | $(BUILD)
	$(SHARED_CC) -O2 -fPIC $(EXTRA_CFLAGS) $(IDRISML_LOG_CFLAG) -include $(BACKEND_RENAME_H) -I$(CJSON_DIR) -c -o $@ $<

$(BUILD)/cJSON.o: $(CJSON_C) $(CJSON_H) $(BUILD)/.backends-cache-stamp | $(BUILD)
	$(SHARED_CC) -O2 -fPIC $(EXTRA_CFLAGS) -c -o $@ $<

$(BUILD)/shared_utils.o: $(BACKENDS_DIR)/shared_utils.c $(BACKENDS_DIR)/shared_utils.h $(BUILD)/.backends-cache-stamp | $(BUILD)
	$(SHARED_CC) -O2 -fPIC $(EXTRA_CFLAGS) -c -o $@ $<

$(BUILD)/idx.o: $(BACKENDS_DIR)/idx.c $(BACKENDS_DIR)/idx.h $(BUILD)/.backends-cache-stamp | $(BUILD)
	$(SHARED_CC) -O2 -fPIC $(EXTRA_CFLAGS) -c -o $@ $<

$(BUILD)/log.o: $(BACKENDS_DIR)/log.c $(BACKENDS_DIR)/log.h $(BUILD)/.backends-cache-stamp | $(BUILD)
	$(SHARED_CC) -O2 -fPIC $(EXTRA_CFLAGS) $(IDRISML_LOG_CFLAG) -c -o $@ $<

$(BUILD)/probes.o: $(BACKENDS_DIR)/probes.c $(BACKENDS_DIR)/probes.h $(BUILD)/.backends-cache-stamp | $(BUILD)
	$(SHARED_CC) -O2 -fPIC $(EXTRA_CFLAGS) -c -o $@ $<

# Drift detector: errors if any method is present in some Executor backend
# files but not all three. Run alongside other check-* targets.
.PHONY: check-executor-method-drift
check-executor-method-drift:
	@python3 scripts/check-executor-method-drift.py

# Regenerate per-backend instance method body lines from ffi_manifest.py
# (Executor/{Tape,Torch,Mlx}.idr). Idempotent — no diff if MANIFEST and
# the instance files are in sync. Hand-written overrides below
# `-- <<< END GENERATED <<<` markers are preserved.
.PHONY: gen-executor-instances
gen-executor-instances:
	@python3 scripts/codegen/gen-executor-instances.py

# CI gate: fails if running gen-executor-instances would change any file
# (i.e. someone hand-edited the marker-bounded blocks without updating
# MANIFEST, or vice versa).
.PHONY: check-executor-instances
check-executor-instances:
	@python3 scripts/codegen/gen-executor-instances.py --check

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

# Content guard for the C build artifacts — the object-file twin of
# mk/config.mk's .library-src-sha. Pure mtime dependency tracking is
# unsound under CI's cross-commit cache restore + git restore-mtime:
# a tree saved later than the current commit's file dates makes every
# restored .o look fresher than a changed .c, so make links a stale
# dylib (run 27434768856's macOS test-unit linked a pre-typed-codes
# safetensors.o and failed the new criterion asserts). The sha file's
# mtime moves only when backend C source content changes, forcing a
# rebuild of the objects regardless of timestamps.
BACKEND_C_SRCS := $(shell find packages/backends \( -name '*.c' -o -name '*.cpp' -o -name '*.h' -o -name '*.hpp' \) 2>/dev/null)

$(BUILD)/.backends-src-sha: FORCE | $(BUILD)
	@cat $(BACKEND_C_SRCS) | shasum -a 256 | cut -d' ' -f1 | cmp -s - $@ 2>/dev/null || \
		cat $(BACKEND_C_SRCS) | shasum -a 256 | cut -d' ' -f1 > $@

# A NORMAL prerequisite of every backend object + the final link. Its
# mtime bumps (via touch) only when .backends-src-sha changed — i.e. when
# backend C source CONTENT changed — so make rebuilds each stale object by
# ordinary "prereq is newer" logic. It must NOT rm the objects: an earlier
# version did, from this recipe, and because the rule is reached while make
# is already building the graph, the rm deleted .o files make had already
# stat'd as up-to-date (fresh from a cross-commit CI cache restore). make
# then never rebuilt them and the link died on the first missing .o (run
# 27499005405's macOS mlx leg: backend_meta.o). Touch-not-delete keeps the
# rebuild coherent without yanking files out from under make's planning.
$(BUILD)/.backends-cache-stamp: $(BUILD)/.backends-src-sha | $(BUILD)
	@echo "[$(BUILD_KEY)] Backend C source changed — forcing object rebuild"
	@touch $@

$(BUILD):
	mkdir -p $(BUILD)

# Final link: all listed backends' .o + shared objects (primary's
# suffix). One dylib, no symlink. Every symbol is reached by its
# suffixed name from the per-instance UserExecutor methods — no aliases.
$(LIB): $(BACKEND_OBJS) $(SHARED_OBJ) $(BUILD)/.backend-stamp $(BUILD)/.backends-cache-stamp | $(BUILD)
	$(LINK_CC) -O2 -shared $(EXTRA_LDFLAGS) -o $@ $(BACKEND_OBJS) $(SHARED_OBJ) $(BACKEND_LDFLAGS)

# Multi-link: one libidrisml.{so,dylib} with all listed BACKENDs in it.
# Primary backend's symbols are exported under both unified
# (`tensor_add`) and suffixed (`tensor_add_<primary>`) names; other
# backends' symbols are reachable only via their suffixed names.
backend: $(LIB)
