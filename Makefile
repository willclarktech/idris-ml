UNAME := $(shell uname)
BUILD := build
BACKEND ?= tape

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
ifeq ($(BACKEND), torch)
  # libtorch detection
  ifndef LIBTORCH_PATH
    LIBTORCH_PATH := $(shell pkg-config --variable=prefix torch 2>/dev/null)
  endif
  ifndef LIBTORCH_PATH
    LIBTORCH_PATH := $(shell python3 -c "import torch, os; print(os.path.dirname(torch.__file__))" 2>/dev/null)
  endif
  ifndef LIBTORCH_PATH
    LIBTORCH_PATH := $(shell packages/pytorch/.venv/bin/python3 -c "import torch, os; print(os.path.dirname(torch.__file__))" 2>/dev/null)
  endif
  ifdef LIBTORCH_PATH
    TORCH_INC := $(LIBTORCH_PATH)/include
    TORCH_INC_API := $(LIBTORCH_PATH)/include/torch/csrc/api/include
    TORCH_LIB := $(LIBTORCH_PATH)/lib
  endif
  BACKEND_SRC := $(BACKENDS_DIR)/backend_torch.cpp
  ifeq ($(UNAME), Darwin)
    LIB := $(BUILD)/libidrisml.dylib
    BACKEND_FLAGS := -std=c++17 -O2 -shared -I$(TORCH_INC) -I$(TORCH_INC_API) -L$(TORCH_LIB) -ltorch -ltorch_cpu -lc10 -Wl,-rpath,$(TORCH_LIB)
    BACKEND_CC := c++
  else
    LIB := $(BUILD)/libidrisml.so
    # GNU ld defaults to --as-needed since binutils 2.x, which drops NEEDED tags
    # for libs whose symbols aren't *directly* referenced in our object files.
    # libidrisml.so calls torch C++ API which transitively pulls c10/torch_cpu
    # symbols at runtime — but the linker can't see those needs at static-link
    # time, so it strips the NEEDED tags and dlopen later trips on
    # `undefined symbol: _ZTIN3c105ErrorE` (c10::Error RTTI).
    # `--no-as-needed` forces NEEDED for the libtorch trio explicitly.
    BACKEND_FLAGS := -std=c++17 -O2 -shared -fPIC -I$(TORCH_INC) -I$(TORCH_INC_API) -L$(TORCH_LIB) -Wl,--no-as-needed -ltorch -ltorch_cpu -lc10 -Wl,--as-needed -Wl,-rpath,$(TORCH_LIB)
    BACKEND_CC := c++
  endif
else ifeq ($(BACKEND), mlx)
  # MLX backend: Apple Metal GPU via MLX C++ API
  #
  # Detection covers both the historical single-package mlx wheel and the
  # 0.31+ namespace-package layout (mlx + mlx-metal). For namespace
  # packages, `mlx.__file__` is None and the C++ headers/libs ship in the
  # mlx-metal package's site-packages dir. Pick whichever package has the
  # `include/` subdir (where the Makefile expects MLX_SITE/include/mlx/*.h).
  ifndef MLX_SITE
    MLX_SITE := $(shell python3 -c "import importlib.util as u, os; print(next((p for n in ('mlx','mlx_metal') for s in [u.find_spec(n)] if s for p in [s.submodule_search_locations[0] if s.submodule_search_locations else os.path.dirname(s.origin)] if os.path.isdir(os.path.join(p,'include'))), ''))" 2>/dev/null)
  endif
  ifeq ($(MLX_SITE),)
    MLX_SITE := $(shell nix build nixpkgs\#python3Packages.mlx --no-link --print-out-paths 2>/dev/null)/lib/python3.13/site-packages/mlx
  endif
  MLX_INC := $(MLX_SITE)/include
  MLX_LIB := $(MLX_SITE)/lib
  BACKEND_SRC := $(BACKENDS_DIR)/backend_mlx.cpp
  ifeq ($(UNAME), Darwin)
    LIB := $(BUILD)/libidrisml.dylib
    BACKEND_FLAGS := -std=c++20 -O2 -shared -I$(MLX_INC) -L$(MLX_LIB) -lmlx -Wl,-rpath,$(MLX_LIB) -framework Accelerate -framework Metal -framework Foundation
    BACKEND_CC := c++
  else
    # MLX is Apple-only. Without an explicit $(error), BACKEND_FLAGS and
    # BACKEND_CC stay empty, the recipe expands to ` -o $@ ...`, and GNU
    # make's leading-`-` "ignore errors" kicks in — silently turning a
    # broken build into success. test-examples then runs examples against
    # a non-existent dylib and they all crash. Errror loudly instead so
    # test-examples' backend-skip path triggers (||).
    $(error MLX backend requires macOS; current UNAME=$(UNAME))
  endif
else
  # Tape backend (default): custom C, no libtorch dependency
  BACKEND_SRC := $(BACKENDS_DIR)/backend_tape.c
  ifeq ($(UNAME), Darwin)
    LIB := $(BUILD)/libidrisml.dylib
    BACKEND_FLAGS := -O2 -shared -DACCELERATE_NEW_LAPACK -framework Accelerate
    BACKEND_CC := cc
  else
    LIB := $(BUILD)/libidrisml.so
    BACKEND_FLAGS := -O2 -shared -fPIC -lm -lblas
    BACKEND_CC := cc
  endif
endif

# Per-backend shared library: each backend compiles to its own file.
# Switching backends = updating a symlink (instant, no recompile).
BACKEND_LIB := $(BUILD)/libidrisml_$(BACKEND).$(LIB_EXT)

# Shared C sources (backend-agnostic: serialization, JSON, data loading)
SHARED_OBJ := $(BUILD)/safetensors.o $(BUILD)/cJSON.o $(BUILD)/mnist.o $(BUILD)/dataloader.o

$(BUILD)/safetensors.o: $(BACKENDS_DIR)/safetensors.c $(BACKENDS_DIR)/backend.h $(BACKENDS_DIR)/cJSON.h | $(BUILD)
	cc -O2 -fPIC -c -o $@ $<

$(BUILD)/cJSON.o: $(BACKENDS_DIR)/cJSON.c $(BACKENDS_DIR)/cJSON.h | $(BUILD)
	cc -O2 -fPIC -c -o $@ $<

$(BUILD)/mnist.o: $(BACKENDS_DIR)/mnist.c $(BACKENDS_DIR)/backend.h | $(BUILD)
	cc -O2 -fPIC -c -o $@ $<

$(BUILD)/dataloader.o: $(BACKENDS_DIR)/dataloader.c | $(BUILD)
	cc -O2 -fPIC -c -o $@ $<

$(BACKEND_LIB): $(BACKEND_SRC) $(BACKENDS_DIR)/backend.h $(SHARED_OBJ) | $(BUILD)
ifeq ($(BACKEND), torch)
  ifndef LIBTORCH_PATH
	$(error libtorch not found. Set LIBTORCH_PATH, install via pkg-config, or run: cd packages/pytorch && uv sync)
  endif
endif
	$(BACKEND_CC) $(BACKEND_FLAGS) -o $@ $< $(SHARED_OBJ)

# Download MNIST dataset
dataset-mnist:
	bash scripts/dataset_mnist.sh

# Download tinyshakespeare corpus (~1 MB, 65-char vocab) for the GPT
# convergence run. Smoke gate uses the small embedded corpus and does
# not need this file.
dataset-tinyshakespeare:
	bash scripts/dataset_tinyshakespeare.sh

# Always update symlink to point to the active backend
backend: $(BACKEND_LIB)
	@ln -sf libidrisml_$(BACKEND).$(LIB_EXT) $(LIB)

# Backend API test suite — runs against whichever backend is active
test-backend: $(BACKENDS_DIR)/test_backend.c backend | $(BUILD)
	cc -o $(BUILD)/test_backend $(BACKENDS_DIR)/test_backend.c -L$(BUILD) -lidrisml -Wl,-rpath,$(BUILD) -lm
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
install-examples: install-core install-gym
	@cd packages/idris-ml-examples && IDRIS2_PREFIX=$(IDRIS2_LOCAL) idris2 --install idris-ml-examples.ipkg >/dev/null

# Install all Idris packages locally
install: install-core install-gym install-notebook install-examples build/.library-cache-stamp

# Idris build (type-check core library)
check: backend
	cd packages/idris-ml && idris2 --build idris-ml.ipkg

# Type-check gym package
check-gym:
	cd packages/idris-gym && idris2 --build idris-gym.ipkg

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

# Unit tests for idris-ml-examples (runs moved Test.Generate)
test-examples-unit: install-examples
	cd packages/idris-ml-examples/test && idris2 --build test.ipkg
	cp $(LIB) packages/idris-ml-examples/test/build/exec/idris-ml-examples-test_app/
	$(STDBUF) ./packages/idris-ml-examples/test/build/exec/idris-ml-examples-test

# Build and run examples (require: make install)
example-supervised: install
	idris2 $(IDRIS_FLAGS) -o supervised $(EXAMPLE_SRC)/Example/Supervised.idr
	cp $(LIB) build/exec/supervised_app/
	./build/exec/supervised $(SUPERVISED_ARGS)

example-rnn: install
	idris2 $(IDRIS_FLAGS) -o rnn $(EXAMPLE_SRC)/Example/Rnn.idr
	cp $(LIB) build/exec/rnn_app/
	./build/exec/rnn $(RNN_ARGS)

example-lstm: install
	idris2 $(IDRIS_FLAGS) -o lstm $(EXAMPLE_SRC)/Example/Lstm.idr
	cp $(LIB) build/exec/lstm_app/
	./build/exec/lstm $(LSTM_ARGS)

example-gru: install
	idris2 $(IDRIS_FLAGS) -o gru $(EXAMPLE_SRC)/Example/Gru.idr
	cp $(LIB) build/exec/gru_app/
	./build/exec/gru $(GRU_ARGS)

example-ntm-copy: install
	idris2 $(IDRIS_FLAGS) -o ntm-copy $(EXAMPLE_SRC)/Example/NtmCopy.idr
	cp $(LIB) build/exec/ntm-copy_app/
	$(STDBUF) ./build/exec/ntm-copy $(NTM_COPY_ARGS)

example-ntm-associative-recall: install
	idris2 $(IDRIS_FLAGS) -o ntm-associative-recall $(EXAMPLE_SRC)/Example/NtmAssociativeRecall.idr
	cp $(LIB) build/exec/ntm-associative-recall_app/
	$(STDBUF) ./build/exec/ntm-associative-recall $(NTM_RECALL_ARGS)

example-dnc-copy: install
	idris2 $(IDRIS_FLAGS) -o dnc-copy $(EXAMPLE_SRC)/Example/DncCopy.idr
	cp $(LIB) build/exec/dnc-copy_app/
	$(STDBUF) ./build/exec/dnc-copy $(DNC_COPY_ARGS)

example-dnc-recall: install
	idris2 $(IDRIS_FLAGS) -o dnc-recall $(EXAMPLE_SRC)/Example/DncAssociativeRecall.idr
	cp $(LIB) build/exec/dnc-recall_app/
	$(STDBUF) ./build/exec/dnc-recall $(DNC_RECALL_ARGS)

example-transformer: install
	idris2 $(IDRIS_FLAGS) -o transformer $(EXAMPLE_SRC)/Example/Transformer.idr
	cp $(LIB) build/exec/transformer_app/
	./build/exec/transformer $(TRANSFORMER_ARGS)

example-gpt: install
	idris2 $(IDRIS_FLAGS) -o gpt $(EXAMPLE_SRC)/Example/Gpt.idr
	cp $(LIB) build/exec/gpt_app/
	$(STDBUF) ./build/exec/gpt $(GPT_ARGS)

# Full-corpus convergence run (~hours on tape). Default `make example-gpt`
# is a ~30s embedded-corpus demo; this target is the real char-LM
# convergence target (matching nanoGPT/train_shakespeare_char.py).
example-gpt-full: install dataset-tinyshakespeare
	idris2 $(IDRIS_FLAGS) -o gpt $(EXAMPLE_SRC)/Example/Gpt.idr
	cp $(LIB) build/exec/gpt_app/
	$(STDBUF) ./build/exec/gpt --corpus tinyshakespeare --epochs 1000 $(GPT_ARGS)

example-mnist: install dataset-mnist
	idris2 $(IDRIS_FLAGS) -o mnist $(EXAMPLE_SRC)/Example/Mnist.idr
	cp $(LIB) build/exec/mnist_app/
	$(STDBUF) ./build/exec/mnist $(MNIST_ARGS)

example-seq-classify: install
	idris2 $(IDRIS_FLAGS) -o seq-classify $(EXAMPLE_SRC)/Example/SeqClassify.idr
	cp $(LIB) build/exec/seq-classify_app/
	$(STDBUF) ./build/exec/seq-classify $(SEQ_ARGS)

example-reinforce: install
	idris2 $(IDRIS_FLAGS) -o reinforce $(EXAMPLE_SRC)/Example/Reinforce.idr
	cp $(LIB) build/exec/reinforce_app/
	./build/exec/reinforce $(REINFORCE_ARGS)

example-q-learning: install
	idris2 $(IDRIS_FLAGS) -o q-learning $(EXAMPLE_SRC)/Example/QLearning.idr
	cp $(LIB) build/exec/q-learning_app/
	./build/exec/q-learning $(Q_LEARNING_ARGS)

example-sarsa: install
	idris2 $(IDRIS_FLAGS) -o sarsa $(EXAMPLE_SRC)/Example/Sarsa.idr
	cp $(LIB) build/exec/sarsa_app/
	./build/exec/sarsa $(SARSA_ARGS)

example-monte-carlo: install
	idris2 $(IDRIS_FLAGS) -o monte-carlo $(EXAMPLE_SRC)/Example/MonteCarlo.idr
	cp $(LIB) build/exec/monte-carlo_app/
	./build/exec/monte-carlo $(MONTE_CARLO_ARGS)

example-frozen-lake: install
	idris2 $(IDRIS_FLAGS) -o frozen-lake $(EXAMPLE_SRC)/Example/FrozenLake.idr
	cp $(LIB) build/exec/frozen-lake_app/
	./build/exec/frozen-lake $(FROZEN_LAKE_ARGS)

example-taxi: install
	idris2 $(IDRIS_FLAGS) -o taxi $(EXAMPLE_SRC)/Example/Taxi.idr
	cp $(LIB) build/exec/taxi_app/
	./build/exec/taxi $(TAXI_ARGS)

example-dqn: install
	idris2 $(IDRIS_FLAGS) -o dqn $(EXAMPLE_SRC)/Example/Dqn.idr
	cp $(LIB) build/exec/dqn_app/
	$(STDBUF) ./build/exec/dqn $(DQN_ARGS)

example-mountain-car: install
	idris2 $(IDRIS_FLAGS) -o mountain-car $(EXAMPLE_SRC)/Example/MountainCar.idr
	cp $(LIB) build/exec/mountain-car_app/
	$(STDBUF) ./build/exec/mountain-car $(MOUNTAIN_CAR_ARGS)

example-mountain-car-cont: install
	idris2 $(IDRIS_FLAGS) -o mountain-car-cont $(EXAMPLE_SRC)/Example/MountainCarCont.idr
	cp $(LIB) build/exec/mountain-car-cont_app/
	$(STDBUF) ./build/exec/mountain-car-cont $(MOUNTAIN_CAR_CONT_ARGS)

example-a2c: install
	idris2 $(IDRIS_FLAGS) -o a2c $(EXAMPLE_SRC)/Example/A2c.idr
	cp $(LIB) build/exec/a2c_app/
	$(STDBUF) ./build/exec/a2c $(A2C_ARGS)

example-ppo: install
	idris2 $(IDRIS_FLAGS) -o ppo $(EXAMPLE_SRC)/Example/Ppo.idr
	cp $(LIB) build/exec/ppo_app/
	$(STDBUF) ./build/exec/ppo $(PPO_ARGS)

example-sac: install
	idris2 $(IDRIS_FLAGS) -o sac $(EXAMPLE_SRC)/Example/Sac.idr
	cp $(LIB) build/exec/sac_app/
	$(STDBUF) ./build/exec/sac $(SAC_ARGS)

example-transfer: install
	idris2 $(IDRIS_FLAGS) -o transfer $(EXAMPLE_SRC)/Example/Transfer.idr
	cp $(LIB) build/exec/transfer_app/
	./build/exec/transfer $(TRANSFER_ARGS)

example-transfer-demo:
	@echo "=== Phase 1: Train on tape ==="
	$(MAKE) BACKEND=tape example-transfer TRANSFER_ARGS="--mode train --epochs 500 --save /tmp/transfer.safetensors"
	@echo ""
	@echo "=== Phase 2: Continue on mlx ==="
	$(MAKE) BACKEND=mlx example-transfer TRANSFER_ARGS="--mode continue --load /tmp/transfer.safetensors --epochs 500 --save /tmp/transfer2.safetensors"
	@echo ""
	@echo "=== Phase 3: Infer on torch ==="
	$(MAKE) BACKEND=torch example-transfer TRANSFER_ARGS="--mode infer --load /tmp/transfer2.safetensors"

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
# Links each bench_ops_<backend> directly against its specific dylib.
bench-ops-compare:
	@for b in tape mlx torch; do \
		if [ ! -f $(BUILD)/libidrisml_$$b.dylib ]; then \
			$(MAKE) --no-print-directory BACKEND=$$b backend 2>/dev/null || continue; \
		fi; \
		cc -o $(BUILD)/bench_ops_$$b $(BACKENDS_DIR)/bench_ops.c \
			$(BUILD)/libidrisml_$$b.dylib -Wl,-rpath,$(CURDIR)/$(BUILD) -lm -lc++ 2>/dev/null \
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
# Use nix Python if available (3.12+), fall back to system python3
NIX_PYTHON := $(shell nix build nixpkgs\#python3 --no-link --print-out-paths 2>/dev/null)/bin/python3
VENV_PYTHON := $(shell [ -x "$(NIX_PYTHON)" ] && echo "$(NIX_PYTHON)" || echo python3)
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
EXAMPLES := example-supervised example-rnn example-lstm example-gru example-transformer example-gpt example-mnist example-seq-classify example-ntm-copy example-ntm-associative-recall example-dnc-copy example-dnc-recall example-reinforce example-q-learning example-sarsa example-monte-carlo example-frozen-lake example-taxi example-dqn example-mountain-car example-mountain-car-cont example-a2c example-ppo example-sac example-transfer
BACKENDS := tape mlx torch

# Crash-only smoke gate: every example × 3 backends, 3-10 epochs each,
# safety-net thresholds in test-examples.expect. Catches crashes / NaN /
# divergence / missing RESULT keys; does NOT require any model to learn.
# See docs/develop/testing.md for the full testing-layer overview.
test-examples:
	@fail=0; skip=""; \
	if command -v timeout >/dev/null 2>&1; then TIMEOUT_PREFIX="timeout $(EXAMPLE_TIMEOUT)"; \
	elif command -v gtimeout >/dev/null 2>&1; then TIMEOUT_PREFIX="gtimeout $(EXAMPLE_TIMEOUT)"; \
	else echo "WARNING: no timeout/gtimeout binary; examples will not be time-bounded"; TIMEOUT_PREFIX=""; fi; \
	for b in tape mlx torch; do \
		backend_output=$$($(MAKE) --no-print-directory BACKEND=$$b backend 2>&1) || { \
			echo "--- backend $$b: build failed, skipping its examples ---"; \
			echo "$$backend_output" | tail -20 | sed 's/^/  | /'; \
			skip="$$skip $$b"; continue; \
		}; \
		for e in $(EXAMPLES); do \
			case " $(SKIP_EXAMPLES) " in *" $$b:$$e "*) \
				echo "skip: $$e [$$b] (in SKIP_EXAMPLES)"; continue ;; \
			esac; \
			echo "--- $$e [$$b] ---"; \
			extra_args=""; \
			case "$$e" in \
				example-supervised)  extra_args="SUPERVISED_ARGS=--epochs 5" ;; \
				example-rnn)         extra_args="RNN_ARGS=--epochs 5" ;; \
				example-lstm)        extra_args="LSTM_ARGS=--epochs 5" ;; \
				example-gru)         extra_args="GRU_ARGS=--epochs 5" ;; \
				example-transformer) extra_args="TRANSFORMER_ARGS=--epochs 5" ;; \
				example-reinforce)   extra_args="REINFORCE_ARGS=--epochs 10" ;; \
				example-gpt)         extra_args="GPT_ARGS=--epochs 3" ;; \
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
				output=$$($$TIMEOUT_PREFIX $(MAKE) --no-print-directory BACKEND=$$b $$e "$$extra_args" 2>&1); rc=$$?; \
			else \
				output=$$($$TIMEOUT_PREFIX $(MAKE) --no-print-directory BACKEND=$$b $$e 2>&1); rc=$$?; \
			fi; \
			t_end=$$(date +%s); elapsed=$$((t_end - t_start)); \
			if [ $$elapsed -lt 60 ]; then elapsed_fmt="$${elapsed}s"; \
			elif [ $$elapsed -lt 3600 ]; then elapsed_fmt="$$((elapsed/60))m$$((elapsed%60))s"; \
			else elapsed_fmt="$$((elapsed/3600))h$$(((elapsed%3600)/60))m"; fi; \
			if [ $$rc -ne 0 ]; then \
				if [ $$rc -eq 124 ]; then \
					echo "FAIL: $$e [$$b] timed out (>$(EXAMPLE_TIMEOUT)s) ($$elapsed_fmt)"; \
				else \
					echo "FAIL: $$e [$$b] crashed (rc=$$rc) ($$elapsed_fmt)"; \
				fi; \
				echo "$$output" | tail -40 | sed 's/^/  | /'; \
				fail=1; continue; \
			fi; \
			result_line=$$(echo "$$output" | grep '^RESULT' | head -1); \
			if [ -z "$$result_line" ]; then \
				echo "FAIL: $$e [$$b] -- no RESULT line ($$elapsed_fmt)"; \
				echo "$$output" | tail -40 | sed 's/^/  | /'; \
				fail=1; \
			else \
				scripts/check-result.sh "$$e" "$$result_line" || fail=1; \
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
        example-gpt example-gpt-full example-mnist example-seq-classify example-transformer \
        example-transfer example-transfer-demo \
        example-bench example-profile sweep sweep-quick clean \
        backend print-torch ref-setup ref-supervised ref-rnn ref-lstm ref-gru ref-ntm-copy \
        ref-ntm-recall ref-dnc-copy ref-dnc-recall \
        ref-transformer bench-py bench-compare bench-ops bench-ops-py bench-ops-compare test-ref ref-test ref-lint \
        ref-typecheck ref-convergence ref-convergence-copy ref-convergence-recall \
        jupyter-install jupyter-lab test-jupyter test-jupyter-unit test-notebooks
