UNAME := $(shell uname)
BUILD := build
BACKEND ?= tape

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
    LIBTORCH_PATH := $(shell pytorch/.venv/bin/python3 -c "import torch, os; print(os.path.dirname(torch.__file__))" 2>/dev/null)
  endif
  ifdef LIBTORCH_PATH
    TORCH_INC := $(LIBTORCH_PATH)/include
    TORCH_INC_API := $(LIBTORCH_PATH)/include/torch/csrc/api/include
    TORCH_LIB := $(LIBTORCH_PATH)/lib
  endif
  BACKEND_SRC := csrc/backend_torch.cpp
  ifeq ($(UNAME), Darwin)
    LIB := $(BUILD)/libidrisml.dylib
    BACKEND_FLAGS := -std=c++17 -O2 -shared -I$(TORCH_INC) -I$(TORCH_INC_API) -L$(TORCH_LIB) -ltorch -ltorch_cpu -lc10 -Wl,-rpath,$(TORCH_LIB)
    BACKEND_CC := c++
  else
    LIB := $(BUILD)/libidrisml.so
    BACKEND_FLAGS := -std=c++17 -O2 -shared -fPIC -I$(TORCH_INC) -I$(TORCH_INC_API) -L$(TORCH_LIB) -ltorch -ltorch_cpu -lc10 -Wl,-rpath,$(TORCH_LIB)
    BACKEND_CC := c++
  endif
else ifeq ($(BACKEND), mlx)
  # MLX backend: Apple Metal GPU via MLX C++ API
  # Auto-detect MLX from nix store (cached after first build)
  ifndef MLX_SITE
    MLX_SITE := $(shell python3 -c "import mlx; import os; print(os.path.dirname(mlx.__file__))" 2>/dev/null)
  endif
  ifeq ($(MLX_SITE),)
    MLX_SITE := $(shell nix build nixpkgs\#python3Packages.mlx --no-link --print-out-paths 2>/dev/null)/lib/python3.13/site-packages/mlx
  endif
  MLX_INC := $(MLX_SITE)/include
  MLX_LIB := $(MLX_SITE)/lib
  BACKEND_SRC := csrc/backend_mlx.cpp
  ifeq ($(UNAME), Darwin)
    LIB := $(BUILD)/libidrisml.dylib
    BACKEND_FLAGS := -std=c++20 -O2 -shared -I$(MLX_INC) -L$(MLX_LIB) -lmlx -Wl,-rpath,$(MLX_LIB) -framework Accelerate -framework Metal -framework Foundation
    BACKEND_CC := c++
  endif
else
  # Tape backend (default): custom C, no libtorch dependency
  BACKEND_SRC := csrc/backend_tape.c
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

# Per-backend dylib: each backend compiles to its own file.
# Switching backends = updating a symlink (instant, no recompile).
BACKEND_LIB := $(BUILD)/libidrisml_$(BACKEND).dylib

$(BACKEND_LIB): $(BACKEND_SRC) csrc/backend.h | $(BUILD)
ifeq ($(BACKEND), torch)
  ifndef LIBTORCH_PATH
	$(error libtorch not found. Set LIBTORCH_PATH, install via pkg-config, or run: cd pytorch && uv sync)
  endif
endif
	$(BACKEND_CC) $(BACKEND_FLAGS) -o $@ $<

# Always update symlink to point to the active backend
backend: $(BACKEND_LIB)
	@ln -sf libidrisml_$(BACKEND).dylib $(LIB)

# Backend-specific test suites
test-backend-torch: csrc/test_backend.c | $(BUILD)
	$(MAKE) BACKEND=torch backend
	cc -o $(BUILD)/test_backend csrc/test_backend.c -L$(BUILD) -lidrisml -Wl,-rpath,$(BUILD) -lm
	./$(BUILD)/test_backend

test-backend-tape: csrc/test_backend_tape.c | $(BUILD)
	$(MAKE) BACKEND=tape backend
	cc -o $(BUILD)/test_tape csrc/test_backend_tape.c -L$(BUILD) -lidrisml -Wl,-rpath,$(BUILD) -lm
	./$(BUILD)/test_tape

print-torch:
	@echo "LIBTORCH_PATH=$(LIBTORCH_PATH)"
	@echo "TORCH_INC=$(TORCH_INC)"
	@echo "TORCH_LIB=$(TORCH_LIB)"
	@echo "BACKEND=$(BACKEND)"
	@echo "LIB=$(LIB)"

# Idris build (type-check library)
check: backend
	idris2 --build idris-ml.ipkg

# Idris tests
test: check
	idris2 --source-dir src --source-dir test/src -p contrib -o test test/src/Main.idr
	cp $(LIB) build/exec/test_app/
	./build/exec/test

# Build and run examples
supervised: backend
	idris2 --source-dir src -p contrib -o supervised src/Example/Supervised.idr
	cp $(LIB) build/exec/supervised_app/
	./build/exec/supervised

rnn: backend
	idris2 --source-dir src -p contrib -o rnn src/Example/Rnn.idr
	cp $(LIB) build/exec/rnn_app/
	./build/exec/rnn

lstm: backend
	idris2 --source-dir src -p contrib -o lstm src/Example/Lstm.idr
	cp $(LIB) build/exec/lstm_app/
	./build/exec/lstm

ntm-copy: backend
	idris2 --source-dir src -p contrib -o ntm-copy src/Example/NtmCopy.idr
	cp $(LIB) build/exec/ntm-copy_app/
	./build/exec/ntm-copy

ntm-associative-recall: backend
	idris2 --source-dir src -p contrib -o ntm-associative-recall src/Example/NtmAssociativeRecall.idr
	cp $(LIB) build/exec/ntm-associative-recall_app/
	./build/exec/ntm-associative-recall

transformer: backend
	idris2 --source-dir src -p contrib -o transformer src/Example/Transformer.idr
	cp $(LIB) build/exec/transformer_app/
	./build/exec/transformer

bench: backend
	idris2 --source-dir src -p contrib -o bench src/Example/Bench.idr
	cp $(LIB) build/exec/bench_app/
	./build/exec/bench

$(BUILD):
	mkdir -p $(BUILD)

profile: backend
	idris2 --source-dir src -p contrib -o profile src/Example/Profile.idr
	cp $(LIB) build/exec/profile_app/
	./build/exec/profile

sweep: backend
	bash scripts/sweep.sh --parallel 4

sweep-quick: backend
	bash scripts/sweep.sh --parallel 4 --quick

# PyTorch reference implementation (uv manages Python)
ref-setup:
	cd pytorch && uv sync --dev

bench-py:
	cd pytorch && uv run python -m torch_ref.benchmark $(BENCH)

bench-compare: backend
	idris2 --source-dir src -p contrib -o bench src/Example/Bench.idr
	cp $(LIB) build/exec/bench_app/
	cd pytorch && uv run python -m torch_ref.compare

ref-supervised:
	cd pytorch && uv run python -m torch_ref.scripts.supervised

ref-rnn:
	cd pytorch && uv run python -m torch_ref.scripts.rnn

ref-lstm:
	cd pytorch && uv run python -m torch_ref.scripts.lstm

ref-ntm-copy:
	cd pytorch && uv run python -m torch_ref.scripts.ntm_copy

ref-ntm-recall:
	cd pytorch && uv run python -m torch_ref.scripts.ntm_recall

ref-transformer:
	cd pytorch && uv run python -m torch_ref.scripts.transformer

ref-test:
	cd pytorch && uv run pytest torch_ref/correctness/ -v

ref-lint:
	cd pytorch && uv run ruff check torch_ref/ && uv run ruff format --check torch_ref/

ref-typecheck:
	cd pytorch && uv run pyright torch_ref/

ref-convergence:
	cd pytorch && uv run python -u -m torch_ref.scripts.convergence --task both

ref-convergence-copy:
	cd pytorch && uv run python -u -m torch_ref.scripts.convergence --task copy

ref-convergence-recall:
	cd pytorch && uv run python -u -m torch_ref.scripts.convergence --task recall

clean:
	rm -f $(BUILD)/libidrisml*.dylib $(BUILD)/test_backend $(BUILD)/test_tape

.PHONY: test test-backend-torch test-backend-tape check supervised rnn lstm ntm-copy ntm-associative-recall transformer bench profile sweep sweep-quick clean backend print-torch ref-setup ref-supervised ref-rnn ref-lstm ref-ntm-copy ref-ntm-recall ref-transformer bench-py bench-compare ref-test ref-lint ref-typecheck ref-convergence ref-convergence-copy ref-convergence-recall
