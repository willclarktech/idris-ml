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

# Shared C sources (backend-agnostic: serialization, JSON, data loading)
SHARED_OBJ := $(BUILD)/safetensors.o $(BUILD)/cJSON.o $(BUILD)/mnist.o

$(BUILD)/safetensors.o: csrc/safetensors.c csrc/backend.h csrc/cJSON.h | $(BUILD)
	cc -O2 -c -o $@ $<

$(BUILD)/cJSON.o: csrc/cJSON.c csrc/cJSON.h | $(BUILD)
	cc -O2 -c -o $@ $<

$(BUILD)/mnist.o: csrc/mnist.c csrc/backend.h | $(BUILD)
	cc -O2 -c -o $@ $<

$(BACKEND_LIB): $(BACKEND_SRC) csrc/backend.h $(SHARED_OBJ) | $(BUILD)
ifeq ($(BACKEND), torch)
  ifndef LIBTORCH_PATH
	$(error libtorch not found. Set LIBTORCH_PATH, install via pkg-config, or run: cd pytorch && uv sync)
  endif
endif
	$(BACKEND_CC) $(BACKEND_FLAGS) -o $@ $< $(SHARED_OBJ)

# Download MNIST dataset
download-mnist:
	bash scripts/download_mnist.sh

# Always update symlink to point to the active backend
backend: $(BACKEND_LIB)
	@ln -sf libidrisml_$(BACKEND).dylib $(LIB)

# Backend API test suite — runs against whichever backend is active
test-backend: csrc/test_backend.c backend | $(BUILD)
	cc -o $(BUILD)/test_backend csrc/test_backend.c -L$(BUILD) -lidrisml -Wl,-rpath,$(BUILD) -lm
	./$(BUILD)/test_backend

# Per-backend convenience targets
test-backend-tape:
	$(MAKE) BACKEND=tape test-backend

test-backend-mlx:
	$(MAKE) BACKEND=mlx test-backend

test-backend-torch:
	$(MAKE) BACKEND=torch test-backend

# Specialized C test suites
test-safetensors: csrc/test_safetensors.c backend | $(BUILD)
	cc -o $(BUILD)/test_safetensors csrc/test_safetensors.c -L$(BUILD) -lidrisml -Wl,-rpath,$(BUILD) -lm
	./$(BUILD)/test_safetensors

test-ntm-grad: csrc/test_ntm_grad.c backend | $(BUILD)
	cc -o $(BUILD)/test_ntm_grad csrc/test_ntm_grad.c -L$(BUILD) -lidrisml -Wl,-rpath,$(BUILD) -lm
	./$(BUILD)/test_ntm_grad

test-ntm-timestep: csrc/test_ntm_timestep.c backend | $(BUILD)
	cc -o $(BUILD)/test_ntm_timestep csrc/test_ntm_timestep.c -L$(BUILD) -lidrisml -Wl,-rpath,$(BUILD) -lm
	./$(BUILD)/test_ntm_timestep

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
example-supervised: backend
	idris2 --source-dir src -p contrib -o supervised src/Example/Supervised.idr
	cp $(LIB) build/exec/supervised_app/
	./build/exec/supervised

example-rnn: backend
	idris2 --source-dir src -p contrib -o rnn src/Example/Rnn.idr
	cp $(LIB) build/exec/rnn_app/
	./build/exec/rnn

example-lstm: backend
	idris2 --source-dir src -p contrib -o lstm src/Example/Lstm.idr
	cp $(LIB) build/exec/lstm_app/
	./build/exec/lstm

example-ntm-copy: backend
	idris2 --source-dir src -p contrib -o ntm-copy src/Example/NtmCopy.idr
	cp $(LIB) build/exec/ntm-copy_app/
	./build/exec/ntm-copy

example-ntm-associative-recall: backend
	idris2 --source-dir src -p contrib -o ntm-associative-recall src/Example/NtmAssociativeRecall.idr
	cp $(LIB) build/exec/ntm-associative-recall_app/
	./build/exec/ntm-associative-recall

example-transformer: backend
	idris2 --source-dir src -p contrib -o transformer src/Example/Transformer.idr
	cp $(LIB) build/exec/transformer_app/
	./build/exec/transformer

example-gpt: backend
	idris2 --source-dir src -p contrib -o gpt src/Example/Gpt.idr
	cp $(LIB) build/exec/gpt_app/
	./build/exec/gpt $(GPT_ARGS)

example-mnist: backend
	idris2 --source-dir src -p contrib -o mnist src/Example/Mnist.idr
	cp $(LIB) build/exec/mnist_app/
	./build/exec/mnist $(MNIST_ARGS)

example-reinforce: backend
	idris2 --source-dir src -p contrib -o reinforce src/Example/Reinforce.idr
	cp $(LIB) build/exec/reinforce_app/
	./build/exec/reinforce $(REINFORCE_ARGS)

example-transfer: backend
	idris2 --source-dir src -p contrib -o transfer src/Example/Transfer.idr
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

example-bench: backend
	idris2 --source-dir src -p contrib -o bench src/Example/Bench.idr
	cp $(LIB) build/exec/bench_app/
	./build/exec/bench

$(BUILD):
	mkdir -p $(BUILD)

example-profile: backend
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

bench-compare: example-bench
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
	rm -f $(BUILD)/libidrisml*.dylib $(BUILD)/test_backend $(BUILD)/test_safetensors \
	      $(BUILD)/test_ntm_grad $(BUILD)/test_ntm_timestep

EXAMPLES := example-supervised example-rnn example-lstm example-transformer example-gpt example-mnist example-ntm-copy example-ntm-associative-recall example-reinforce
BACKENDS := tape mlx torch

# Run all examples on all available backends, validate RESULT lines.
# Tries to build each backend; skips gracefully if libraries not installed.
test-examples:
	@fail=0; skip=""; \
	for b in tape mlx torch; do \
		$(MAKE) --no-print-directory BACKEND=$$b backend 2>/dev/null || { skip="$$skip $$b"; continue; }; \
		for e in $(EXAMPLES); do \
			echo "--- $$e [$$b] ---"; \
			extra_args=""; \
			if [ "$$e" = "example-reinforce" ]; then extra_args="REINFORCE_ARGS=--epochs 200"; fi; \
			if [ "$$e" = "example-gpt" ]; then extra_args="GPT_ARGS=--epochs 200"; fi; \
			if [ "$$e" = "example-mnist" ]; then extra_args="MNIST_ARGS=--epochs 5"; fi; \
			output=$$($(MAKE) --no-print-directory BACKEND=$$b $$e $$extra_args 2>&1) || { echo "FAIL: $$e [$$b] crashed"; fail=1; continue; }; \
			result_line=$$(echo "$$output" | grep '^RESULT'); \
			if [ -z "$$result_line" ]; then \
				echo "FAIL: $$e [$$b] -- no RESULT line"; \
				fail=1; \
			else \
				echo "ok: $$result_line"; \
			fi; \
		done; \
	done; \
	if [ -n "$$skip" ]; then echo "Skipped backends (not installed):$$skip"; fi; \
	if [ $$fail -ne 0 ]; then echo "Some integration tests FAILED"; exit 1; fi; \
	echo "All integration tests passed."

all-backends: test-examples

# Run everything: Idris unit tests, C backend tests, specialized tests,
# integration tests, PyTorch reference tests (if available)
test-all:
	@echo "=== Idris unit tests ==="
	$(MAKE) test
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
	@if command -v uv >/dev/null 2>&1 && [ -f pytorch/pyproject.toml ]; then \
		echo "=== PyTorch reference tests ==="; \
		$(MAKE) ref-test; \
	else \
		echo "=== PyTorch reference tests SKIPPED (uv not found) ==="; \
	fi
	@echo ""
	@echo "=== All tests complete ==="

.PHONY: all-backends test test-all download-mnist test-backend test-backend-tape test-backend-mlx \
        test-backend-torch test-safetensors test-ntm-grad test-ntm-timestep \
        test-examples check example-supervised example-rnn example-lstm \
        example-ntm-copy example-ntm-associative-recall example-reinforce \
        example-gpt example-mnist example-transformer example-transfer example-transfer-demo \
        example-bench example-profile sweep sweep-quick clean \
        backend print-torch ref-setup ref-supervised ref-rnn ref-lstm ref-ntm-copy \
        ref-ntm-recall ref-transformer bench-py bench-compare ref-test ref-lint \
        ref-typecheck ref-convergence ref-convergence-copy ref-convergence-recall
