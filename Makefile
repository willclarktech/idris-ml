UNAME := $(shell uname)
BUILD := build
BACKEND ?= tape

# Per-example wall-clock cap for test-examples. Examples exceeding this are
# killed and reported as timeouts. Override with `EXAMPLE_TIMEOUT=900 make ...`.
EXAMPLE_TIMEOUT ?= 600

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
export IDRIS2_PACKAGE_PATH := $(IDRIS2_LOCAL)/idris2-0.8.0

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
    BACKEND_FLAGS := -std=c++17 -O2 -shared -fPIC -I$(TORCH_INC) -I$(TORCH_INC_API) -L$(TORCH_LIB) -ltorch -ltorch_cpu -lc10 -Wl,-rpath,$(TORCH_LIB)
    BACKEND_CC := c++
  endif
else ifeq ($(BACKEND), mlx)
  # MLX backend: Apple Metal GPU via MLX C++ API
  ifndef MLX_SITE
    MLX_SITE := $(shell python3 -c "import mlx; import os; print(os.path.dirname(mlx.__file__))" 2>/dev/null)
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

# Per-backend dylib: each backend compiles to its own file.
# Switching backends = updating a symlink (instant, no recompile).
BACKEND_LIB := $(BUILD)/libidrisml_$(BACKEND).dylib

# Shared C sources (backend-agnostic: serialization, JSON, data loading)
SHARED_OBJ := $(BUILD)/safetensors.o $(BUILD)/cJSON.o $(BUILD)/mnist.o $(BUILD)/dataloader.o

$(BUILD)/safetensors.o: $(BACKENDS_DIR)/safetensors.c $(BACKENDS_DIR)/backend.h $(BACKENDS_DIR)/cJSON.h | $(BUILD)
	cc -O2 -c -o $@ $<

$(BUILD)/cJSON.o: $(BACKENDS_DIR)/cJSON.c $(BACKENDS_DIR)/cJSON.h | $(BUILD)
	cc -O2 -c -o $@ $<

$(BUILD)/mnist.o: $(BACKENDS_DIR)/mnist.c $(BACKENDS_DIR)/backend.h | $(BUILD)
	cc -O2 -c -o $@ $<

$(BUILD)/dataloader.o: $(BACKENDS_DIR)/dataloader.c | $(BUILD)
	cc -O2 -c -o $@ $<

$(BACKEND_LIB): $(BACKEND_SRC) $(BACKENDS_DIR)/backend.h $(SHARED_OBJ) | $(BUILD)
ifeq ($(BACKEND), torch)
  ifndef LIBTORCH_PATH
	$(error libtorch not found. Set LIBTORCH_PATH, install via pkg-config, or run: cd packages/pytorch && uv sync)
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
	stdbuf -oL ./packages/idris-gym/test/build/exec/idris-gym-test

# Unit tests for idris-ml-examples (runs moved Test.Generate)
test-examples-unit: install-examples
	cd packages/idris-ml-examples/test && idris2 --build test.ipkg
	cp $(LIB) packages/idris-ml-examples/test/build/exec/idris-ml-examples-test_app/
	stdbuf -oL ./packages/idris-ml-examples/test/build/exec/idris-ml-examples-test

# Build and run examples (require: make install)
example-supervised: install
	idris2 $(IDRIS_FLAGS) -o supervised $(EXAMPLE_SRC)/Example/Supervised.idr
	cp $(LIB) build/exec/supervised_app/
	./build/exec/supervised

example-rnn: install
	idris2 $(IDRIS_FLAGS) -o rnn $(EXAMPLE_SRC)/Example/Rnn.idr
	cp $(LIB) build/exec/rnn_app/
	./build/exec/rnn

example-lstm: install
	idris2 $(IDRIS_FLAGS) -o lstm $(EXAMPLE_SRC)/Example/Lstm.idr
	cp $(LIB) build/exec/lstm_app/
	./build/exec/lstm

example-ntm-copy: install
	idris2 $(IDRIS_FLAGS) -o ntm-copy $(EXAMPLE_SRC)/Example/NtmCopy.idr
	cp $(LIB) build/exec/ntm-copy_app/
	stdbuf -oL ./build/exec/ntm-copy

example-ntm-associative-recall: install
	idris2 $(IDRIS_FLAGS) -o ntm-associative-recall $(EXAMPLE_SRC)/Example/NtmAssociativeRecall.idr
	cp $(LIB) build/exec/ntm-associative-recall_app/
	stdbuf -oL ./build/exec/ntm-associative-recall

example-dnc-copy: install
	idris2 $(IDRIS_FLAGS) -o dnc-copy $(EXAMPLE_SRC)/Example/DncCopy.idr
	cp $(LIB) build/exec/dnc-copy_app/
	stdbuf -oL ./build/exec/dnc-copy

example-dnc-recall: install
	idris2 $(IDRIS_FLAGS) -o dnc-recall $(EXAMPLE_SRC)/Example/DncAssociativeRecall.idr
	cp $(LIB) build/exec/dnc-recall_app/
	stdbuf -oL ./build/exec/dnc-recall

example-transformer: install
	idris2 $(IDRIS_FLAGS) -o transformer $(EXAMPLE_SRC)/Example/Transformer.idr
	cp $(LIB) build/exec/transformer_app/
	./build/exec/transformer

example-gpt: install
	idris2 $(IDRIS_FLAGS) -o gpt $(EXAMPLE_SRC)/Example/Gpt.idr
	cp $(LIB) build/exec/gpt_app/
	stdbuf -oL ./build/exec/gpt $(GPT_ARGS)

example-mnist: install download-mnist
	idris2 $(IDRIS_FLAGS) -o mnist $(EXAMPLE_SRC)/Example/Mnist.idr
	cp $(LIB) build/exec/mnist_app/
	stdbuf -oL ./build/exec/mnist $(MNIST_ARGS)

example-seq-classify: install
	idris2 $(IDRIS_FLAGS) -o seq-classify $(EXAMPLE_SRC)/Example/SeqClassify.idr
	cp $(LIB) build/exec/seq-classify_app/
	stdbuf -oL ./build/exec/seq-classify $(SEQ_ARGS)

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

example-dqn: install
	idris2 $(IDRIS_FLAGS) -o dqn $(EXAMPLE_SRC)/Example/Dqn.idr
	cp $(LIB) build/exec/dqn_app/
	stdbuf -oL ./build/exec/dqn $(DQN_ARGS)

example-a2c: install
	idris2 $(IDRIS_FLAGS) -o a2c $(EXAMPLE_SRC)/Example/A2c.idr
	cp $(LIB) build/exec/a2c_app/
	stdbuf -oL ./build/exec/a2c $(A2C_ARGS)

example-ppo: install
	idris2 $(IDRIS_FLAGS) -o ppo $(EXAMPLE_SRC)/Example/Ppo.idr
	cp $(LIB) build/exec/ppo_app/
	stdbuf -oL ./build/exec/ppo $(PPO_ARGS)

example-sac: install
	idris2 $(IDRIS_FLAGS) -o sac $(EXAMPLE_SRC)/Example/Sac.idr
	cp $(LIB) build/exec/sac_app/
	stdbuf -oL ./build/exec/sac $(SAC_ARGS)

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
	./build/exec/bench

$(BUILD):
	mkdir -p $(BUILD)

example-profile: install
	idris2 $(IDRIS_FLAGS) -o profile $(EXAMPLE_SRC)/Example/Profile.idr
	cp $(LIB) build/exec/profile_app/
	./build/exec/profile

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
	rm -f $(BUILD)/libidrisml*.dylib $(BUILD)/test_backend $(BUILD)/test_safetensors \
	      $(BUILD)/test_ntm_grad $(BUILD)/test_ntm_timestep $(BUILD)/bench_ops

# Examples run on every built backend. Keep in sync with packages/idris-ml-examples/src/Example/.
# Excluded intentionally: Bench (bench-compare target), Profile (example-profile target) —
# they don't emit RESULT lines and are covered elsewhere.
EXAMPLES := example-supervised example-rnn example-lstm example-transformer example-gpt example-mnist example-seq-classify example-ntm-copy example-ntm-associative-recall example-dnc-copy example-dnc-recall example-reinforce example-q-learning example-sarsa example-monte-carlo example-dqn example-a2c example-ppo example-sac example-transfer
BACKENDS := tape mlx torch

# Run all examples on all available backends, validate RESULT lines.
# Tries to build each backend; skips gracefully if libraries not installed.
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
			echo "--- $$e [$$b] ---"; \
			extra_args=""; \
			case "$$e" in \
				example-reinforce)   extra_args="REINFORCE_ARGS=--epochs 200" ;; \
				example-gpt)         extra_args="GPT_ARGS=--epochs 200" ;; \
				example-mnist)       extra_args="MNIST_ARGS=--epochs 5" ;; \
				example-seq-classify) extra_args="SEQ_ARGS=--epochs 200" ;; \
				example-dqn)         extra_args="DQN_ARGS=--epochs 50" ;; \
				example-a2c)         extra_args="A2C_ARGS=--epochs 1000" ;; \
				example-ppo)         extra_args="PPO_ARGS=--epochs 20" ;; \
				example-sac)         extra_args="SAC_ARGS=--epochs 1500" ;; \
			esac; \
			if [ -n "$$extra_args" ]; then \
				output=$$($$TIMEOUT_PREFIX $(MAKE) --no-print-directory BACKEND=$$b $$e "$$extra_args" 2>&1); rc=$$?; \
			else \
				output=$$($$TIMEOUT_PREFIX $(MAKE) --no-print-directory BACKEND=$$b $$e 2>&1); rc=$$?; \
			fi; \
			if [ $$rc -ne 0 ]; then \
				if [ $$rc -eq 124 ]; then \
					echo "FAIL: $$e [$$b] timed out (>$(EXAMPLE_TIMEOUT)s)"; \
				else \
					echo "FAIL: $$e [$$b] crashed (rc=$$rc)"; \
				fi; \
				echo "$$output" | tail -40 | sed 's/^/  | /'; \
				fail=1; continue; \
			fi; \
			result_line=$$(echo "$$output" | grep '^RESULT' | head -1); \
			if [ -z "$$result_line" ]; then \
				echo "FAIL: $$e [$$b] -- no RESULT line"; \
				echo "$$output" | tail -40 | sed 's/^/  | /'; \
				fail=1; \
			else \
				scripts/check-result.sh "$$e" "$$result_line" || fail=1; \
			fi; \
		done; \
	done; \
	if [ -z "$$skip" ]; then \
		echo "--- example-transfer-demo (tape->mlx->torch round-trip) ---"; \
		demo_out=$$($$TIMEOUT_PREFIX $(MAKE) --no-print-directory example-transfer-demo 2>&1); demo_rc=$$?; \
		if [ $$demo_rc -ne 0 ]; then \
			if [ $$demo_rc -eq 124 ]; then echo "FAIL: example-transfer-demo timed out (>$(EXAMPLE_TIMEOUT)s)"; \
			else echo "FAIL: example-transfer-demo crashed (rc=$$demo_rc)"; fi; \
			echo "$$demo_out" | tail -40 | sed 's/^/  | /'; \
			fail=1; \
		else \
			result_line=$$(echo "$$demo_out" | grep '^RESULT' | tail -1); \
			if [ -z "$$result_line" ]; then \
				echo "FAIL: example-transfer-demo -- no RESULT line"; \
				echo "$$demo_out" | tail -40 | sed 's/^/  | /'; \
				fail=1; \
			else \
				scripts/check-result.sh "example-transfer-demo" "$$result_line" || fail=1; \
			fi; \
		fi; \
	else \
		echo "--- example-transfer-demo: skipped (requires tape+mlx+torch; skipped:$$skip) ---"; \
	fi; \
	if [ -n "$$skip" ]; then echo "Skipped backends (not installed or build failed):$$skip"; fi; \
	if [ $$fail -ne 0 ]; then echo "Some integration tests FAILED"; exit 1; fi; \
	echo "All integration tests passed."

all-backends: test-examples

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

.PHONY: all check-all all-backends test test-gym test-examples-unit test-all download-mnist test-backend test-backend-tape test-backend-mlx \
        test-backend-torch test-safetensors test-ntm-grad test-ntm-timestep \
        test-examples check check-gym check-notebook check-examples install install-core install-gym install-notebook install-examples \
        example-supervised example-rnn example-lstm \
        example-ntm-copy example-ntm-associative-recall example-dnc-copy example-dnc-recall \
        example-reinforce \
        example-gpt example-mnist example-seq-classify example-transformer \
        example-transfer example-transfer-demo \
        example-bench example-profile sweep sweep-quick clean \
        backend print-torch ref-setup ref-supervised ref-rnn ref-lstm ref-ntm-copy \
        ref-ntm-recall ref-dnc-copy ref-dnc-recall \
        ref-transformer bench-py bench-compare bench-ops bench-ops-py bench-ops-compare test-ref ref-test ref-lint \
        ref-typecheck ref-convergence ref-convergence-copy ref-convergence-recall \
        jupyter-install jupyter-lab test-jupyter test-jupyter-unit test-notebooks
