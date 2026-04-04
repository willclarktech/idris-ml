UNAME := $(shell uname)
BUILD := build

# --- libtorch detection (LIBTORCH_PATH > pkg-config > python3 torch) ---
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

# C shared library (legacy backend)
CSRC := csrc/tensor.c
ifeq ($(UNAME), Darwin)
  CLIB := $(BUILD)/libidrisml.dylib
  CFLAGS := -O2 -shared -framework Accelerate
else
  CLIB := $(BUILD)/libidrisml.so
  CFLAGS := -O2 -shared -fPIC -lm
endif

$(CLIB): $(CSRC) | $(BUILD)
	cc $(CFLAGS) -o $@ $<

# libtorch backend
BACKEND_SRC := csrc/backend_torch.cpp
ifeq ($(UNAME), Darwin)
  BACKEND_LIB := $(BUILD)/libidrisml_torch.dylib
  BACKEND_CXXFLAGS := -std=c++17 -O2 -shared -I$(TORCH_INC) -I$(TORCH_INC_API) -L$(TORCH_LIB) -ltorch -ltorch_cpu -lc10 -Wl,-rpath,$(TORCH_LIB)
else
  BACKEND_LIB := $(BUILD)/libidrisml_torch.so
  BACKEND_CXXFLAGS := -std=c++17 -O2 -shared -fPIC -I$(TORCH_INC) -I$(TORCH_INC_API) -L$(TORCH_LIB) -ltorch -ltorch_cpu -lc10 -Wl,-rpath,$(TORCH_LIB)
endif

$(BACKEND_LIB): $(BACKEND_SRC) csrc/backend.h | $(BUILD)
ifdef LIBTORCH_PATH
	c++ $(BACKEND_CXXFLAGS) -o $@ $<
else
	$(error libtorch not found. Set LIBTORCH_PATH, install via pkg-config, or run: cd pytorch && uv sync)
endif

backend: $(BACKEND_LIB)

test-backend: $(BACKEND_LIB) csrc/test_backend.c | $(BUILD)
	cc -o $(BUILD)/test_backend csrc/test_backend.c -L$(BUILD) -lidrisml_torch -Wl,-rpath,$(BUILD) -lm
	./$(BUILD)/test_backend

print-torch:
	@echo "LIBTORCH_PATH=$(LIBTORCH_PATH)"
	@echo "TORCH_INC=$(TORCH_INC)"
	@echo "TORCH_LIB=$(TORCH_LIB)"

# Idris tests (check builds library .ttc files first)
test: check
	idris2 --source-dir src --source-dir test/src -p contrib -o test test/src/Main.idr
	cp $(CLIB) build/exec/test_app/
	./build/exec/test

# C tests
test-c: csrc/test_tensor.c $(CSRC) | $(BUILD)
	cc -O2 -o $(BUILD)/test_tensor csrc/test_tensor.c $(if $(filter Darwin,$(UNAME)),-framework Accelerate,-lm)
	./$(BUILD)/test_tensor

# Idris build (type-check library)
check: $(CLIB)
	idris2 --build idris-ml.ipkg

# Build and run examples
supervised: $(CLIB)
	idris2 --source-dir src -p contrib -o supervised src/Example/Supervised.idr
	cp $(CLIB) build/exec/supervised_app/
	./build/exec/supervised

rnn: $(CLIB)
	idris2 --source-dir src -p contrib -o rnn src/Example/Rnn.idr
	cp $(CLIB) build/exec/rnn_app/
	./build/exec/rnn

lstm: $(CLIB)
	idris2 --source-dir src -p contrib -o lstm src/Example/Lstm.idr
	cp $(CLIB) build/exec/lstm_app/
	./build/exec/lstm

ntm-copy: $(CLIB)
	idris2 --source-dir src -p contrib -o ntm-copy src/Example/NtmCopy.idr
	cp $(CLIB) build/exec/ntm-copy_app/
	./build/exec/ntm-copy

ntm-associative-recall: $(CLIB)
	idris2 --source-dir src -p contrib -o ntm-associative-recall src/Example/NtmAssociativeRecall.idr
	cp $(CLIB) build/exec/ntm-associative-recall_app/
	./build/exec/ntm-associative-recall

bench: $(CLIB)
	idris2 --source-dir src -p contrib -o bench src/Example/Bench.idr
	cp $(CLIB) build/exec/bench_app/
	./build/exec/bench

$(BUILD):
	mkdir -p $(BUILD)

profile: $(CLIB)
	idris2 --source-dir src -p contrib -o profile src/Example/Profile.idr
	cp $(CLIB) build/exec/profile_app/
	./build/exec/profile

sweep: $(CLIB)
	bash scripts/sweep.sh --parallel 4

sweep-quick: $(CLIB)
	bash scripts/sweep.sh --parallel 4 --quick

# PyTorch reference implementation (uv manages Python)
ref-setup:
	cd pytorch && uv sync --dev

bench-py:
	cd pytorch && uv run python -m torch_ref.benchmark

bench-compare: $(CLIB)
	idris2 --source-dir src -p contrib -o bench src/Example/Bench.idr
	cp $(CLIB) build/exec/bench_app/
	cd pytorch && uv run python -m torch_ref.compare

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
	rm -f $(CLIB) $(BUILD)/test_tensor

.PHONY: test test-c test-backend check supervised rnn lstm ntm-copy ntm-associative-recall bench profile sweep sweep-quick clean backend print-torch ref-setup bench-py bench-compare ref-test ref-lint ref-typecheck ref-convergence ref-convergence-copy ref-convergence-recall
