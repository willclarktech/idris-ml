UNAME := $(shell uname)
BUILD := build

# C shared library
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

.PHONY: test test-c check supervised rnn lstm ntm-copy ntm-associative-recall bench sweep sweep-quick clean ref-setup bench-py bench-compare ref-test ref-lint ref-typecheck ref-convergence ref-convergence-copy ref-convergence-recall
