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
	./build/exec/supervised

rnn: $(CLIB)
	idris2 --source-dir src -p contrib -o rnn src/Example/Rnn.idr
	./build/exec/rnn

ntm-copy: $(CLIB)
	idris2 --source-dir src -p contrib -o ntm-copy src/Example/NtmCopy.idr
	./build/exec/ntm-copy

ntm-associative-recall: $(CLIB)
	idris2 --source-dir src -p contrib -o ntm-associative-recall src/Example/NtmAssociativeRecall.idr
	./build/exec/ntm-associative-recall

bench: $(CLIB)
	idris2 --source-dir src -p contrib -o bench src/Example/Bench.idr
	./build/exec/bench

$(BUILD):
	mkdir -p $(BUILD)

sweep: $(CLIB)
	bash scripts/sweep.sh --parallel 4

sweep-quick: $(CLIB)
	bash scripts/sweep.sh --parallel 4 --quick

# PyTorch benchmarks (uv manages Python)
bench-setup:
	cd bench && uv sync --dev

bench-py:
	cd bench && uv run python -m bench.benchmark

bench-compare: $(CLIB)
	idris2 --source-dir src -p contrib -o bench src/Example/Bench.idr
	cd bench && uv run python -m bench.compare

bench-test:
	cd bench && uv run pytest bench/correctness/ -v

bench-lint:
	cd bench && uv run ruff check bench/ && uv run ruff format --check bench/

bench-typecheck:
	cd bench && uv run pyright bench/

bench-convergence:
	cd bench && uv run python -u -m bench.scripts.convergence --task both

bench-convergence-copy:
	cd bench && uv run python -u -m bench.scripts.convergence --task copy

bench-convergence-recall:
	cd bench && uv run python -u -m bench.scripts.convergence --task recall

bench-convergence-recall-lstm:
	cd bench && uv run python -u -m bench.scripts.convergence --task recall --recall-controller lstm --recall-n 128

bench-convergence-recall-lstm-small:
	cd bench && uv run python -u -m bench.scripts.convergence --task recall --recall-controller lstm --recall-n 16

bench-convergence-recall-rnn-small:
	cd bench && uv run python -u -m bench.scripts.convergence --task recall --recall-controller rnn --recall-n 16

clean:
	rm -f $(CLIB) $(BUILD)/test_tensor

.PHONY: test test-c check supervised rnn ntm-copy ntm-associative-recall bench sweep sweep-quick clean bench-setup bench-py bench-compare bench-test bench-lint bench-typecheck bench-convergence bench-convergence-copy bench-convergence-recall bench-convergence-recall-lstm bench-convergence-recall-lstm-small bench-convergence-recall-rnn-small
