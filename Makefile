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

ntm: $(CLIB)
	idris2 --source-dir src -p contrib -o ntm src/Example/Ntm.idr
	./build/exec/ntm

bench: $(CLIB)
	idris2 --source-dir src -p contrib -o bench src/Example/Bench.idr
	./build/exec/bench

$(BUILD):
	mkdir -p $(BUILD)

sweep: $(CLIB)
	bash scripts/sweep.sh --parallel 4

sweep-quick: $(CLIB)
	bash scripts/sweep.sh --parallel 4 --quick

clean:
	rm -f $(CLIB) $(BUILD)/test_tensor

.PHONY: test-c check supervised rnn ntm bench sweep sweep-quick clean
