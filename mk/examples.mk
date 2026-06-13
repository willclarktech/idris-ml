# mk/examples.mk — all example-* targets, grouped by domain, plus
# the data they need: dataset fetch rules, SST-2, tokenized corpora,
# the HF checkpoint pattern rule, HF roundtrip gates, profile, sweep.

# Datasets: file-as-target so Make skips the fetch when the data is
# already on disk (same pattern as HF_MODELS_DIR's HF safetensors).
# Sentinel files anchor the recipe — `dataset_mnist.sh` writes 4
# files in one shot; using the first as the Make target is enough
# to gate the recipe.
.PHONY: dataset-mnist dataset-tinyshakespeare example-supervised \
        example-bert-classify-finetune data-sst2 \
        example-bert-classify-sst2-finetune \
        example-bert-classify-sst2-lora data-tinyshakespeare-distilgpt2 \
        data-tinyshakespeare-bert-tiny example-gpt2-lm-finetune \
        ref-gpt2-lm-finetune example-bert-mlm-finetune \
        ref-bert-mlm-finetune example-hf-bert-inference \
        test-e2e-hf-bert-roundtrip example-hf-gpt2-inference \
        example-hf-llama-inference \
        test-integration-lint-hf-llama-inference \
        example-hf-bitnet-inference test-e2e-hf-bitnet-roundtrip \
        test-e2e-hf-gpt2-roundtrip test-e2e-hf-llama-roundtrip \
        test-e2e-hf-llama-generate-roundtrip \
        test-e2e-transformers-oracle-llama-generate example-rnn \
        example-lstm example-gru example-bring-your-own example-ntm-copy \
        example-ntm-associative-recall example-dnc-copy \
        example-dnc-recall example-transformer example-tcast-demo \
        example-dtype-serialize example-index-ops example-dtype-pitch \
        example-precision-checkpoint test-integration-checkpoint-resume \
        example-mlx-stream-demo example-gpt example-gpt-full \
        example-mnist example-seq-classify example-reinforce \
        example-q-learning example-sarsa example-monte-carlo \
        example-frozen-lake example-taxi example-dqn \
        example-mountain-car example-mountain-car-cont example-a2c \
        example-ppo example-sac example-transfer example-precision-demo \
        example-checkpoint example-checkpoint-demo example-matmul-bench \
        example-rank-broadcast-bench example-bench example-profile sweep \
        sweep-quick

TINYSHAKESPEARE_FILE := data/tinyshakespeare/input.txt
MNIST_SENTINEL       := data/mnist/train-images-idx3-ubyte

$(TINYSHAKESPEARE_FILE):
	bash scripts/dataset_tinyshakespeare.sh

$(MNIST_SENTINEL):
	bash scripts/dataset_mnist.sh

# Convenience phony aliases preserving the public `make dataset-*`
# interface. Existing CI / docs / users referencing these names keep
# working; they just no-op when the data is already on disk.
dataset-mnist: $(MNIST_SENTINEL)

# Download tinyshakespeare corpus (~1 MB, 65-char vocab) for the GPT
# convergence run. Smoke gate uses the small embedded corpus and does
# not need this file.
dataset-tinyshakespeare: $(TINYSHAKESPEARE_FILE)

# Build and run examples (require: make install)
example-supervised: install
	idris2 $(IDRIS_FLAGS) -o supervised $(EXAMPLE_SRC)/Example/Supervised.idr
	cp $(LIB) $(BUILD)/exec/supervised_app/
	./$(BUILD)/exec/supervised $(SEED_FLAG) $(SUPERVISED_ARGS)

# BERT classification fine-tune on a synthetic 3-class task. Demonstrates
# the FT1+FT2 surface (`BertForSequenceClassification` head + optional
# `freezeByPrefix`). Tiny config (vocab=64, hidden=32, layers=1) so the
# example converges from-scratch in seconds without a pretrained
# checkpoint; the real-text warm-start workflow is parked as a
# follow-up TODO row.
example-bert-classify-finetune: install install-transformers
	idris2 $(IDRIS_FLAGS) -p idris-transformers -o bert-classify-finetune $(EXAMPLE_SRC)/Example/BertClassifyFinetune.idr
	cp $(LIB) $(BUILD)/exec/bert-classify-finetune_app/
	./$(BUILD)/exec/bert-classify-finetune $(SEED_FLAG) $(BERT_FINETUNE_ARGS)

# SST-2 dataset pattern: $(SST2_DATA_DIR)/{train,validation}.tsv. Pattern
# rule fires when either file is missing; downloader is idempotent.
SST2_DATA_DIR := data/hf-datasets/glue-sst2

$(SST2_DATA_DIR)/train.tsv:
	bash packages/idris-transformers/scripts/hf-download-dataset.sh glue train sst2

$(SST2_DATA_DIR)/validation.tsv:
	bash packages/idris-transformers/scripts/hf-download-dataset.sh glue validation sst2

# Convenience alias for fetching both splits.
data-sst2: $(SST2_DATA_DIR)/train.tsv $(SST2_DATA_DIR)/validation.tsv

# Real-text fine-tune: warm-starts the bert_uncased_L-2_H-128_A-2 backbone
# from disk, fine-tunes on SST-2 with attention-mask threading. Deps on
# the bert-tiny checkpoint (via the HF pattern rule defined further down
# — hardcode `models/...` since HF_MODELS_DIR := assignment lands later
# in the file) plus the SST-2 TSV files (via the dataset rule above).
example-bert-classify-sst2-finetune: install install-transformers \
		models/google/bert_uncased_L-2_H-128_A-2/config.json \
		$(SST2_DATA_DIR)/train.tsv $(SST2_DATA_DIR)/validation.tsv
	idris2 $(IDRIS_FLAGS) -p idris-transformers -o bert-classify-sst2-finetune \
		$(EXAMPLE_SRC)/Example/BertClassifySst2Finetune.idr
	cp $(LIB) $(BUILD)/exec/bert-classify-sst2-finetune_app/
	./$(BUILD)/exec/bert-classify-sst2-finetune $(SEED_FLAG) $(BERT_SST2_ARGS)

# LoRA fine-tune variant: same backbone + dataset + classifier, but
# freezes the backbone weights and trains only the LoRA adapters
# (Q+V projections, rank=8 default) + the classifier head. Trainable
# param count drops from ~4.4M to ~6K (~0.13% of the model). Saved
# adapter is ~80KB on disk and round-trips with HF peft via
# `make validate-lora-adapter` (cross-tool gate).
example-bert-classify-sst2-lora: install install-transformers \
		models/google/bert_uncased_L-2_H-128_A-2/config.json \
		$(SST2_DATA_DIR)/train.tsv $(SST2_DATA_DIR)/validation.tsv
	idris2 $(IDRIS_FLAGS) -p idris-transformers -o bert-classify-sst2-lora \
		$(EXAMPLE_SRC)/Example/BertClassifySst2Lora.idr
	cp $(LIB) $(BUILD)/exec/bert-classify-sst2-lora_app/
	./$(BUILD)/exec/bert-classify-sst2-lora $(SEED_FLAG) $(BERT_SST2_LORA_ARGS)

# Tokenize Tiny Shakespeare via distilgpt2's BPE for use by the GPT-2
# LM continued-pretraining example. Lands a flat comma-separated
# integer token-id file (~338K tokens). Skipped if file is already on
# disk (the script's `[[ -s OUT_PATH ]]` check guards re-tokenization).
data/tinyshakespeare/input.distilgpt2.tokens: data/tinyshakespeare/input.txt
	bash packages/idris-transformers/scripts/tokenize-text-corpus.sh \
		data/tinyshakespeare/input.txt distilgpt2 \
		data/tinyshakespeare/input.distilgpt2.tokens

data-tinyshakespeare-distilgpt2: data/tinyshakespeare/input.distilgpt2.tokens

# Tokenize Tiny Shakespeare via google/bert_uncased_L-2_H-128_A-2's
# WordPiece for use by the BERT MLM continued-pretraining example
# (~289K tokens).
data/tinyshakespeare/input.bert-tiny.tokens: data/tinyshakespeare/input.txt
	bash packages/idris-transformers/scripts/tokenize-text-corpus.sh \
		data/tinyshakespeare/input.txt google/bert_uncased_L-2_H-128_A-2 \
		data/tinyshakespeare/input.bert-tiny.tokens

data-tinyshakespeare-bert-tiny: data/tinyshakespeare/input.bert-tiny.tokens

# GPT-2 LM continued pretraining: distilgpt2 backbone + sliding-window
# next-token CE loss on Tiny Shakespeare. Deps on the distilgpt2
# checkpoint (HF pattern rule) + the tokenized corpus.
example-gpt2-lm-finetune: install install-transformers \
		models/distilgpt2/config.json \
		data/tinyshakespeare/input.distilgpt2.tokens
	idris2 $(IDRIS_FLAGS) -p idris-transformers -o gpt2-lm-finetune \
		$(EXAMPLE_SRC)/Example/Gpt2LmFinetune.idr
	cp $(LIB) $(BUILD)/exec/gpt2-lm-finetune_app/
	./$(BUILD)/exec/gpt2-lm-finetune $(SEED_FLAG) $(GPT2_LM_ARGS)

ref-gpt2-lm-finetune: models/distilgpt2/config.json \
		data/tinyshakespeare/input.distilgpt2.tokens
	cd packages/pytorch && uv run python -m torch_ref.scripts.gpt2_lm_finetune $(GPT2_LM_ARGS)

# BERT MLM continued pretraining: bert-tiny backbone + MLM head; 80/10/10
# masking + position-selective CE loss on Tiny Shakespeare-via-WordPiece.
example-bert-mlm-finetune: install install-transformers \
		models/google/bert_uncased_L-2_H-128_A-2/config.json \
		data/tinyshakespeare/input.bert-tiny.tokens
	idris2 $(IDRIS_FLAGS) -p idris-transformers -o bert-mlm-finetune \
		$(EXAMPLE_SRC)/Example/BertMlmFinetune.idr
	cp $(LIB) $(BUILD)/exec/bert-mlm-finetune_app/
	./$(BUILD)/exec/bert-mlm-finetune $(SEED_FLAG) $(BERT_MLM_ARGS)

ref-bert-mlm-finetune: models/google/bert_uncased_L-2_H-128_A-2/config.json \
		data/tinyshakespeare/input.bert-tiny.tokens
	cd packages/pytorch && uv run python -m torch_ref.scripts.bert_mlm_finetune $(BERT_MLM_ARGS)

# HuggingFace BERT inference example. Loads google/bert_uncased_L-2_H-128_A-2
# weights via the HF-aligned HfBert layer module (from idris-transformers)
# and dumps the 128-dim pooled [CLS] output to stdout, one value per line.
# The checkpoint is fetched on demand via the pattern rule below.
# Pattern rule for any HuggingFace single-file checkpoint. Make-native
# dep tracking: each example/gate declares the safetensors path as a
# prerequisite; Make skips the recipe when the file is already on disk.
# Replaces the older shape (unconditional `bash hf-download.sh …` in
# every recipe + an internal cache check inside the script).
#
# `%` matches the HF repo path (e.g. `meta-llama/Llama-3.2-1B`). HF_TOKEN
# is checked here (the one place that actually fetches) rather than in
# every consumer recipe. Gated models that need the token surface a
# clear error; ungated models (BERT-tiny, distilgpt2) ignore the check.
# (HF_MODELS_DIR itself is defined in mk/config.mk — tests.mk parses
# before this fragment and declares fixture prerequisites under it.)

$(HF_MODELS_DIR)/%/config.json:
	@if echo "$*" | grep -q '^meta-llama/' && [ -z "$$HF_TOKEN" ]; then \
		echo "ERR: HF_TOKEN must be set ($* is gated)."; \
		echo "     1. Accept the license at https://huggingface.co/$*"; \
		echo "     2. Get a token at https://huggingface.co/settings/tokens"; \
		echo "     3. export HF_TOKEN=hf_..."; \
		exit 1; \
	fi
	bash packages/idris-transformers/scripts/hf-download.sh $*

# Prefetch the small tokenizer-test fixtures (bert-tiny + distilgpt2)
# without building anything. Run by the CI HF job so the shared
# hf-models cache carries them for the test-unit legs' restore.
.PHONY: hf-fixtures
hf-fixtures: $(TRANSFORMERS_TEST_FIXTURES)

example-hf-bert-inference: install $(HF_MODELS_DIR)/google/bert_uncased_L-2_H-128_A-2/config.json
	idris2 $(IDRIS_FLAGS) -o hf-bert-inference $(EXAMPLE_SRC)/Example/HfBertInference.idr
	cp $(LIB) $(BUILD)/exec/hf-bert-inference_app/
	./$(BUILD)/exec/hf-bert-inference

# Cross-language correctness gate for HfBert: regenerates the Python
# oracle via save_oracle.py, then runs the Idris example and compares
# stdout against the oracle within F32 tolerance.
test-e2e-hf-bert-roundtrip: install $(HF_MODELS_DIR)/google/bert_uncased_L-2_H-128_A-2/config.json
	cd packages/pytorch && uv run pytest \
		../idris-transformers/scripts/test_save_oracle.py -v
	idris2 $(IDRIS_FLAGS) -o hf-bert-inference $(EXAMPLE_SRC)/Example/HfBertInference.idr
	cp $(LIB) $(BUILD)/exec/hf-bert-inference_app/
	./$(BUILD)/exec/hf-bert-inference --dump-pooled > $(BUILD)/hf-bert-idris-out.txt
	cd packages/pytorch && uv run python \
		../idris-transformers/scripts/compare_inference.py \
		../../$(BUILD)/hf-bert-idris-out.txt \
		../../models/bert-tiny-oracle.safetensors \
		1e-3

# Build + run Example/HfGpt2Inference. Fetches distilgpt2 once via the
# pattern rule above.
example-hf-gpt2-inference: install $(HF_MODELS_DIR)/distilgpt2/config.json
	idris2 $(IDRIS_FLAGS) -o hf-gpt2-inference $(EXAMPLE_SRC)/Example/HfGpt2Inference.idr
	cp $(LIB) $(BUILD)/exec/hf-gpt2-inference_app/
	./$(BUILD)/exec/hf-gpt2-inference

# Cross-language correctness gate for HfGpt2: regenerate the Python
# oracle from distilgpt2 + run the Idris example + compare
# stdout against the oracle within F32 tolerance. The Idris example
# prints the final-position hidden state (the `last_hidden_state[-1]`
# row) which the comparator diffs elementwise.
# Build + run the Llama 3.2 1B inference example. Requires HF_TOKEN
# with Llama 3.2 license accepted on huggingface.co. The first
# invocation fetches the ~2.5 GB safetensors; subsequent runs reuse
# the cached file (Make's existence check handles it — the pattern
# rule's recipe doesn't fire).
#
# Tape lane (F64) doesn't fit in 16 GB; build with
# `BACKEND=torch TORCH_DEVICE=mps make example-hf-llama-inference`
# or `BACKEND=mlx MLX_DEVICE=gpu make example-hf-llama-inference` for
# the F32 / GPU paths.
#
# HF inference targets auto-set TORCH_DTYPE/MLX_DTYPE/TAPE_DTYPE
# to F32 (see the MAKECMDGOALS conditional near BUILD_KEY); the
# 1.24B-param Llama at F64 is ~10 GB which doesn't fit comfortably
# on a 16 GB VM. Override by setting TORCH_DTYPE=F64 (etc) on the
# command line if you genuinely want F64 (e.g. for numerical
# bisection vs the F64 oracle in `save_oracle_llama.py`).
example-hf-llama-inference: install $(HF_MODELS_DIR)/unsloth/Llama-3.2-1B/config.json
	idris2 $(IDRIS_FLAGS) -o hf-llama-inference $(EXAMPLE_SRC)/Example/HfLlamaInference.idr
	cp $(LIB) $(BUILD)/exec/hf-llama-inference_app/
	./$(BUILD)/exec/hf-llama-inference

# Fast feedback loop for HfLlamaInference: type-check only (`--check`),
# skip Scheme codegen + linking. Turns around in tens of seconds vs the
# multi-minute `example-hf-llama-inference` build. Useful when iterating
# on the typed surface (signatures, implicit-resolution, totality)
# without caring about an executable binary yet.
#
# Same install dep as the full build so dependent libraries (idris-ml,
# idris-transformers) are present; the difference is that the example
# file itself is `--check`ed rather than `-o`'d.
#
# Shares $(BUILD) with the example builds (not an isolated check dir):
# elaborating HfLlamaInference cold peaks past CI runner RAM — it
# OOM-killed the Ubuntu test-integration leg and burned the macOS
# leg's whole 60-min budget in run 27373449876. With the shared dir
# the check is nearly free whenever the example (or the roundtrip
# gate) already elaborated in this build set, and the --check ttc
# warms the later `-o` build in turn.
test-integration-lint-hf-llama-inference: install
	IDRIS2_PREFIX=$(IDRIS2_LOCAL) idris2 -p contrib -p idris-ml -p idris-gym -p idris-transformers \
		--build-dir $(BUILD) --source-dir $(EXAMPLE_SRC) \
		--check $(EXAMPLE_SRC)/Example/HfLlamaInference.idr

# Build + run Example/HfBitNetInference. Fetches microsoft/bitnet-b1.58-2B-4T
# once via the pattern rule (1.18 GB, not gated). Default mode runs the
# fixed-prompt forward and prints the top 5 logits; `--dump-logits` mode
# prints all 128256 logits for the roundtrip gate.
#
# Tape lane (F64) won't fit in 16 GB; build with
# `BACKEND=torch TORCH_DEVICE=mps make example-hf-bitnet-inference` or
# `BACKEND=mlx MLX_DEVICE=gpu make example-hf-bitnet-inference`.
example-hf-bitnet-inference: install $(HF_MODELS_DIR)/microsoft/bitnet-b1.58-2B-4T/config.json
	idris2 $(IDRIS_FLAGS) -o hf-bitnet-inference $(EXAMPLE_SRC)/Example/HfBitNetInference.idr
	cp $(LIB) $(BUILD)/exec/hf-bitnet-inference_app/
	./$(BUILD)/exec/hf-bitnet-inference

# Cross-language correctness gate for HfBitNet: regenerate the Python
# oracle from microsoft/bitnet-b1.58-2B-4T, run the Idris example
# in --dump-logits mode, compare stdout against the oracle.
# Tolerance is 1.0 max-abs-diff + an argmax-match assertion. The
# tolerance is loose because BitNet's BF16-storage + ternary-weight
# noise compounds across 30 decoder blocks: per-element diff at the
# logits layer settles at ~0.7 even with the kernel math correct.
# The argmax-match assertion catches the meaningful regression class
# (the model picking a different next token) without burdening the
# gate with the per-element noise floor.
test-e2e-hf-bitnet-roundtrip: install $(HF_MODELS_DIR)/microsoft/bitnet-b1.58-2B-4T/config.json
	cd packages/pytorch && uv run python \
		../idris-transformers/scripts/save_oracle_bitnet.py
	idris2 $(IDRIS_FLAGS) -o hf-bitnet-inference $(EXAMPLE_SRC)/Example/HfBitNetInference.idr
	cp $(LIB) $(BUILD)/exec/hf-bitnet-inference_app/
	./$(BUILD)/exec/hf-bitnet-inference --dump-logits > $(BUILD)/hf-bitnet-idris-out.txt
	cd packages/pytorch && uv run python \
		../idris-transformers/scripts/compare_inference.py \
		../../$(BUILD)/hf-bitnet-idris-out.txt \
		../../models/bitnet-2b-4t-oracle.safetensors \
		1.0 --argmax-match

test-e2e-hf-gpt2-roundtrip: install $(HF_MODELS_DIR)/distilgpt2/config.json
	cd packages/pytorch && uv run pytest \
		../idris-transformers/scripts/test_save_oracle_gpt2.py -v
	idris2 $(IDRIS_FLAGS) -o hf-gpt2-inference $(EXAMPLE_SRC)/Example/HfGpt2Inference.idr
	cp $(LIB) $(BUILD)/exec/hf-gpt2-inference_app/
	./$(BUILD)/exec/hf-gpt2-inference --dump-final-hidden > $(BUILD)/hf-gpt2-idris-out.txt
	cd packages/pytorch && uv run python \
		../idris-transformers/scripts/compare_inference.py \
		../../$(BUILD)/hf-gpt2-idris-out.txt \
		../../models/distilgpt2-oracle.safetensors \
		1e-3

# Cross-language correctness gate for HfLlama: regenerate the Python
# oracle from unsloth/Llama-3.2-1B (public mirror of Meta's weights;
# no license-gate / no HF_TOKEN required), run the Idris example in
# --dump-final-hidden mode, compare stdout
# against the oracle's last-position hidden state. Tolerance is 1.0
# max-abs-diff — Llama 3.2 1B is 16 layers × hidden=2048 with on-disk
# BF16 cast to F32, so per-element drift accumulates; the gate's job
# is catching macro regressions (broken forward, broken param load,
# bad RoPE), not pinning numerics to BF16-noise-floor precision.
# Tighten if measurements show consistent tighter alignment.
test-e2e-hf-llama-roundtrip: install $(HF_MODELS_DIR)/unsloth/Llama-3.2-1B/config.json
	cd packages/pytorch && uv run pytest \
		../idris-transformers/scripts/test_save_oracle_llama.py -v
	idris2 $(IDRIS_FLAGS) -o hf-llama-inference $(EXAMPLE_SRC)/Example/HfLlamaInference.idr
	cp $(LIB) $(BUILD)/exec/hf-llama-inference_app/
	./$(BUILD)/exec/hf-llama-inference --dump-final-hidden > $(BUILD)/hf-llama-idris-out.txt
	cd packages/pytorch && uv run python \
		../idris-transformers/scripts/compare_inference.py \
		../../$(BUILD)/hf-llama-idris-out.txt \
		../../models/llama-3.2-1b-oracle.safetensors \
		1.0

# Multi-step generation gate for HfLlama. Regenerates the Python
# oracle by greedy-decoding 8 tokens from `model.generate(do_sample=
# False, use_cache=True)` on the same prompt the user-facing demo
# uses ("The capital of France is"), runs the Idris example in
# --dump-tokens mode for the same prompt + budget, and asserts the
# resulting token-ID sequences match element-wise. Catches
# generation-path drift the single-forward
# `test-e2e-hf-llama-roundtrip` can't see.
#
# Budget bumped 2026-06-04 from 4 to 8 after the KV cache landed
# (commits `b5443135` ... `3b87291f`): with cached decode each step
# is constant-cost in Q/K/V projection (vs the no-cache path's
# growing prefix), so 8 tokens is cheap.
#
# Tape lane (F64) doesn't fit in 16 GB; build with
# `BACKEND=torch TORCH_DEVICE=cpu` for CI or
# `BACKEND=torch TORCH_DEVICE=mps` / `BACKEND=mlx MLX_DEVICE=gpu`
# for paired-lane dev verification.
test-e2e-hf-llama-generate-roundtrip: install $(HF_MODELS_DIR)/unsloth/Llama-3.2-1B/config.json
	cd packages/pytorch && uv run pytest \
		../idris-transformers/scripts/test_save_oracle_llama_generate.py -v
	idris2 $(IDRIS_FLAGS) -o hf-llama-inference $(EXAMPLE_SRC)/Example/HfLlamaInference.idr
	cp $(LIB) $(BUILD)/exec/hf-llama-inference_app/
	./$(BUILD)/exec/hf-llama-inference --dump-tokens --num-tokens 8 > $(BUILD)/hf-llama-tokens-out.txt
	cd packages/pytorch && uv run python \
		../idris-transformers/scripts/compare_inference.py \
		../../$(BUILD)/hf-llama-tokens-out.txt \
		../../models/llama-3.2-1b-generate-oracle.safetensors \
		--token-sequence

# Manual oracle-regen entry point (pytest harness pairs with
# `test-e2e-hf-llama-generate-roundtrip` above). Useful when bumping
# the budget after KV cache lands.
test-e2e-transformers-oracle-llama-generate:
	cd packages/pytorch && uv run pytest \
		../idris-transformers/scripts/test_save_oracle_llama_generate.py -v

example-rnn: install
	idris2 $(IDRIS_FLAGS) -o rnn $(EXAMPLE_SRC)/Example/Rnn.idr
	cp $(LIB) $(BUILD)/exec/rnn_app/
	./$(BUILD)/exec/rnn $(SEED_FLAG) $(RNN_ARGS)

example-lstm: install
	idris2 $(IDRIS_FLAGS) -o lstm $(EXAMPLE_SRC)/Example/Lstm.idr
	cp $(LIB) $(BUILD)/exec/lstm_app/
	./$(BUILD)/exec/lstm $(SEED_FLAG) $(LSTM_ARGS)

example-gru: install
	idris2 $(IDRIS_FLAGS) -o gru $(EXAMPLE_SRC)/Example/Gru.idr
	cp $(LIB) $(BUILD)/exec/gru_app/
	./$(BUILD)/exec/gru $(SEED_FLAG) $(GRU_ARGS)

# BringYourOwn — worked example of a user-supplied backend. Builds
# libbyo.dylib alongside the active libidrisml so the example app
# can dlopen both: the BYO instance dispatches to `byo_*` symbols
# in libbyo, and the built-in CPU instance dispatches to unified
# names in libidrisml. See packages/backends/backend_byo.c +
# Example/BringYourOwn.idr.
$(BUILD)/libbyo.$(LIB_EXT): $(BACKENDS_DIR)/backend_byo.c | $(BUILD)
	cc -O2 -shared -fPIC -o $@ $<

example-bring-your-own: install $(BUILD)/libbyo.$(LIB_EXT)
	idris2 $(IDRIS_FLAGS) -o bring-your-own $(EXAMPLE_SRC)/Example/BringYourOwn.idr
	cp $(LIB) $(BUILD)/libbyo.$(LIB_EXT) $(BUILD)/exec/bring-your-own_app/
	./$(BUILD)/exec/bring-your-own

example-ntm-copy: install
	idris2 $(IDRIS_FLAGS) -o ntm-copy $(EXAMPLE_SRC)/Example/NtmCopy.idr
	cp $(LIB) $(BUILD)/exec/ntm-copy_app/
	$(STDBUF) ./$(BUILD)/exec/ntm-copy $(SEED_FLAG) $(NTM_COPY_ARGS)

example-ntm-associative-recall: install
	idris2 $(IDRIS_FLAGS) -o ntm-associative-recall $(EXAMPLE_SRC)/Example/NtmAssociativeRecall.idr
	cp $(LIB) $(BUILD)/exec/ntm-associative-recall_app/
	$(STDBUF) ./$(BUILD)/exec/ntm-associative-recall $(SEED_FLAG) $(NTM_ASSOCIATIVE_RECALL_ARGS)

example-dnc-copy: install
	idris2 $(IDRIS_FLAGS) -o dnc-copy $(EXAMPLE_SRC)/Example/DncCopy.idr
	cp $(LIB) $(BUILD)/exec/dnc-copy_app/
	$(STDBUF) ./$(BUILD)/exec/dnc-copy $(SEED_FLAG) $(DNC_COPY_ARGS)

example-dnc-recall: install
	idris2 $(IDRIS_FLAGS) -o dnc-recall $(EXAMPLE_SRC)/Example/DncAssociativeRecall.idr
	cp $(LIB) $(BUILD)/exec/dnc-recall_app/
	$(STDBUF) ./$(BUILD)/exec/dnc-recall $(SEED_FLAG) $(DNC_RECALL_ARGS)

example-transformer: install
	idris2 $(IDRIS_FLAGS) -o transformer $(EXAMPLE_SRC)/Example/Transformer.idr
	cp $(LIB) $(BUILD)/exec/transformer_app/
	./$(BUILD)/exec/transformer $(SEED_FLAG) $(TRANSFORMER_ARGS)

example-tcast-demo: install
	idris2 $(IDRIS_FLAGS) -o tcast-demo $(EXAMPLE_SRC)/Example/TCastDemo.idr
	cp $(LIB) $(BUILD)/exec/tcast-demo_app/
	./$(BUILD)/exec/tcast-demo $(TCAST_DEMO_ARGS)

# Cross-language dtype serialization demo. Forces BACKEND=torch (bf16/f16/
# int are Compatible only on torch), writes a multi-dtype .safetensors from
# Idris, then verifies the byte layout via the reference safetensors.torch
# reader (Python). Verifier is skipped if the pytorch venv is absent.
example-dtype-serialize:
	$(MAKE) BACKEND=torch install >/dev/null
	idris2 $(IDRIS_FLAGS) -o dtype-serialize $(EXAMPLE_SRC)/Example/DTypeSerialize.idr
	cp $(LIB) $(BUILD)/exec/dtype-serialize_app/
	./$(BUILD)/exec/dtype-serialize /tmp/idrisml-dtypes.safetensors
	@if [ -x packages/pytorch/.venv/bin/python3 ]; then \
		echo "=== cross-language verify (safetensors.torch) ==="; \
		packages/pytorch/.venv/bin/python3 packages/idris-ml-examples/scripts/verify_dtypes.py /tmp/idrisml-dtypes.safetensors; \
	else \
		echo "=== cross-language verify SKIPPED (pytorch venv not found) ==="; \
	fi

# Type-safe integral index API demo. Forces BACKEND=torch (an I64 index
# tensor is Compatible only on torch-cpu/cuda), then runs the typed
# targsort/tgather/tscatterAdd round-trip with order-sensitive readouts.
# The whole recipe must run under the torch build key: an inner
# `$(MAKE) BACKEND=torch install` alone leaves IDRIS_FLAGS/BUILD/LIB
# expanded for the caller's backend, and the example then elaborates
# against a tree whose generated HwConfig lacks
# Linked (TorchExecutor TCpu) — latent since the constraint-bundle
# sweep put Linked inside Backend; surfaced in CI run 27434768856
# once the ttc cache fixes let test-integration reach this step.
ifeq ($(BACKEND),torch)
example-index-ops: install
	idris2 $(IDRIS_FLAGS) -o index-ops $(EXAMPLE_SRC)/Example/IndexOps.idr
	cp $(LIB) $(BUILD)/exec/index-ops_app/
	./$(BUILD)/exec/index-ops
else
example-index-ops:
	$(MAKE) BACKEND=torch example-index-ops
endif

# Compile-time (device, dtype) Compatible gate demo. The example's `ok*`
# witnesses typecheck against the real constructor across all backends;
# main constructs on the build-selected cell, so it runs on any BACKEND.
example-dtype-pitch: install
	idris2 $(IDRIS_FLAGS) -o dtype-pitch $(EXAMPLE_SRC)/Example/DTypePitch.idr
	cp $(LIB) $(BUILD)/exec/dtype-pitch_app/
	./$(BUILD)/exec/dtype-pitch

# Cross-dtype SafeTensors round-trip smoke test for L63.
#   1. Save F32 (BACKEND=mlx MLX_DEVICE=gpu): writes a checkpoint with
#      "dtype":"F32" headers and 4-byte-per-element data.
#   2. Load-strict in F64 (BACKEND=mlx): expects to FAIL with a dtype
#      mismatch — `loadModel` returns False, the example exits nonzero.
#   3. Load-cast in F64 (BACKEND=mlx): expects to PASS — bytes widened
#      f32 -> f64 at load time, eval loss reproduces the trained loss.
example-precision-checkpoint:
	@rm -f /tmp/precision-checkpoint.safetensors
	@echo "=== Step 1: save F32 (BACKEND=mlx MLX_DEVICE=gpu) ==="
	$(MAKE) BACKEND=mlx MLX_DEVICE=gpu install >/dev/null
	idris2 $(IDRIS_FLAGS) -o precision-checkpoint $(EXAMPLE_SRC)/Example/PrecisionCheckpoint.idr
	cp $(LIB) $(BUILD)/exec/precision-checkpoint_app/
	./$(BUILD)/exec/precision-checkpoint --mode save --path /tmp/precision-checkpoint.safetensors --expect pass
	@echo ""
	@echo "=== Step 2: load-strict into F64 (BACKEND=mlx), expect FAIL ==="
	$(MAKE) BACKEND=mlx install >/dev/null
	idris2 $(IDRIS_FLAGS) -o precision-checkpoint $(EXAMPLE_SRC)/Example/PrecisionCheckpoint.idr
	cp $(LIB) $(BUILD)/exec/precision-checkpoint_app/
	./$(BUILD)/exec/precision-checkpoint --mode load-strict --path /tmp/precision-checkpoint.safetensors --expect fail
	@echo ""
	@echo "=== Step 3: load-cast into F64 (BACKEND=mlx), expect PASS ==="
	./$(BUILD)/exec/precision-checkpoint --mode load-cast --path /tmp/precision-checkpoint.safetensors --expect pass
	@echo ""
	@echo "All three steps passed (PrecisionCheckpoint L63 round-trip)."

# Training-loop checkpoint/resume smoke test (tape backend, fast).
# Trains gpt 10 epochs to a checkpoint dir, resumes to 20, asserts the
# sidecar epoch + resume log + completion. Gates the Train/Checkpoint
# integration. See scripts/test-checkpoint-resume.sh.
test-integration-checkpoint-resume: install
	idris2 $(IDRIS_FLAGS) -o gpt $(EXAMPLE_SRC)/Example/Gpt.idr
	cp $(LIB) $(BUILD)/exec/gpt_app/
	bash scripts/test-checkpoint-resume.sh ./$(BUILD)/exec/gpt

# Mlx-only: cross-stream MlxCpu F64 / MlxGpu F32 smoke test. Builds
# under any BACKEND list that includes mlx; references MlxCpu / MlxGpu
# directly, so won't link under tape-only or torch-only builds.
example-mlx-stream-demo: install
	idris2 $(IDRIS_FLAGS) -o mlx-stream-demo $(EXAMPLE_SRC)/Example/MlxStreamDemo.idr
	cp $(LIB) $(BUILD)/exec/mlx-stream-demo_app/
	./$(BUILD)/exec/mlx-stream-demo $(MLX_STREAM_DEMO_ARGS)

example-gpt: install
	idris2 $(IDRIS_FLAGS) -o gpt $(EXAMPLE_SRC)/Example/Gpt.idr
	cp $(LIB) $(BUILD)/exec/gpt_app/
	$(STDBUF) ./$(BUILD)/exec/gpt $(SEED_FLAG) $(GPT_ARGS)

# Full-corpus convergence run (~hours on tape). Default `make example-gpt`
# is a ~30s embedded-corpus demo; this target is the real char-LM
# convergence target (matching nanoGPT/train_shakespeare_char.py).
example-gpt-full: install $(TINYSHAKESPEARE_FILE)
	idris2 $(IDRIS_FLAGS) -o gpt $(EXAMPLE_SRC)/Example/Gpt.idr
	cp $(LIB) $(BUILD)/exec/gpt_app/
	$(STDBUF) ./$(BUILD)/exec/gpt $(SEED_FLAG) --corpus tinyshakespeare --epochs 1000 $(GPT_ARGS)

example-mnist: install $(MNIST_SENTINEL)
	idris2 $(IDRIS_FLAGS) -o mnist $(EXAMPLE_SRC)/Example/Mnist.idr
	cp $(LIB) $(BUILD)/exec/mnist_app/
	$(STDBUF) ./$(BUILD)/exec/mnist $(SEED_FLAG) $(MNIST_ARGS)

example-seq-classify: install
	idris2 $(IDRIS_FLAGS) -o seq-classify $(EXAMPLE_SRC)/Example/SeqClassify.idr
	cp $(LIB) $(BUILD)/exec/seq-classify_app/
	$(STDBUF) ./$(BUILD)/exec/seq-classify $(SEED_FLAG) $(SEQ_CLASSIFY_ARGS)

example-reinforce: install
	idris2 $(IDRIS_FLAGS) -o reinforce $(EXAMPLE_SRC)/Example/Reinforce.idr
	cp $(LIB) $(BUILD)/exec/reinforce_app/
	./$(BUILD)/exec/reinforce $(SEED_FLAG) $(REINFORCE_ARGS)

example-q-learning: install
	idris2 $(IDRIS_FLAGS) -o q-learning $(EXAMPLE_SRC)/Example/QLearning.idr
	cp $(LIB) $(BUILD)/exec/q-learning_app/
	./$(BUILD)/exec/q-learning $(SEED_FLAG) $(Q_LEARNING_ARGS)

example-sarsa: install
	idris2 $(IDRIS_FLAGS) -o sarsa $(EXAMPLE_SRC)/Example/Sarsa.idr
	cp $(LIB) $(BUILD)/exec/sarsa_app/
	./$(BUILD)/exec/sarsa $(SEED_FLAG) $(SARSA_ARGS)

example-monte-carlo: install
	idris2 $(IDRIS_FLAGS) -o monte-carlo $(EXAMPLE_SRC)/Example/MonteCarlo.idr
	cp $(LIB) $(BUILD)/exec/monte-carlo_app/
	./$(BUILD)/exec/monte-carlo $(SEED_FLAG) $(MONTE_CARLO_ARGS)

example-frozen-lake: install
	idris2 $(IDRIS_FLAGS) -o frozen-lake $(EXAMPLE_SRC)/Example/FrozenLake.idr
	cp $(LIB) $(BUILD)/exec/frozen-lake_app/
	./$(BUILD)/exec/frozen-lake $(SEED_FLAG) $(FROZEN_LAKE_ARGS)

example-taxi: install
	idris2 $(IDRIS_FLAGS) -o taxi $(EXAMPLE_SRC)/Example/Taxi.idr
	cp $(LIB) $(BUILD)/exec/taxi_app/
	./$(BUILD)/exec/taxi $(SEED_FLAG) $(TAXI_ARGS)

example-dqn: install
	idris2 $(IDRIS_FLAGS) -o dqn $(EXAMPLE_SRC)/Example/Dqn.idr
	cp $(LIB) $(BUILD)/exec/dqn_app/
	$(STDBUF) ./$(BUILD)/exec/dqn $(SEED_FLAG) $(DQN_ARGS)

example-mountain-car: install
	idris2 $(IDRIS_FLAGS) -o mountain-car $(EXAMPLE_SRC)/Example/MountainCar.idr
	cp $(LIB) $(BUILD)/exec/mountain-car_app/
	$(STDBUF) ./$(BUILD)/exec/mountain-car $(SEED_FLAG) $(MOUNTAIN_CAR_ARGS)

example-mountain-car-cont: install
	idris2 $(IDRIS_FLAGS) -o mountain-car-cont $(EXAMPLE_SRC)/Example/MountainCarCont.idr
	cp $(LIB) $(BUILD)/exec/mountain-car-cont_app/
	$(STDBUF) ./$(BUILD)/exec/mountain-car-cont $(SEED_FLAG) $(MOUNTAIN_CAR_CONT_ARGS)

example-a2c: install
	idris2 $(IDRIS_FLAGS) -o a2c $(EXAMPLE_SRC)/Example/A2c.idr
	cp $(LIB) $(BUILD)/exec/a2c_app/
	$(STDBUF) ./$(BUILD)/exec/a2c $(SEED_FLAG) $(A2C_ARGS)

example-ppo: install
	idris2 $(IDRIS_FLAGS) -o ppo $(EXAMPLE_SRC)/Example/Ppo.idr
	cp $(LIB) $(BUILD)/exec/ppo_app/
	$(STDBUF) ./$(BUILD)/exec/ppo $(SEED_FLAG) $(PPO_ARGS)

example-sac: install
	idris2 $(IDRIS_FLAGS) -o sac $(EXAMPLE_SRC)/Example/Sac.idr
	cp $(LIB) $(BUILD)/exec/sac_app/
	$(STDBUF) ./$(BUILD)/exec/sac $(SEED_FLAG) $(SAC_ARGS)

# Live cross-backend Tensor transfer demo. Builds with all three
# backends linked so the example can call tape / torch / mlx C
# symbols in a single process. Exits 0 with RESULT line on success;
# crashes at FFI resolution if any backend's symbols are missing.
# Every leg constructs in its declared dtype directly —
# `primCreateFromHost` threads the RuntimeDType dtag through each
# backend's `tensor_create_streamed`, so the F32 hop needs no cast
# workaround and the primary choice carries no dtype significance.
example-transfer:
	$(MAKE) BACKEND=torch,tape,mlx install
	idris2 $(IDRIS_FLAGS) -o transfer $(EXAMPLE_SRC)/Example/Transfer.idr
	cp $(LIB) $(BUILD)/exec/transfer_app/
	./$(BUILD)/exec/transfer $(TRANSFER_ARGS)

# F32/F64 precision artifact + cross-backend hop demo. References
# TapeExecutor/TorchExecutor/MlxExecutor directly, so it needs all three backends
# linked (same as `example-transfer`). Unblocked by tape's F32 storage
# + kernel coverage — every cell is first-class for both precisions.
example-precision-demo:
	$(MAKE) BACKEND=tape,torch,mlx install
	idris2 $(IDRIS_FLAGS) -o precision-demo $(EXAMPLE_SRC)/Example/PrecisionDemo.idr
	cp $(LIB) $(BUILD)/exec/precision-demo_app/
	./$(BUILD)/exec/precision-demo $(PRECISION_DEMO_ARGS)

# SafeTensors checkpoint demo (formerly the Example/Transfer.idr
# content). Per-phase BACKEND= invocation; `example-checkpoint-demo`
# drives the tape→mlx→torch on-disk round-trip via three calls.
example-checkpoint: install
	idris2 $(IDRIS_FLAGS) -o checkpoint $(EXAMPLE_SRC)/Example/Checkpoint.idr
	cp $(LIB) $(BUILD)/exec/checkpoint_app/
	./$(BUILD)/exec/checkpoint $(SEED_FLAG) $(CHECKPOINT_ARGS)

example-checkpoint-demo:
	@echo "=== Phase 1: Train on tape ==="
	$(MAKE) BACKEND=tape example-checkpoint CHECKPOINT_ARGS="--mode train --epochs 500 --save /tmp/checkpoint.safetensors"
	@echo ""
	@echo "=== Phase 2: Continue on mlx ==="
	$(MAKE) BACKEND=mlx example-checkpoint CHECKPOINT_ARGS="--mode continue --load /tmp/checkpoint.safetensors --epochs 500 --save /tmp/checkpoint2.safetensors"
	@echo ""
	@echo "=== Phase 3: Infer on torch ==="
	$(MAKE) BACKEND=torch example-checkpoint CHECKPOINT_ARGS="--mode infer --load /tmp/checkpoint2.safetensors"

example-matmul-bench: install
	idris2 $(IDRIS_FLAGS) -o matmul-bench $(EXAMPLE_SRC)/Example/MatmulBench.idr
	cp $(LIB) $(BUILD)/exec/matmul-bench_app/
	$(STDBUF) ./$(BUILD)/exec/matmul-bench $(MATMUL_BENCH_ARGS)

# #402 Idris-level rank-3 broadcast microbench. Counterpart to the
# `bench-rank3-broadcast{,-wrapped}` C harnesses; calls `primMul` in a
# tight loop on `[6, 32, 32] x [6, 1, 32]` — same shape and iteration
# counts. The delta vs the wrapped C bench is the Scheme wrap layer
# (cached foreign-procedure dispatch + tensor-handle-v2 unwrap/wrap +
# guardian register). Identical wrap structure across all three
# backends — any wrap-layer overhead measured here applies symmetrically.
example-rank-broadcast-bench: install
	idris2 $(IDRIS_FLAGS) -o rank-broadcast-bench $(EXAMPLE_SRC)/Example/RankBroadcastBench.idr
	cp $(LIB) $(BUILD)/exec/rank-broadcast-bench_app/
	$(STDBUF) ./$(BUILD)/exec/rank-broadcast-bench $(RANK_BROADCAST_BENCH_ARGS)

example-bench: install
	idris2 $(IDRIS_FLAGS) -o bench $(EXAMPLE_SRC)/Example/Bench.idr
	cp $(LIB) $(BUILD)/exec/bench_app/
	@# Each benchmark runs in its own process. Sharing one process across
	@# all six accumulates allocator state that nondeterministically trips
	@# the unresolved tape stale-reader bug (see TODO.md High Priority).
	@for b in supervised rnn ntm ntm-copy ntm-copy-1k ntm-recall; do \
	    ./$(BUILD)/exec/bench $$b || exit $$?; \
	done

example-profile: install
	idris2 $(IDRIS_FLAGS) -o profile $(EXAMPLE_SRC)/Example/Profile.idr
	cp $(LIB) $(BUILD)/exec/profile_app/
	./$(BUILD)/exec/profile

sweep: backend
	python3 scripts/sweep.py --parallel 4

sweep-quick: backend
	python3 scripts/sweep.py --parallel 4 --quick
