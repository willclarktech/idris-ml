# mk/ref.mk — PyTorch reference runs (torch_ref). ref-setup, the
# collapsed ref-* static pattern rules + exceptions, HF reference
# inference, oracle gates, ref-convergence, test-e2e-cuda.

# PyTorch reference implementation (uv manages Python)
.PHONY: ref-setup ref-bert-classify-sst2-finetune \
        ref-bert-classify-sst2-lora-finetune validate-lora-adapter \
        ref-hf-bert ref-hf-gpt2 ref-hf-llama test-e2e-pytorch-ref \
        ref-lint ref-typecheck test-e2e-transformers-oracle-bert \
        test-e2e-rope-oracle test-e2e-transformers-oracle-gpt2 \
        test-e2e-transformers-oracle-llama ref-convergence \
        ref-convergence-copy ref-convergence-recall test-e2e-cuda \
        test-convergence-ref-campaign

ref-setup:
	cd packages/pytorch && uv sync --dev

# Plain torch_ref reference runs, collapsed into one static pattern
# rule: ref-foo-bar -> `python -m torch_ref.scripts.foo_bar`. Static
# (not a bare `ref-%`) so the member list stays explicit/greppable,
# .PHONY applies, and a stray file named ref-<x> can't shadow a rule.
# Exceptions keep explicit rules (file deps, ARGS, or different shapes
# entirely): ref-setup ref-lint ref-typecheck ref-convergence{,-copy,
# -recall} ref-hf-{bert,gpt2,llama} ref-bert-classify-sst2-{finetune,
# lora-finetune} validate-lora-adapter, plus ref-gpt2-lm-finetune and
# ref-bert-mlm-finetune (colocated with their examples further up).
REF_SCRIPT_NAMES := supervised bert-classify-finetune rnn lstm gru \
	ntm-copy ntm-recall dnc-copy dnc-recall transformer gpt mnist \
	seq-classify reinforce a2c ppo dqn double-dqn sac mountain-car \
	mountain-car-cont q-learning sarsa frozen-lake taxi monte-carlo
REF_SCRIPT_TARGETS := $(addprefix ref-,$(REF_SCRIPT_NAMES))
.PHONY: $(REF_SCRIPT_TARGETS)
$(REF_SCRIPT_TARGETS): ref-%:
	cd packages/pytorch && uv run python -m torch_ref.scripts.$(subst -,_,$*)

ref-bert-classify-sst2-finetune: models/google/bert_uncased_L-2_H-128_A-2/config.json \
		$(SST2_DATA_DIR)/train.tsv $(SST2_DATA_DIR)/validation.tsv
	cd packages/pytorch && uv run python -m torch_ref.scripts.bert_classify_sst2_finetune $(BERT_SST2_ARGS)

ref-bert-classify-sst2-lora-finetune: models/google/bert_uncased_L-2_H-128_A-2/config.json \
		$(SST2_DATA_DIR)/train.tsv $(SST2_DATA_DIR)/validation.tsv
	cd packages/pytorch && uv run python -m torch_ref.scripts.bert_classify_sst2_lora_finetune $(BERT_SST2_LORA_ARGS)

# Cross-tool gate: load an idris-ml-saved LoRA adapter via peft and run a
# forward pass. ADAPTER_DIR points at the directory written by the
# `--save-adapter` flag on the worked example. Default = /tmp/idris-ml-lora-out;
# override on the command line: `make validate-lora-adapter ADAPTER_DIR=...`.
ADAPTER_DIR ?= /tmp/idris-ml-lora-out
validate-lora-adapter: models/google/bert_uncased_L-2_H-128_A-2/config.json
	cd packages/pytorch && uv run python torch_ref/scripts/validate_lora_adapter.py \
		--adapter-dir $(realpath $(ADAPTER_DIR)) \
		--base-model $(realpath models/google/bert_uncased_L-2_H-128_A-2) \
		--num-labels 2

# PyTorch reference inference for the HF-aligned models. Each invokes the
# canonical HF transformers forward pass for the same model the matching
# Idris example runs, so users can eyeball PyTorch's output (or wall
# time) for direct comparison with `make example-{bert,gpt2,llama}-inference`.
#
# bert + gpt2 reuse the oracle scripts (load via HF, run forward, save
# the comparison-target tensor) — re-running them refreshes the oracle
# files used by `test-hf-{bert,gpt2}-roundtrip`. llama uses
# `time_inference_llama.py` (PyTorch greedy decode, stage timers
# mirroring the Idris example).
ref-hf-bert:
	cd packages/pytorch && uv run python \
		../idris-transformers/scripts/save_oracle.py

ref-hf-gpt2:
	cd packages/pytorch && uv run python \
		../idris-transformers/scripts/save_oracle_gpt2.py

ref-hf-llama:
	cd packages/pytorch && uv run python \
		../idris-transformers/scripts/time_inference_llama.py

test-e2e-pytorch-ref:
	cd packages/pytorch && uv run pytest torch_ref/correctness/ -v

ref-lint:
	cd packages/pytorch && uv run ruff check torch_ref/ && uv run ruff format --check torch_ref/

ref-typecheck:
	cd packages/pytorch && uv run pyright torch_ref/

# Regenerate + validate the HfBert forward-pass oracle. Runs
# packages/idris-transformers/scripts/save_oracle.py through pytest
# under the pytorch package's uv-managed venv (which carries the
# `transformers` dep). The pytest is colocated with the script per
# feedback_paired_side_alignment. Wire into CI alongside test-transformers.
#
# This target only runs the generator + asserts the fixture is
# well-formed (shape, dtype, finite, nontrivial). The cross-language
# Idris-vs-Python comparison gate lands in Phase 6 as
# test-e2e-bert-roundtrip.
test-e2e-transformers-oracle-bert: $(HF_MODELS_DIR)/google/bert_uncased_L-2_H-128_A-2/config.json
	cd packages/pytorch && uv run pytest \
		../idris-transformers/scripts/test_save_oracle.py -v

# Produce the Llama-3 RoPE table oracle (inv_freq + a slice of
# cos/sin tables). Pinned by Test.RoPE in the idris-ml unit suite;
# this target lets you regenerate the oracle if the upstream Llama-3
# rope_scaling formula changes.
test-e2e-rope-oracle:
	cd packages/pytorch && uv run python \
		../idris-transformers/scripts/save_rope_oracle.py

# Same shape as test-e2e-transformers-oracle-bert, paired with HfGpt2.idr:
# generates `models/tiny-gpt2-oracle.safetensors` from
# `distilgpt2`'s last-hidden-state for [15496, 995] and
# asserts the fixture is well-formed. The cross-language gate lands
# as test-e2e-gpt2-roundtrip alongside the Idris example.
test-e2e-transformers-oracle-gpt2: $(HF_MODELS_DIR)/distilgpt2/config.json
	cd packages/pytorch && uv run pytest \
		../idris-transformers/scripts/test_save_oracle_gpt2.py -v

# Same shape, paired with HfLlama.idr: generates
# `models/llama-3.2-1b-oracle.safetensors` from `unsloth/Llama-3.2-1B`'s
# last-hidden-state for [9906] ("Hello") and asserts the fixture is
# well-formed. The cross-language gate lands as test-e2e-llama-roundtrip
# alongside the Idris example.
# Depends on the model file-target so a cache miss downloads it (hf-download.sh,
# gated -> needs HF_TOKEN) BEFORE the oracle pytest, which only asserts the model
# is present. Without this the oracle (first llama step in CI) fails "model not
# found" on any cold cache, before the roundtrip that would have fetched it runs.
test-e2e-transformers-oracle-llama: $(HF_MODELS_DIR)/unsloth/Llama-3.2-1B/config.json
	cd packages/pytorch && uv run pytest \
		../idris-transformers/scripts/test_save_oracle_llama.py -v

ref-convergence:
	cd packages/pytorch && uv run python -u -m torch_ref.scripts.convergence --task both

ref-convergence-copy:
	cd packages/pytorch && uv run python -u -m torch_ref.scripts.convergence --task copy

ref-convergence-recall:
	cd packages/pytorch && uv run python -u -m torch_ref.scripts.convergence --task recall

# Reference-side multi-seed convergence CAMPAIGN — the peer of
# test-convergence-campaign in mk/e2e.mk. Same seeds, same thresholds, same
# resumable TSV shape, so the two pass-rate tables answer the same question
# and sit side by side in reference-alignment.md.
#
# The memory models joined on 2026-07-31, when their init was aligned; they run
# in ~30s-2min per seed, not the grind their reputation suggests. Tabular RL
# carries no dense layer at all, but it is cheap and shares every hyperparameter
# default, so it stays in as a free cross-check. Only the transformer pair is
# absent, and only because its reference has no seeded convergence bar.
CONVERGENCE_REF_MODULES := supervised rnn lstm gru mnist seq_classify \
	reinforce dqn double_dqn mountain_car mountain_car_cont a2c ppo sac \
	q_learning sarsa monte_carlo frozen_lake taxi \
	ntm_copy ntm_recall dnc_copy dnc_recall
CONVERGENCE_REF_EXPECT  := test-refs-convergence.expect
CONVERGENCE_REF_OUT     ?= docs/develop/convergence-campaign-ref.tsv

test-convergence-ref-campaign:
	@MODULES='$(CONVERGENCE_REF_MODULES)' SEEDS='$(CONVERGENCE_SEEDS)' \
		CONVERGENCE_TIMEOUT='$(CONVERGENCE_TIMEOUT)' \
		CONVERGENCE_EXPECT='$(CONVERGENCE_REF_EXPECT)' \
		CONVERGENCE_REF_OUT='$(CONVERGENCE_REF_OUT)' \
		bash scripts/test-convergence-ref.sh

# CUDA test (run on Colab or Linux with CUDA GPU)
test-e2e-cuda:
	bash scripts/test_cuda_colab.sh
