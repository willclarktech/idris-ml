#!/usr/bin/env bash
# scripts/perf-run.sh — run an example to convergence (or early-stop)
# and auto-log a structured entry into docs/develop/perf-log.jsonl.
#
# Each invocation appends a single JSON object on its own line to
# `perf-log.jsonl`. Schema and conventions are documented in
# `perf-log.md` (the companion markdown file). Entry construction
# lives in `scripts/mltools/perf_log.py` — this script just dispatches
# to it via the `append-run` subcommand.
#
# Usage:
#   scripts/perf-run.sh <example-key> <backend> [example-args...]
#
# Examples:
#   scripts/perf-run.sh ntm-copy tape --seed 42 --batch 1
#   scripts/perf-run.sh dnc-copy torch --seed 1
#
# Querying:
#   jq 'select(.example == "dnc-copy" and .backend == "tape")' \
#     docs/develop/perf-log.jsonl
#
# Per CLAUDE.md "Performance results" convention: every measurement
# is appended; never edit or delete prior entries. If a measurement
# is invalid, append a follow-up entry that says so.
set -euo pipefail

source "$( dirname "${BASH_SOURCE[0]}" )/perf_lib.sh"

if [ $# -lt 2 ]; then
	cat <<EOF >&2
usage: $0 <example-key> <backend> [example-args...]

example-keys: ntm-copy, ntm-recall, dnc-copy, dnc-recall, supervised,
							rnn, lstm, gru, transformer, gpt, matmul-bench, mnist, seq-classify,
							reinforce, dqn, mountain-car, mountain-car-cont, a2c,
							ppo, sac, hf-bert, hf-gpt2, hf-llama, hf-llama-generate, hf-bitnet,
							bert-classify-finetune, bert-classify-sst2-finetune,
							bert-classify-sst2-lora,
							gpt2-lm-finetune, bert-mlm-finetune
backends:     tape, mlx, torch
EOF
	exit 2
fi

EXAMPLE_KEY="$1"; shift
BACKEND="$1"; shift
ARGS=("$@")

case "$EXAMPLE_KEY" in
	ntm-copy)            TGT=example-ntm-copy;                   AVAR=NTM_COPY_ARGS ;;
	ntm-recall)          TGT=example-ntm-associative-recall;     AVAR=NTM_ASSOCIATIVE_RECALL_ARGS ;;
	dnc-copy)            TGT=example-dnc-copy;                   AVAR=DNC_COPY_ARGS ;;
	dnc-recall)          TGT=example-dnc-recall;                 AVAR=DNC_RECALL_ARGS ;;
	supervised)          TGT=example-supervised;                 AVAR=SUPERVISED_ARGS ;;
	rnn)                 TGT=example-rnn;                        AVAR=RNN_ARGS ;;
	lstm)                TGT=example-lstm;                       AVAR=LSTM_ARGS ;;
	gru)                 TGT=example-gru;                        AVAR=GRU_ARGS ;;
	transformer)         TGT=example-transformer;                AVAR=TRANSFORMER_ARGS ;;
	gpt)                 TGT=example-gpt;                        AVAR=GPT_ARGS ;;
	matmul-bench)        TGT=example-matmul-bench;               AVAR=MATMUL_BENCH_ARGS ;;
	mnist)               TGT=example-mnist;                      AVAR=MNIST_ARGS ;;
	seq-classify)        TGT=example-seq-classify;               AVAR=SEQ_CLASSIFY_ARGS ;;
	reinforce)           TGT=example-reinforce;                  AVAR=REINFORCE_ARGS ;;
	dqn)                 TGT=example-dqn;                        AVAR=DQN_ARGS ;;
	mountain-car)        TGT=example-mountain-car;               AVAR=MOUNTAIN_CAR_ARGS ;;
	mountain-car-cont)   TGT=example-mountain-car-cont;          AVAR=MOUNTAIN_CAR_CONT_ARGS ;;
	a2c)                 TGT=example-a2c;                        AVAR=A2C_ARGS ;;
	ppo)                 TGT=example-ppo;                        AVAR=PPO_ARGS ;;
	sac)                 TGT=example-sac;                        AVAR=SAC_ARGS ;;
	bert-classify-finetune)
											 TGT=example-bert-classify-finetune;     AVAR=BERT_FINETUNE_ARGS ;;
	bert-classify-sst2-finetune)
											 TGT=example-bert-classify-sst2-finetune; AVAR=BERT_SST2_ARGS ;;
	bert-classify-sst2-lora)
											 TGT=example-bert-classify-sst2-lora;     AVAR=BERT_SST2_LORA_ARGS ;;
	gpt2-lm-finetune)
											 TGT=example-gpt2-lm-finetune;            AVAR=GPT2_LM_ARGS ;;
	bert-mlm-finetune)
											 TGT=example-bert-mlm-finetune;           AVAR=BERT_MLM_ARGS ;;
	# HF inference examples — no training loop, no RESULT line; the
	# mltools.perf_log writer parses `[stage] [hh:mm:ss] <label>` lines
	# into entry.stages instead. AVAR is set to a no-op make-variable
	# name so the existing AVAR=ARGS plumbing doesn't fight us (the
	# inference examples don't take CLI args via *_ARGS).
	hf-bert)             TGT=example-hf-bert-inference;          AVAR=_HF_NOARGS ;;
	hf-gpt2)             TGT=example-hf-gpt2-inference;          AVAR=_HF_NOARGS ;;
	hf-llama)            TGT=example-hf-llama-inference;         AVAR=_HF_NOARGS ;;
	# Same example as hf-llama, but with the --dump-tokens / multi-step
	# generation gate path exercised. Distinguished from `hf-llama` so
	# the perf-log entry carries the gate's wall-clock separately from
	# the user-facing-demo wall-clock (they decode for different default
	# budgets; the gate is fixed at --num-tokens 4 in the Makefile).
	hf-llama-generate)   TGT=test-e2e-hf-llama-generate-roundtrip; AVAR=_HF_NOARGS ;;
	hf-bitnet)           TGT=example-hf-bitnet-inference;        AVAR=_HF_NOARGS ;;
	*) echo "unknown example-key: $EXAMPLE_KEY" >&2; exit 2 ;;
esac

COMMIT=$( perf_commit_with_dirty )
DEVICE=$( perf_device_for "$BACKEND" )

# Dtype override of record. Only set when caller explicitly chose
# TORCH_DTYPE / MLX_DTYPE / TAPE_DTYPE; empty otherwise (the
# BuildConfig default for the (backend, device) cell applies — F32 for
# torch-mps and mlx-gpu, F64 elsewhere). Tracked in the JSONL entry so
# a BF16/F16 run is visibly distinct from the default-F32 run on the
# same example/backend/device.
TORCH_DTYPE_STATE="${TORCH_DTYPE:-}"
MLX_DTYPE_STATE="${MLX_DTYPE:-}"
TAPE_DTYPE_STATE="${TAPE_DTYPE:-}"

MLX_COMPILE_STATE=$( perf_mlx_compile_state "$BACKEND" )

# Run the example and capture output. `tee` so make's stdout/stderr
# is saved into $LOG (for the grep-based summary below) AND streamed
# to this script's stdout. The stream-through matters when this runs
# as a background task or under a wrapper that captures stdout
# (perf-run-quiet.sh + the harness's bash-task .output file): without
# the tee, the captured file stays empty until the run completes, so
# a wedge inside `make` (download stall, elaboration hang, dylib
# relink stuck) is invisible from outside. With tee, the operator can
# read the captured file mid-run and see live progress.
LOG=$( mktemp )
T0=$( perf_now_ms )
set +e
BACKEND="$BACKEND" stdbuf -oL make --no-print-directory "$TGT" \
	"${AVAR}=${ARGS[*]}" 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
set -e
T1=$( perf_now_ms )

ELAPSED_MS=$(( T1 - T0 ))
ELAPSED_PRETTY=$( perf_pretty_elapsed_ms "$ELAPSED_MS" )

# Extract the canonical lines for operator display. The writer
# re-greps the log itself for the structured fields (the parsing
# regexes are unified in mltools.perf_log.extract_run_lines); the
# bash side just needs them for the human-readable summary.
RESULT_LINE=$( grep '^RESULT' "$LOG" | tail -1 || true )
COMPLETED_LINE=$( grep '^Completed' "$LOG" | tail -1 || true )
CONVERGED_LINE=$( grep -E '^\s*\[[^]]+\]\s+Converged' "$LOG" | tail -1 || true )
DIVERGED_LINE=$( grep -E '^\s*\[[^]]+\]\s+Diverged' "$LOG" | tail -1 || true )
STAGE_LINES=$( grep -E '^\[stage\] \[[0-9]{2}:[0-9]{2}:[0-9]{2}\]' "$LOG" || true )
PERF_LINES=$( grep -E '^\[perf\]' "$LOG" || true )

ARG_SUMMARY="${ARGS[*]:-defaults}"

python3 -m mltools.perf_log append-run \
	--example "$EXAMPLE_KEY" \
	--backend "$BACKEND" \
	--device "$DEVICE" \
	--mlx-compile "$MLX_COMPILE_STATE" \
	--commit "$COMMIT" \
	--cli-args "$ARG_SUMMARY" \
	--exit-code "$RC" \
	--wall-ms "$ELAPSED_MS" \
	--wall-human "$ELAPSED_PRETTY" \
	--torch-dtype "$TORCH_DTYPE_STATE" \
	--mlx-dtype "$MLX_DTYPE_STATE" \
	--tape-dtype "$TAPE_DTYPE_STATE" \
	--parse-log "$LOG"

rm -f "$LOG"

# Mirror to stdout for the operator.
DTYPE_TAG="${TORCH_DTYPE_STATE}${MLX_DTYPE_STATE}${TAPE_DTYPE_STATE}"
echo "=== ${EXAMPLE_KEY} [${BACKEND}/${DEVICE}${DTYPE_TAG:+/$DTYPE_TAG}] @ ${COMMIT} ==="
echo "wall:    ${ELAPSED_PRETTY} (exit ${RC})"
[ -n "$CONVERGED_LINE" ] && echo "${CONVERGED_LINE}"
[ -n "$DIVERGED_LINE"  ] && echo "${DIVERGED_LINE}"
[ -n "$COMPLETED_LINE" ] && echo "${COMPLETED_LINE}"
[ -n "$RESULT_LINE"    ] && echo "${RESULT_LINE}"
[ -n "$STAGE_LINES"    ] && printf '%s\n' "$STAGE_LINES"
[ -n "$PERF_LINES"     ] && printf '%s\n' "$PERF_LINES"
echo "Logged to ${PERF_LOG_PATH}"

exit "$RC"
