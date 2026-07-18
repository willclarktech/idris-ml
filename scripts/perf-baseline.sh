#!/usr/bin/env bash
# scripts/perf-baseline.sh
#
# Measure ms/epoch for an example on a given backend, plus the matching
# PyTorch reference. Output one CSV row to stdout:
#
#   example,backend,idris_ms_per_epoch,pytorch_ms_per_epoch,ratio,epochs,seed,commit,notes
#
# `commit` is the abbreviated git hash of HEAD, with `+dirty` appended
# if the working tree has uncommitted changes (mirrors perf-run.sh).
#
# A `kind: "baseline"` entry is also appended to `docs/develop/perf-log.jsonl`
# via the `mltools.perf_log append-baseline` writer.
#
# Usage:
#   scripts/perf-baseline.sh <example-key> <backend>
#
# Examples (key -> idris-target -> ref-script -> CLI args):
#   See EXAMPLES table below. Each row maps a friendly key to:
#     - the make target
#     - the args var name (e.g. NTM_COPY_ARGS)
#     - the python ref module
#     - the per-example budget (epochs)
#
# Notes:
#  - Timing methodology: both sides print a `PERF_MS_PER_EP=<float>`
#    marker after the training loop, before eval. The script greps the
#    marker from stdout. Eliminates startup / build / eval / Python
#    import variance from the measurement — the timer is *inside* the
#    training process. Previous version did wall-clock two-point
#    subtraction, which collapsed for short-converging RL refs (the
#    signal was below the startup-variance noise floor; see
#    `docs/develop/perf-changes.md` 2026-05-19 entry).
#  - Each side runs once at N=$N_LONG. Marker emit-points:
#      Idris:   `Train.runTrainingIO` → `formatPerfMsPerEp` (Util.idr)
#      PyTorch: `runner.run_training` for non-RL refs; per-script for RL.
#  - All idris-side runs use BACKEND=$2 from the env.
#  - All runs use --seed 42.
set -euo pipefail

source "$( dirname "${BASH_SOURCE[0]}" )/perf_lib.sh"

EXAMPLE_KEY="${1:-}"
BACKEND="${2:-tape}"
SEED=42

if [ -z "$EXAMPLE_KEY" ]; then
	echo "usage: $0 <example-key> <backend>"
	echo "  example-keys: see source"
	exit 2
fi

# example-key -> (idris-make-target, idris-args-var, ref-py-module, n_long)
case "$EXAMPLE_KEY" in
	supervised)        IDRIS_TGT=example-supervised;             IDRIS_VAR=SUPERVISED_ARGS;        REF_MOD=torch_ref.scripts.supervised;        N_LONG=200  ;;
	rnn)               IDRIS_TGT=example-rnn;                    IDRIS_VAR=RNN_ARGS;               REF_MOD=torch_ref.scripts.rnn;               N_LONG=200  ;;
	lstm)              IDRIS_TGT=example-lstm;                   IDRIS_VAR=LSTM_ARGS;              REF_MOD=torch_ref.scripts.lstm;              N_LONG=200  ;;
	gru)               IDRIS_TGT=example-gru;                    IDRIS_VAR=GRU_ARGS;               REF_MOD=torch_ref.scripts.gru;               N_LONG=200  ;;
	transformer)       IDRIS_TGT=example-transformer;            IDRIS_VAR=TRANSFORMER_ARGS;       REF_MOD=torch_ref.scripts.transformer;       N_LONG=200  ;;
	ntm-copy)          IDRIS_TGT=example-ntm-copy;               IDRIS_VAR=NTM_COPY_ARGS;          REF_MOD=torch_ref.scripts.ntm_copy;          N_LONG=40   ;;
	ntm-recall)        IDRIS_TGT=example-ntm-associative-recall; IDRIS_VAR=NTM_ASSOCIATIVE_RECALL_ARGS;        REF_MOD=torch_ref.scripts.ntm_recall;        N_LONG=40   ;;
	dnc-copy)          IDRIS_TGT=example-dnc-copy;               IDRIS_VAR=DNC_COPY_ARGS;          REF_MOD=torch_ref.scripts.dnc_copy;          N_LONG=80   ;;
	dnc-recall)        IDRIS_TGT=example-dnc-recall;             IDRIS_VAR=DNC_RECALL_ARGS;        REF_MOD=torch_ref.scripts.dnc_recall;        N_LONG=40   ;;
	reinforce)         IDRIS_TGT=example-reinforce;              IDRIS_VAR=REINFORCE_ARGS;         REF_MOD=torch_ref.scripts.reinforce;         N_LONG=200  ;;
	dqn)               IDRIS_TGT=example-dqn;                    IDRIS_VAR=DQN_ARGS;               REF_MOD=torch_ref.scripts.dqn;               N_LONG=80   ;;
	mountain-car)      IDRIS_TGT=example-mountain-car;           IDRIS_VAR=MOUNTAIN_CAR_ARGS;      REF_MOD=torch_ref.scripts.mountain_car;      N_LONG=80   ;;
	mountain-car-cont) IDRIS_TGT=example-mountain-car-cont;      IDRIS_VAR=MOUNTAIN_CAR_CONT_ARGS; REF_MOD=torch_ref.scripts.mountain_car_cont; N_LONG=2000 ;;
	a2c)               IDRIS_TGT=example-a2c;                    IDRIS_VAR=A2C_ARGS;               REF_MOD=torch_ref.scripts.a2c;               N_LONG=200  ;;
	ppo)               IDRIS_TGT=example-ppo;                    IDRIS_VAR=PPO_ARGS;               REF_MOD=torch_ref.scripts.ppo;               N_LONG=40   ;;
	sac)               IDRIS_TGT=example-sac;                    IDRIS_VAR=SAC_ARGS;               REF_MOD=torch_ref.scripts.sac;               N_LONG=2000 ;;
	*)
		echo "unknown example-key: $EXAMPLE_KEY" >&2
		exit 2 ;;
esac

# Build the example binary ONCE so `run_idris` can invoke it directly
# without paying make's per-call overhead (dependency check, dylib
# copy, fork — ~50-500 ms variable, larger than the per-epoch signal
# on tiny tape examples like rnn/gru). The recipe always runs the
# binary at the end; we pass `--epochs 1` so the post-build run is
# cheap and serves as a warmup we discard.
build_idris_binary() {
	BACKEND="$BACKEND" make --no-print-directory "$IDRIS_TGT" \
		"$IDRIS_VAR=--epochs 1 --seed $SEED" >/dev/null 2>&1 || true
}

# Derive binary path from make target: example-rnn ->
# ./build/<BUILD_KEY>/exec/rnn. Ask make for the key (mk/config.mk
# `print-%`) — replicating it here rotted when BUILD_KEY grew the
# mach/hw axes (same fix as perf-sweep.sh's build_key_for_cell).
binary_for_target() {
	local build_key
	build_key=$( BACKEND="$BACKEND" make -s --no-print-directory print-BUILD_KEY )
	echo "./build/${build_key}/exec/${IDRIS_TGT#example-}"
}

# Run idris example with `--epochs N --seed S`, capture stdout, return
# PERF_MS_PER_EP value or "crashed"/"missing".
run_idris() {
	local n="$1"
	local bin rc stdout_path errlog
	bin=$( binary_for_target )
	stdout_path=$( mktemp "${TMPDIR:-/tmp}/perf-baseline-out.XXXXXX" )
	errlog=$( mktemp "${TMPDIR:-/tmp}/perf-baseline-err.XXXXXX" )
	rc=0
	"$bin" --epochs "$n" --seed "$SEED" >"$stdout_path" 2>"$errlog" || rc=$?
	if [ "$rc" -ne 0 ]; then
		echo "[CRASH] $bin (epochs=$n) exit=$rc" >&2
		tail -3 "$errlog" >&2
		rm -f "$stdout_path" "$errlog"
		echo "crashed"
		return 0
	fi
	local val
	val=$( perf_extract_marker "$stdout_path" )
	if [ "$val" = "missing" ]; then
		# The Idris fit epilogue prints the marker via logInfo, which goes
		# to STDERR since the INFO-gating change — fall back to the errlog.
		val=$( perf_extract_marker "$errlog" )
	fi
	rm -f "$errlog" "$stdout_path"
	echo "$val"
}

run_pytorch() {
	local n="$1"
	local stdout_path rc
	stdout_path=$( mktemp "${TMPDIR:-/tmp}/perf-baseline-out.XXXXXX" )
	rc=0
	( cd packages/pytorch && uv run python -m "$REF_MOD" --epochs "$n" --seed "$SEED" ) \
		>"$stdout_path" 2>/dev/null || rc=$?
	if [ "$rc" -ne 0 ]; then
		rm -f "$stdout_path"
		echo "crashed"
		return 0
	fi
	perf_extract_marker "$stdout_path"
	rm -f "$stdout_path"
}

measure_idris() {
	build_idris_binary
	run_idris "$N_LONG"
}

measure_pytorch() {
	run_pytorch "$N_LONG"
}

COMMIT=$( perf_commit_with_dirty )

echo "[perf-baseline] $EXAMPLE_KEY [$BACKEND]: idris N=$N_LONG..." >&2
IDRIS_MS=$( measure_idris )
echo "[perf-baseline] $EXAMPLE_KEY: pytorch N=$N_LONG..." >&2
PY_MS=$( measure_pytorch )
if [ "$IDRIS_MS" = "crashed" ] || [ "$IDRIS_MS" = "missing" ]; then
	RATIO="N/A"
elif [ "$PY_MS" = "crashed" ] || [ "$PY_MS" = "missing" ]; then
	RATIO="N/A"
else
	RATIO=$( python3 -c "print(round(${IDRIS_MS} / ${PY_MS}, 2) if ${PY_MS} > 0 else 'inf')" )
fi

echo "${EXAMPLE_KEY},${BACKEND},${IDRIS_MS},${PY_MS},${RATIO},${N_LONG},${SEED},${COMMIT},"

DEVICE=$( perf_device_for "$BACKEND" )
python3 -m mltools.perf_log append-baseline \
	--example "$EXAMPLE_KEY" \
	--backend "$BACKEND" \
	--device "$DEVICE" \
	--commit "$COMMIT" \
	--idris-ms "$IDRIS_MS" \
	--pytorch-ms "$PY_MS" \
	--ratio "$RATIO" \
	--n-long "$N_LONG" \
	--seed "$SEED"
