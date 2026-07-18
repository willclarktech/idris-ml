#!/usr/bin/env bash
# scripts/perf-sweep.sh — sweep ms/epoch across (example × cell) cells.
#
# For each example: run the PyTorch reference ONCE (cached); then run
# Idris on each requested cell. One JSONL entry per (example, cell) is
# appended to docs/develop/perf-log.jsonl, plus a tabular summary to
# stdout. Entry construction delegates to `mltools.perf_log
# append-baseline`.
#
# A "cell" is a (backend, device) pair. The supported cells are:
#   tape       — CPU, tape backend
#   torch      — CPU, libtorch backend
#   torch-cpu  — alias for torch
#   torch-mps  — Metal Performance Shaders
#   torch-cuda — NVIDIA GPU via libtorch (Linux/Colab CUDA boxes)
#   mlx-cpu    — mlx backend, MLX_DEVICE=cpu (default)
#   mlx-gpu    — mlx backend, MLX_DEVICE=gpu (Metal)
#
# Usage:
#   scripts/perf-sweep.sh [--examples a,b,...] [--cells tape,torch,mlx-cpu,mlx-gpu]
#                         [--seed N]
#
# Differences vs running scripts/perf-baseline.sh per cell:
#  - One PyTorch reference run per example (not per cell). On a 6-
#    example × 4-cell sweep that's 18 saved ref runs.
#  - mlx-gpu is a first-class cell (perf-baseline.sh has no flag for it).
#
# Timing methodology: per cell, build the example binary ONCE via
# `make example-<name>` (which links the right backend dylib), then
# run it at N_LONG and grep `PERF_MS_PER_EP=<float>` from stdout.
# Each side (Idris + PyTorch ref) emits the marker after the timed
# training loop, before eval — so the measurement is the per-epoch
# wall *inside* the training process, with no startup / build / eval
# variance folded in.
set -euo pipefail

source "$( dirname "${BASH_SOURCE[0]}" )/perf_lib.sh"

EXAMPLES_CSV="rnn,lstm,gru,transformer,ntm-copy,ntm-recall"
CELLS_CSV="tape,torch-cpu,torch-mps,mlx-cpu,mlx-gpu"
SEED=42

while [ $# -gt 0 ]; do
	case "$1" in
		--examples) EXAMPLES_CSV="$2"; shift 2 ;;
		--cells)    CELLS_CSV="$2"; shift 2 ;;
		--seed)     SEED="$2"; shift 2 ;;
		-h|--help)  sed -n '/^# scripts/,/^set -e/p' "$0"; exit 0 ;;
		*) echo "unknown flag: $1" >&2; exit 2 ;;
	esac
done

# example -> "make-target args-var pytorch-module N_LONG"
spec_for() {
	case "$1" in
		supervised)        echo "example-supervised SUPERVISED_ARGS torch_ref.scripts.supervised 200" ;;
		rnn)               echo "example-rnn RNN_ARGS torch_ref.scripts.rnn 200" ;;
		lstm)              echo "example-lstm LSTM_ARGS torch_ref.scripts.lstm 200" ;;
		gru)               echo "example-gru GRU_ARGS torch_ref.scripts.gru 200" ;;
		transformer)       echo "example-transformer TRANSFORMER_ARGS torch_ref.scripts.transformer 200" ;;
		ntm-copy)          echo "example-ntm-copy NTM_COPY_ARGS torch_ref.scripts.ntm_copy 40" ;;
		ntm-recall)        echo "example-ntm-associative-recall NTM_ASSOCIATIVE_RECALL_ARGS torch_ref.scripts.ntm_recall 40" ;;
		dnc-copy)          echo "example-dnc-copy DNC_COPY_ARGS torch_ref.scripts.dnc_copy 80" ;;
		dnc-recall)        echo "example-dnc-recall DNC_RECALL_ARGS torch_ref.scripts.dnc_recall 40" ;;
		reinforce)         echo "example-reinforce REINFORCE_ARGS torch_ref.scripts.reinforce 200" ;;
		dqn)               echo "example-dqn DQN_ARGS torch_ref.scripts.dqn 80" ;;
		mountain-car)      echo "example-mountain-car MOUNTAIN_CAR_ARGS torch_ref.scripts.mountain_car 80" ;;
		mountain-car-cont) echo "example-mountain-car-cont MOUNTAIN_CAR_CONT_ARGS torch_ref.scripts.mountain_car_cont 2000" ;;
		a2c)               echo "example-a2c A2C_ARGS torch_ref.scripts.a2c 200" ;;
		ppo)               echo "example-ppo PPO_ARGS torch_ref.scripts.ppo 40" ;;
		sac)               echo "example-sac SAC_ARGS torch_ref.scripts.sac 2000" ;;
		*) return 1 ;;
	esac
}

# cell -> "BACKEND DEVICE"
cell_to_backend_device() {
	case "$1" in
		tape)      echo "tape cpu" ;;
		torch)     echo "torch cpu" ;;
		torch-cpu) echo "torch cpu" ;;
		torch-mps) echo "torch mps" ;;
		torch-cuda) echo "torch cuda" ;;
		mlx-cpu)   echo "mlx cpu" ;;
		mlx-gpu)   echo "mlx gpu" ;;
		*) return 1 ;;
	esac
}

# Per-cell BUILD_KEY, asked from make itself (mk/config.mk `print-%`).
# Replicating the key here rotted when BUILD_KEY grew the mach/hw axes
# — the derived path silently pointed at a nonexistent build tree.
build_key_for_cell() {
	local backend="$1" device="$2"
	local mlx_dev=cpu torch_dev=cpu
	case "$backend" in
		mlx)   mlx_dev="$device"   ;;
		torch) torch_dev="$device" ;;
		tape)  ;;
	esac
	BACKEND="$backend" MLX_DEVICE="$mlx_dev" TORCH_DEVICE="$torch_dev" \
		make -s --no-print-directory print-BUILD_KEY
}

# example-rnn (tape, cpu) -> ./build/<BUILD_KEY>/exec/rnn
binary_for_target() {
	local target="$1" backend="$2" device="$3"
	local build_key
	build_key=$( build_key_for_cell "$backend" "$device" )
	echo "./build/${build_key}/exec/${target#example-}"
}

build_idris_binary() {
	local target="$1" var="$2" backend="$3" device="$4"
	local errlog rc=0
	errlog=$( mktemp "${TMPDIR:-/tmp}/perf-sweep-build.XXXXXX" )
	case "$backend" in
		mlx)
			MLX_DEVICE="$device" BACKEND="$backend" make --no-print-directory "$target" \
				"$var=--epochs 1 --seed $SEED" >/dev/null 2>"$errlog" || rc=$?
			;;
		torch)
			TORCH_DEVICE="$device" BACKEND="$backend" make --no-print-directory "$target" \
				"$var=--epochs 1 --seed $SEED" >/dev/null 2>"$errlog" || rc=$?
			;;
		*)
			BACKEND="$backend" make --no-print-directory "$target" \
				"$var=--epochs 1 --seed $SEED" >/dev/null 2>"$errlog" || rc=$?
			;;
	esac
	if [ "$rc" -ne 0 ]; then
		echo "[BUILD FAIL] make $target backend=$backend device=$device exit=$rc" >&2
		tail -5 "$errlog" >&2
		rm -f "$errlog"
		return "$rc"
	fi
	rm -f "$errlog"
}

run_idris_once() {
	local target="$1" _var="$2" n="$3" backend="$4" device="$5"
	local bin rc stdout_path errlog
	bin=$( binary_for_target "$target" "$backend" "$device" )
	stdout_path=$( mktemp "${TMPDIR:-/tmp}/perf-sweep-out.XXXXXX" )
	errlog=$( mktemp "${TMPDIR:-/tmp}/perf-sweep-err.XXXXXX" )
	rc=0
	if [ "$backend" = "mlx" ]; then
		MLX_DEVICE="$device" "$bin" --epochs "$n" --seed "$SEED" >"$stdout_path" 2>"$errlog" || rc=$?
	else
		"$bin" --epochs "$n" --seed "$SEED" >"$stdout_path" 2>"$errlog" || rc=$?
	fi
	if [ "$rc" -ne 0 ]; then
		echo "[CRASH] $bin (backend=$backend device=$device epochs=$n) exit=$rc" >&2
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

run_pytorch_once() {
	local mod="$1" n="$2"
	local stdout_path rc
	stdout_path=$( mktemp "${TMPDIR:-/tmp}/perf-sweep-out.XXXXXX" )
	rc=0
	( cd packages/pytorch && uv run python -m "$mod" --epochs "$n" --seed "$SEED" ) \
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
	local target="$1" var="$2" n_long="$3" backend="$4" device="$5"
	if ! build_idris_binary "$target" "$var" "$backend" "$device"; then
		echo "crashed"
		return 0
	fi
	run_idris_once "$target" "$var" "$n_long" "$backend" "$device"
}

measure_pytorch() {
	local mod="$1" n_long="$2"
	run_pytorch_once "$mod" "$n_long"
}

COMMIT=$( perf_commit_with_dirty )

echo
printf '%-18s %-8s %-6s %12s %12s %8s\n' example backend device idris_ms py_ms ratio
printf '%-18s %-8s %-6s %12s %12s %8s\n' '------' '------' '----' '--------' '-----' '-----'

IFS=, read -r -a EXAMPLES <<<"$EXAMPLES_CSV"
IFS=, read -r -a CELLS <<<"$CELLS_CSV"

for example in "${EXAMPLES[@]}"; do
	spec=$( spec_for "$example" ) || { echo "unknown example: $example" >&2; exit 2; }
	read -r IDRIS_TGT IDRIS_VAR REF_MOD N_LONG <<<"$spec"

	echo "[$example] pytorch ref N=$N_LONG..." >&2
	PY_MS=$( measure_pytorch "$REF_MOD" "$N_LONG" )

	for cell in "${CELLS[@]}"; do
		bd=$( cell_to_backend_device "$cell" ) || { echo "unknown cell: $cell" >&2; exit 2; }
		read -r BACKEND DEVICE <<<"$bd"
		echo "[$example/$cell] idris N=$N_LONG..." >&2
		IDRIS_MS=$( measure_idris "$IDRIS_TGT" "$IDRIS_VAR" "$N_LONG" "$BACKEND" "$DEVICE" )
		if [ "$IDRIS_MS" = "crashed" ] || [ "$IDRIS_MS" = "missing" ]; then
			RATIO="N/A"
		elif [ "$PY_MS" = "crashed" ] || [ "$PY_MS" = "missing" ]; then
			RATIO="N/A"
		else
			RATIO=$( python3 -c "print(round($IDRIS_MS / $PY_MS, 2) if $PY_MS > 0 else float('inf'))" )
		fi
		python3 -m mltools.perf_log append-baseline \
			--example "$example" --backend "$BACKEND" --device "$DEVICE" \
			--commit "$COMMIT" --idris-ms "$IDRIS_MS" --pytorch-ms "$PY_MS" \
			--ratio "$RATIO" --n-long "$N_LONG" --seed "$SEED"
		printf '%-18s %-8s %-6s %12s %12s %8s\n' "$example" "$BACKEND" "$DEVICE" "$IDRIS_MS" "$PY_MS" "$RATIO"
	done
done

echo
echo "Sweep complete. JSONL entries appended to $PERF_LOG_PATH (kind=baseline)."
