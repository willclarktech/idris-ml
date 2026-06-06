#!/usr/bin/env bash
# scripts/perf-nightly.sh — Tier 2 perf gate (Axes A+B+C+D, tape only).
#
# Runs the Tier-1 op-kernel + layer benches via perf-fast.sh, then
# extends with Axis C (end-to-end training, one workload per training
# mode, capped to a short epoch count) and Axis D (HF inference, per-
# token wall around the inference window). All entries are appended
# via `mltools.perf_log append-axis-row`.
#
# Designed for a daily GitHub Actions schedule — total wall ≤ 20 min
# on tape. Per CLAUDE.md "all backends first-class" Tier 2 stays
# tape-only by design; Tier 3 (perf-sweep.sh) covers cross-backend.
#
# Schema (Axis C / D entries):
#   { "kind": "op_bench", "axis": "C"|"D", "section": "<example>-train",
#     "label": "<example>-train", "runtime": "tape" | "pytorch",
#     "wall_ms": ..., "iters": N, "ms_per_iter": ms_per_ep,
#     "commit": "...", "ts": "...", "date": "..." }
#
# Usage:
#   scripts/perf-nightly.sh                # full Tier 1 + Axes C + D
#   scripts/perf-nightly.sh --axis-c-only  # skip Tier 1, skip Axis D
#   scripts/perf-nightly.sh --axis-d-only  # skip Tier 1, skip Axis C
#
# Workloads (one per distinct training mode):
#   supervised   — feedforward
#   lstm         — recurrent
#   transformer  — sequence-to-sequence
#   ntm-copy     — memory / two-phase
#   reinforce    — RL on-policy

set -euo pipefail

source "$( dirname "${BASH_SOURCE[0]}" )/perf_lib.sh"

cd "$PERF_REPO_ROOT"

SEED=42

DO_TIER1=1
DO_AXIS_C=1
DO_AXIS_D=1
for arg in "$@"; do
	case "$arg" in
		--axis-c-only) DO_TIER1=0; DO_AXIS_D=0 ;;
		--axis-d-only) DO_TIER1=0; DO_AXIS_C=0 ;;
		-h|--help)
			sed -n '2,30p' "$0"
			exit 0
			;;
		*)
			echo "perf-nightly.sh: unknown arg: $arg" >&2
			exit 2
			;;
	esac
done

if [ "$DO_TIER1" = "1" ]; then
	echo "==> perf-nightly: running Tier 1 (Axes A + B)"
	bash scripts/perf-fast.sh --no-render
fi

COMMIT=$( perf_commit_with_dirty )

# example-key  idris-make-target               idris-args-var  ref-module                      epochs
AXIS_C_WORKLOADS=(
	"supervised  example-supervised  SUPERVISED_ARGS  torch_ref.scripts.supervised  50"
	"lstm        example-lstm        LSTM_ARGS        torch_ref.scripts.lstm        20"
	"transformer example-transformer TRANSFORMER_ARGS torch_ref.scripts.transformer 10"
	"ntm-copy    example-ntm-copy    NTM_COPY_ARGS    torch_ref.scripts.ntm_copy    10"
	"reinforce   example-reinforce   REINFORCE_ARGS   torch_ref.scripts.reinforce   50"
)

# Marker → axis-C row. Logs both sides; missing-marker side is
# silently skipped (matches the previous heredoc behaviour).
log_axis_c_pair() {
	local key="$1" epochs="$2" idris_ms="$3" py_ms="$4"
	local label="${key}-train"
	for pair in "tape:$idris_ms" "pytorch:$py_ms"; do
		local runtime="${pair%%:*}"
		local ms="${pair#*:}"
		[ -z "$ms" ] && continue
		if ! python3 -c "float('$ms')" 2>/dev/null; then
			continue
		fi
		local wall_ms
		wall_ms=$( python3 -c "print(float('$ms') * $epochs)" )
		python3 -m mltools.perf_log append-axis-row \
			--axis C --runtime "$runtime" --label "$label" --section "$label" \
			--wall-ms "$wall_ms" --iters "$epochs" --commit "$COMMIT"
	done
}

if [ "$DO_AXIS_C" = "1" ]; then
	echo "==> perf-nightly: running Axis C (e2e training)"
	for row in "${AXIS_C_WORKLOADS[@]}"; do
		read -r key idris_tgt idris_var ref_mod epochs <<< "$row"
		echo "  -> $key (epochs=$epochs)"

		IDRIS_OUT=$( mktemp )
		PY_OUT=$( mktemp )

		set +e
		caffeinate -i nice -n 19 env MAKEFLAGS=-j2 \
			make --no-print-directory "$idris_tgt" \
			"$idris_var=--epochs $epochs --seed $SEED" \
			>"$IDRIS_OUT" 2>&1
		set -e
		IDRIS_MS_RAW=$( perf_extract_marker "$IDRIS_OUT" )

		set +e
		( cd packages/pytorch && uv run python -m "$ref_mod" \
				--epochs "$epochs" --seed "$SEED" ) >"$PY_OUT" 2>&1
		set -e
		PY_MS_RAW=$( perf_extract_marker "$PY_OUT" )

		# perf_extract_marker emits "missing" / "crashed" sentinels; the
		# axis-row logger treats those as "skip side" (only numeric values
		# become rows).
		[ "$IDRIS_MS_RAW" = "missing" ] && IDRIS_MS_RAW=""
		[ "$IDRIS_MS_RAW" = "crashed" ] && IDRIS_MS_RAW=""
		[ "$PY_MS_RAW" = "missing" ] && PY_MS_RAW=""
		[ "$PY_MS_RAW" = "crashed" ] && PY_MS_RAW=""

		log_axis_c_pair "$key" "$epochs" "$IDRIS_MS_RAW" "$PY_MS_RAW"
		rm -f "$IDRIS_OUT" "$PY_OUT"
	done
fi

# AXIS D — HuggingFace inference.
# Each workload emits PERF_GENERATE_TOKENS + PERF_GENERATE_WALL_MS
# from both sides. ms_per_iter = wall / tokens; iters = tokens.
AXIS_D_WORKLOADS=(
	"hf-bert            example-hf-bert-inference   torch_ref.scripts.hf_bert_inference  -"
	"hf-gpt2            example-hf-gpt2-inference   torch_ref.scripts.hf_gpt2_inference  -"
	"hf-llama-generate  example-hf-llama-inference  torch_ref.scripts.hf_llama_inference BACKEND=torch:TORCH_DEVICE=mps"
)

log_axis_d_pair() {
	local key="$1" idris_tok="$2" idris_wall="$3" py_tok="$4" py_wall="$5"
	local label="${key}-inference"
	for triple in "tape:$idris_tok:$idris_wall" "pytorch:$py_tok:$py_wall"; do
		local runtime="${triple%%:*}"
		local rest="${triple#*:}"
		local tok="${rest%%:*}"
		local wall="${rest#*:}"
		[ -z "$tok" ] || [ -z "$wall" ] && continue
		if ! python3 -c "int('$tok'); float('$wall')" 2>/dev/null; then
			continue
		fi
		if [ "$tok" -le 0 ]; then
			continue
		fi
		python3 -m mltools.perf_log append-axis-row \
			--axis D --runtime "$runtime" --label "$label" --section "$label" \
			--wall-ms "$wall" --iters "$tok" --commit "$COMMIT"
	done
}

if [ "$DO_AXIS_D" = "1" ]; then
	echo "==> perf-nightly: running Axis D (HF inference)"
	for row in "${AXIS_D_WORKLOADS[@]}"; do
		read -r key idris_tgt ref_mod idris_env <<< "$row"
		echo "  -> $key"

		IDRIS_OUT=$( mktemp )
		PY_OUT=$( mktemp )

		env_args=()
		if [ "$idris_env" != "-" ]; then
			IFS=':' read -ra env_pairs <<< "$idris_env"
			for pair in "${env_pairs[@]}"; do
				env_args+=("$pair")
			done
		fi

		set +e
		caffeinate -i nice -n 19 env MAKEFLAGS=-j2 "${env_args[@]}" \
			make --no-print-directory "$idris_tgt" >"$IDRIS_OUT" 2>&1
		set -e
		IDRIS_TOK=$( perf_extract_axis_d_tokens "$IDRIS_OUT" )
		IDRIS_WALL=$( perf_extract_axis_d_wall "$IDRIS_OUT" )

		set +e
		( cd packages/pytorch && uv run python -m "$ref_mod" ) >"$PY_OUT" 2>&1
		set -e
		PY_TOK=$( perf_extract_axis_d_tokens "$PY_OUT" )
		PY_WALL=$( perf_extract_axis_d_wall "$PY_OUT" )

		log_axis_d_pair "$key" "$IDRIS_TOK" "$IDRIS_WALL" "$PY_TOK" "$PY_WALL"
		rm -f "$IDRIS_OUT" "$PY_OUT"
	done
fi

echo "==> perf-nightly: rendering BENCHMARKS.md"
python3 scripts/render-benchmarks.py

echo "==> perf-nightly: done"
