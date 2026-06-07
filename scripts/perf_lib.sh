# scripts/perf_lib.sh — common helpers for the perf-* shell scripts.
#
# Sourced, not executed:
#
#     source "$(dirname "$0")/perf_lib.sh"
#
# Exports:
#   PERF_REPO_ROOT   absolute repo root (parent of scripts/)
#   PERF_LOG_PATH    docs/develop/perf-log.jsonl (absolute)
#   PYTHONPATH       prefixed with $PERF_REPO_ROOT/scripts so the
#                    sibling `python3 -m mltools.<module>` invocations
#                    can find the package.
#
# Functions:
#   perf_quiet_run CMD...        run CMD under caffeinate -i + nice -n 19
#                                (plain nice on Linux — no caffeinate there)
#   perf_commit_with_dirty       short HEAD hash; +dirty if any tracked
#                                file outside perf-log.jsonl / BENCHMARKS.md
#                                is modified
#   perf_now_ms                  ms since epoch (int)
#   perf_extract_marker FILE     PERF_MS_PER_EP value, or "missing"
#   perf_extract_axis_d_tokens FILE
#   perf_extract_axis_d_wall FILE
#   perf_pretty_elapsed_ms MS    "1h 2m 3s" / "4m 5s" / "5.123s"
#   perf_device_for BACKEND      lookup MLX_DEVICE / TORCH_DEVICE, normalize
#                                "metal" → "gpu", default "cpu"
#   perf_mlx_compile_state BACKEND
#                                "on" / "off" / "n/a" (mlx-only)

# Resolve repo root from this file's location (not $PWD — the script
# may be invoked from anywhere). BASH_SOURCE[0] is this file; its
# parent is scripts/, its grandparent is the repo root.
PERF_REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
PERF_LOG_PATH="$PERF_REPO_ROOT/docs/develop/perf-log.jsonl"
export PYTHONPATH="$PERF_REPO_ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"

# Run a command under the heavy-command wrapper (caffeinate + nice).
# caffeinate is macOS-only — on Linux (CI runners) the binary doesn't
# exist and the idle-sleep concern doesn't apply, so degrade to plain
# nice. Callers pass the full command incl. any `env VAR=...` prefix.
perf_quiet_run() {
	if [ "$(uname -s)" = "Darwin" ] && command -v caffeinate >/dev/null 2>&1; then
		caffeinate -i nice -n 19 "$@"
	else
		nice -n 19 "$@"
	fi
}

perf_commit_with_dirty() {
	local commit
	commit=$( git -C "$PERF_REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo "unknown" )
	if [ -n "$( git -C "$PERF_REPO_ROOT" status --porcelain -- ':!docs/develop/perf-log.jsonl' ':!BENCHMARKS.md' 2>/dev/null )" ]; then
		commit="${commit}+dirty"
	fi
	printf '%s' "$commit"
}

perf_now_ms() {
	python3 -c 'import time; print(int(time.time_ns()/1_000_000))'
}

perf_extract_marker() {
	local stdout_path="$1"
	local val
	val=$( { grep -E '^PERF_MS_PER_EP=' "$stdout_path" || true; } | tail -1 | sed 's/^PERF_MS_PER_EP=//' )
	if [ -z "$val" ]; then
		echo "missing"
	else
		python3 -c "print(round(float('$val'), 2))"
	fi
}

perf_extract_axis_d_tokens() {
	local stdout_path="$1"
	{ grep -E '^PERF_GENERATE_TOKENS=' "$stdout_path" || true; } | tail -1 \
		| sed 's/^PERF_GENERATE_TOKENS=//'
}

perf_extract_axis_d_wall() {
	local stdout_path="$1"
	{ grep -E '^PERF_GENERATE_WALL_MS=' "$stdout_path" || true; } | tail -1 \
		| sed 's/^PERF_GENERATE_WALL_MS=//'
}

perf_pretty_elapsed_ms() {
	python3 -c "
ms = $1
s = ms // 1000
m = s // 60
s = s % 60
h = m // 60
m = m % 60
if h > 0: print(f'{h}h {m}m {s}s')
elif m > 0: print(f'{m}m {s}s')
else: print(f'{s}.{ms%1000:03d}s')
"
}

perf_device_for() {
	local backend="$1"
	local dev
	case "$backend" in
		mlx)   dev="${MLX_DEVICE:-cpu}" ;;
		tape)  dev="cpu" ;;
		torch) dev="${TORCH_DEVICE:-cpu}" ;;
		*)     dev="unknown" ;;
	esac
	[ "$dev" = "metal" ] && dev="gpu"
	printf '%s' "$dev"
}

perf_mlx_compile_state() {
	local backend="$1"
	case "$backend" in
		mlx)
			case "${MLX_COMPILE:-}" in
				1|true|yes) printf 'on'  ;;
				*)          printf 'off' ;;
			esac
			;;
		*) printf 'n/a' ;;
	esac
}
