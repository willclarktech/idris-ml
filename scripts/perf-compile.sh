#!/usr/bin/env bash
# scripts/perf-compile.sh — measure COLD full Idris compilation time for
# one unit (or a sweep) and append kind=compile records to
# docs/develop/perf-log.jsonl.
#
# Compilation is the developer-felt cost the linear-types migration can
# move — linearity is compile-time only, so it can't touch runtime (which
# perf-run.sh / perf-baseline.sh cover). "Full compilation" means the
# whole build, not just the typecheck phase: libraries via `--build`
# (all modules → ttc), examples via `-o` (elaborate + Chez codegen +
# executable). See docs/develop/linear-types-and-effects.md.
#
# Usage:
#   scripts/perf-compile.sh <target> <backend> [unit-label]
#
#   <target>:
#     lib | idris-ml             build packages/idris-ml/idris-ml.ipkg
#     transformers | idris-transformers
#     gym | idris-gym
#     <path/to/Example/Foo.idr>  -o build that single example (full codegen)
#     sweep | all                every library package + every example
#                                module (the comprehensive / campaign pass)
#   <backend>:    tape | mlx | torch
#   [unit-label]: override the recorded `unit` (default: derived)
#
# Cold by construction: builds into a throwaway --build-dir so no cached
# ttc is reused. Does NOT relink the C dylib (reused). Build vars
# (EXAMPLE_SRC / IDRIS2_LOCAL / IDRIS2_PACKAGE_PATH) are queried from
# `make print-%` so the timer reuses the build's own flag/prefix
# resolution (single source of truth).
#
# Env:
#   PERF_COMPILE_DRYRUN=1  resolve + print the idris2 command per unit;
#                          run nothing, append nothing.
#   PERF_COMPILE_LOG=path  override the perf-log.jsonl target (smoke tests).
set -euo pipefail

source "$( dirname "${BASH_SOURCE[0]}" )/perf_lib.sh"
cd "$PERF_REPO_ROOT"

if [ $# -lt 2 ]; then
	cat <<EOF >&2
usage: $0 <target> <backend> [unit-label]
  targets:  lib | idris-ml | transformers | idris-transformers | gym | idris-gym
            | <path/to/Example/Foo.idr> | sweep
  backends: tape | mlx | torch
EOF
	exit 2
fi

TARGET="$1"; BACKEND="$2"; LABEL="${3:-}"

DEVICE=$( perf_device_for "$BACKEND" )
COMMIT=$( perf_commit_with_dirty )

# Build-var resolution via make introspection (single source of truth).
EXAMPLE_SRC=$( make -s print-EXAMPLE_SRC BACKEND="$BACKEND" )
IDRIS2_LOCAL=$( make -s print-IDRIS2_LOCAL BACKEND="$BACKEND" )
IDRIS2_PACKAGE_PATH=$( make -s print-IDRIS2_PACKAGE_PATH BACKEND="$BACKEND" )
export IDRIS2_PACKAGE_PATH
# Single compiler: pack's idris2 (same one the build installed against).
IDRIS2=$( make -s print-IDRIS2 BACKEND="$BACKEND" )

LOG_ARGS=()
[ -n "${PERF_COMPILE_LOG:-}" ] && LOG_ARGS=( --log-path "$PERF_COMPILE_LOG" )

# Full cold compile of one unit → one kind=compile record. Library units
# `--build` the .ipkg from the package dir (mirroring mk/install.mk);
# example units `-o` a full executable build against the installed packages.
measure_one() {  # <target> <label-override>
	local target="$1" label_override="$2"
	local pkgdir="" ipkg="" unit="" base kebab name="" tmp rundir rc ms pretty
	local -a run
	case "$target" in
		lib|idris-ml)
			pkgdir="packages/idris-ml";           ipkg="idris-ml.ipkg";           unit="idris-ml" ;;
		transformers|idris-transformers)
			pkgdir="packages/idris-transformers"; ipkg="idris-transformers.ipkg"; unit="idris-transformers" ;;
		gym|idris-gym)
			pkgdir="packages/idris-gym";          ipkg="idris-gym.ipkg";          unit="idris-gym" ;;
		*.idr)
			[ -f "$target" ] || { echo "no such source file: $target" >&2; return 2; }
			base=$( basename "$target" .idr )    # e.g. Mnist
			# kebab-case the stem: Mnist → mnist, SeqClassify → seq-classify
			kebab=$( printf '%s' "$base" | sed 's/\([a-z0-9]\)\([A-Z]\)/\1-\2/g' | tr '[:upper:]' '[:lower:]' )
			unit="example-$kebab"; name="$base" ;;
		*)
			echo "unknown target: $target (expected lib / transformers / gym / a .idr path)" >&2
			return 2 ;;
	esac
	[ -n "$label_override" ] && unit="$label_override"

	tmp=$( mktemp -d )
	if [ -n "$pkgdir" ]; then
		run=( env IDRIS2_PREFIX="$IDRIS2_LOCAL" "$IDRIS2" --build-dir "$tmp" --build "$ipkg" )
		rundir="$pkgdir"
	else
		run=( "$IDRIS2" --build-dir "$tmp" --source-dir "$EXAMPLE_SRC"
		      -p contrib -p linear -p idris-ml -p idris-gym -p idris-transformers
		      -o "$name" "$target" )
		rundir="."
	fi

	if [ "${PERF_COMPILE_DRYRUN:-}" = "1" ]; then
		echo "[dry-run] unit=$unit backend=$BACKEND device=$DEVICE commit=$COMMIT"
		echo "[dry-run] (cd $rundir && ${run[*]})"
		rm -rf "$tmp"
		return 0
	fi

	echo "=== compile: $unit [$BACKEND/$DEVICE] @ $COMMIT (cold) ==="
	local t0 t1
	t0=$( perf_now_ms )
	set +e
	( cd "$rundir" && perf_quiet_run "${run[@]}" )
	rc=$?
	set -e
	t1=$( perf_now_ms )
	rm -rf "$tmp"

	ms=$(( t1 - t0 ))
	pretty=$( perf_pretty_elapsed_ms "$ms" )
	python3 -m mltools.perf_log "${LOG_ARGS[@]}" append-compile \
		--unit "$unit" --backend "$BACKEND" --device "$DEVICE" --commit "$COMMIT" \
		--compile-ms "$ms" --compile-human "$pretty" --exit-code "$rc"
	echo "compile: $pretty (exit $rc)"
	return "$rc"
}

case "$TARGET" in
	sweep|all)
		units="idris-ml idris-transformers idris-gym"
		units="$units $( find packages/idris-ml-examples/src/Example -name '*.idr' | sort )"
		n=0; fail=0
		for u in $units; do
			n=$(( n + 1 ))
			echo "--- [$n] $u ($BACKEND) ---"
			measure_one "$u" "" || fail=$(( fail + 1 ))
		done
		echo "compile sweep done: $n units, $fail non-zero exit(s) on $BACKEND"
		echo "Logged to ${PERF_COMPILE_LOG:-$PERF_LOG_PATH}"
		exit 0 ;;
	*)
		measure_one "$TARGET" "$LABEL" && rc=0 || rc=$?
		echo "Logged to ${PERF_COMPILE_LOG:-$PERF_LOG_PATH}"
		exit "$rc" ;;
esac
