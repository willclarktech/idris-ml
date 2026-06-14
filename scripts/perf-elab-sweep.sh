#!/usr/bin/env bash
# scripts/perf-elab-sweep.sh — cold-elaboration sweep over the library
# packages + every example module, appending one kind=elab record each.
#
# Intended for the FINAL measurement pass (run alongside the convergence
# campaign on the final tree), not mid-development — elaboration is the
# one perf axis the linear-types migration can move. See
# docs/develop/linear-types-and-effects.md.
#
# Usage:
#   scripts/perf-elab-sweep.sh [backend] [--units "u1 u2 ..."]
#
#   backend:   tape (default) | mlx | torch
#   --units:   override the swept unit list (default: the 3 library
#              packages + every packages/idris-ml-examples/src/Example/*.idr)
#
# Honors PERF_ELAB_DRYRUN=1 (pass-through) so the whole sweep can be
# resolved without elaborating anything. Each unit is independent; a
# failing typecheck is logged (exit!=0) and the sweep continues.
set -uo pipefail

HERE="$( dirname "${BASH_SOURCE[0]}" )"
ROOT="$( cd "$HERE/.." && pwd )"
cd "$ROOT" || exit 1

BACKEND="tape"
UNITS=""
while [ $# -gt 0 ]; do
	case "$1" in
		--units) UNITS="$2"; shift 2 ;;
		tape|mlx|torch) BACKEND="$1"; shift ;;
		*) echo "usage: $0 [tape|mlx|torch] [--units \"...\"]" >&2; exit 2 ;;
	esac
done

# Default sweep: the three library packages (the heavy elaboration units
# where the linear-types machinery lives), then every example module.
if [ -z "$UNITS" ]; then
	UNITS="idris-ml idris-transformers idris-gym"
	UNITS="$UNITS $( find packages/idris-ml-examples/src/Example -name '*.idr' | sort )"
fi

n=0; fail=0
for u in $UNITS; do
	n=$(( n + 1 ))
	echo "--- [$n] elab: $u ($BACKEND) ---"
	"$HERE/perf-elab.sh" "$u" "$BACKEND" || fail=$(( fail + 1 ))
done

echo "elab sweep done: $n units, $fail non-zero exit(s) on $BACKEND"
# Non-zero typechecks are recorded data (exit field), not a sweep failure.
exit 0
