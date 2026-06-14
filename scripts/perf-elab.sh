#!/usr/bin/env bash
# scripts/perf-elab.sh — measure COLD Idris elaboration (typecheck) time
# for one unit and append a kind=elab record to docs/develop/perf-log.jsonl.
#
# Elaboration (typecheck) is the ONE perf axis the linear-types migration
# can move — linearity is compile-time only, so it cannot touch runtime
# (which perf-run.sh / perf-baseline.sh already cover). See
# docs/develop/linear-types-and-effects.md.
#
# Usage:
#   scripts/perf-elab.sh <target> <backend> [unit-label]
#
#   <target>:
#     lib | idris-ml             typecheck packages/idris-ml/idris-ml.ipkg
#     transformers | idris-transformers
#     gym | idris-gym
#     <path/to/Example/Foo.idr>  --check that single example module
#   <backend>:    tape | mlx | torch
#   [unit-label]: override the recorded `unit` (default: derived)
#
# Cold by construction: elaborates into a throwaway --build-dir so no
# cached ttc is reused. Measures the typecheck phase only — no codegen,
# no C dylib build. Build vars (BUILD / EXAMPLE_SRC / IDRIS2_LOCAL /
# IDRIS2_PACKAGE_PATH) are queried from `make print-%` so the timer
# reuses the build's own flag/prefix resolution (single source of truth).
#
# Env:
#   PERF_ELAB_DRYRUN=1   resolve + print the idris2 command and the record
#                        that WOULD be written; run nothing, append nothing.
#   PERF_ELAB_LOG=path   override the perf-log.jsonl target (for smoke tests).
set -euo pipefail

source "$( dirname "${BASH_SOURCE[0]}" )/perf_lib.sh"

if [ $# -lt 2 ]; then
	cat <<EOF >&2
usage: $0 <target> <backend> [unit-label]
  targets:  lib | idris-ml | transformers | idris-transformers | gym | idris-gym
            | <path/to/Example/Foo.idr>
  backends: tape | mlx | torch
EOF
	exit 2
fi

TARGET="$1"; BACKEND="$2"; LABEL="${3:-}"

# Build-var resolution via make introspection (single source of truth).
EXAMPLE_SRC=$( make -s print-EXAMPLE_SRC BACKEND="$BACKEND" )
IDRIS2_LOCAL=$( make -s print-IDRIS2_LOCAL BACKEND="$BACKEND" )
IDRIS2_PACKAGE_PATH=$( make -s print-IDRIS2_PACKAGE_PATH BACKEND="$BACKEND" )
export IDRIS2_PACKAGE_PATH

DEVICE=$( perf_device_for "$BACKEND" )
COMMIT=$( perf_commit_with_dirty )

# Throwaway build-dir → guaranteed cold elaboration; always cleaned up.
TMP=$( mktemp -d )
trap 'rm -rf "$TMP"' EXIT

# Resolve <target> → (the idris2 command, the recorded unit label). Library
# units typecheck the .ipkg from the package dir (mirroring mk/install.mk);
# example units --check the single module against the installed packages.
PKGDIR=""
case "$TARGET" in
	lib|idris-ml)
		PKGDIR="packages/idris-ml";            IPKG="idris-ml.ipkg";            UNIT="idris-ml" ;;
	transformers|idris-transformers)
		PKGDIR="packages/idris-transformers";  IPKG="idris-transformers.ipkg";  UNIT="idris-transformers" ;;
	gym|idris-gym)
		PKGDIR="packages/idris-gym";           IPKG="idris-gym.ipkg";           UNIT="idris-gym" ;;
	*.idr)
		[ -f "$TARGET" ] || { echo "no such source file: $TARGET" >&2; exit 2; }
		base=$( basename "$TARGET" .idr )       # e.g. Mnist
		# kebab-case the module stem: Mnist → mnist, SeqClassify → seq-classify
		kebab=$( printf '%s' "$base" | sed 's/\([a-z0-9]\)\([A-Z]\)/\1-\2/g' | tr '[:upper:]' '[:lower:]' )
		UNIT="example-$kebab" ;;
	*)
		echo "unknown target: $TARGET (expected lib / transformers / gym / a .idr path)" >&2
		exit 2 ;;
esac
[ -n "$LABEL" ] && UNIT="$LABEL"

# Assemble the cold-typecheck command.
if [ -n "$PKGDIR" ]; then
	RUN=( env IDRIS2_PREFIX="$IDRIS2_LOCAL" idris2 --build-dir "$TMP" --typecheck "$IPKG" )
	RUNDIR="$PKGDIR"
else
	RUN=( idris2 --build-dir "$TMP" --source-dir "$EXAMPLE_SRC"
	      -p contrib -p linear -p idris-ml -p idris-gym -p idris-transformers
	      --check "$TARGET" )
	RUNDIR="."
fi

if [ "${PERF_ELAB_DRYRUN:-}" = "1" ]; then
	echo "[dry-run] unit=$UNIT backend=$BACKEND device=$DEVICE commit=$COMMIT"
	echo "[dry-run] (cd $RUNDIR && ${RUN[*]})"
	echo "[dry-run] would append kind=elab to ${PERF_ELAB_LOG:-$PERF_LOG_PATH}"
	exit 0
fi

echo "=== elab: $UNIT [$BACKEND/$DEVICE] @ $COMMIT (cold) ==="
T0=$( perf_now_ms )
set +e
( cd "$RUNDIR" && perf_quiet_run "${RUN[@]}" )
RC=$?
set -e
T1=$( perf_now_ms )

ELAB_MS=$(( T1 - T0 ))
ELAB_PRETTY=$( perf_pretty_elapsed_ms "$ELAB_MS" )

LOG_ARGS=()
[ -n "${PERF_ELAB_LOG:-}" ] && LOG_ARGS=( --log-path "$PERF_ELAB_LOG" )

python3 -m mltools.perf_log "${LOG_ARGS[@]}" append-elab \
	--unit "$UNIT" \
	--backend "$BACKEND" \
	--device "$DEVICE" \
	--commit "$COMMIT" \
	--elab-ms "$ELAB_MS" \
	--elab-human "$ELAB_PRETTY" \
	--exit-code "$RC"

echo "elab:    $ELAB_PRETTY (exit $RC)"
echo "Logged to ${PERF_ELAB_LOG:-$PERF_LOG_PATH}"
exit "$RC"
