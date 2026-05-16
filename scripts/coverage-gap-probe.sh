#!/usr/bin/env bash
# coverage-gap-probe.sh — emit per-OP and per-FFI-symbol test coverage CSVs
# for the tape + mlx backends.
#
# Usage: scripts/coverage-gap-probe.sh [OUTDIR]
#
# Algorithm:
#   1. Parse OP_* enum entries from backend_<b>/tape.h (skip OP_COUNT and
#      OP_CONST — the latter is a leaf marker with no backward).
#   2. For each OP_FOO, find the source file containing the canonical
#      registration anchor:
#        - tape: `TAPE_REGISTER_OP(OP_FOO, ...)`
#        - mlx:  `MLX_REGISTER_REPLAY(OP_FOO, ...)`
#      That source file holds the forward + backward for this OP.
#   3. Extract the non-static FFI symbols defined in that source file
#      (return type + name, excluding `static` defs). These are the
#      forward entry points users call.
#   4. Grep test/ for ANY of those FFI symbols. If at least one appears
#      in a test file, mark OP_FOO as covered. (Test files like
#      `test_unary.c` cover multiple OPs in one file — symbol-based
#      detection handles that correctly.)
#   5. Parse top-level decls from backend.h, apply the documented
#      exclusion list (diagnostics, mnist, refcount glue), grep test/
#      for each remaining symbol, count file hits.
#   6. Emit two CSVs and print a per-backend summary.
#
# Exit code is advisory (always 0) until W3+W4 close the OP_* gaps; then
# flip to non-zero on any MISSING.

set -uo pipefail
# NOTE: deliberately NOT using `set -e` — `grep | head -1` returns
# non-zero under pipefail when no match, which is a normal "missing"
# signal, not a script error. We check explicitly where we care.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BACKENDS="$REPO_ROOT/packages/backends"
TEST_DIR="$BACKENDS/test"
OUTDIR="${1:-$REPO_ROOT/build}"
mkdir -p "$OUTDIR"

# 1. Extract OP_* tags from a tape.h header.
extract_ops() {
    sed -n '/^enum {/,/^};/p' "$1" | \
        grep -oE 'OP_[A-Z][A-Z0-9_]*' | \
        grep -vE '^(OP_COUNT|OP_CONST)$' | \
        sort -u
}

# 2. Find the source file holding the registration anchor for OP_FOO.
find_source() {
    local op="$1"
    local backend_dir="$2"
    local anchor="$3"  # TAPE_REGISTER_OP or MLX_REGISTER_REPLAY
    grep -rln --include='*.c' --include='*.cpp' \
        "${anchor}(${op}," "$backend_dir" 2>/dev/null | head -1 || true
}

# 3. From a source file, extract non-static function names that are
#    plausibly FFI entry points (return type then name then `(`).
extract_ffi_symbols() {
    local source="$1"
    [ -z "$source" ] && return
    # Match lines beginning with a return type and a non-static fn name.
    # Common return types in the backends: TensorHandle, void, int,
    # double, float, char*, TensorHandle*.
    grep -hE '^(extern "C"[[:space:]]+)?(TensorHandle\*?|void\*?|int|double|float|char\*?)[[:space:]]+[a-z_][a-z0-9_]*[[:space:]]*\(' "$source" 2>/dev/null | \
        grep -v '^static' | \
        sed -E 's/^(extern "C"[[:space:]]+)?//' | \
        sed -E 's/^[A-Za-z_]+\*?[[:space:]]+([a-z_][a-z0-9_]*)[[:space:]]*\(.*/\1/' | \
        sort -u
}

# 4. Check if any of the FFI symbols for OP_FOO appears in a test file.
#    Returns the first matching test file path, or empty.
find_test_by_symbols() {
    local symbols="$1"
    [ -z "$symbols" ] && return
    for sym in $symbols; do
        # Word-boundary grep, .c only (test files are all .c).
        # Use -w for whole-word match so `tensor_add` doesn't match
        # `tensor_add_scalar` etc.
        local hit
        hit=$(grep -rlw --include='*.c' "$sym" "$TEST_DIR" 2>/dev/null | head -1 || true)
        if [ -n "$hit" ]; then
            echo "$hit"
            return
        fi
    done
}

# 5. Emit OPs gap CSV.
ops_csv="$OUTDIR/coverage-gap-ops.csv"
{
    echo "backend,op,source_file,ffi_symbols,test_file,status"
    for backend in tape mlx; do
        header="$BACKENDS/backend_${backend}/tape.h"
        backend_dir="$BACKENDS/backend_${backend}"
        [ -f "$header" ] || continue
        if [ "$backend" = "tape" ]; then
            anchor="TAPE_REGISTER_OP"
        else
            anchor="MLX_REGISTER_REPLAY"
        fi
        for op in $(extract_ops "$header"); do
            src="$(find_source "$op" "$backend_dir" "$anchor")"
            symbols="$(extract_ffi_symbols "$src" | tr '\n' ' ')"
            symbols_trim="$(echo "$symbols" | sed 's/[[:space:]]*$//')"
            test="$(find_test_by_symbols "$symbols_trim")"
            src_rel="${src#$BACKENDS/}"
            test_rel="${test#$BACKENDS/}"
            if [ -z "$src_rel" ]; then
                src_rel="NO_REGISTRATION"
            fi
            if [ -z "$test_rel" ]; then
                status="MISSING"
                test_rel="MISSING"
            else
                status="present"
            fi
            # Pipe-separate symbols for CSV embed (commas would break columns).
            symbols_csv_safe="$(echo "$symbols_trim" | tr ' ' '|')"
            echo "$backend,$op,$src_rel,$symbols_csv_safe,$test_rel,$status"
        done
    done
} > "$ops_csv"

# 6. Emit FFI symbols CSV.
symbols_csv="$OUTDIR/coverage-gap-symbols.csv"
{
    echo "symbol,test_hits"
    grep -hE '^[A-Za-z_][A-Za-z0-9_ *]*\*?[ \t]+[a-z_][a-z0-9_]+[ \t]*\(' "$BACKENDS/backend.h" 2>/dev/null | \
        sed -E 's/.*[ *]([a-z_][a-z0-9_]+)[ \t]*\(.*/\1/' | \
        sort -u | while read -r sym; do
        [ -z "$sym" ] && continue
        # Apply the documented exclusion list (see coverage-policy.md, W7).
        case "$sym" in
            tensor_print|tensor_live_count|tensor_peak_live_count) continue ;;
            backend_profile_reset|backend_profile_report|backend_reset_for_eval|backend_epoch_begin|backend_name) continue ;;
            backend_profile_reset_return|backend_profile_report_return|backend_reset_for_eval_return) continue ;;
            get_rss_mb|get_current_rss_mb) continue ;;
            mnist_load|mnist_count|mnist_get_image|mnist_get_label|mnist_free) continue ;;
            tensor_retain_handle|tensor_release_handle) continue ;;
            idrisml_seq) continue ;;
            tensor_mlx_compile_enabled|tensor_mlx_compile_invocations) continue ;;
        esac
        hits=$(grep -rlw --include='*.c' "$sym" "$TEST_DIR" 2>/dev/null | wc -l | tr -d ' ')
        echo "$sym,$hits"
    done
} > "$symbols_csv"

# 7. Summary + advisory exit.
gap_count=$(awk -F, 'NR>1 && $6=="MISSING"' "$ops_csv" | wc -l | tr -d ' ')
tape_gaps=$(awk -F, 'NR>1 && $6=="MISSING" && $1=="tape"' "$ops_csv" | wc -l | tr -d ' ')
mlx_gaps=$(awk -F, 'NR>1 && $6=="MISSING" && $1=="mlx"' "$ops_csv" | wc -l | tr -d ' ')
symbols_zero=$(awk -F, 'NR>1 && $2=="0"' "$symbols_csv" | wc -l | tr -d ' ')

echo ""
echo "=== Coverage gap probe ==="
echo "OP_* without any FFI test hit: $gap_count  (tape=$tape_gaps, mlx=$mlx_gaps)"
echo "FFI symbols with 0 test hits:  $symbols_zero"
echo ""
echo "Reports:"
echo "  $ops_csv"
echo "  $symbols_csv"

if [ "$gap_count" -gt 0 ]; then
    echo ""
    echo "Missing OP_* tests (sample):"
    awk -F, 'NR>1 && $6=="MISSING" {print "  " $1 "  " $2 "  (source: " $3 ", symbols: " $4 ")"}' "$ops_csv" | head -20
    if [ "$gap_count" -gt 20 ]; then
        echo "  ... and $((gap_count - 20)) more — see CSV"
    fi
fi

if [ "$symbols_zero" -gt 0 ]; then
    echo ""
    echo "FFI symbols with 0 test hits (sample):"
    awk -F, 'NR>1 && $2=="0" {print "  " $1}' "$symbols_csv" | head -15
    if [ "$symbols_zero" -gt 15 ]; then
        echo "  ... and $((symbols_zero - 15)) more — see CSV"
    fi
fi

# Advisory only: returns 0 even on gaps. CI gate flips to non-zero exit
# after W3+W4 close (TODO: see docs/develop/coverage-policy.md).
exit 0
