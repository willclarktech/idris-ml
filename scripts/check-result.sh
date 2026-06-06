#!/usr/bin/env bash
# Validate a test-examples RESULT line against the threshold for <target>
# in <expect_file> (defaults to test-examples.expect).
#
# Usage: check-result.sh <target> <result_line> [<expect_file>]
# Exit 0: passed (or no threshold defined; presence-only)
# Exit 1: failed threshold, malformed expect, or missing key in RESULT

set -u
target=$1
result_line=$2
expect_file=${3:-$(dirname "$0")/../test-examples.expect}
if [ ! -f "$expect_file" ]; then
		echo "ERROR: $expect_file missing" >&2
		exit 1
fi

expect=$(awk -v t="$target" '$1==t {print $2, $3, $4; exit}' "$expect_file")
if [ -z "$expect" ]; then
		echo "ok: $result_line"
		exit 0
fi

key=$(echo "$expect" | awk '{print $1}')
op=$(echo "$expect" | awk '{print $2}')
thr=$(echo "$expect" | awk '{print $3}')

val=$(echo "$result_line" | tr '\t' '\n' | grep "^$key=" | head -1 | cut -d= -f2-)
case "$val" in
		*/*) val=$(echo "$val" | awk -F/ 'BEGIN{OFMT="%.6f"} {print $1/$2}') ;;
esac

if [ -z "$val" ]; then
		echo "FAIL: $target -- key '$key' not in RESULT"
		echo "  | $result_line"
		exit 1
fi

if awk -v a="$val" -v thr="$thr" "BEGIN { exit !(a $op thr) }"; then
		echo "ok: $result_line  [$key=$val $op $thr]"
		exit 0
else
		echo "FAIL: $target -- $key=$val fails check ($op $thr)"
		echo "  | $result_line"
		exit 1
fi
