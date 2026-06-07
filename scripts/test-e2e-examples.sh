#!/usr/bin/env bash
# Crash-only smoke gate: every example × backend lane, 3-10 epochs each,
# safety-net thresholds in test-examples.expect. Catches crashes / NaN /
# divergence / missing RESULT keys; does NOT require any model to learn.
# See docs/develop/testing.md for the full testing-layer overview.
#
# Invoked by `make test-e2e-examples` (mandatory `+` recipe prefix keeps
# the jobserver alive for the $MAKE sub-builds). Direct invocation works
# too — every input defaults below.
#
# Env interface (all passed by the Make recipe):
#   MAKE                  make binary for sub-builds (default: make)
#   EXAMPLES              space-separated example-* targets
#   BACKENDS              space-separated lanes (tape mlx mlx-gpu torch torch-mps)
#   EXAMPLE_TIMEOUT       per-example timeout in seconds
#   FAIL_FAST             non-empty = bail on first failure
#   PRECISION_DEMO_READY  "1" = run the example-precision-demo post-matrix step

set -u

MAKE=${MAKE:-make}
EXAMPLES=${EXAMPLES:-}
BACKENDS=${BACKENDS:-tape}
EXAMPLE_TIMEOUT=${EXAMPLE_TIMEOUT:-600}
FAIL_FAST=${FAIL_FAST:-}
PRECISION_DEMO_READY=${PRECISION_DEMO_READY:-1}

# The ARGS-var guard greps every file that can define example recipes.
# nullglob makes mk/*.mk vanish while the mk/ split hasn't landed yet.
shopt -s nullglob
makefiles=(Makefile mk/*.mk)

fail=0
skip=""
if command -v timeout >/dev/null 2>&1; then TIMEOUT_PREFIX="timeout $EXAMPLE_TIMEOUT"
elif command -v gtimeout >/dev/null 2>&1; then TIMEOUT_PREFIX="gtimeout $EXAMPLE_TIMEOUT"
else echo "WARNING: no timeout/gtimeout binary; examples will not be time-bounded"; TIMEOUT_PREFIX=""; fi

for lane in $BACKENDS; do
	case "$lane" in
		mlx-gpu)   b=mlx;   lane_env="MLX_DEVICE=gpu";   expect_suffix=.mlx-gpu ;;
		torch-mps) b=torch; lane_env="TORCH_DEVICE=mps"; expect_suffix=.torch-mps ;;
		*)         b=$lane; lane_env="";                 expect_suffix="" ;;
	esac
	backend_output=$(env $lane_env $MAKE --no-print-directory BACKEND=$b backend 2>&1) || {
		echo "--- backend $lane: build failed, skipping its examples ---"
		echo "$backend_output" | tail -20 | sed 's/^/  | /'
		skip="$skip $lane"; continue
	}
	for e in $EXAMPLES; do
		echo "--- $e [$lane] ---"
		extra_args=""
		smoke_args=""
		case "$e" in
			example-supervised)              smoke_args="--epochs 5" ;;
			example-rnn)                     smoke_args="--epochs 5" ;;
			example-lstm)                    smoke_args="--epochs 5" ;;
			example-gru)                     smoke_args="--epochs 5" ;;
			example-transformer)             smoke_args="--epochs 5" ;;
			example-reinforce)               smoke_args="--epochs 10" ;;
			example-gpt)                     smoke_args="--epochs 3" ;;
			example-matmul-bench)            smoke_args="--size 1024 --iters 3" ;;
			example-mnist)                   smoke_args="--epochs 1 --train-count 6000" ;;
			example-seq-classify)            smoke_args="--epochs 5" ;;
			example-dqn)                     smoke_args="--epochs 10" ;;
			example-mountain-car)            smoke_args="--epochs 5" ;;
			example-mountain-car-cont)       smoke_args="--epochs 5" ;;
			example-a2c)                     smoke_args="--epochs 50" ;;
			example-ppo)                     smoke_args="--epochs 5" ;;
			example-sac)                     smoke_args="--epochs 100" ;;
			example-ntm-copy)                smoke_args="--epochs 5" ;;
			example-ntm-associative-recall)  smoke_args="--epochs 5" ;;
			example-dnc-copy)                smoke_args="--epochs 5 --max-len 3 --batch 1" ;;
			example-dnc-recall)              smoke_args="--epochs 5 --max-items 2 --batch 1" ;;
		esac
		if [ -n "$smoke_args" ]; then
			args_var=$(echo "${e#example-}" | tr 'a-z-' 'A-Z_')_ARGS
			if ! grep -qF "\$(${args_var})" "${makefiles[@]}"; then
				echo "FAIL: $e: test-examples derived '$args_var' but no Makefile recipe consumes it."
				echo "  (Naming convention: example-foo-bar => FOO_BAR_ARGS. Likely cause: the var was renamed in the recipe without updating the test-examples case-arm, or vice versa.)"
				fail=1; [ -n "$FAIL_FAST" ] && exit 1; continue
			fi
			extra_args="$args_var=$smoke_args"
		fi
		t_start=$(date +%s)
		if [ -n "$extra_args" ]; then
			output=$(env $lane_env $TIMEOUT_PREFIX $MAKE --no-print-directory BACKEND=$b $e "$extra_args" 2>&1); rc=$?
		else
			output=$(env $lane_env $TIMEOUT_PREFIX $MAKE --no-print-directory BACKEND=$b $e 2>&1); rc=$?
		fi
		t_end=$(date +%s); elapsed=$((t_end - t_start))
		if [ $elapsed -lt 60 ]; then elapsed_fmt="${elapsed}s"
		elif [ $elapsed -lt 3600 ]; then elapsed_fmt="$((elapsed/60))m$((elapsed%60))s"
		else elapsed_fmt="$((elapsed/3600))h$(((elapsed%3600)/60))m"; fi
		if [ $rc -ne 0 ]; then
			if [ $rc -eq 124 ]; then
				echo "FAIL: $e [$lane] timed out (>${EXAMPLE_TIMEOUT}s) ($elapsed_fmt)"
			else
				echo "FAIL: $e [$lane] crashed (rc=$rc) ($elapsed_fmt)"
			fi
			echo "$output" | tail -40 | sed 's/^/  | /'
			fail=1; [ -n "$FAIL_FAST" ] && { echo "FAIL_FAST: bail on first failure ($e [$lane])"; exit 1; }; continue
		fi
		result_line=$(echo "$output" | grep '^RESULT' | head -1)
		if [ -z "$result_line" ]; then
			echo "FAIL: $e [$lane] -- no RESULT line ($elapsed_fmt)"
			echo "$output" | tail -40 | sed 's/^/  | /'
			fail=1; [ -n "$FAIL_FAST" ] && { echo "FAIL_FAST: bail on first failure ($e [$lane])"; exit 1; }
		else
			if [ -f "test-examples.expect$expect_suffix" ]; then
				scripts/check-result.sh "$e" "$result_line" "test-examples.expect$expect_suffix" || { fail=1; [ -n "$FAIL_FAST" ] && { echo "FAIL_FAST: bail ($e [$lane])"; exit 1; }; }
			else
				scripts/check-result.sh "$e" "$result_line" || { fail=1; [ -n "$FAIL_FAST" ] && { echo "FAIL_FAST: bail ($e [$lane])"; exit 1; }; }
			fi
			echo "  ($elapsed_fmt)"
		fi
	done
done

if [ -z "$skip" ]; then
	echo "--- example-checkpoint-demo (tape->mlx->torch round-trip) ---"
	t_start=$(date +%s)
	demo_out=$($TIMEOUT_PREFIX $MAKE --no-print-directory example-checkpoint-demo 2>&1); demo_rc=$?
	t_end=$(date +%s); elapsed=$((t_end - t_start))
	if [ $elapsed -lt 60 ]; then elapsed_fmt="${elapsed}s"
	elif [ $elapsed -lt 3600 ]; then elapsed_fmt="$((elapsed/60))m$((elapsed%60))s"
	else elapsed_fmt="$((elapsed/3600))h$(((elapsed%3600)/60))m"; fi
	if [ $demo_rc -ne 0 ]; then
		if [ $demo_rc -eq 124 ]; then echo "FAIL: example-checkpoint-demo timed out (>${EXAMPLE_TIMEOUT}s) ($elapsed_fmt)"
		else echo "FAIL: example-checkpoint-demo crashed (rc=$demo_rc) ($elapsed_fmt)"; fi
		echo "$demo_out" | tail -40 | sed 's/^/  | /'
		fail=1
	else
		result_line=$(echo "$demo_out" | grep '^RESULT' | tail -1)
		if [ -z "$result_line" ]; then
			echo "FAIL: example-checkpoint-demo -- no RESULT line ($elapsed_fmt)"
			echo "$demo_out" | tail -40 | sed 's/^/  | /'
			fail=1
		else
			scripts/check-result.sh "example-checkpoint-demo" "$result_line" || fail=1
			echo "  ($elapsed_fmt)"
		fi
	fi
else
	echo "--- example-checkpoint-demo: skipped (requires tape+mlx+torch; skipped:$skip) ---"
fi

if [ "$PRECISION_DEMO_READY" = "1" ] && [ -z "$skip" ]; then
	echo "--- example-precision-demo (F32/F64 cast + cross-backend hop) ---"
	t_start=$(date +%s)
	pdemo_out=$($TIMEOUT_PREFIX $MAKE --no-print-directory example-precision-demo 2>&1); pdemo_rc=$?
	t_end=$(date +%s); elapsed=$((t_end - t_start))
	if [ $elapsed -lt 60 ]; then elapsed_fmt="${elapsed}s"
	elif [ $elapsed -lt 3600 ]; then elapsed_fmt="$((elapsed/60))m$((elapsed%60))s"
	else elapsed_fmt="$((elapsed/3600))h$(((elapsed%3600)/60))m"; fi
	if [ $pdemo_rc -ne 0 ]; then
		if [ $pdemo_rc -eq 124 ]; then echo "FAIL: example-precision-demo timed out (>${EXAMPLE_TIMEOUT}s) ($elapsed_fmt)"
		else echo "FAIL: example-precision-demo crashed (rc=$pdemo_rc) ($elapsed_fmt)"; fi
		echo "$pdemo_out" | tail -40 | sed 's/^/  | /'
		fail=1
	else
		result_line=$(echo "$pdemo_out" | grep '^RESULT' | tail -1)
		if [ -z "$result_line" ]; then
			echo "FAIL: example-precision-demo -- no RESULT line ($elapsed_fmt)"
			echo "$pdemo_out" | tail -40 | sed 's/^/  | /'
			fail=1
		else
			scripts/check-result.sh "example-precision-demo" "$result_line" || fail=1
			echo "  ($elapsed_fmt)"
		fi
	fi
elif [ "$PRECISION_DEMO_READY" != "1" ]; then
	echo "--- example-precision-demo: skipped (PRECISION_DEMO_READY=0; example not yet landed) ---"
else
	echo "--- example-precision-demo: skipped (requires tape+mlx+torch; skipped:$skip) ---"
fi

if [ -n "$skip" ]; then echo "Skipped backends (not installed or build failed):$skip"; fi
if [ $fail -ne 0 ]; then echo "Some integration tests FAILED"; exit 1; fi
echo "All integration tests passed."
