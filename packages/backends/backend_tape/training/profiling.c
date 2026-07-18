/* training/profiling.c — wall-time accounting and per-op timing buckets.
 *
 * _wall_ms (gettimeofday-based monotonic-ish), all the
 * prof_* globals (forward/backward/optimizer wall time, per-op buckets
 * for forward & backward, kernel-direct buckets, binop-path classifiers)
 * plus the public profile entry points:
 *   backend_epoch_begin / backend_profile_reset / backend_profile_report.
 *
 * op_name[] is the human-readable name table indexed by OP_COUNT —
 * compile-time _Static_assert keeps it in sync with the enum.
 */

#include <stdio.h>
#include <string.h>
#include "../tape.h"
#include "../tensor.h"
#include "../../backend.h"

/* _wall_ms lives in shared/training/profiler.c — same gettimeofday-
   based monotonic-ish reading every backend uses. The extern decl
   keeps existing usages here unchanged. */
extern double _wall_ms(void);

double prof_forward_ms = 0, prof_backward_ms = 0, prof_optimizer_ms = 0;
int prof_forward_ops = 0, prof_backward_ops = 0, prof_epochs = 0;
double prof_epoch_start = 0;
int prof_backward_processed = 0, prof_backward_skipped = 0;
double prof_backward_per_op[OP_COUNT] = {0};
int prof_backward_count_per_op[OP_COUNT] = {0};
double prof_forward_per_op[OP_COUNT] = {0};
int prof_forward_count_per_op[OP_COUNT] = {0};
double prof_kernel_per_op[OP_COUNT] = {0};
int prof_kernel_count_per_op[OP_COUNT] = {0};
int prof_binop_path_count[3] = {0};
double prof_binop_general_ms = 0;
double prof_binop_inside_ms[OP_COUNT] = {0};
int prof_binop_inside_count[OP_COUNT] = {0};
double prof_op_t_prev = 0;

void backend_epoch_begin(void) {
	double t = _wall_ms();
	prof_epoch_start = t;
	prof_op_t_prev = t;
}

void backend_profile_reset(void) {
	prof_forward_ms = prof_backward_ms = prof_optimizer_ms = 0;
	prof_forward_ops = prof_backward_ops = prof_epochs = 0;
	prof_epoch_start = 0;
	prof_op_t_prev = 0;
	prof_backward_processed = prof_backward_skipped = 0;
	memset(prof_backward_per_op, 0, sizeof(prof_backward_per_op));
	memset(prof_backward_count_per_op, 0, sizeof(prof_backward_count_per_op));
	memset(prof_forward_per_op, 0, sizeof(prof_forward_per_op));
	memset(prof_forward_count_per_op, 0, sizeof(prof_forward_count_per_op));
	memset(prof_kernel_per_op, 0, sizeof(prof_kernel_per_op));
	memset(prof_kernel_count_per_op, 0, sizeof(prof_kernel_count_per_op));
	memset(prof_binop_inside_ms, 0, sizeof(prof_binop_inside_ms));
	memset(prof_binop_inside_count, 0, sizeof(prof_binop_inside_count));
	memset(prof_binop_path_count, 0, sizeof(prof_binop_path_count));
	prof_binop_general_ms = 0;
}

/* TODO #393 op-submission counter stubs — diagnostic surface
 * implemented on torch (counts at::Tensor wraps in from_tensor); tape
 * doesn't have the same per-op submission model, so the counter
 * always reads 0 and reset is a no-op. */
void tensor_perf_reset(void) {
}
long tensor_perf_op_count(void) {
	return 0;
}

static const char* op_name(int op) {
	static const char* names[] = {
	    "CONST",       "ADD",         "SUB",      "MUL",         "DIV",         "NEG",
	    "ABS",         "EXP",         "LOG",      "SQRT",        "POW",         "SIGMOID",
	    "TANH",        "MV",          "LINEAR",   "DOT",         "OUTER",       "SOFTMAX",
	    "LOG_SOFTMAX", "SUM",         "MEAN",     "BCE_LOGITS",  "LSTM_GATES",  "ADD_S",
	    "MUL_S",       "CLAMP",       "COS_SIM",  "CONV1D_CIRC", "LSTM_CELL",   "STACK",
	    "RESHAPE",     "SELECT",      "VECMAT",   "CAT",         "NARROW",      "LOG_SM_2D",
	    "MM",          "TRANS_2D",    "SM_2D",    "MASK_FILL",   "LN_2D",       "BMM",
	    "BMM_3X3",     "SM_3D",       "TRANS_L2", "GELU",        "GRU",         "EMBED",
	    "BATCH_NORM",  "DROPOUT",     "AVGP1D",   "AVGP2D",      "CONV1D",      "MAXP1D",
	    "CONV2D",      "CONV2D_B",    "MAXP2D",   "MAXP2D_B",    "CUMPROD",     "GATHER",
	    "SCATTER_ADD", "GATHER_ROWS", "MAX_ROWS", "LEAKY_RELU",  "SILU",        "LINEAR_2D",
	    "CONCAT_2D",   "SOFTPLUS",    "TILE_2D",  "CAST_DTYPE",  "RMS_NORM_2D", "SWIGLU_2D",
	    "SFTMX_XENT"};
	/* Compile-time check: names[] must cover every op tag.
	   Add to BOTH this list and the enum when introducing new ops. */
	_Static_assert(sizeof(names) / sizeof(names[0]) == OP_COUNT,
	               "op_name names[] out of sync with OP_COUNT — add new ops here");
	if (op >= 0 && op < OP_COUNT) return names[op];
	return "???";
}

void backend_profile_report(void) {
	fprintf(stderr, "=== Profile Report ===\n");
	fprintf(stderr, "  Epochs: %d\n", prof_epochs);
	fprintf(stderr, "  Tape entries (last fwd): %d\n", tape_size);
	fprintf(stderr, "  Params: %d tensors, %d elements\n", param_count(), ({
		        int n = 0;
		        for (int i = 0; i < param_count(); i++)
			        n += ((Tensor*)param_tensor(i))->numel;
		        // Value of the GCC statement-expression `({ ... })`, not a useless statement.
		        // cppcheck-suppress constStatement
		        n;
	        }));
	fprintf(stderr, "  Forward:   %.1fms total (%.1fms/epoch)\n", prof_forward_ms,
	        prof_epochs > 0 ? prof_forward_ms / prof_epochs : 0);
	fprintf(stderr, "  Backward:  %.1fms total (%.1fms/epoch)\n", prof_backward_ms,
	        prof_epochs > 0 ? prof_backward_ms / prof_epochs : 0);
	fprintf(stderr, "  Optimizer: %.1fms total (%.1fms/epoch)\n", prof_optimizer_ms,
	        prof_epochs > 0 ? prof_optimizer_ms / prof_epochs : 0);
	double total = prof_forward_ms + prof_backward_ms + prof_optimizer_ms;
	fprintf(stderr, "  C total:   %.1fms total (%.1fms/epoch)\n", total,
	        prof_epochs > 0 ? total / prof_epochs : 0);
	/* Tape walk stats */
	int total_visited = prof_backward_processed + prof_backward_skipped;
	if (total_visited > 0) {
		fprintf(stderr, "  Backward walk: %d processed, %d skipped (%.0f%% dead)\n",
		        prof_backward_processed, prof_backward_skipped,
		        100.0 * prof_backward_skipped / total_visited);
	}
	/* Top-5 ops by backward time */
	fprintf(stderr, "  Top backward ops:\n");
	for (int rank = 0; rank < 5; rank++) {
		int best = -1;
		double best_time = 0;
		for (int j = 0; j < OP_COUNT; j++) {
			if (prof_backward_per_op[j] > best_time) {
				/* Skip already printed */
				int already = 0;
				for (int k = 0; k < rank; k++) {
					/* Find k-th best again to skip it */
					double kt = 0;
					int ki = -1;
					for (int m = 0; m < OP_COUNT; m++) {
						if (prof_backward_per_op[m] > kt) {
							kt = prof_backward_per_op[m];
							ki = m;
						}
					}
					/* This naive approach doesn't work for rank>0. Use simpler method. */
					(void)kt;
					(void)ki;
				}
				(void)already;
				best = j;
				best_time = prof_backward_per_op[j];
			}
		}
		if (best < 0 || best_time < 0.001) break;
		fprintf(stderr, "    %-12s %.2fms (%d calls)\n", op_name(best), best_time,
		        prof_backward_count_per_op[best]);
		prof_backward_per_op[best] = -1; /* mark as printed (will be reset on next profile_reset) */
	}
	/* Top-10 ops by forward time (broader than backward — more ops contribute) */
	fprintf(stderr, "  Top forward ops:\n");
	for (int rank = 0; rank < 10; rank++) {
		int best = -1;
		double best_time = 0;
		for (int j = 0; j < OP_COUNT; j++) {
			if (prof_forward_per_op[j] > best_time) {
				best = j;
				best_time = prof_forward_per_op[j];
			}
		}
		if (best < 0 || best_time < 0.001) break;
		int n = prof_forward_count_per_op[best];
		double per_call_us = n > 0 ? (best_time * 1000.0 / n) : 0.0;
		fprintf(stderr, "    %-12s %.2fms (%d calls, %.2f us/call)\n", op_name(best), best_time, n,
		        per_call_us);
		prof_forward_per_op[best] = -1; /* mark as printed */
	}
	/* Kernel-only timer (elementwise vDSP path only, today). Shows the
	   actual kernel time independent of the tape_append attribution. */
	int any_kernel = 0;
	for (int j = 0; j < OP_COUNT; j++) {
		if (prof_kernel_count_per_op[j] > 0) {
			any_kernel = 1;
			break;
		}
	}
	if (any_kernel) {
		fprintf(stderr, "  Direct-kernel timing (subset of ops):\n");
		for (int j = 0; j < OP_COUNT; j++) {
			int n = prof_kernel_count_per_op[j];
			if (n == 0) continue;
			double per_call_us = prof_kernel_per_op[j] * 1000.0 / n;
			fprintf(stderr, "    %-12s %.2fms (%d calls, %.2f us/call) [kernel only]\n", op_name(j),
			        prof_kernel_per_op[j], n, per_call_us);
		}
	}
	fprintf(stderr,
	        "  binop_elementwise paths: fast=%d scalar_bcast=%d general_bcast=%d  "
	        "general_bcast_total=%.2fms\n",
	        prof_binop_path_count[0], prof_binop_path_count[1], prof_binop_path_count[2],
	        prof_binop_general_ms);
	int any_inside = 0;
	for (int j = 0; j < OP_COUNT; j++) {
		if (prof_binop_inside_count[j] > 0) {
			any_inside = 1;
			break;
		}
	}
	if (any_inside) {
		fprintf(stderr, "  binop_elementwise inside (entry-to-exit):\n");
		for (int j = 0; j < OP_COUNT; j++) {
			int n = prof_binop_inside_count[j];
			if (n == 0) continue;
			double per_call_us = prof_binop_inside_ms[j] * 1000.0 / n;
			fprintf(stderr, "    %-12s %.2fms (%d calls, %.2f us/call) [in-function]\n", op_name(j),
			        prof_binop_inside_ms[j], n, per_call_us);
		}
	}
}
