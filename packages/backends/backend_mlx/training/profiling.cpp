/* Profiling counters + report helper for the mlx backend.
 *
 * Counters accumulate across tensor_backward / optimizer_step;
 * backend_profile_reset zeroes them and backend_profile_report
 * prints the summary. The tape-append counter is mirrored here
 * (declared via tape.h, defined in backend_mlx.cpp). */
#include <cstdio>
#include <sys/time.h>
#include "../tensor.h"
#include "../tape.h" /* prof_tape_appends_mlx */
#include "profiling.h"

double prof_backward_ms_mlx = 0;
double prof_optimizer_ms_mlx = 0;
double prof_optimizer_math_ms_mlx = 0;
int prof_epochs_mlx = 0;

double _wall_ms_mlx(void) {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

extern "C" int param_count(void);

extern "C" void backend_epoch_begin(void) {
	/* No-op: profiling is backward + optimizer only. */
}

extern "C" void backend_profile_reset(void) {
	prof_backward_ms_mlx = prof_optimizer_ms_mlx = 0;
	prof_optimizer_math_ms_mlx = 0;
	prof_epochs_mlx = 0;
	prof_tape_appends_mlx = 0;
}

/* TODO #393 op-submission counter stubs — diagnostic surface
 * implemented on torch (counts at::Tensor wraps in from_tensor); mlx
 * already batches via lazy mx::array graphs (the analog metric is
 * prof_tape_appends_mlx above), so this counter always reads 0 and
 * reset is a no-op. */
extern "C" void tensor_perf_reset(void) {
}
extern "C" long tensor_perf_op_count(void) {
	return 0;
}

extern "C" void backend_profile_report(void) {
	fprintf(stderr, "=== Profile Report (MLX backend) ===\n");
	fprintf(stderr, "  Epochs: %d\n", prof_epochs_mlx);
	fprintf(stderr, "  Params: %d tensors\n", param_count());
	fprintf(stderr, "  Backward:  %.1fms total (%.1fms/epoch)\n", prof_backward_ms_mlx,
	        prof_epochs_mlx > 0 ? prof_backward_ms_mlx / prof_epochs_mlx : 0);
	fprintf(stderr, "  Optimizer: %.1fms total (%.1fms/epoch)\n", prof_optimizer_ms_mlx,
	        prof_epochs_mlx > 0 ? prof_optimizer_ms_mlx / prof_epochs_mlx : 0);
	fprintf(stderr, "    of which math: %.1fms total (%.1fms/epoch)\n", prof_optimizer_math_ms_mlx,
	        prof_epochs_mlx > 0 ? prof_optimizer_math_ms_mlx / prof_epochs_mlx : 0);
	double total = prof_backward_ms_mlx + prof_optimizer_ms_mlx;
	fprintf(stderr, "  C total:   %.1fms total (%.1fms/epoch)\n", total,
	        prof_epochs_mlx > 0 ? total / prof_epochs_mlx : 0);
	fprintf(stderr, "  Forward tape_appends (grad-tracked ops): %ld total (%.0f/epoch)\n",
	        prof_tape_appends_mlx,
	        prof_epochs_mlx > 0 ? (double)prof_tape_appends_mlx / prof_epochs_mlx : 0);
}
