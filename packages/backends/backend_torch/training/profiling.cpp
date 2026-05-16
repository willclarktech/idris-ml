/* Profiling counters + report helper for the torch backend.
 *
 * The counters accumulate across tensor_backward / optimizer_step calls;
 * `backend_profile_reset` zeroes them and `backend_profile_report` prints
 * the summary to stderr. `_wall_ms_torch` is the gettimeofday-backed
 * timestamp helper used by every site that increments these. */
#include <cstdio>
#include <sys/time.h>
#include "../tensor.h"
#include "profiling.h"

double prof_backward_ms_torch = 0;
double prof_optimizer_ms_torch = 0;
double prof_optimizer_math_ms_torch = 0;
int    prof_epochs_torch = 0;

double _wall_ms_torch(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/* param_count() is exposed by shared/training/param_registry. */
extern "C" int param_count(void);

extern "C" void backend_epoch_begin(void) {
    /* No-op for torch: profiling is backward + optimizer only. */
}

extern "C" void backend_profile_reset(void) {
    prof_backward_ms_torch = prof_optimizer_ms_torch = prof_optimizer_math_ms_torch = 0;
    prof_epochs_torch = 0;
}

extern "C" void backend_profile_report(void) {
    fprintf(stderr, "=== Profile Report (torch backend) ===\n");
    fprintf(stderr, "  Epochs: %d\n", prof_epochs_torch);
    fprintf(stderr, "  Params: %d tensors\n", (int)param_count());
    fprintf(stderr, "  Backward:  %.1fms total (%.1fms/epoch)\n",
            prof_backward_ms_torch, prof_epochs_torch > 0 ? prof_backward_ms_torch / prof_epochs_torch : 0);
    fprintf(stderr, "  Optimizer: %.1fms total (%.1fms/epoch)\n",
            prof_optimizer_ms_torch, prof_epochs_torch > 0 ? prof_optimizer_ms_torch / prof_epochs_torch : 0);
    fprintf(stderr, "    of which math: %.1fms total (%.2fms/epoch)\n",
            prof_optimizer_math_ms_torch, prof_epochs_torch > 0 ? prof_optimizer_math_ms_torch / prof_epochs_torch : 0);
    double total = prof_backward_ms_torch + prof_optimizer_ms_torch;
    fprintf(stderr, "  C total:   %.1fms total (%.1fms/epoch)\n",
            total, prof_epochs_torch > 0 ? total / prof_epochs_torch : 0);
}
