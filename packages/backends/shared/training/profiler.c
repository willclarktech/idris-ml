/* shared/training/profiler.c — backend-agnostic wall-clock provider.
 *
 * One small function: `_wall_ms()` returning a monotonic-ish millisecond
 * count from gettimeofday. Used by the shared optimizer for its
 * per-step timing and by every backend's per-op accumulators via
 * extern decls.
 *
 * Why a separate TU rather than inline in port.h? Compilation-firewall
 * cost: gettimeofday + sys/time.h on every site bloats the parse cost
 * for files that just want the symbol. One non-inline definition is
 * the cheapest split.
 *
 * Compiled once per backend in TRAINING_ADAPTER_BACKENDS so the
 * unsuffixed name resolves to the right backend's instance at link
 * time. Tape's other profiling state (per-OP timing arrays,
 * backend_profile_report's tape-specific output format, the op_name
 * table) stays in backend_tape/training/profiling.c — none of that
 * lifts cleanly because OP_COUNT and the report format are
 * tape-specific.
 */

#include <sys/time.h>

double _wall_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}
