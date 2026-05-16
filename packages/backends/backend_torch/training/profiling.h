/* Profiling counters for the torch backend's training surface.
 *
 * `prof_backward_ms_torch` / `prof_optimizer_ms_torch` accumulate
 * wall-clock time across tensor_backward / optimizer_step calls.
 * `prof_epochs_torch` counts completed epoch reports. They live in
 * profiling.cpp; per-op TUs increment them via these externs.
 *
 * The `_torch` suffix sidesteps a tri-link collision with the tape
 * backend's same-named counters (which are exposed across tape's own
 * TU boundary, so it can't be static-scoped).
 *
 * `_wall_ms_torch()` is the shared timestamp helper. */
#ifndef IDRISML_BACKEND_TORCH_PROFILING_H
#define IDRISML_BACKEND_TORCH_PROFILING_H

extern double prof_backward_ms_torch;
extern double prof_optimizer_ms_torch;
extern double prof_optimizer_math_ms_torch;
extern int    prof_epochs_torch;

double _wall_ms_torch(void);

#endif /* IDRISML_BACKEND_TORCH_PROFILING_H */
