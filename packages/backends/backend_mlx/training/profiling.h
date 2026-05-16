/* Profiling counters for the mlx backend's training surface.
 *
 * prof_backward_ms_mlx / prof_optimizer_ms_mlx accumulate wall-clock
 * across tensor_backward / optimizer_step calls; prof_epochs_mlx
 * counts completed epoch reports; prof_tape_appends_mlx counts
 * grad-tracked forward ops (already declared via tape.h since
 * tape_append references it directly).
 *
 * The `_mlx` suffix dodges a tri-link collision with the same-named
 * counters in tape's modular tree (which can't be static-scoped).
 *
 * `_wall_ms_mlx()` is the shared gettimeofday helper. */
#ifndef IDRISML_BACKEND_MLX_PROFILING_H
#define IDRISML_BACKEND_MLX_PROFILING_H

extern double prof_backward_ms_mlx;
extern double prof_optimizer_ms_mlx;
extern double prof_optimizer_math_ms_mlx;
extern int    prof_epochs_mlx;
/* prof_tape_appends_mlx is declared by backend_mlx/tape.h. */

double _wall_ms_mlx(void);

#endif /* IDRISML_BACKEND_MLX_PROFILING_H */
