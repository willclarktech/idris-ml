/* test_helpers.h — backend-aware tolerance + readout helpers for the
 * common Criterion suite.
 *
 * The suite was originally authored against tape (F64-flat storage,
 * 1e-12 tolerance everywhere). Running the same suite against torch
 * and mlx requires two adaptations:
 *
 *   - Tolerance: mlx's default storage is F32 (the framework is F32-
 *     pervasive; the F64 lingua-franca path narrows to F32 at the
 *     boundary). Single-precision elementwise error swamps a 1e-12
 *     bar — `0.1 + 0.2 + 0.3` lands ~1e-7 off `0.6`. We provide
 *     `TEST_TOL_TIGHT` (F64-exact backends) and `TEST_TOL_RELAXED`
 *     (F32-storage backends) and pick at compile time via the
 *     `-DBACKEND_<NAME>` flag the Makefile already passes.
 *
 *   - Flat-buffer readouts: tape's `tensor_item_1d` reads a flat data
 *     buffer at index `idx`. Torch's wraps `at::Tensor::flatten()`
 *     after this commit, so both match mlx (which uses
 *     `mx_read_double` on flattened storage). All three backends
 *     now treat `tensor_item_1d` as "flat[idx]", but the
 *     `read_flat(t, buf, n)` helper here uses `tensor_to_doubles`,
 *     which is the most explicit "give me the entire flat buffer"
 *     path and the one we recommend for new tests.
 */

#ifndef BACKENDS_TEST_HELPERS_H
#define BACKENDS_TEST_HELPERS_H

#include <stdlib.h>
#include <string.h>
#include "backend.h"

/* Heap-copy `n` doubles. The shaped tensor creators — tensor_create_{1,2}d_*,
   tensor_create_param_*, tensor_create_state_*, tensor_create_*_streamed — OWN
   and free() their `data` argument (they own the marshalling buffer; see
   Handle.idr:462 / lifecycle_ext.c:25). Only the base `tensor_create` copies.
   So route every shaped-creator call's data through hcopy (a fresh heap copy);
   passing a STACK array frees stack memory -> heap corruption (ASan: bad-free).
   For the common 2D case, prefer mk2d() below, which hides the copy entirely. */
static inline double* hcopy(const double* src, int n) {
	double* b = (double*)malloc((size_t)n * sizeof(double));
	memcpy(b, src, (size_t)n * sizeof(double));
	return b;
}

/* Stack-safe 2D constructor: heap-copies `src` then hands ownership to
   tensor_create_2d (which frees it). Use instead of
   tensor_create_2d(rows, cols, <stack array>, rg). */
static inline TensorHandle mk2d(int rows, int cols, const double* src, int rg) {
	return tensor_create_2d(rows, cols, hcopy(src, rows * cols), rg);
}

#if defined(BACKEND_MLX)
/* mlx defaults to F32 storage; tolerance bar set to where summed
   single-precision error reliably lives. Tightened later if a test
   truly is F32-tight (e.g. trivial roundtrips). */
#define TEST_TOL_TIGHT 1e-5
#define TEST_TOL_RELAXED 1e-4
#else
/* tape + torch CPU keep F64 storage. */
#define TEST_TOL_TIGHT 1e-12
#define TEST_TOL_RELAXED 1e-10
#endif

/* Disabled-test flags for backend-specific known bugs. Used as the
   `.disabled = SKIP_ON_<backend>` argument to a Criterion `Test()`
   declaration so the runtime emits SKIP rather than FAIL and the
   suite stays green while the underlying bug has its own follow-up
   row in TODO.md. */
#if defined(BACKEND_MLX)
#define SKIP_ON_MLX 1
#else
#define SKIP_ON_MLX 0
#endif

#endif /* BACKENDS_TEST_HELPERS_H */
