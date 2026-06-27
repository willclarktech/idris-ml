/* port_assert.h — ASSERT_NEAR / ASSERT_TRUE Criterion shims, backend-aware
 * finite-difference vs value tolerances. The heap-copy helper lives in
 * test_helpers.h (hcopy / mk2d), re-exported here via #include.
 *
 * Shared by the per-area C suites (test_autograd / test_linalg /
 * test_nn_layers / test_activations / test_losses / test_lstm /
 * test_tensor_misc, plus the per-backend dtype-scaffolding suites under
 * packages/backends/test/<backend>/). These were carried over verbatim
 * from the original bulk backend port; the macros redirect ASSERT_NEAR to
 * cr_assert_float_eq and ASSERT_TRUE to cr_assert so the assertion sites
 * need no per-line edits, keeping the in-file fp64-vs-mlx tolerance
 * discipline (FD_TOL / VAL_TOL).
 */

#ifndef IDRIS_ML_PORT_ASSERT_H
#define IDRIS_ML_PORT_ASSERT_H

#include <criterion/criterion.h>
#include "backend.h"
#include "shared_utils.h"
#include "test_helpers.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* The heap-copy helper for callee-frees creators is hcopy(), shared from
   test_helpers.h (included above). For the common 2D case prefer mk2d(). */

/* `got` is evaluated exactly once — required for sites that pass
   destructive readers like param_grad_item_and_zero(i) whose second
   call would observe the zeroed state. */
#define ASSERT_NEAR(msg, got, expected, tol)                                                       \
	do {                                                                                           \
		double _an_got = (got);                                                                    \
		double _an_exp = (expected);                                                               \
		cr_assert_float_eq(_an_got, _an_exp, (tol), "%s: got %.6f expected %.6f", msg, _an_got,    \
		                   _an_exp);                                                               \
	} while (0)

#define ASSERT_TRUE(msg, cond) cr_assert((cond), "%s", msg)

/* Backend-aware FD vs VAL tolerance pair — mlx is fp32 internally so
   finite-difference noise is multi-op amplified; fp64 backends (tape,
   torch) come within ~1e-5 of FD. */
#if defined(BACKEND_MLX)
#define FD_TOL 5e-1
#define VAL_TOL 1e-5
#else
#define FD_TOL 1e-3
#define VAL_TOL 1e-10
#endif

#endif /* IDRIS_ML_PORT_ASSERT_H */
