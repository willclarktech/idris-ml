/* mlx-only Criterion suite for training/adapter.cpp — the BackendPort
 * accessors the shared param_registry.c uses to talk to mlx.
 *
 * The common (tape-shared) param-registry suite already drives most of
 * the mlx adapter via the public FFI, but it leans on a handful of
 * slots. This file pins the mlx-specific behavior of the reachable
 * port slots so a regression in mlx's immutable-array element rebuild
 * (grad_write realizes host-side, mutates one element, rebuilds via
 * mx_array_from_doubles) shows up here rather than as a silent wrong
 * gradient in a training run.
 *
 * Reachable slots exercised (via public FFI, on the mlx-CPU lane):
 *   - tensor_numel        -> param_load_data's numel-vs-shape guard
 *   - tensor_has_grad     -> param_grad_item* short-circuit
 *   - grad_read           -> param_grad_item_at value
 *   - grad_write          -> param_grad_item_and_zero (read-then-zero,
 *                            the documented sole grad_write caller)
 *   - zero_grad           -> param_zero_all_grads
 *   - load_doubles        -> param_load_data overwrite + readback
 *   - load_int64          -> param_load_data_int64 (lossy double pivot)
 *
 * The remaining adapter slots (data_write, tensor_requires_grad) have
 * no consumer on the mlx lane — mlx is not in SHARED_BACKENDS_optimizer,
 * so the shared optimizer.c that would invoke them via polyak / native
 * train-step is never linked here, and param_registry.c never calls
 * them. They are GCOVR_EXCL'd in adapter.cpp with that reason.
 *
 * Params/inputs use the F64 dtag (15) so the default mlx-cpu F64 path is
 * exercised; value asserts use TEST_TOL_TIGHT (1e-5 — mlx readback
 * carries ~1e-6 error even on the F64 path) except exact-zero / exact-
 * int checks which use 0.0.
 */

#include <criterion/criterion.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_MLX

/* Heap-copy a stack array — the streamed creators take ownership and
   free it after building the mx::array. */
static double* hcopy(const double* s, int n) {
	double* b = (double*)malloc((size_t)n * sizeof(double));
	memcpy(b, s, (size_t)n * sizeof(double));
	return b;
}

/* ---- load_doubles + tensor_numel: param_load_data overwrite ---- */

Test(mlx_training_adapter, load_doubles_overwrites_in_place) {
	/* param_load_data routes through mlx_port_load_doubles, which
	   rebuilds t->data from the host buffer keeping the original
	   shape/dtype. The numel guard in param_registry.c first calls
	   mlx_port_tensor_numel. Both adapter slots fire here. */
	param_clear();
	double init[] = {0.0, 0.0, 0.0, 0.0};
	TensorHandle p = tensor_create_param_1d_streamed(4, hcopy(init, 4), 0, 15);
	param_register("p", p);

	double fresh[] = {1.5, -2.25, 3.75, -4.5};
	param_load_data(0, fresh, 4);

	double buf[4];
	tensor_to_doubles(p, buf);
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(buf[i], fresh[i], TEST_TOL_TIGHT,
		                   "param_load_data overwrite [%d]: expected %.4f got %.6f", i, fresh[i],
		                   buf[i]);
	param_clear();
}

Test(mlx_training_adapter, load_doubles_wrong_numel_is_rejected) {
	/* The numel guard (param_registry.c, via mlx_port_tensor_numel)
	   rejects a buffer whose length differs from the tensor's numel:
	   data must be left unchanged. */
	param_clear();
	double init[] = {7.0, 8.0, 9.0};
	TensorHandle p = tensor_create_param_1d_streamed(3, hcopy(init, 3), 0, 15);
	param_register("p", p);

	double too_long[] = {1.0, 2.0, 3.0, 4.0, 5.0};
	param_load_data(0, too_long, 5); /* numel mismatch -> no-op */

	double buf[3];
	tensor_to_doubles(p, buf);
	for (int i = 0; i < 3; i++)
		cr_assert_float_eq(buf[i], init[i], TEST_TOL_TIGHT,
		                   "wrong-numel load left data unchanged [%d]: expected %.1f got %.6f", i,
		                   init[i], buf[i]);
	param_clear();
}

/* ---- load_int64: lossy double-pivot loader ---- */

Test(mlx_training_adapter, load_int64_pivots_through_double) {
	/* param_load_data_int64 routes through mlx_port_load_int64, which
	   widens each i64 to double then rebuilds the array (mlx has no
	   native integer storage). Small magnitudes survive exactly. */
	param_clear();
	double init[] = {0.0, 0.0, 0.0};
	TensorHandle p = tensor_create_param_1d_streamed(3, hcopy(init, 3), 0, 15);
	param_register("p", p);

	int64_t vals[] = {5, -17, 1000};
	param_load_data_int64(0, vals, 3);

	double buf[3];
	tensor_to_doubles(p, buf);
	cr_assert_float_eq(buf[0], 5.0, 0.0, "i64 load [0] exact (got %.6f)", buf[0]);
	cr_assert_float_eq(buf[1], -17.0, 0.0, "i64 load [1] exact (got %.6f)", buf[1]);
	cr_assert_float_eq(buf[2], 1000.0, 0.0, "i64 load [2] exact (got %.6f)", buf[2]);
	param_clear();
}

/* ---- has_grad short-circuit: no backward => grad reads are zero ---- */

Test(mlx_training_adapter, grad_read_before_backward_is_zero) {
	/* param_grad_item_at -> param_registry.c checks mlx_port_tensor_has_grad
	   first; with no backward pass run, has_grad is false and the
	   adapter short-circuits to 0.0 without touching t->grad. */
	param_clear();
	double init[] = {1.0, 2.0};
	TensorHandle p = tensor_create_param_1d_streamed(2, hcopy(init, 2), 0, 15);
	param_register("p", p);
	cr_assert_float_eq(param_grad_item_at(0, 0), 0.0, 0.0,
	                   "grad before backward should be 0 (has_grad false)");
	cr_assert_float_eq(param_grad_item_at(0, 1), 0.0, 0.0,
	                   "grad before backward should be 0 (has_grad false)");
	param_clear();
}

/* ---- grad_read after backward: real gradient values ---- */

Test(mlx_training_adapter, grad_read_after_backward) {
	/* loss = sum(p*p); d/dp[i] = 2*p[i]. Drives grad_read through
	   param_grad_item_at with a non-trivial per-element gradient. */
	param_clear();
	double init[] = {3.0, -4.0, 5.0};
	TensorHandle p = tensor_create_param_1d_streamed(3, hcopy(init, 3), 0, 15);
	param_register("p", p);
	TensorHandle sq = tensor_mul(p, p);
	TensorHandle loss = tensor_sum(sq);
	tensor_backward(loss);
	double expect[] = {6.0, -8.0, 10.0};
	for (int i = 0; i < 3; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), expect[i], TEST_TOL_TIGHT,
		                   "d(sum(p*p))/dp[%d] should be %.1f (got %.6f)", i, expect[i],
		                   param_grad_item_at(0, i));
	param_clear();
}

/* ---- grad_write: param_grad_item_and_zero (the sole grad_write caller) ---- */

Test(mlx_training_adapter, grad_item_and_zero_reads_then_zeros) {
	/* param_grad_item_and_zero -> grad_read(t,0) then grad_write(t,0,0).
	   grad_write is the per-element immutable rebuild the adapter doc
	   calls out: realize grad host-side, set element 0 to 0, rebuild.
	   This is the documented sole grad_write caller — the path that
	   keeps mlx_port_grad_write live. */
	param_clear();
	TensorHandle s = tensor_create_scalar(4.0, 1); /* scalar param */
	param_register("s", s);
	TensorHandle loss = tensor_mul(s, s); /* d/ds = 2*4 = 8 */
	tensor_backward(loss);

	double first = param_grad_item_and_zero(0);
	cr_assert_float_eq(first, 8.0, TEST_TOL_TIGHT, "grad before zero should be 8 (got %.6f)",
	                   first);
	/* After grad_write set element 0 to 0, the next read returns 0. */
	double after = param_grad_item(0);
	cr_assert_float_eq(after, 0.0, TEST_TOL_TIGHT,
	                   "grad after item_and_zero should be 0 (got %.6f)", after);
	param_clear();
}

/* ---- zero_grad: param_zero_all_grads ---- */

Test(mlx_training_adapter, zero_all_grads_clears_grad) {
	/* param_zero_all_grads -> mlx_port_zero_grad per param: rebuilds
	   t->grad as mx::zeros while leaving has_grad set. Subsequent
	   grad reads return 0 (not the short-circuit path — has_grad is
	   still true after a backward). */
	param_clear();
	double init[] = {2.0, -3.0};
	TensorHandle p = tensor_create_param_1d_streamed(2, hcopy(init, 2), 0, 15);
	param_register("p", p);
	TensorHandle loss = tensor_sum(tensor_mul(p, p));
	tensor_backward(loss); /* grads = 2*p = {4, -6} */
	cr_assert_float_eq(param_grad_item_at(0, 0), 4.0, TEST_TOL_TIGHT, "pre-zero grad[0]");

	param_zero_all_grads();
	cr_assert_float_eq(param_grad_item_at(0, 0), 0.0, TEST_TOL_TIGHT,
	                   "zeroed grad[0] should be 0 (got %.6f)", param_grad_item_at(0, 0));
	cr_assert_float_eq(param_grad_item_at(0, 1), 0.0, TEST_TOL_TIGHT,
	                   "zeroed grad[1] should be 0 (got %.6f)", param_grad_item_at(0, 1));
	param_clear();
}

#endif /* BACKEND_MLX */
