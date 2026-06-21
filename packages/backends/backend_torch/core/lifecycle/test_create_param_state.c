/* torch-only Criterion suite for the F64 2d base creator.
 *
 * Targets tensor_create_2d in create_param_state.cpp — the raw F64 2d
 * creator (torch::from_blob {rows,cols} kFloat64 .clone()) with the
 * optional requires_grad leaf. The common tape suite never reaches the
 * torch implementation; this drives both the plain readback path and the
 * requires_grad != 0 branch (grad-eligible leaf, sum->backward gives an
 * elementwise grad of 1).
 *
 * torch CPU base dtype is F64; integer values are exact at 1e-12.
 */

#include <criterion/criterion.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_TORCH

Test(torch_core_lifecycle_create_param_state, create_2d_f64_readback) {
	/* tensor_create_2d: F64 from a host buffer, no grad. */
	double xd[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	TensorHandle x = mk2d(2, 3, xd, /*requires_grad=*/0);
	cr_assert_str_eq(tensor_dtype_name(x), "F64", "tensor_create_2d should yield F64 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_eq(tensor_numel(x), 6, "numel should be 6");
	cr_assert_eq(tensor_dim(x), 2, "rank should be 2");
	cr_assert_eq(tensor_size(x, 0), 2, "dim 0 should be 2");
	cr_assert_eq(tensor_size(x, 1), 3, "dim 1 should be 3");
	double buf[6];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 6; i++) {
		cr_assert_float_eq(buf[i], xd[i], 1e-12, "F64 readback [%d]: expected %.1f got %.12f", i,
		                   xd[i], buf[i]);
	}
}

Test(torch_core_lifecycle_create_param_state, create_2d_f64_requires_grad) {
	/* requires_grad != 0 branch: grad-eligible leaf, sum->backward gives
	   elementwise grad of 1 across all 6 elements. */
	param_clear();
	double xd[] = {2.0, 4.0, 6.0, 8.0, 10.0, 12.0};
	TensorHandle x = mk2d(2, 3, xd, /*requires_grad=*/1);
	param_register("x", x);
	TensorHandle loss = tensor_sum(x);
	cr_assert_float_eq(tensor_item(loss), 42.0, 1e-12, "sum should be 42 (got %.6f)",
	                   tensor_item(loss));
	tensor_backward(loss);
	for (int i = 0; i < 6; i++) {
		cr_assert_float_eq(param_grad_item_at(0, i), 1.0, 1e-12,
		                   "grad x[%d] should be 1 (got %.6f)", i, param_grad_item_at(0, i));
	}
}

#endif /* BACKEND_TORCH */
