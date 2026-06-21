/* torch-only Criterion suite — adapter.cpp port-slot edge paths the
 * existing test_training_torch.c leaves uncovered.
 *
 * The common (tape-shared) param-registry suite never runs on torch, and
 * test_training_torch.c only drives the F32 / F16 grad branches plus the
 * dtag-streamed creators. That leaves these F64-fast / loader / zero_grad
 * arms of adapter.cpp's port shims unexercised:
 *
 *   - torch_port_grad_read    F64 fast branch (g cpu+contig, kFloat64)
 *   - torch_port_grad_write   F64 fast branch (kFloat64) via the sole
 *                             caller param_grad_item_and_zero
 *   - torch_port_zero_grad    BOTH branches: g.defined() true (after a
 *                             backward) and g.defined() false (a param
 *                             with no backward run yet -> no-op)
 *   - torch_port_load_doubles via param_load_data (overwrite + the
 *                             numel-mismatch reject in param_registry.c)
 *   - torch_port_load_int64   via param_load_data_int64 (int64 staging
 *                             copy_ into an F64 destination view)
 *   - torch_port_tensor_numel via param_load_data's dest-numel guard
 *
 * torch CPU base dtype is F64 (exact at 1e-12). Params/inputs use the
 * F64 dtag (15) so the default F64 storage path is hit.
 */

#include <criterion/criterion.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_TORCH

/* DType.Core dtag values (kind-major layout): 13=F16, 14=F32, 15=F64. */
#define DTAG_F64 15

/* ---- load_doubles + tensor_numel: param_load_data overwrite ---- */

Test(adapter_cov, load_doubles_overwrites_in_place) {
	/* param_load_data first checks tensor_numel (torch_port_tensor_numel)
	   then routes through torch_port_load_doubles, which stages the host
	   buffer via from_blob(kFloat64) and copy_'s it into t->view({n}). */
	param_clear();
	double init[] = {0.0, 0.0, 0.0, 0.0};
	TensorHandle p = tensor_create_param_1d_streamed(4, hcopy(init, 4), /*stream_tag=*/0, DTAG_F64);
	param_register("p", p);

	double fresh[] = {1.5, -2.25, 3.75, -4.5};
	param_load_data(0, fresh, 4);

	double buf[4];
	tensor_to_doubles(p, buf);
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(buf[i], fresh[i], TEST_TOL_TIGHT,
		                   "param_load_data overwrite [%d]: expected %.4f got %.12f", i, fresh[i],
		                   buf[i]);
	param_clear();
}

Test(adapter_cov, load_doubles_wrong_numel_is_rejected) {
	/* The numel guard (param_registry.c, via torch_port_tensor_numel)
	   rejects a buffer whose length differs from the tensor's numel:
	   load_doubles is never reached, data must be left unchanged. */
	param_clear();
	double init[] = {7.0, 8.0, 9.0};
	TensorHandle p = tensor_create_param_1d_streamed(3, hcopy(init, 3), /*stream_tag=*/0, DTAG_F64);
	param_register("p", p);

	double too_long[] = {1.0, 2.0, 3.0, 4.0, 5.0};
	param_load_data(0, too_long, 5); /* numel mismatch -> no-op */

	double buf[3];
	tensor_to_doubles(p, buf);
	for (int i = 0; i < 3; i++)
		cr_assert_float_eq(buf[i], init[i], TEST_TOL_TIGHT,
		                   "wrong-numel load left data unchanged [%d]: expected %.1f got %.12f", i,
		                   init[i], buf[i]);
	param_clear();
}

/* ---- load_int64: int64 staging copy into an F64 destination ---- */

Test(adapter_cov, load_int64_into_f64_param) {
	/* param_load_data_int64 routes through torch_port_load_int64, which
	   stages via from_blob(kInt64) and copy_'s into the F64 view (narrowing
	   each i64 to double). Small magnitudes survive exactly. */
	param_clear();
	double init[] = {0.0, 0.0, 0.0};
	TensorHandle p = tensor_create_param_1d_streamed(3, hcopy(init, 3), /*stream_tag=*/0, DTAG_F64);
	param_register("p", p);

	int64_t vals[] = {5, -17, 1000};
	param_load_data_int64(0, vals, 3);

	double buf[3];
	tensor_to_doubles(p, buf);
	cr_assert_float_eq(buf[0], 5.0, 0.0, "i64 load [0] exact (got %.12f)", buf[0]);
	cr_assert_float_eq(buf[1], -17.0, 0.0, "i64 load [1] exact (got %.12f)", buf[1]);
	cr_assert_float_eq(buf[2], 1000.0, 0.0, "i64 load [2] exact (got %.12f)", buf[2]);
	param_clear();
}

/* ---- grad_read F64 fast branch after backward ---- */

Test(adapter_cov, grad_read_f64_after_backward) {
	/* loss = sum(p*p); d/dp[i] = 2*p[i]. The F64 grad is cpu+contiguous
	   so torch_port_grad_read hits the kFloat64 data_ptr fast branch. */
	param_clear();
	double init[] = {3.0, -4.0, 5.0};
	TensorHandle p = tensor_create_param_1d_streamed(3, hcopy(init, 3), /*stream_tag=*/0, DTAG_F64);
	param_register("p", p);
	TensorHandle sq = tensor_mul(p, p);
	TensorHandle loss = tensor_sum(sq);
	tensor_backward(loss);
	double expect[] = {6.0, -8.0, 10.0};
	for (int i = 0; i < 3; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), expect[i], TEST_TOL_TIGHT,
		                   "d(sum(p*p))/dp[%d] should be %.1f (got %.12f)", i, expect[i],
		                   param_grad_item_at(0, i));
	param_clear();
}

/* ---- grad_write F64 fast branch via param_grad_item_and_zero ---- */

Test(adapter_cov, grad_write_f64_item_and_zero) {
	/* param_grad_item_and_zero reads grad[0] then writes 0 — the sole
	   grad_write caller. With an F64 grad it hits torch_port_grad_write's
	   kFloat64 data_ptr fast branch. */
	param_clear();
	TensorHandle s = tensor_create_scalar(4.0, /*requires_grad=*/1);
	param_register("s", s);
	TensorHandle loss = tensor_mul(s, s); /* d/ds = 2*4 = 8 */
	tensor_backward(loss);

	double first = param_grad_item_and_zero(0);
	cr_assert_float_eq(first, 8.0, TEST_TOL_TIGHT, "grad before zero should be 8 (got %.12f)",
	                   first);
	double after = param_grad_item(0);
	cr_assert_float_eq(after, 0.0, TEST_TOL_TIGHT,
	                   "grad after item_and_zero should be 0 (got %.12f)", after);
	param_clear();
}

/* ---- zero_grad: defined branch (g.zero_()) ---- */

Test(adapter_cov, zero_all_grads_defined) {
	/* param_zero_all_grads -> torch_port_zero_grad per param. After a
	   backward the grad is defined, so the g.defined() branch runs g.zero_().
	   Subsequent grad reads return 0 (has_grad stays true). */
	param_clear();
	double init[] = {2.0, -3.0};
	TensorHandle p = tensor_create_param_1d_streamed(2, hcopy(init, 2), /*stream_tag=*/0, DTAG_F64);
	param_register("p", p);
	TensorHandle loss = tensor_sum(tensor_mul(p, p));
	tensor_backward(loss); /* grads = 2*p = {4, -6} */
	cr_assert_float_eq(param_grad_item_at(0, 0), 4.0, TEST_TOL_TIGHT, "pre-zero grad[0]");

	param_zero_all_grads();
	cr_assert_float_eq(param_grad_item_at(0, 0), 0.0, TEST_TOL_TIGHT,
	                   "zeroed grad[0] should be 0 (got %.12f)", param_grad_item_at(0, 0));
	cr_assert_float_eq(param_grad_item_at(0, 1), 0.0, TEST_TOL_TIGHT,
	                   "zeroed grad[1] should be 0 (got %.12f)", param_grad_item_at(0, 1));
	param_clear();
}

/* ---- zero_grad: undefined branch (no-op when grad not yet defined) ---- */

Test(adapter_cov, zero_all_grads_undefined_is_noop) {
	/* A param registered without any backward pass has an undefined grad,
	   so torch_port_zero_grad takes the g.defined()==false no-op branch.
	   Must not crash; grad reads remain 0 (has_grad false short-circuit). */
	param_clear();
	double init[] = {1.0, 2.0, 3.0};
	TensorHandle p = tensor_create_param_1d_streamed(3, hcopy(init, 3), /*stream_tag=*/0, DTAG_F64);
	param_register("p", p);

	param_zero_all_grads(); /* exercises the undefined-grad no-op branch */

	for (int i = 0; i < 3; i++)
		cr_assert_float_eq(param_grad_item_at(0, i), 0.0, 0.0,
		                   "grad with no backward should read 0 [%d] (got %.12f)", i,
		                   param_grad_item_at(0, i));
	param_clear();
}

#endif /* BACKEND_TORCH */
