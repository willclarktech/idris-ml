/* torch-only Criterion suite for the training port adapter + autograd TU.
 *
 * Two product files the common tape suite never reaches on torch:
 *
 *   adapter.cpp — the static port trampolines wired into g_active_port.
 *     The dtag-streamed creators (tensor_create_param_*_streamed,
 *     tensor_create_state_*_streamed, the fused *_normal / *_const
 *     creators, tensor_set_init_seed_streamed) forward through these
 *     shims, so driving the public FFI exercises the torch_port_create_*
 *     bodies. The per-element grad accessor shims (torch_port_grad_read /
 *     torch_port_grad_write) are reached via the shared param_registry.c
 *     FFIs (param_grad_item_at / param_grad_item_and_zero) — torch IS in
 *     SHARED_BACKENDS_param_registry. The F32 / F16 grad branches and the
 *     slow flatten().index() fallback only fire when the param carries a
 *     non-F64 grad, which the F64-default tape suite never produces.
 *
 *   autograd.cpp — tensor_zero_grad (defined-grad branch), the nested
 *     no-grad counter (tensor_no_grad_begin / _end), and the no-op
 *     tensor_epoch_begin / _end.
 *
 * torch CPU base dtype is F64 (exact ints at 1e-12); F32 asserts at 1e-5;
 * F16 small-int values are exact at a relaxed tolerance.
 */

#include <criterion/criterion.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_TORCH

/* DType.Core dtag values (kind-major layout): 13=F16, 14=F32, 15=F64. */
#define DTAG_F16 13
#define DTAG_F32 14
#define DTAG_F64 15

#define TOL_RELAXED 1e-2

/* Streamed creators are callee-owns on the data arg → hand them a heap copy. */

/* ----------------------------------------------------------------------
   adapter.cpp — dtag-streamed param / state creators (torch_port_create_*).
   ---------------------------------------------------------------------- */

Test(torch_training_adapter, create_param_3d_streamed_f64) {
	/* torch_port_create_param_3d -> torch_create_param_3d_dtag. */
	double xd[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
	TensorHandle x =
	    tensor_create_param_3d_streamed(2, 2, 2, hcopy(xd, 8), /*stream_tag=*/0, DTAG_F64);
	cr_assert_str_eq(tensor_dtype_name(x), "F64", "param_3d should be F64 (got %s)",
	                 tensor_dtype_name(x));
	cr_assert_eq(tensor_numel(x), 8, "numel should be 8");
	cr_assert_eq(tensor_dim(x), 3, "rank should be 3");
	cr_assert_eq(tensor_requires_grad(x), 1, "param should require grad");
	double buf[8];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 8; i++) {
		cr_assert_float_eq(buf[i], xd[i], 1e-12, "param_3d [%d]: expected %.1f got %.12f", i, xd[i],
		                   buf[i]);
	}
}

Test(torch_training_adapter, create_param_4d_streamed_f64) {
	/* torch_port_create_param_4d -> torch_create_param_4d_dtag. */
	double xd[] = {1.0, 2.0, 3.0, 4.0};
	TensorHandle x =
	    tensor_create_param_4d_streamed(1, 1, 2, 2, hcopy(xd, 4), /*stream_tag=*/0, DTAG_F64);
	cr_assert_eq(tensor_numel(x), 4, "numel should be 4");
	cr_assert_eq(tensor_dim(x), 4, "rank should be 4");
	cr_assert_eq(tensor_requires_grad(x), 1, "param should require grad");
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], xd[i], 1e-12, "param_4d [%d]: expected %.1f got %.12f", i, xd[i],
		                   buf[i]);
	}
}

Test(torch_training_adapter, create_state_1d_streamed_f64) {
	/* torch_port_create_state_1d -> torch_create_state_1d_dtag. State
	   tensors are leaves without requires_grad. */
	double xd[] = {9.0, 8.0, 7.0};
	TensorHandle x = tensor_create_state_1d_streamed(3, hcopy(xd, 3), /*stream_tag=*/0, DTAG_F64);
	cr_assert_eq(tensor_numel(x), 3, "numel should be 3");
	cr_assert_eq(tensor_requires_grad(x), 0, "state should not require grad");
	double buf[3];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 3; i++) {
		cr_assert_float_eq(buf[i], xd[i], 1e-12, "state_1d [%d]: expected %.1f got %.12f", i, xd[i],
		                   buf[i]);
	}
}

Test(torch_training_adapter, create_state_2d_streamed_f64) {
	/* torch_port_create_state_2d -> torch_create_state_2d_dtag. */
	double xd[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	TensorHandle x =
	    tensor_create_state_2d_streamed(2, 3, hcopy(xd, 6), /*stream_tag=*/0, DTAG_F64);
	cr_assert_eq(tensor_numel(x), 6, "numel should be 6");
	cr_assert_eq(tensor_dim(x), 2, "rank should be 2");
	cr_assert_eq(tensor_requires_grad(x), 0, "state should not require grad");
}

/* ----------------------------------------------------------------------
   adapter.cpp — fused param create + init shims (torch_port_create_param_*
   _normal / _const) and the seed shim (torch_port_set_init_seed).
   ---------------------------------------------------------------------- */

Test(torch_training_adapter, create_param_1d_const_streamed) {
	/* torch_port_create_param_1d_const: fill_ then leaf. */
	TensorHandle x =
	    tensor_create_param_1d_const_streamed(4, /*value=*/2.5, /*stream_tag=*/0, DTAG_F64);
	cr_assert_eq(tensor_numel(x), 4, "numel should be 4");
	cr_assert_eq(tensor_requires_grad(x), 1, "param should require grad");
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], 2.5, 1e-12, "const [%d]: expected 2.5 got %.12f", i, buf[i]);
	}
}

Test(torch_training_adapter, create_param_2d_const_streamed) {
	/* torch_port_create_param_2d_const. */
	TensorHandle x =
	    tensor_create_param_2d_const_streamed(2, 3, /*value=*/-1.0, /*stream_tag=*/0, DTAG_F64);
	cr_assert_eq(tensor_numel(x), 6, "numel should be 6");
	double buf[6];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 6; i++) {
		cr_assert_float_eq(buf[i], -1.0, 1e-12, "const [%d]: expected -1 got %.12f", i, buf[i]);
	}
}

Test(torch_training_adapter, create_param_3d_const_streamed) {
	/* torch_port_create_param_3d_const. */
	TensorHandle x =
	    tensor_create_param_3d_const_streamed(2, 1, 2, /*value=*/0.5, /*stream_tag=*/0, DTAG_F64);
	cr_assert_eq(tensor_numel(x), 4, "numel should be 4");
	cr_assert_eq(tensor_dim(x), 3, "rank should be 3");
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], 0.5, 1e-12, "const [%d]: expected 0.5 got %.12f", i, buf[i]);
	}
}

Test(torch_training_adapter, create_param_4d_const_streamed) {
	/* torch_port_create_param_4d_const. */
	TensorHandle x = tensor_create_param_4d_const_streamed(1, 2, 1, 2, /*value=*/3.0,
	                                                       /*stream_tag=*/0, DTAG_F64);
	cr_assert_eq(tensor_numel(x), 4, "numel should be 4");
	cr_assert_eq(tensor_dim(x), 4, "rank should be 4");
	double buf[4];
	tensor_to_doubles(x, buf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(buf[i], 3.0, 1e-12, "const [%d]: expected 3 got %.12f", i, buf[i]);
	}
}

Test(torch_training_adapter, create_param_normal_streamed_seeded) {
	/* torch_port_set_init_seed + the four normal creators. Seeding makes
	   the draw deterministic; we assert the *shape/dtype* and that the
	   mean of a wide draw is in the right ballpark (the normal_ fill ran),
	   not exact values (RNG impl-defined). */
	tensor_set_init_seed_streamed(/*seed=*/1234ULL, /*stream_tag=*/0);

	TensorHandle a = tensor_create_param_1d_normal_streamed(64, /*mean=*/0.0, /*std=*/1.0,
	                                                        /*stream_tag=*/0, DTAG_F64);
	cr_assert_eq(tensor_numel(a), 64, "1d normal numel should be 64");
	cr_assert_eq(tensor_requires_grad(a), 1, "normal param should require grad");
	double abuf[64];
	tensor_to_doubles(a, abuf);
	double mean = 0.0;
	for (int i = 0; i < 64; i++)
		mean += abuf[i];
	mean /= 64.0;
	cr_assert(mean > -0.6 && mean < 0.6, "1d normal sample mean %.4f out of plausible band", mean);

	TensorHandle b = tensor_create_param_2d_normal_streamed(2, 4, /*mean=*/5.0, /*std=*/0.0,
	                                                        /*stream_tag=*/0, DTAG_F64);
	cr_assert_eq(tensor_numel(b), 8, "2d normal numel should be 8");
	double bbuf[8];
	tensor_to_doubles(b, bbuf);
	for (int i = 0; i < 8; i++) {
		/* std=0 collapses the normal to a constant at the mean. */
		cr_assert_float_eq(bbuf[i], 5.0, 1e-9, "2d normal std=0 [%d]: expected 5 got %.9f", i,
		                   bbuf[i]);
	}

	TensorHandle c = tensor_create_param_3d_normal_streamed(2, 2, 2, /*mean=*/-3.0, /*std=*/0.0,
	                                                        /*stream_tag=*/0, DTAG_F64);
	cr_assert_eq(tensor_numel(c), 8, "3d normal numel should be 8");
	cr_assert_eq(tensor_dim(c), 3, "3d normal rank should be 3");
	double cbuf[8];
	tensor_to_doubles(c, cbuf);
	for (int i = 0; i < 8; i++) {
		cr_assert_float_eq(cbuf[i], -3.0, 1e-9, "3d normal std=0 [%d]: expected -3 got %.9f", i,
		                   cbuf[i]);
	}

	TensorHandle d = tensor_create_param_4d_normal_streamed(1, 2, 2, 1, /*mean=*/7.0, /*std=*/0.0,
	                                                        /*stream_tag=*/0, DTAG_F64);
	cr_assert_eq(tensor_numel(d), 4, "4d normal numel should be 4");
	cr_assert_eq(tensor_dim(d), 4, "4d normal rank should be 4");
	double dbuf[4];
	tensor_to_doubles(d, dbuf);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(dbuf[i], 7.0, 1e-9, "4d normal std=0 [%d]: expected 7 got %.9f", i,
		                   dbuf[i]);
	}
}

/* ----------------------------------------------------------------------
   adapter.cpp — per-element grad accessor shims (torch_port_grad_read /
   torch_port_grad_write) via the shared param_registry FFIs, driving the
   F32 and F16 (slow flatten) branches.
   ---------------------------------------------------------------------- */

Test(torch_training_adapter, grad_read_f32_param) {
	/* F32 param -> F32 grad -> torch_port_grad_read F32 fast branch
	   (line 114). param_grad_item_at routes through grad_read. */
	param_clear();
	double xd[] = {2.0, 4.0, 6.0};
	TensorHandle x = tensor_create_param_1d_streamed(3, hcopy(xd, 3), /*stream_tag=*/0, DTAG_F32);
	cr_assert_str_eq(tensor_dtype_name(x), "F32", "param should be F32 (got %s)",
	                 tensor_dtype_name(x));
	param_register("x", x);
	TensorHandle loss = tensor_sum(x);
	tensor_backward(loss);
	for (int i = 0; i < 3; i++) {
		cr_assert_float_eq(param_grad_item_at(0, i), 1.0, 1e-5,
		                   "F32 grad [%d] should be 1 (got %.6f)", i, param_grad_item_at(0, i));
	}
}

Test(torch_training_adapter, grad_read_f16_slow_path) {
	/* F16 grad is cpu+contiguous but neither F64 nor F32, so grad_read
	   falls through to the slow flatten().index() path (line 116). */
	param_clear();
	double xd[] = {1.0, 2.0, 4.0};
	TensorHandle x = tensor_create_param_1d_streamed(3, hcopy(xd, 3), /*stream_tag=*/0, DTAG_F16);
	cr_assert_str_eq(tensor_dtype_name(x), "F16", "param should be F16 (got %s)",
	                 tensor_dtype_name(x));
	param_register("x", x);
	TensorHandle loss = tensor_sum(x);
	tensor_backward(loss);
	for (int i = 0; i < 3; i++) {
		cr_assert_float_eq(param_grad_item_at(0, i), 1.0, TOL_RELAXED,
		                   "F16 grad [%d] should be 1 (got %.6f)", i, param_grad_item_at(0, i));
	}
}

Test(torch_training_adapter, grad_write_f32_zero) {
	/* param_grad_item_and_zero reads grad[0] then writes 0 — drives
	   torch_port_grad_write's F32 fast branch (lines 128-130). */
	param_clear();
	double xd[] = {3.0, 5.0};
	TensorHandle x = tensor_create_param_1d_streamed(2, hcopy(xd, 2), /*stream_tag=*/0, DTAG_F32);
	param_register("x", x);
	TensorHandle loss = tensor_sum(x);
	tensor_backward(loss);
	double g0 = param_grad_item_and_zero(0);
	cr_assert_float_eq(g0, 1.0, 1e-5, "F32 grad[0] before zero should be 1 (got %.6f)", g0);
	cr_assert_float_eq(param_grad_item_at(0, 0), 0.0, 1e-5,
	                   "F32 grad[0] should be 0 after zero (got %.6f)", param_grad_item_at(0, 0));
}

Test(torch_training_adapter, grad_write_f16_slow_zero) {
	/* F16 grad routes grad_write through the slow index_put_ path
	   (line 133). */
	param_clear();
	double xd[] = {2.0, 6.0};
	TensorHandle x = tensor_create_param_1d_streamed(2, hcopy(xd, 2), /*stream_tag=*/0, DTAG_F16);
	param_register("x", x);
	TensorHandle loss = tensor_sum(x);
	tensor_backward(loss);
	double g0 = param_grad_item_and_zero(0);
	cr_assert_float_eq(g0, 1.0, TOL_RELAXED, "F16 grad[0] before zero should be 1 (got %.6f)", g0);
	cr_assert_float_eq(param_grad_item_at(0, 0), 0.0, TOL_RELAXED,
	                   "F16 grad[0] should be 0 after zero (got %.6f)", param_grad_item_at(0, 0));
}

/* ----------------------------------------------------------------------
   autograd.cpp — tensor_zero_grad, nested no-grad scope, epoch no-ops.
   ---------------------------------------------------------------------- */

Test(torch_training_autograd, zero_grad_defined) {
	/* tensor_zero_grad with a defined grad takes the .zero_() branch
	   (autograd.cpp lines 41-45). */
	param_clear();
	double xd[] = {1.0, 2.0, 3.0, 4.0};
	TensorHandle x = tensor_create_1d_f64(4, hcopy(xd, 4), /*requires_grad=*/1);
	param_register("x", x);
	TensorHandle loss = tensor_sum(x);
	tensor_backward(loss);
	cr_assert_float_eq(param_grad_item_at(0, 0), 1.0, 1e-12, "grad[0] should be 1 pre-zero");
	tensor_zero_grad(x);
	for (int i = 0; i < 4; i++) {
		cr_assert_float_eq(param_grad_item_at(0, i), 0.0, 1e-12,
		                   "grad [%d] should be 0 after tensor_zero_grad (got %.12f)", i,
		                   param_grad_item_at(0, i));
	}
}

Test(torch_training_autograd, no_grad_nested_scope) {
	/* tensor_no_grad_begin/end nesting counter (autograd.cpp lines 78-92).
	   A leaf's requires_grad_ flag survives a NoGradGuard (the guard only
	   suppresses op-graph recording), so the observable effect is on
	   OPERATION results: an op over a grad-requiring input produces a
	   non-grad result inside the scope, a grad-tracked result outside.
	   The nested begin must NOT create a second guard; only the matched
	   outermost end releases it. */
	double xd[] = {1.0, 2.0};
	double yd[] = {3.0, 4.0};
	TensorHandle x = tensor_create_1d_f64(2, hcopy(xd, 2), /*requires_grad=*/1);
	TensorHandle y = tensor_create_1d_f64(2, hcopy(yd, 2), /*requires_grad=*/1);

	tensor_no_grad_begin();
	tensor_no_grad_begin(); /* nested: no new guard, depth->2 */
	TensorHandle inner = tensor_add(x, y);
	tensor_retain_handle(inner);
	cr_assert_eq(tensor_requires_grad(inner), 0,
	             "op result under NoGradGuard must not require grad");
	tensor_no_grad_end(); /* depth->1, still guarded */
	TensorHandle mid = tensor_add(x, y);
	tensor_retain_handle(mid);
	cr_assert_eq(tensor_requires_grad(mid), 0, "still guarded at depth 1");
	tensor_no_grad_end(); /* depth->0, guard released */

	TensorHandle outer = tensor_add(x, y);
	tensor_retain_handle(outer);
	cr_assert_eq(tensor_requires_grad(outer), 1, "grad tracking live again after matched ends");
}

Test(torch_training_autograd, no_grad_end_underflow_safe) {
	/* tensor_no_grad_end with depth already 0 is a no-op (the depth>0
	   guard) — must not underflow or leave a stray guard installed. */
	tensor_no_grad_end();
	tensor_no_grad_end();
	double xd[] = {1.0};
	double yd[] = {2.0};
	TensorHandle x = tensor_create_1d_f64(1, hcopy(xd, 1), /*requires_grad=*/1);
	TensorHandle y = tensor_create_1d_f64(1, hcopy(yd, 1), /*requires_grad=*/1);
	TensorHandle r = tensor_add(x, y);
	tensor_retain_handle(r);
	cr_assert_eq(tensor_requires_grad(r), 1, "grad must still be live after spurious ends");
}

Test(torch_training_autograd, epoch_begin_end_noop) {
	/* tensor_epoch_begin/end are no-ops on torch (no Metal buffer ceiling);
	   just confirm they're callable without disturbing grad state. */
	tensor_epoch_begin();
	double xd[] = {1.0, 2.0};
	double yd[] = {3.0, 4.0};
	TensorHandle x = tensor_create_1d_f64(2, hcopy(xd, 2), /*requires_grad=*/1);
	TensorHandle y = tensor_create_1d_f64(2, hcopy(yd, 2), /*requires_grad=*/1);
	TensorHandle r = tensor_add(x, y);
	tensor_retain_handle(r);
	cr_assert_eq(tensor_requires_grad(r), 1, "epoch_begin must not touch grad mode");
	tensor_epoch_end();
}

#endif /* BACKEND_TORCH */
