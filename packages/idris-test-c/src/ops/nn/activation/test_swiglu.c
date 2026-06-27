/* Criterion suite for tensor_swiglu_2d (forward).
 *
 *   swiglu_2d(gate, up):
 *     out[i, j] = silu(gate[i, j]) * up[i, j]
 *              = gate[i, j] * sigmoid(gate[i, j]) * up[i, j]
 *
 * Replaces the tsilu + tmul pair in HfLlama.applyMlp with one fused
 * FFI call. Both inputs share shape [M, N].
 */

#include <criterion/criterion.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"
#include "test_helpers.h"

static double silu_ref(double x) {
	return x / (1.0 + exp(-x));
}

static double sigmoid_ref(double x) {
	return 1.0 / (1.0 + exp(-x));
}

/* dL/d(gate) for loss = sum(silu(gate)*up), dout=1. */
static double swiglu_dgate_ref(double gate, double up) {
	double s = sigmoid_ref(gate);
	return up * s * (1.0 + gate * (1.0 - s));
}

/* dL/d(up) for loss = sum(silu(gate)*up), dout=1. */
static double swiglu_dup_ref(double gate, double up) {
	return gate * sigmoid_ref(gate);
}

Test(nn_activation_swiglu, forward_zero_gate) {
	/* silu(0) = 0 * 0.5 = 0, so out should be 0 regardless of up. */
	param_clear();
	double g_d[] = {0.0, 0.0, 0.0, 0.0};
	double u_d[] = {1.0, 2.0, 3.0, 4.0};
	TensorHandle gate = tensor_create_2d_f64(1, 4, hcopy(g_d, 4), 0);
	TensorHandle up = tensor_create_2d_f64(1, 4, hcopy(u_d, 4), 0);
	TensorHandle r = tensor_swiglu_2d(gate, up);
	double buf[4];
	tensor_to_doubles(r, buf);
	for (int j = 0; j < 4; j++) {
		cr_assert_float_eq(buf[j], 0.0, TEST_TOL_RELAXED,
		                   "swiglu[%d] should be 0 when gate=0 (got %.9f)", j, buf[j]);
	}
}

Test(nn_activation_swiglu, forward_unit_up) {
	/* up = 1, so out[j] = silu(gate[j]). Reduces to silu reference. */
	param_clear();
	double g_d[] = {1.0, -1.0, 2.0, -2.0};
	double u_d[] = {1.0, 1.0, 1.0, 1.0};
	TensorHandle gate = tensor_create_2d_f64(1, 4, hcopy(g_d, 4), 0);
	TensorHandle up = tensor_create_2d_f64(1, 4, hcopy(u_d, 4), 0);
	TensorHandle r = tensor_swiglu_2d(gate, up);
	double buf[4];
	tensor_to_doubles(r, buf);
	for (int j = 0; j < 4; j++) {
		double expect = silu_ref(g_d[j]);
		cr_assert_float_eq(buf[j], expect, TEST_TOL_RELAXED,
		                   "swiglu[%d] expected %.9f got %.9f (gate=%.3f)", j, expect, buf[j],
		                   g_d[j]);
	}
}

Test(nn_activation_swiglu, forward_per_row_independent) {
	/* Two rows, distinct gate/up combinations.
	 * row 0: gate=[1, -1], up=[2, 2] -> [2*silu(1), 2*silu(-1)]
	 * row 1: gate=[0.5, 0.5], up=[1, -1] -> [silu(0.5), -silu(0.5)]
	 */
	param_clear();
	double g_d[] = {1.0, -1.0, 0.5, 0.5};
	double u_d[] = {2.0, 2.0, 1.0, -1.0};
	TensorHandle gate = tensor_create_2d_f64(2, 2, hcopy(g_d, 4), 0);
	TensorHandle up = tensor_create_2d_f64(2, 2, hcopy(u_d, 4), 0);
	TensorHandle r = tensor_swiglu_2d(gate, up);
	double buf[4];
	tensor_to_doubles(r, buf);
	for (int k = 0; k < 4; k++) {
		double expect = silu_ref(g_d[k]) * u_d[k];
		cr_assert_float_eq(buf[k], expect, TEST_TOL_RELAXED,
		                   "swiglu[%d] expected %.9f got %.9f (gate=%.3f up=%.3f)", k, expect,
		                   buf[k], g_d[k], u_d[k]);
	}
}

Test(nn_activation_swiglu, forward_matches_decomposed_chain) {
	/* Strongest correctness check: fused op must match host-side
	 * silu(g) * u over a non-trivial [seq, intermediate] grid. Same
	 * shape class HfLlama's MLP hits per token at miniature scale.
	 */
	param_clear();
	double g_d[32];
	double u_d[32];
	for (int i = 0; i < 32; i++) {
		g_d[i] = (i % 5 == 0) ? -0.7 : 0.3 + (i * 0.11);
		u_d[i] = 0.5 - (i * 0.07);
	}
	TensorHandle gate = tensor_create_2d_f64(4, 8, hcopy(g_d, 32), 0);
	TensorHandle up = tensor_create_2d_f64(4, 8, hcopy(u_d, 32), 0);
	TensorHandle r = tensor_swiglu_2d(gate, up);
	double got[32];
	tensor_to_doubles(r, got);
	for (int k = 0; k < 32; k++) {
		double expect = silu_ref(g_d[k]) * u_d[k];
		cr_assert_float_eq(got[k], expect, TEST_TOL_RELAXED, "swiglu[%d] expected %.9f got %.9f", k,
		                   expect, got[k]);
	}
}

/* Backward through OP_SWIGLU_2D with both inputs requiring grad. gate is
 * param 0, up is param 1. loss = sum(out) so dout = 1 everywhere. Covers the
 * F64 tape-append branch and tape_backward_swiglu_2d (both grad paths). */
Test(nn_activation_swiglu, backward_both_inputs) {
	param_clear();
	double g_d[] = {1.0, -1.0, 0.5, 2.0};
	double u_d[] = {2.0, 3.0, -1.0, 0.5};
	TensorHandle gate = tensor_create_2d_f64(2, 2, hcopy(g_d, 4), 1);
	TensorHandle up = tensor_create_2d_f64(2, 2, hcopy(u_d, 4), 1);
	param_register("gate", gate);
	param_register("up", up);
	TensorHandle loss = tensor_sum(tensor_swiglu_2d(gate, up));
	tensor_backward(loss);
	for (int k = 0; k < 4; k++) {
		double dg = swiglu_dgate_ref(g_d[k], u_d[k]);
		double du = swiglu_dup_ref(g_d[k], u_d[k]);
		cr_assert_float_eq(param_grad_item_at(0, k), dg, TEST_TOL_RELAXED,
		                   "d/d gate[%d] expected %.9f got %.9f", k, dg, param_grad_item_at(0, k));
		cr_assert_float_eq(param_grad_item_at(1, k), du, TEST_TOL_RELAXED,
		                   "d/d up[%d] expected %.9f got %.9f", k, du, param_grad_item_at(1, k));
	}
}

/* Only gate requires grad: exercises the gNeedsGrad-but-not-uNeedsGrad branch
 * in tape_backward_swiglu_2d. up still participates in the forward (rg = OR). */
Test(nn_activation_swiglu, backward_gate_only) {
	param_clear();
	double g_d[] = {0.5, -0.5, 1.5, -1.5};
	double u_d[] = {1.0, 2.0, -2.0, 0.5};
	TensorHandle gate = tensor_create_2d_f64(2, 2, hcopy(g_d, 4), 1);
	TensorHandle up = tensor_create_2d_f64(2, 2, hcopy(u_d, 4), 0);
	param_register("gate", gate);
	TensorHandle loss = tensor_sum(tensor_swiglu_2d(gate, up));
	tensor_backward(loss);
	for (int k = 0; k < 4; k++) {
		double dg = swiglu_dgate_ref(g_d[k], u_d[k]);
		cr_assert_float_eq(param_grad_item_at(0, k), dg, TEST_TOL_RELAXED,
		                   "d/d gate[%d] expected %.9f got %.9f", k, dg, param_grad_item_at(0, k));
	}
}

/* Only up requires grad: exercises the uNeedsGrad-but-not-gNeedsGrad branch. */
Test(nn_activation_swiglu, backward_up_only) {
	param_clear();
	double g_d[] = {0.3, -0.8, 1.2, 2.0};
	double u_d[] = {1.5, -1.0, 0.5, 2.0};
	TensorHandle gate = tensor_create_2d_f64(2, 2, hcopy(g_d, 4), 0);
	TensorHandle up = tensor_create_2d_f64(2, 2, hcopy(u_d, 4), 1);
	param_register("up", up);
	TensorHandle loss = tensor_sum(tensor_swiglu_2d(gate, up));
	tensor_backward(loss);
	for (int k = 0; k < 4; k++) {
		double du = swiglu_dup_ref(g_d[k], u_d[k]);
		cr_assert_float_eq(param_grad_item_at(0, k), du, TEST_TOL_RELAXED,
		                   "d/d up[%d] expected %.9f got %.9f", k, du, param_grad_item_at(0, k));
	}
}

/* F32 forward path: when both inputs carry the F32 tag (tag 14 via the
 * streamed constructor — tape's *_f32 convenience constructors are abort
 * stubs), tensor_swiglu_2d takes the DT_F32 branch (swiglu_2d.c:43-51) and
 * make_tensor_arena_f32 with NO grad (sig_g==NULL since rg==0). F32 readback
 * tolerance 1e-5. */
Test(nn_activation_swiglu, f32_forward_no_grad) {
	param_clear();
	double g_d[] = {1.0, -1.0, 0.5, 2.0, -2.0, 0.0};
	double u_d[] = {2.0, 3.0, -1.0, 0.5, 1.0, 4.0};
	TensorHandle gate = tensor_create_2d_streamed(2, 3, hcopy(g_d, 6), 0, 0, 14);
	TensorHandle up = tensor_create_2d_streamed(2, 3, hcopy(u_d, 6), 0, 0, 14);
	cr_assert_str_eq(tensor_dtype_name(gate), "F32", "gate should be F32-tagged");
	TensorHandle r = tensor_swiglu_2d(gate, up);
	cr_assert_str_eq(tensor_dtype_name(r), "F32", "swiglu F32 output should propagate F32 tag");
	double buf[6];
	tensor_to_doubles(r, buf);
	for (int k = 0; k < 6; k++) {
		double expect = silu_ref(g_d[k]) * u_d[k];
		cr_assert_float_eq(buf[k], expect, 1e-5, "f32 swiglu[%d] expected %.7f got %.7f", k, expect,
		                   buf[k]);
	}
}

/* F32 forward + backward: both inputs F32 params (rg==1) so the DT_F32 branch
 * allocates sig_g and appends the tape entry (swiglu_2d.c:53-60), then
 * tape_backward_swiglu_2d runs over the cached sigmoids. Closes the F32 grad
 * gap. F32 tolerance 1e-5. */
Test(nn_activation_swiglu, f32_backward_both_inputs) {
	param_clear();
	double g_d[] = {1.0, -1.0, 0.5, 2.0};
	double u_d[] = {2.0, 3.0, -1.0, 0.5};
	TensorHandle gate = tensor_create_param_2d_streamed(2, 2, hcopy(g_d, 4), 0, 14);
	TensorHandle up = tensor_create_param_2d_streamed(2, 2, hcopy(u_d, 4), 0, 14);
	param_register("gate", gate);
	param_register("up", up);
	cr_assert_str_eq(tensor_dtype_name(gate), "F32", "gate param should be F32-tagged");
	TensorHandle r = tensor_swiglu_2d(gate, up);
	cr_assert_str_eq(tensor_dtype_name(r), "F32", "swiglu F32 output should propagate F32 tag");
	TensorHandle loss = tensor_sum(r);
	tensor_backward(loss);
	for (int k = 0; k < 4; k++) {
		double dg = swiglu_dgate_ref(g_d[k], u_d[k]);
		double du = swiglu_dup_ref(g_d[k], u_d[k]);
		cr_assert_float_eq(param_grad_item_at(0, k), dg, 1e-5,
		                   "f32 d/d gate[%d] expected %.7f got %.7f", k, dg,
		                   param_grad_item_at(0, k));
		cr_assert_float_eq(param_grad_item_at(1, k), du, 1e-5,
		                   "f32 d/d up[%d] expected %.7f got %.7f", k, du,
		                   param_grad_item_at(1, k));
	}
}
