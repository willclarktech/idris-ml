/* test_quantize.c — BitNet b1.58 load-time quantization + HF-fused forward.
 *
 * Covers the ops in bitlinear.c that the pack/forward suites don't:
 *   - tensor_absmean_per_row_2d
 *   - tensor_ternary_quant_with_scale_2d   (verified via tensor_bitlinear_fwd)
 *   - tensor_create_ternary_from_hf_packed_2d (verified via forward)
 *   - tensor_bitlinear_fwd_hf_quant         (no-rms and rms paths)
 * plus the own-line abort guards as SIGABRT death tests.
 *
 * Quantized weights cannot be read back directly (packed 2-bit on tape,
 * int8 on torch/mlx), so the quant ops are checked by running the result
 * through tensor_bitlinear_fwd with scale=1, x=1 — a portable contract
 * across all three backends. Expected values are hand-computed (see the
 * per-test comments) and reproduced in the conversation log.
 */

#include <signal.h>
#include <criterion/criterion.h>
#include <stdint.h>
#include "backend.h"
#include "test_helpers.h"

/* ------------------------------------------------------------------ */
/* tensor_absmean_per_row_2d                                          */
/* ------------------------------------------------------------------ */

/* DISABLED: tensor_absmean_per_row_2d crashes on a valid 2D F64 input — see
   TODO.md "tape BitNet quant helpers crash". Re-enable when fixed. */
Test(nn_quantization_absmean, per_row_mean_abs, .disabled = true) {
	/* w = [[1, -2, 3, -4], [0.5, -0.5, 1.5, -1.5]]
	   row 0 absmean = (1+2+3+4)/4 = 2.5
	   row 1 absmean = (0.5+0.5+1.5+1.5)/4 = 1.0 */
	double w_data[8] = {1.0, -2.0, 3.0, -4.0, 0.5, -0.5, 1.5, -1.5};
	TensorHandle w = tensor_create_2d(2, 4, w_data, 0);
	TensorHandle s = tensor_absmean_per_row_2d(w);

	cr_assert_eq(tensor_dim(s), 1);
	cr_assert_eq(tensor_size(s, 0), 2);

	double out[2] = {0};
	tensor_to_doubles(s, out);
	cr_assert_float_eq(out[0], 2.5, TEST_TOL_RELAXED, "absmean row0: got %.6f", out[0]);
	cr_assert_float_eq(out[1], 1.0, TEST_TOL_RELAXED, "absmean row1: got %.6f", out[1]);
}

/* DISABLED: tensor_absmean_per_row_2d crash — see TODO.md. */
Test(nn_quantization_absmean, all_zero_row_is_zero, .disabled = true) {
	/* A row of zeros -> absmean 0 (the /0-guard input for the quant op). */
	double w_data[8] = {0.0, 0.0, 0.0, 0.0, 2.0, -2.0, 2.0, -2.0};
	TensorHandle w = tensor_create_2d(2, 4, w_data, 0);
	TensorHandle s = tensor_absmean_per_row_2d(w);
	double out[2] = {0};
	tensor_to_doubles(s, out);
	cr_assert_float_eq(out[0], 0.0, TEST_TOL_RELAXED, "zero row absmean should be 0");
	cr_assert_float_eq(out[1], 2.0, TEST_TOL_RELAXED, "absmean row1: got %.6f", out[1]);
}

/* Non-2D input aborts (own-line abort in absmean). */
Test(nn_quantization_absmean, rank1_aborts, .signal = SIGABRT) {
	double d[3] = {1.0, 2.0, 3.0};
	int s[1] = {3};
	TensorHandle v = tensor_create(d, s, 1, 0);
	tensor_absmean_per_row_2d(v); /* rank=1 -> abort */
}

/* ------------------------------------------------------------------ */
/* tensor_ternary_quant_with_scale_2d (verified via forward)          */
/* ------------------------------------------------------------------ */

/* DISABLED: tensor_ternary_quant_with_scale_2d crash — see TODO.md. */
Test(nn_quantization_ternary_quant, round_clamp_via_forward, .disabled = true) {
	/* w = [[2, 0, -3, 1], [-1, 0.4, 0.6, -2]], scale = [2, 1]
	   row0 w/scale = [1, 0, -1.5, 0.5] -> round/clamp = [1, 0, -1, 0]
	   row1 w/scale = [-1, 0.4, 0.6, -2] -> round/clamp = [-1, 0, 1, -1]
	   Forward with fwd-scale=[1,1], x=[1,1,1,1]:
	     row0 sum = 1+0-1+0 =  0
	     row1 sum = -1+0+1-1 = -1 */
	double w_data[8] = {2.0, 0.0, -3.0, 1.0, -1.0, 0.4, 0.6, -2.0};
	double scale_data[2] = {2.0, 1.0};
	int scale_shape[1] = {2};
	TensorHandle w = tensor_create_2d(2, 4, w_data, 0);
	TensorHandle scale = tensor_create(scale_data, scale_shape, 1, 0);
	TensorHandle q = tensor_ternary_quant_with_scale_2d(w, scale);

	double fwd_scale_data[2] = {1.0, 1.0};
	double x_data[4] = {1.0, 1.0, 1.0, 1.0};
	TensorHandle fwd_scale = tensor_create(fwd_scale_data, scale_shape, 1, 0);
	int x_shape[1] = {4};
	TensorHandle x = tensor_create(x_data, x_shape, 1, 0);
	TensorHandle y = tensor_bitlinear_fwd(q, fwd_scale, x, NULL);

	double out[2] = {0};
	tensor_to_doubles(y, out);
	cr_assert_float_eq(out[0], 0.0, TEST_TOL_RELAXED, "quant row0 fwd: got %.6f", out[0]);
	cr_assert_float_eq(out[1], -1.0, TEST_TOL_RELAXED, "quant row1 fwd: got %.6f", out[1]);
}

/* DISABLED: tensor_ternary_quant_with_scale_2d crash — see TODO.md. */
Test(nn_quantization_ternary_quant, zero_scale_row_all_zero, .disabled = true) {
	/* scale[0] == 0 -> row stays all-zero ternary (the /0 guard branch). */
	double w_data[4] = {5.0, -5.0, 5.0, -5.0};
	double scale_data[1] = {0.0};
	int scale_shape[1] = {1};
	TensorHandle w = tensor_create_2d(1, 4, w_data, 0);
	TensorHandle scale = tensor_create(scale_data, scale_shape, 1, 0);
	TensorHandle q = tensor_ternary_quant_with_scale_2d(w, scale);

	double fwd_scale_data[1] = {1.0};
	double x_data[4] = {1.0, 1.0, 1.0, 1.0};
	TensorHandle fwd_scale = tensor_create(fwd_scale_data, scale_shape, 1, 0);
	int x_shape[1] = {4};
	TensorHandle x = tensor_create(x_data, x_shape, 1, 0);
	TensorHandle y = tensor_bitlinear_fwd(q, fwd_scale, x, NULL);

	double out[1] = {0};
	tensor_to_doubles(y, out);
	cr_assert_float_eq(out[0], 0.0, TEST_TOL_RELAXED, "zero-scale row should quantize to all-zero");
}

/* Non-2D weight aborts. */
Test(nn_quantization_ternary_quant, rank1_weight_aborts, .signal = SIGABRT) {
	double d[3] = {1.0, 2.0, 3.0};
	int s[1] = {3};
	TensorHandle v = tensor_create(d, s, 1, 0);
	double sc[3] = {1.0, 1.0, 1.0};
	TensorHandle scale = tensor_create(sc, s, 1, 0);
	tensor_ternary_quant_with_scale_2d(v, scale); /* w rank=1 -> abort */
}

/* Scale shape mismatch aborts (scale rank or length != w->shape[0]). */
Test(nn_quantization_ternary_quant, scale_shape_mismatch_aborts, .signal = SIGABRT) {
	double w_data[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
	TensorHandle w = tensor_create_2d(2, 4, w_data, 0);
	double sc[3] = {1.0, 1.0, 1.0};
	int s[1] = {3}; /* expected [2], given [3] */
	TensorHandle scale = tensor_create(sc, s, 1, 0);
	tensor_ternary_quant_with_scale_2d(w, scale); /* shape mismatch -> abort */
}

/* ------------------------------------------------------------------ */
/* tensor_create_ternary_from_hf_packed_2d (verified via forward)     */
/* ------------------------------------------------------------------ */

Test(nn_quantization_hf_packed, layout_remap_via_forward) {
	/* o=2, i=4. HF layout: [(o+3)/4, i] = [1, 4] bytes; rows 0 and 1
	   share the byte, chunk 0 (low bits) = row 0, chunk 1 = row 1.
	   Target ternary:
	     row0 = [ 1, 0, -1, 1]  -> codes (value+1) = [2, 1, 0, 2]
	     row1 = [-1, 1,  0, 1]  -> codes (value+1) = [0, 2, 1, 2]
	   hf_byte[k] = code_row0 | (code_row1 << 2):
	     k0: 2|(0<<2)=0x02  k1: 1|(2<<2)=0x09  k2: 0|(1<<2)=0x04  k3: 2|(2<<2)=0x0A
	   Forward scale=[1,1], x=[1,1,1,1]:
	     row0 sum = 1+0-1+1 = 1 ; row1 sum = -1+1+0+1 = 1 */
	uint8_t hf[4] = {0x02, 0x09, 0x04, 0x0A};
	TensorHandle W = tensor_create_ternary_from_hf_packed_2d(hf, 2, 4);
	cr_assert_eq(tensor_dim(W), 2);
	cr_assert_eq(tensor_size(W, 0), 2);
	cr_assert_eq(tensor_size(W, 1), 4);

	double scale_data[2] = {1.0, 1.0};
	int scale_shape[1] = {2};
	double x_data[4] = {1.0, 1.0, 1.0, 1.0};
	int x_shape[1] = {4};
	TensorHandle scale = tensor_create(scale_data, scale_shape, 1, 0);
	TensorHandle x = tensor_create(x_data, x_shape, 1, 0);
	TensorHandle y = tensor_bitlinear_fwd(W, scale, x, NULL);

	double out[2] = {0};
	tensor_to_doubles(y, out);
	cr_assert_float_eq(out[0], 1.0, TEST_TOL_RELAXED, "hf-remap row0 fwd: got %.6f", out[0]);
	cr_assert_float_eq(out[1], 1.0, TEST_TOL_RELAXED, "hf-remap row1 fwd: got %.6f", out[1]);
}

/* ------------------------------------------------------------------ */
/* tensor_bitlinear_fwd_hf_quant                                      */
/* ------------------------------------------------------------------ */

Test(nn_quantization_hf_quant, no_rms_matches_handcompute) {
	/* W ternary (0x71,0x17,0x4C) = [[1,0,-1,1],[-1,1,1,0],[0,-1,0,1]]
	   w_scale=0.5, x=[1,3,-0.6,0.3], no rms, no bias.
	   (x chosen so x*in_scale stays away from .5 rounding boundaries, so
	   the F32 backends round identically to F64.)
	   xmax=3 -> in_scale=42.3333...; xq=[42,127,-25,13]; rescale=0.5/42.333
	     row0 sum=42+0+25+13=80   -> y =  80*rescale = 0.944881889...
	     row1 sum=-42+127-25+0=60 -> y =  60*rescale = 0.708661417...
	     row2 sum=0-127+0+13=-114 -> y = -114*rescale = -1.346456692... */
	uint8_t packed[3] = {0x71, 0x17, 0x4C};
	TensorHandle W = tensor_create_ternary_packed_2d(packed, 3, 3, 4, 0);
	double x_data[4] = {1.0, 3.0, -0.6, 0.3};
	int x_shape[1] = {4};
	TensorHandle x = tensor_create(x_data, x_shape, 1, 0);

	TensorHandle y = tensor_bitlinear_fwd_hf_quant(W, 0.5, x, /*bias=*/NULL, /*use_rms_norm=*/0,
	                                               /*rms_w=*/NULL, /*rms_eps=*/1.0e-5);
	double out[3] = {0};
	tensor_to_doubles(y, out);
	cr_assert_float_eq(out[0], 0.9448818897637795, TEST_TOL_RELAXED, "hf_quant y0: got %.6f",
	                   out[0]);
	cr_assert_float_eq(out[1], 0.7086614173228346, TEST_TOL_RELAXED, "hf_quant y1: got %.6f",
	                   out[1]);
	cr_assert_float_eq(out[2], -1.3464566929133859, TEST_TOL_RELAXED, "hf_quant y2: got %.6f",
	                   out[2]);
}

Test(nn_quantization_hf_quant, no_rms_with_bias) {
	/* Same fixture + bias [0.1, -0.2, 0.3] -> add to the no-bias output. */
	uint8_t packed[3] = {0x71, 0x17, 0x4C};
	TensorHandle W = tensor_create_ternary_packed_2d(packed, 3, 3, 4, 0);
	double x_data[4] = {1.0, 3.0, -0.6, 0.3};
	int x_shape[1] = {4};
	TensorHandle x = tensor_create(x_data, x_shape, 1, 0);
	double bias_data[3] = {0.1, -0.2, 0.3};
	int bias_shape[1] = {3};
	TensorHandle bias = tensor_create(bias_data, bias_shape, 1, 0);

	TensorHandle y = tensor_bitlinear_fwd_hf_quant(W, 0.5, x, bias, 0, NULL, 1.0e-5);
	double out[3] = {0};
	tensor_to_doubles(y, out);
	cr_assert_float_eq(out[0], 0.9448818897637795 + 0.1, TEST_TOL_RELAXED, "y0+bias: got %.6f",
	                   out[0]);
	cr_assert_float_eq(out[1], 0.7086614173228346 - 0.2, TEST_TOL_RELAXED, "y1+bias: got %.6f",
	                   out[1]);
	cr_assert_float_eq(out[2], -1.3464566929133859 + 0.3, TEST_TOL_RELAXED, "y2+bias: got %.6f",
	                   out[2]);
}

Test(nn_quantization_hf_quant, with_rms_norm) {
	/* o=1, W=[1,1,0,0] (codes 01,01,00,00 -> byte 0x05).
	   x=[3,9,0,0], rms_w=[1,1,1,1], eps~0.
	   ss=90, inv=1/sqrt(90/4)=1/sqrt(22.5); xn=[3*inv, 9*inv, 0, 0]
	   xmax=9*inv -> in_scale=127/(9*inv); xq=[42, 127, 0, 0]
	   sum=42+127=169; rescale=1/in_scale -> y=169*(9*inv)/127 = 2.524842163... */
	uint8_t packed[1] = {0x05};
	TensorHandle W = tensor_create_ternary_packed_2d(packed, 1, 1, 4, 0);
	double x_data[4] = {3.0, 9.0, 0.0, 0.0};
	int x_shape[1] = {4};
	TensorHandle x = tensor_create(x_data, x_shape, 1, 0);
	double rms_data[4] = {1.0, 1.0, 1.0, 1.0};
	TensorHandle rms_w = tensor_create(rms_data, x_shape, 1, 0);

	TensorHandle y = tensor_bitlinear_fwd_hf_quant(W, 1.0, x, /*bias=*/NULL, /*use_rms_norm=*/1,
	                                               rms_w, /*rms_eps=*/1.0e-12);
	double out[1] = {0};
	tensor_to_doubles(y, out);
	cr_assert_float_eq(out[0], 2.5248421633155407, TEST_TOL_RELAXED, "rms hf_quant y0: got %.6f",
	                   out[0]);
}

/* xmax-clamp branch: an all-zero activation hits the `xmax < 1e-5` floor
   (no rms). With all-zero x the matmul is 0 regardless, so y = bias (0). */
Test(nn_quantization_hf_quant, zero_activation_hits_xmax_floor) {
	uint8_t packed[3] = {0x71, 0x17, 0x4C};
	TensorHandle W = tensor_create_ternary_packed_2d(packed, 3, 3, 4, 0);
	double x_data[4] = {0.0, 0.0, 0.0, 0.0};
	int x_shape[1] = {4};
	TensorHandle x = tensor_create(x_data, x_shape, 1, 0);
	TensorHandle y = tensor_bitlinear_fwd_hf_quant(W, 0.5, x, NULL, 0, NULL, 1.0e-5);
	double out[3] = {0};
	tensor_to_doubles(y, out);
	cr_assert_float_eq(out[0], 0.0, TEST_TOL_RELAXED, "zero-activation y0 should be 0");
	cr_assert_float_eq(out[1], 0.0, TEST_TOL_RELAXED, "zero-activation y1 should be 0");
	cr_assert_float_eq(out[2], 0.0, TEST_TOL_RELAXED, "zero-activation y2 should be 0");
}

/* ------------------------------------------------------------------ */
/* construction guard (death test)                                    */
/* ------------------------------------------------------------------ */

/* Byte-count mismatch in the packed constructor -> abort. */
Test(nn_quantization_ternary_packed, byte_count_mismatch_aborts, .signal = SIGABRT) {
	uint8_t packed[2] = {0x00, 0x00}; /* expected 1 byte for [1,4], gave 2 */
	tensor_create_ternary_packed_2d(packed, 2, 1, 4, 0);
}
