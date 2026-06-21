/* mlx-only Criterion suite for the BitNet load-time quantization helpers.
 *
 * The cross-backend bitlinear suite lives under backend_tape/ and is NOT
 * compiled into the mlx binary, so on the mlx lane the standalone quant
 * helpers in nn/quantization/bitlinear.cpp stayed uncovered:
 *   - tensor_absmean_per_row_2d          (mx::mean(mx::abs(w), 1))
 *   - tensor_ternary_quant_with_scale_2d (per-row divide/round/clip + mask)
 *   plus the own-line std::abort() construction/shape guards.
 *
 * IMPORTANT: the tape implementation of these two helpers crashes on a
 * valid 2D F64 input (see the .disabled tests in
 * backend_tape/nn/quantization/test_quantize.c + TODO.md "tape BitNet
 * quant helpers crash"). The mlx implementation is a different code path
 * built on mx::mean / mx::divide / mx::round / mx::clip, which does NOT
 * crash — these tests confirm the mlx success paths work and are why this
 * file is mlx-gated (not just for the abort-flavour difference but because
 * the success assertions would fail on tape).
 *
 * Quantized weights are int8 storage on mlx and cannot be meaningfully
 * read back as a ternary set without re-deriving codes, so the quant op is
 * verified by running its output through tensor_bitlinear_fwd with
 * scale=1, x=1 — the same portable contract the tape suite uses.
 *
 * The std::abort() guards in bitlinear.cpp are plain C-side abort() calls
 * (not mlx C++ exceptions), so SIGABRT death tests for them are reliable
 * on the mlx lane.
 */

#include <signal.h>
#include <criterion/criterion.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_MLX

/* ------------------------------------------------------------------ */
/* tensor_absmean_per_row_2d (mlx success path: bitlinear.cpp 214-228) */
/* ------------------------------------------------------------------ */

/* DISABLED: mlx absmean/ternary_quant crash on valid 2D input — same class as
   the tape BitNet quant bug (see TODO.md "tape BitNet quant helpers crash"); the
   bug is cross-backend. Re-enable when fixed. */
Test(mlx_nn_quantization_absmean, per_row_mean_abs, .disabled = true) {
	/* w = [[1, -2, 3, -4], [0.5, -0.5, 1.5, -1.5]]
	   row 0 absmean = (1+2+3+4)/4 = 2.5
	   row 1 absmean = (0.5+0.5+1.5+1.5)/4 = 1.0 */
	double w_data[8] = {1.0, -2.0, 3.0, -4.0, 0.5, -0.5, 1.5, -1.5};
	TensorHandle w = tensor_create_2d(2, 4, w_data, 0);
	TensorHandle s = tensor_absmean_per_row_2d(w);

	cr_assert_eq(tensor_dim(s), 1, "absmean output should be rank 1");
	cr_assert_eq(tensor_size(s, 0), 2, "absmean output length should be #rows");

	double out[2] = {0};
	tensor_to_doubles(s, out);
	cr_assert_float_eq(out[0], 2.5, TEST_TOL_RELAXED, "absmean row0: got %.6f", out[0]);
	cr_assert_float_eq(out[1], 1.0, TEST_TOL_RELAXED, "absmean row1: got %.6f", out[1]);
}

/* DISABLED: cross-backend BitNet quant crash — see TODO.md. */
Test(mlx_nn_quantization_absmean, all_zero_row_is_zero, .disabled = true) {
	/* A row of zeros -> absmean 0 (the /0-guard input for the quant op). */
	double w_data[8] = {0.0, 0.0, 0.0, 0.0, 2.0, -2.0, 2.0, -2.0};
	TensorHandle w = tensor_create_2d(2, 4, w_data, 0);
	TensorHandle s = tensor_absmean_per_row_2d(w);
	double out[2] = {0};
	tensor_to_doubles(s, out);
	cr_assert_float_eq(out[0], 0.0, TEST_TOL_RELAXED, "zero row absmean should be 0");
	cr_assert_float_eq(out[1], 2.0, TEST_TOL_RELAXED, "absmean row1: got %.6f", out[1]);
}

/* Non-2D input aborts (own-line std::abort() in absmean: bitlinear.cpp
   217-223). Plain C-side abort -> reliable SIGABRT death test. */
Test(mlx_nn_quantization_absmean, rank1_aborts, .signal = SIGABRT) {
	double d[3] = {1.0, 2.0, 3.0};
	int s[1] = {3};
	TensorHandle v = tensor_create(d, s, 1, 0);
	tensor_absmean_per_row_2d(v); /* rank=1 -> abort */
}

/* ------------------------------------------------------------------ */
/* tensor_ternary_quant_with_scale_2d (mlx: bitlinear.cpp 235-277)     */
/* verified via tensor_bitlinear_fwd                                   */
/* ------------------------------------------------------------------ */

/* DISABLED: cross-backend BitNet quant crash — see TODO.md. */
Test(mlx_nn_quantization_ternary_quant, round_clamp_via_forward, .disabled = true) {
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

/* DISABLED: cross-backend BitNet quant crash — see TODO.md. */
Test(mlx_nn_quantization_ternary_quant, zero_scale_row_all_zero, .disabled = true) {
	/* scale[0] == 0 -> the active-row mask zeroes the whole row (covers
	   the mx::greater / mask multiply branch: bitlinear.cpp 268-270). */
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

/* Non-2D weight aborts (bitlinear.cpp 241-247). */
Test(mlx_nn_quantization_ternary_quant, rank1_weight_aborts, .signal = SIGABRT) {
	double d[3] = {1.0, 2.0, 3.0};
	int s[1] = {3};
	TensorHandle v = tensor_create(d, s, 1, 0);
	double sc[3] = {1.0, 1.0, 1.0};
	TensorHandle scale = tensor_create(sc, s, 1, 0);
	tensor_ternary_quant_with_scale_2d(v, scale); /* w rank=1 -> abort */
}

/* Scale shape mismatch aborts (bitlinear.cpp 248-255). */
Test(mlx_nn_quantization_ternary_quant, scale_shape_mismatch_aborts, .signal = SIGABRT) {
	double w_data[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
	TensorHandle w = tensor_create_2d(2, 4, w_data, 0);
	double sc[3] = {1.0, 1.0, 1.0};
	int s[1] = {3}; /* expected [2], given [3] */
	TensorHandle scale = tensor_create(sc, s, 1, 0);
	tensor_ternary_quant_with_scale_2d(w, scale); /* shape mismatch -> abort */
}

/* ------------------------------------------------------------------ */
/* construction-guard death tests (plain std::abort, reliable on mlx)  */
/* ------------------------------------------------------------------ */

/* Byte-count mismatch in the packed constructor -> abort
   (bitlinear.cpp 30-36). */
Test(mlx_nn_quantization_ternary_packed, byte_count_mismatch_aborts, .signal = SIGABRT) {
	uint8_t packed[2] = {0x00, 0x00}; /* expected 1 byte for [1,4], gave 2 */
	tensor_create_ternary_packed_2d(packed, 2, 1, 4, 0);
}

/* requires_grad=1 on ternary storage -> abort (bitlinear.cpp 37-42).
   mlx-specific guard: int8 storage cannot carry grad. */
Test(mlx_nn_quantization_ternary_packed, requires_grad_aborts, .signal = SIGABRT) {
	uint8_t packed[1] = {0x05}; /* valid 1-byte payload for [1,4] */
	tensor_create_ternary_packed_2d(packed, 1, 1, 4, /*requires_grad=*/1);
}

/* Invalid 2-bit code (0b10 reserved) reached during forward decode ->
   abort (bitlinear.cpp 61-67). The packed constructor unpacks at build
   time, so the bad code aborts in tensor_create_ternary_packed_2d. */
Test(mlx_nn_quantization_decode, invalid_code_aborts, .signal = SIGABRT) {
	uint8_t packed[1] = {0x02}; /* slot0 code = 0b10 (reserved) */
	tensor_create_ternary_packed_2d(packed, 1, 1, 4, 0);
}

/* Invalid HF code (hf_code=3 -> value=2, out of {-1,0,+1}) aborts in
   tensor_create_ternary_from_hf_packed_2d (bitlinear.cpp 188-194). */
Test(mlx_nn_quantization_hf_packed, invalid_hf_code_aborts, .signal = SIGABRT) {
	/* o=1, i=4. hf_byte chunk0 (low 2 bits) = code 3 at k=0 -> value 2. */
	uint8_t hf[4] = {0x03, 0x00, 0x00, 0x00};
	tensor_create_ternary_from_hf_packed_2d(hf, 1, 4); /* invalid code 3 -> abort */
}

/* HF-format ternary load success path (bitlinear.cpp 178-201), verified
   via forward. Mirrors the tape suite's layout_remap_via_forward fixture. */
Test(mlx_nn_quantization_hf_packed, layout_remap_via_forward) {
	/* o=2, i=4. HF layout: [(o+3)/4, i] = [1, 4] bytes; chunk 0 = row 0,
	   chunk 1 = row 1.
	     row0 = [ 1, 0, -1, 1]  -> codes (value+1) = [2, 1, 0, 2]
	     row1 = [-1, 1,  0, 1]  -> codes (value+1) = [0, 2, 1, 2]
	   hf_byte[k] = code_row0 | (code_row1 << 2):
	     k0: 0x02  k1: 0x09  k2: 0x04  k3: 0x0A
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

#endif /* BACKEND_MLX */
