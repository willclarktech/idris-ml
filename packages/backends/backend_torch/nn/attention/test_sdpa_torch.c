/* torch-only Criterion suite for tensor_sdpa_2d.
 *
 * The common tape suite never reaches the asymmetric-causal branch in
 * attention/scaled_dot_product_attention.cpp (uncovered 69-79): the
 * explicit lower-right additive-mask construction that fires when
 * q_seq != kv_seq AND is_causal. That is the cache-aware decode path
 * (Q is a single new token, K/V cover the full history). These tests
 * drive both the symmetric route (is_causal kernel) and the asymmetric
 * route (explicit mask), plus the GQA branch (numHeads != numKvHeads ->
 * enable_gqa) and the non-causal route, asserting output shape and the
 * decode-attention values.
 *
 * torch CPU base dtype is F64; value asserts use whole numbers where the
 * softmax collapses to a clean average, else a relaxed tolerance.
 */

#include <criterion/criterion.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"
#include "test_helpers.h"

#ifdef BACKEND_TORCH

static double* hcopy(const double* src, int n) {
	double* buf = (double*)malloc(n * sizeof(double));
	memcpy(buf, src, n * sizeof(double));
	return buf;
}

/* Decode: q_seq=1, kv_seq=3, 1 head, headDim=2, causal. Drives the
   asymmetric-causal branch (lines 69-79): offset = kv_seq - q_seq = 2,
   the single query (absolute position 2) sees all 3 KV positions. So
   the output is the softmax-weighted sum over all 3 V rows — NOT
   collapsed to just V[0] (which is the bug the explicit mask fixes). */
Test(torch_nn_attention_sdpa, decode_causal_asymmetric_sees_all_kv) {
	/* Q [1, 1*2] all-zeros -> uniform attention (all scores 0). */
	double qd[] = {0.0, 0.0};
	/* K [3, 1*2] all-zeros -> every QK^T score is 0 -> softmax uniform. */
	double kd[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
	/* V [3, 1*2] distinct rows; uniform attention -> column means. */
	double vd[] = {1.0, 10.0, 2.0, 20.0, 3.0, 30.0};
	TensorHandle q = tensor_create_2d(1, 2, hcopy(qd, 2), /*rg=*/0);
	TensorHandle k = tensor_create_2d(3, 2, hcopy(kd, 6), /*rg=*/0);
	TensorHandle v = tensor_create_2d(3, 2, hcopy(vd, 6), /*rg=*/0);
	TensorHandle out =
	    tensor_sdpa_2d(q, k, v, /*numHeads=*/1, /*numKvHeads=*/1, /*headDim=*/2, /*isCausal=*/1);
	cr_assert_eq(tensor_dim(out), 2, "out rank should be 2");
	cr_assert_eq(tensor_size(out, 0), 1, "out q_seq should be 1");
	cr_assert_eq(tensor_size(out, 1), 2, "out feature dim should be 2");
	double buf[2];
	tensor_to_doubles(out, buf);
	/* mean of column 0 = (1+2+3)/3 = 2, column 1 = (10+20+30)/3 = 20.
	   If the bug were present (collapse to V[0]) we'd get {1, 10}. */
	cr_assert_float_eq(buf[0], 2.0, TEST_TOL_RELAXED, "col0 mean exp 2 got %.6f", buf[0]);
	cr_assert_float_eq(buf[1], 20.0, TEST_TOL_RELAXED, "col1 mean exp 20 got %.6f", buf[1]);
}

/* Symmetric causal (q_seq == kv_seq): stays on the is_causal kernel
   route (causal_flag stays true, attn_mask stays nullopt). First query
   sees only K[0]; with all-zero scores its output is exactly V[0]. */
Test(torch_nn_attention_sdpa, prefill_causal_symmetric_first_row_is_v0) {
	double qd[] = {0.0, 0.0, 0.0, 0.0};
	double kd[] = {0.0, 0.0, 0.0, 0.0};
	double vd[] = {5.0, 50.0, 7.0, 70.0};
	TensorHandle q = tensor_create_2d(2, 2, hcopy(qd, 4), /*rg=*/0);
	TensorHandle k = tensor_create_2d(2, 2, hcopy(kd, 4), /*rg=*/0);
	TensorHandle v = tensor_create_2d(2, 2, hcopy(vd, 4), /*rg=*/0);
	TensorHandle out =
	    tensor_sdpa_2d(q, k, v, /*numHeads=*/1, /*numKvHeads=*/1, /*headDim=*/2, /*isCausal=*/1);
	cr_assert_eq(tensor_size(out, 0), 2, "out q_seq should be 2");
	double buf[4];
	tensor_to_doubles(out, buf);
	/* Row 0 sees only V[0] = {5, 50}. */
	cr_assert_float_eq(buf[0], 5.0, TEST_TOL_RELAXED, "row0 col0 exp 5 got %.6f", buf[0]);
	cr_assert_float_eq(buf[1], 50.0, TEST_TOL_RELAXED, "row0 col1 exp 50 got %.6f", buf[1]);
	/* Row 1 sees both V rows uniformly -> means {6, 60}. */
	cr_assert_float_eq(buf[2], 6.0, TEST_TOL_RELAXED, "row1 col0 exp 6 got %.6f", buf[2]);
	cr_assert_float_eq(buf[3], 60.0, TEST_TOL_RELAXED, "row1 col1 exp 60 got %.6f", buf[3]);
}

/* Non-causal symmetric: every query sees every key (causal_flag false,
   no mask). Both rows -> uniform average over both V rows. */
Test(torch_nn_attention_sdpa, non_causal_full_attention) {
	double qd[] = {0.0, 0.0, 0.0, 0.0};
	double kd[] = {0.0, 0.0, 0.0, 0.0};
	double vd[] = {2.0, 4.0, 8.0, 16.0};
	TensorHandle q = tensor_create_2d(2, 2, hcopy(qd, 4), /*rg=*/0);
	TensorHandle k = tensor_create_2d(2, 2, hcopy(kd, 4), /*rg=*/0);
	TensorHandle v = tensor_create_2d(2, 2, hcopy(vd, 4), /*rg=*/0);
	TensorHandle out =
	    tensor_sdpa_2d(q, k, v, /*numHeads=*/1, /*numKvHeads=*/1, /*headDim=*/2, /*isCausal=*/0);
	double buf[4];
	tensor_to_doubles(out, buf);
	/* col means: (2+8)/2=5, (4+16)/2=10; identical for both rows. */
	cr_assert_float_eq(buf[0], 5.0, TEST_TOL_RELAXED, "row0 col0 exp 5 got %.6f", buf[0]);
	cr_assert_float_eq(buf[1], 10.0, TEST_TOL_RELAXED, "row0 col1 exp 10 got %.6f", buf[1]);
	cr_assert_float_eq(buf[2], 5.0, TEST_TOL_RELAXED, "row1 col0 exp 5 got %.6f", buf[2]);
	cr_assert_float_eq(buf[3], 10.0, TEST_TOL_RELAXED, "row1 col1 exp 10 got %.6f", buf[3]);
}

/* GQA: numHeads=2, numKvHeads=1 (the enable_gqa=true branch). headDim=2,
   so Q feature dim = 2*2 = 4, K/V feature dim = 1*2 = 2, single KV head
   broadcast to both query heads. All-zero scores -> each head averages
   the (single) KV head's V rows. q_seq=kv_seq=2, non-causal. */
Test(torch_nn_attention_sdpa, gqa_broadcast_single_kv_head) {
	double qd[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; /* [2, 4] */
	double kd[] = {0.0, 0.0, 0.0, 0.0};                     /* [2, 2] */
	double vd[] = {1.0, 100.0, 3.0, 300.0};                 /* [2, 2] */
	TensorHandle q = tensor_create_2d(2, 4, hcopy(qd, 8), /*rg=*/0);
	TensorHandle k = tensor_create_2d(2, 2, hcopy(kd, 4), /*rg=*/0);
	TensorHandle v = tensor_create_2d(2, 2, hcopy(vd, 4), /*rg=*/0);
	TensorHandle out =
	    tensor_sdpa_2d(q, k, v, /*numHeads=*/2, /*numKvHeads=*/1, /*headDim=*/2, /*isCausal=*/0);
	cr_assert_eq(tensor_size(out, 1), 4, "out feature dim should be 2*2=4");
	double buf[8];
	tensor_to_doubles(out, buf);
	/* Each of the 2 query heads sees the same single KV head -> col means
	   {2, 200}. Output [q_seq=2, nH*hd=4]; both heads identical, both
	   rows identical. */
	for (int r = 0; r < 2; r++) {
		cr_assert_float_eq(buf[r * 4 + 0], 2.0, TEST_TOL_RELAXED, "h0 col0 exp 2 got %.6f",
		                   buf[r * 4 + 0]);
		cr_assert_float_eq(buf[r * 4 + 1], 200.0, TEST_TOL_RELAXED, "h0 col1 exp 200 got %.6f",
		                   buf[r * 4 + 1]);
		cr_assert_float_eq(buf[r * 4 + 2], 2.0, TEST_TOL_RELAXED, "h1 col0 exp 2 got %.6f",
		                   buf[r * 4 + 2]);
		cr_assert_float_eq(buf[r * 4 + 3], 200.0, TEST_TOL_RELAXED, "h1 col1 exp 200 got %.6f",
		                   buf[r * 4 + 3]);
	}
}

#endif /* BACKEND_TORCH */
