/* Criterion suite for `tensor_sdpa_2d` (fused scaled dot-product attention).
 *
 * Forward-only: the fused op composes narrow/transpose/mm/softmax/mask
 * via direct C calls. It was at 0% line coverage — the probe self-matched
 * its impl and reported `tensor_sdpa_2d` "covered" while no test ran it.
 *
 * Covers: single-head non-causal vs a host reference, causal masking
 * (future positions zeroed), and GQA (numKvHeads < numHeads sharing KV).
 *
 * RED before this commit: each assertion drives tensor_sdpa_2d; e.g. the
 * causal test asserts out[row 0] == V[0] (row 0 attends only to key 0),
 * which fails if the mask is unapplied or the op is unrun.
 */

#include <math.h>
#include <string.h>
#include <criterion/criterion.h>
#include "test_helpers.h"

/* Host reference: single-head SDPA, no mask. q,k,v are [seq, hd] row-major. */
static void sdpa_ref(const double* q, const double* k, const double* v, int qs, int ks, int hd,
                     double* out) {
	double scale = 1.0 / sqrt((double)hd);
	for (int i = 0; i < qs; i++) {
		double scores[64];
		double maxs = -1e300;
		for (int j = 0; j < ks; j++) {
			double dot = 0;
			for (int d = 0; d < hd; d++)
				dot += q[i * hd + d] * k[j * hd + d];
			scores[j] = dot * scale;
			if (scores[j] > maxs) maxs = scores[j];
		}
		double sum = 0;
		for (int j = 0; j < ks; j++) {
			scores[j] = exp(scores[j] - maxs);
			sum += scores[j];
		}
		for (int d = 0; d < hd; d++) {
			double acc = 0;
			for (int j = 0; j < ks; j++)
				acc += (scores[j] / sum) * v[j * hd + d];
			out[i * hd + d] = acc;
		}
	}
}

Test(nn_attention_sdpa, single_head_noncausal_matches_ref) {
	double qd[] = {1.0, 0.0, 0.0, 1.0}; /* [2,2] */
	double kd[] = {1.0, 0.0, 0.0, 1.0};
	double vd[] = {1.0, 2.0, 3.0, 4.0};
	int s[] = {2, 2};
	TensorHandle q = tensor_create(qd, s, 2, 0);
	TensorHandle k = tensor_create(kd, s, 2, 0);
	TensorHandle v = tensor_create(vd, s, 2, 0);
	TensorHandle r = tensor_sdpa_2d(q, k, v, /*numHeads=*/1, /*numKvHeads=*/1, /*headDim=*/2,
	                                /*isCausal=*/0);
	double out[4];
	tensor_to_doubles(r, out);
	double ref[4];
	sdpa_ref(qd, kd, vd, 2, 2, 2, ref);
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(out[i], ref[i], TEST_TOL_RELAXED, "sdpa[%d]: got %.6f ref %.6f", i,
		                   out[i], ref[i]);
}

Test(nn_attention_sdpa, causal_row0_attends_only_to_key0) {
	/* q_seq == kv_seq == 2, causal: row 0 may attend only to key 0, so
	   out[row 0] == V[row 0] exactly (softmax over a single key = 1). */
	double qd[] = {1.0, 1.0, 2.0, 2.0};
	double kd[] = {0.5, 0.5, 1.5, 1.5};
	double vd[] = {10.0, 20.0, 30.0, 40.0};
	int s[] = {2, 2};
	TensorHandle q = tensor_create(qd, s, 2, 0);
	TensorHandle k = tensor_create(kd, s, 2, 0);
	TensorHandle v = tensor_create(vd, s, 2, 0);
	TensorHandle r = tensor_sdpa_2d(q, k, v, 1, 1, 2, /*isCausal=*/1);
	double out[4];
	tensor_to_doubles(r, out);
	cr_assert_float_eq(out[0], 10.0, TEST_TOL_RELAXED, "causal out[0,0] == V[0,0]");
	cr_assert_float_eq(out[1], 20.0, TEST_TOL_RELAXED, "causal out[0,1] == V[0,1]");
}

Test(nn_attention_sdpa, gqa_shared_kv) {
	/* numHeads=2, numKvHeads=1: both query heads share the single KV head.
	   The decode shape q_seq=1, kv_seq=2. With identical Q across the two
	   head slices, both heads produce the same per-head attention, so the
	   concatenated output is [h, h] — and each h matches the single-head
	   reference. */
	double qd[] = {1.0, 1.0, 1.0, 1.0}; /* [1, 4] = 2 heads x hd 2, slices equal */
	double kd[] = {1.0, 0.0, 0.0, 1.0}; /* [2, 2] = 1 kv head x hd 2, kv_seq=2 */
	double vd[] = {1.0, 2.0, 3.0, 4.0};
	int sq[] = {1, 4};
	int skv[] = {2, 2};
	TensorHandle q = tensor_create(qd, sq, 2, 0);
	TensorHandle k = tensor_create(kd, skv, 2, 0);
	TensorHandle v = tensor_create(vd, skv, 2, 0);
	TensorHandle r = tensor_sdpa_2d(q, k, v, /*numHeads=*/2, /*numKvHeads=*/1, /*headDim=*/2, 0);
	double out[4];
	tensor_to_doubles(r, out);
	double ref[2];
	double q_head[] = {1.0, 1.0};
	sdpa_ref(q_head, kd, vd, 1, 2, 2, ref); /* single shared head */
	double expected[] = {ref[0], ref[1], ref[0], ref[1]};
	for (int i = 0; i < 4; i++)
		cr_assert_float_eq(out[i], expected[i], TEST_TOL_RELAXED, "gqa out[%d]: got %.6f exp %.6f",
		                   i, out[i], expected[i]);
}
