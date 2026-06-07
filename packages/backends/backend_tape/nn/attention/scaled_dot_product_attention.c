/* tensor_sdpa_2d for the tape backend.
 *
 * tape has no fused MPSGraph kernel to call; the win on this backend
 * comes from collapsing the Idris-side per-head loop (32 FFI hops per
 * primitive * ~5 primitives per head * 16 layers = ~2500 Idris-to-C
 * dlsym walks per Llama forward) into a single FFI hop here that calls
 * the existing tape kernels directly via C function calls (no FFI). The
 * math is identical to oneHeadAttention's body; only the dispatch layer
 * changes.
 *
 * Composes via direct C calls (not FFI): tensor_narrow (per-head Q/K/V
 * slice), tensor_transpose_2d (K^T), tensor_mm (Q@K^T), tensor_mul_scalar
 * (1/sqrt(d) scale), tensor_masked_fill (causal mask), tensor_softmax_2d,
 * tensor_mm (attn@V), tensor_cat2 (accumulate per-head outputs along
 * axis=1).
 *
 * Mask construction: if isCausal, build a `[q_seq, kv_seq]` mask aligned
 * to the lower-right corner. mask[i, j] = 1.0 if j > (kv_seq - q_seq + i)
 * else 0.0 — entries with mask==1.0 get the fill value (-1e20) so the
 * softmax sends them to 0. This matches torch/mlx's `is_causal=true`
 * lower-right alignment under asymmetric Q/KV sequence dims (cache-
 * aware decode, where query position is offset by cache_len).
 *
 * I/O:
 *   Q : [q_seq,  numHeads   * headDim]
 *   K : [kv_seq, numKvHeads * headDim]
 *   V : [kv_seq, numKvHeads * headDim]
 *   out [q_seq, numHeads * headDim]
 *
 * For prefill / training, q_seq == kv_seq and the mask reduces to the
 * standard strict-upper-triangle form.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../arena.h"
#include "../../tensor.h"
#include "../../../backend.h"

extern TensorHandle tensor_narrow(TensorHandle h, int dim, int start, int len);
extern TensorHandle tensor_transpose_2d(TensorHandle h);
extern TensorHandle tensor_mm(TensorHandle a, TensorHandle b);
extern TensorHandle tensor_mul_scalar(TensorHandle h, double s);
extern TensorHandle tensor_masked_fill(TensorHandle h, TensorHandle mask, double val);
extern TensorHandle tensor_softmax_2d(TensorHandle h);
extern TensorHandle tensor_concat_2d_axis1(TensorHandle a, TensorHandle b);

/* Build a [q_seq, kv_seq] causal mask, lower-right aligned.
 * mask[i, j] = 1.0 if j > (kv_seq - q_seq + i) else 0.0. Under the
 * symmetric case (q_seq == kv_seq), the offset is 0 and this reduces
 * to mask[i, j] = 1.0 if j > i — the standard strict-upper-triangle.
 * Under cache-aware decode (q_seq=1, kv_seq=N), no positions get
 * masked (the single query attends to all history). dtype follows
 * Q's storage so the masked_fill broadcast matches. */
static Tensor* build_causal_mask(int q_seq, int kv_seq, int dtype_tag) {
	Tensor* m = arena_alloc(sizeof(Tensor));
	memset(m, 0, sizeof(Tensor));
	m->dtype_tag = dtype_tag;
	m->rank = 2;
	m->numel = q_seq * kv_seq;
	m->shape = arena_alloc(2 * sizeof(int));
	m->shape[0] = q_seq;
	m->shape[1] = kv_seq;
	m->requires_grad = 0;
	m->tape_idx = -1;
	int offset = kv_seq - q_seq;
	if (dtype_tag == DT_F32) {
		float* d = arena_alloc((size_t)q_seq * (size_t)kv_seq * sizeof(float));
		for (int i = 0; i < q_seq; i++)
			for (int j = 0; j < kv_seq; j++)
				d[i * kv_seq + j] = (j > offset + i) ? 1.0f : 0.0f;
		m->data = d;
	} else {
		double* d = arena_alloc((size_t)q_seq * (size_t)kv_seq * sizeof(double));
		for (int i = 0; i < q_seq; i++)
			for (int j = 0; j < kv_seq; j++)
				d[i * kv_seq + j] = (j > offset + i) ? 1.0 : 0.0;
		m->data = d;
	}
	return m;
}

TensorHandle tensor_sdpa_2d(TensorHandle hq, TensorHandle hk, TensorHandle hv, int numHeads,
                            int numKvHeads, int headDim, int isCausal) {
	Tensor* q = (Tensor*)hq;
	Tensor* k = (Tensor*)hk;
	int q_seq = q->shape[0];
	int kv_seq = k->shape[0];
	double scale = 1.0 / sqrt((double)headDim);
	int kvRatio = numHeads / numKvHeads; /* GQA: numHeads/numKvHeads query heads per KV head */

	Tensor* mask = isCausal ? build_causal_mask(q_seq, kv_seq, q->dtype_tag) : NULL;

	TensorHandle result = NULL;
	for (int h = 0; h < numHeads; h++) {
		int kvIdx = h / kvRatio;
		TensorHandle qh = tensor_narrow(hq, 1, h * headDim, headDim);     /* [q_seq, hd]  */
		TensorHandle kh = tensor_narrow(hk, 1, kvIdx * headDim, headDim); /* [kv_seq, hd] */
		TensorHandle vh = tensor_narrow(hv, 1, kvIdx * headDim, headDim);
		TensorHandle kt = tensor_transpose_2d(kh); /* [hd, kv_seq] */
		TensorHandle scores = tensor_mm(qh, kt);   /* [q_seq, kv_seq] */
		scores = tensor_mul_scalar(scores, scale);
		if (isCausal) {
			scores = tensor_masked_fill(scores, (TensorHandle)mask, -1.0e20);
		}
		TensorHandle attn = tensor_softmax_2d(scores); /* [q_seq, kv_seq] */
		TensorHandle out_h = tensor_mm(attn, vh);      /* [q_seq, hd] */
		if (result == NULL) {
			result = out_h;
		} else {
			result = tensor_concat_2d_axis1(result, out_h); /* [q_seq, growing] cat along axis=1 */
		}
	}
	return result;
}
