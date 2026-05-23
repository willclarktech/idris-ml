/* tensor_sdpa_2d for the tape backend (TODO #399 Commit B).
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
 * Mask construction: if isCausal, build a strict-upper-triangle mask
 * tensor inline (Idris-side has `writeCausalMask`; the same loop here).
 * The mask flag indicates "where to clamp to -inf" — entries with
 * mask==1.0 get the fill value.
 *
 * I/O:
 *   Q : [seq, numHeads   * headDim]
 *   K : [seq, numKvHeads * headDim]
 *   V : [seq, numKvHeads * headDim]
 *   out [seq, numHeads * headDim]
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

/* Build a [seq, seq] strict-upper-triangle mask tensor in the arena.
 * mask[i, j] = 1.0 if j > i else 0.0. dtype follows Q's storage so the
 * masked_fill broadcast matches. */
static Tensor* build_causal_mask(int seq, int dtype_tag) {
    Tensor* m = arena_alloc(sizeof(Tensor));
    memset(m, 0, sizeof(Tensor));
    m->dtype_tag = dtype_tag;
    m->rank = 2;
    m->numel = seq * seq;
    m->shape = arena_alloc(2 * sizeof(int));
    m->shape[0] = seq;
    m->shape[1] = seq;
    m->requires_grad = 0;
    m->tape_idx = -1;
    if (dtype_tag == DT_F32) {
        float* d = arena_alloc((size_t)seq * (size_t)seq * sizeof(float));
        for (int i = 0; i < seq; i++)
            for (int j = 0; j < seq; j++)
                d[i * seq + j] = (j > i) ? 1.0f : 0.0f;
        m->data = d;
    } else {
        double* d = arena_alloc((size_t)seq * (size_t)seq * sizeof(double));
        for (int i = 0; i < seq; i++)
            for (int j = 0; j < seq; j++)
                d[i * seq + j] = (j > i) ? 1.0 : 0.0;
        m->data = d;
    }
    return m;
}

TensorHandle tensor_sdpa_2d(TensorHandle hq, TensorHandle hk, TensorHandle hv,
                            int numHeads, int numKvHeads, int headDim,
                            int isCausal) {
    Tensor* q = (Tensor*)hq;
    int seq = q->shape[0];
    double scale = 1.0 / sqrt((double)headDim);
    int kvRatio = numHeads / numKvHeads;  /* GQA: numHeads/numKvHeads query heads per KV head */

    Tensor* mask = isCausal ? build_causal_mask(seq, q->dtype_tag) : NULL;

    TensorHandle result = NULL;
    for (int h = 0; h < numHeads; h++) {
        int kvIdx = h / kvRatio;
        TensorHandle qh = tensor_narrow(hq, 1, h * headDim, headDim);     /* [seq, hd] */
        TensorHandle kh = tensor_narrow(hk, 1, kvIdx * headDim, headDim);
        TensorHandle vh = tensor_narrow(hv, 1, kvIdx * headDim, headDim);
        TensorHandle kt = tensor_transpose_2d(kh);                          /* [hd, seq] */
        TensorHandle scores = tensor_mm(qh, kt);                            /* [seq, seq] */
        scores = tensor_mul_scalar(scores, scale);
        if (isCausal) {
            scores = tensor_masked_fill(scores, (TensorHandle)mask, -1.0e20);
        }
        TensorHandle attn = tensor_softmax_2d(scores);                      /* [seq, seq] */
        TensorHandle out_h = tensor_mm(attn, vh);                           /* [seq, hd] */
        if (result == NULL) {
            result = out_h;
        } else {
            result = tensor_concat_2d_axis1(result, out_h);                 /* [seq, growing] cat along axis=1 */
        }
    }
    return result;
}
