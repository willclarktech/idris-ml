/* Criterion suite for tensor_embedding_2d (forward).
 *
 * The 2D-returning variant of tensor_embedding. Rather than returning
 * a flat [n * embedDim] handle that the Idris layer immediately
 * reshapes back to [n, embedDim] via primReshape2d, this variant
 * returns the natural [n, embedDim] shape from the underlying op
 * (torch::embedding / mx::take both return 2D natively).
 *
 * Replaces the primEmbedding + primReshape2d pair at every transformer
 * adapter's input layer (HfLlama, HfBert, HfGpt2, HfBitNet,
 * Layer/Transformer). Saves 1 op_count + 1 underlying backend op per
 * forward pass.
 */

#include <criterion/criterion.h>
#include <stdlib.h>
#include <string.h>
#include "backend.h"
#include "test_helpers.h"

static double* heap_copy(const double* src, int n) {
    double* buf = (double*)malloc(n * sizeof(double));
    memcpy(buf, src, n * sizeof(double));
    return buf;
}

Test(nn_attention_embedding_2d, shape_is_rank_2) {
    /* The whole point: rank should be 2, not 1. The pre-fusion
     * tensor_embedding returns rank 1 ([n * embedDim]); this is what
     * the new path is intended to fix. */
    param_clear();
    double w_d[] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        10.0, 11.0, 12.0,
    };
    double idx_d[] = {0.0, 2.0};
    TensorHandle w   = tensor_create_2d_f64(4, 3, heap_copy(w_d, 12), 0);
    TensorHandle idx = tensor_create_1d_f64(2, heap_copy(idx_d, 2), 0);
    TensorHandle r = tensor_embedding_2d(w, idx, 2, 3);
    cr_assert_eq(tensor_dim(r), 2,
        "tensor_embedding_2d output should be rank 2, got %d", tensor_dim(r));
    cr_assert_eq(tensor_size(r, 0), 2,
        "output dim 0 expected 2 (n), got %d", tensor_size(r, 0));
    cr_assert_eq(tensor_size(r, 1), 3,
        "output dim 1 expected 3 (embedDim), got %d", tensor_size(r, 1));
}

Test(nn_attention_embedding_2d, gathers_correct_rows) {
    /* Out[i, j] = weight[idx[i], j]. */
    param_clear();
    double w_d[] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        10.0, 11.0, 12.0,
    };
    double idx_d[] = {2.0, 0.0, 3.0};
    TensorHandle w   = tensor_create_2d_f64(4, 3, heap_copy(w_d, 12), 0);
    TensorHandle idx = tensor_create_1d_f64(3, heap_copy(idx_d, 3), 0);
    TensorHandle r = tensor_embedding_2d(w, idx, 3, 3);
    double got[9];
    tensor_to_doubles(r, got);
    double expected[] = {
        7.0, 8.0, 9.0,
        1.0, 2.0, 3.0,
        10.0, 11.0, 12.0,
    };
    for (int k = 0; k < 9; k++) {
        cr_assert_float_eq(got[k], expected[k], TEST_TOL_RELAXED,
            "embedding_2d[%d] expected %.3f got %.3f", k, expected[k], got[k]);
    }
}

Test(nn_attention_embedding_2d, single_row_lookup) {
    /* n=1, embedDim=4: degenerate case but must still return [1, 4]. */
    param_clear();
    double w_d[] = {
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
    };
    double idx_d[] = {1.0};
    TensorHandle w   = tensor_create_2d_f64(2, 4, heap_copy(w_d, 8), 0);
    TensorHandle idx = tensor_create_1d_f64(1, heap_copy(idx_d, 1), 0);
    TensorHandle r = tensor_embedding_2d(w, idx, 1, 4);
    cr_assert_eq(tensor_dim(r), 2,
        "single-row output should still be rank 2, got %d", tensor_dim(r));
    cr_assert_eq(tensor_size(r, 0), 1, "dim 0 should be 1");
    cr_assert_eq(tensor_size(r, 1), 4, "dim 1 should be 4");
    double got[4];
    tensor_to_doubles(r, got);
    cr_assert_float_eq(got[0], 5.0, TEST_TOL_RELAXED, "elt 0");
    cr_assert_float_eq(got[1], 6.0, TEST_TOL_RELAXED, "elt 1");
    cr_assert_float_eq(got[2], 7.0, TEST_TOL_RELAXED, "elt 2");
    cr_assert_float_eq(got[3], 8.0, TEST_TOL_RELAXED, "elt 3");
}

Test(nn_attention_embedding_2d, matches_decomposed_chain) {
    /* Strongest correctness check: fused 2D path must produce
     * identical values to the legacy flat path. */
    param_clear();
    double w_d[24];
    for (int i = 0; i < 24; i++) w_d[i] = (i * 0.13) - 1.5;
    double idx_d[] = {5.0, 0.0, 3.0, 7.0, 1.0};
    TensorHandle w1 = tensor_create_2d_f64(8, 3, heap_copy(w_d, 24), 0);
    TensorHandle w2 = tensor_create_2d_f64(8, 3, heap_copy(w_d, 24), 0);
    TensorHandle idx1 = tensor_create_1d_f64(5, heap_copy(idx_d, 5), 0);
    TensorHandle idx2 = tensor_create_1d_f64(5, heap_copy(idx_d, 5), 0);
    TensorHandle flat = tensor_embedding(w1, idx1, 5, 3);
    TensorHandle twoD = tensor_embedding_2d(w2, idx2, 5, 3);
    double flat_buf[15];
    double twoD_buf[15];
    tensor_to_doubles(flat, flat_buf);
    tensor_to_doubles(twoD, twoD_buf);
    for (int k = 0; k < 15; k++) {
        cr_assert_float_eq(twoD_buf[k], flat_buf[k], TEST_TOL_RELAXED,
            "embedding_2d[%d]=%.6f vs flat[%d]=%.6f", k, twoD_buf[k], k, flat_buf[k]);
    }
}
