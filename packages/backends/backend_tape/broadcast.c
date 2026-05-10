/* backend_tape/broadcast.c — numpy-style broadcasting helpers.
 *
 * Phase 1e.10. Implementations of the three functions declared in
 * broadcast.h. Used by elementwise backward kernels (and a few
 * reduction ones) to walk an output's flat index back into each
 * operand's flat index given right-aligned broadcast strides.
 */

#include "broadcast.h"
#include "tensor.h"

int shapes_equal(Tensor* a, Tensor* r) {
    if (a->numel != r->numel || a->rank != r->rank) return 0;
    for (int k = 0; k < a->rank; k++) {
        if (a->shape[k] != r->shape[k]) return 0;
    }
    return 1;
}

int compute_bcast_shape(Tensor* a, Tensor* b,
                        int* r_shape, int* r_rank, int* r_numel) {
    int rank = a->rank > b->rank ? a->rank : b->rank;
    if (rank > MAX_BCAST_RANK) return 0;
    int numel = 1;
    for (int k = rank - 1; k >= 0; k--) {
        int ai = k - (rank - a->rank);
        int bi = k - (rank - b->rank);
        int sa = (ai >= 0) ? a->shape[ai] : 1;
        int sb = (bi >= 0) ? b->shape[bi] : 1;
        if (sa != sb && sa != 1 && sb != 1) return 0;
        int so = sa > sb ? sa : sb;
        r_shape[k] = so;
        numel *= so;
    }
    *r_rank = rank;
    *r_numel = numel;
    return 1;
}

void compute_bcast_strides(Tensor* a, int r_rank, int* r_shape,
                           int* out_strides) {
    int a_rank = a->rank;
    int natural[MAX_BCAST_RANK];
    int s = 1;
    for (int k = a_rank - 1; k >= 0; k--) { natural[k] = s; s *= a->shape[k]; }
    int offset = r_rank - a_rank;
    for (int j = 0; j < r_rank; j++) {
        int ai = j - offset;
        if (ai < 0) {
            out_strides[j] = 0;  /* phantom dim from rank-padding */
        } else {
            int sa = a->shape[ai];
            int sr = r_shape[j];
            out_strides[j] = (sa == 1 && sr > 1) ? 0 : natural[ai];
        }
    }
}
