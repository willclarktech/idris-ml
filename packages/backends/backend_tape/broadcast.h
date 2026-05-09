/* backend_tape/broadcast.h — numpy-style broadcasting helpers used
 * across elementwise backward kernels (and a few reduction ones).
 *
 * Phase 1a.2: extracted as extern from backend_tape.c lines 142-194.
 * Each per-op file that needs broadcast routing includes this header.
 */

#ifndef IDRISML_BACKEND_TAPE_BROADCAST_H
#define IDRISML_BACKEND_TAPE_BROADCAST_H

#include "tensor.h"

#define MAX_BCAST_RANK 8

/* True if `a`'s shape exactly matches `r`'s shape (no broadcast). */
int shapes_equal(Tensor* a, Tensor* r);

/* Compute broadcast output shape from a and b (right-aligned, numpy rules).
   Returns 1 on success, 0 on incompatible shapes. */
int compute_bcast_shape(Tensor* a, Tensor* b,
                        int* r_shape, int* r_rank, int* r_numel);

/* Compute right-aligned broadcast strides for `a` w.r.t. output shape r_shape.
   out_strides[k] is the increment to a's flat index when output dim k advances;
   0 means broadcast on that dim. */
void compute_bcast_strides(Tensor* a, int r_rank, int* r_shape,
                           int* out_strides);

#endif /* IDRISML_BACKEND_TAPE_BROADCAST_H */
