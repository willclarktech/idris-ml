/* linear/shape/reshape_3d.c — 3D-shape reshape delegation.
 *
 * Phase 1b.9 (mechanical). Wraps tensor_reshape with a fixed-rank 3
 * shape vector; backward goes through OP_RESHAPE in reshape.c.
 */

#include "../../../backend.h"

TensorHandle tensor_reshape_3d(TensorHandle h, int d0, int d1, int d2) {
    int shape[] = {d0, d1, d2};
    return tensor_reshape(h, shape, 3);
}
