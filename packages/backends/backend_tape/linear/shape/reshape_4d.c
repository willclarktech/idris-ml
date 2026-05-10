/* linear/shape/reshape_4d.c — 4D-shape reshape delegation.
 *
 * Phase 1b.9 (mechanical). Wraps tensor_reshape with a fixed-rank 4
 * shape vector; backward goes through OP_RESHAPE in reshape.c.
 */

#include "../../../backend.h"

TensorHandle tensor_reshape_4d(TensorHandle h, int d0, int d1, int d2, int d3) {
    int shape[] = {d0, d1, d2, d3};
    return tensor_reshape(h, shape, 4);
}
