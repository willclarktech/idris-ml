/* nn/attention/cross_attention.c — Q @ K^T * scale [+ mask] -> softmax -> @ V.
 *
 * Composed from existing ops (transpose_last2, mul_scalar,
 * bmm_3x3, masked_fill, softmax_3d). Backward is handled per-op by
 * the tape entries from each step — no dedicated OP for cross_attention,
 * so no backward arm to migrate.
 */

#include "../../tensor.h"
#include "../../../backend.h"

TensorHandle tensor_cross_attention(TensorHandle hQ, TensorHandle hK, TensorHandle hV,
                                    TensorHandle hmask, double scale) {
    TensorHandle KT = tensor_transpose_last2(hK);
    TensorHandle scores = tensor_mul_scalar(tensor_bmm_3x3(hQ, KT), scale);
    if (hmask) scores = tensor_masked_fill(scores, hmask, -1.0e20);
    TensorHandle attn = tensor_softmax_3d(scores);
    return tensor_bmm_3x3(attn, hV);
}
