/* tensor_concat_2d_axis1 for the mlx backend.
 * A: [m, n], B: [m, k] -> [m, n+k] along axis 1. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"

extern "C" TensorHandle tensor_concat_2d_axis1_mlx_streamed(TensorHandle hA, TensorHandle hB, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto A = (Tensor*)hA; auto B = (Tensor*)hB;
    bool rg = A->requires_grad || B->requires_grad;
    auto r = new Tensor(mx::concatenate({A->data, B->data}, 1), rg);
    if (rg) tape_append(OP_CONCAT_2D_AXIS1, r, A, B, 0);
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_concat_2d_axis1(TensorHandle hA, TensorHandle hB) {
    return tensor_concat_2d_axis1_mlx_streamed(hA, hB, default_stream_tag());
}
