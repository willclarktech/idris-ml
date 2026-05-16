/* tensor_mm and tensor_bmm for the mlx backend.
 *
 * Both back-end functions resolve to mx::matmul — mlx's matmul handles
 * 2D, 3D, and broadcast cases uniformly. OP_MM / OP_BMM tape entries
 * differ at backward replay (broadcast shape of the gradient). */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"

extern "C" TensorHandle tensor_mm_mlx_streamed(TensorHandle ha, TensorHandle hb, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto a = (Tensor*)ha; auto b = (Tensor*)hb;
    bool rg = a->requires_grad || b->requires_grad;
    auto r = new Tensor(mx::matmul(a->data, b->data), rg);
    if (rg) tape_append(OP_MM, r, a, b, 0);
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_mm(TensorHandle ha, TensorHandle hb) {
    return tensor_mm_mlx_streamed(ha, hb, default_stream_tag());
}

extern "C" TensorHandle tensor_bmm_mlx_streamed(TensorHandle ha, TensorHandle hb, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto a = (Tensor*)ha; auto b = (Tensor*)hb;
    bool rg = a->requires_grad || b->requires_grad;
    auto r = new Tensor(mx::matmul(a->data, b->data), rg);
    if (rg) tape_append(OP_BMM, r, a, b, 0);
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_bmm(TensorHandle ha, TensorHandle hb) {
    return tensor_bmm_mlx_streamed(ha, hb, default_stream_tag());
}

extern "C" TensorHandle tensor_bmm_3x3_mlx_streamed(TensorHandle ha, TensorHandle hb, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto a = (Tensor*)ha; auto b = (Tensor*)hb;
    bool rg = a->requires_grad || b->requires_grad;
    auto r = new Tensor(mx::matmul(a->data, b->data), rg);
    if (rg) tape_append(OP_BMM_3X3, r, a, b, 0);
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_bmm_3x3(TensorHandle ha, TensorHandle hb) {
    return tensor_bmm_3x3_mlx_streamed(ha, hb, default_stream_tag());
}
