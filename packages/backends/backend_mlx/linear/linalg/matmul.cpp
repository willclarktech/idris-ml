/* tensor_matmul for the mlx backend. mx::matmul handles arbitrary rank
 * (broadcasts the leading dims) — backward replay reuses OP_MM since
 * both forms are mathematically equivalent for vjp purposes. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"

extern "C" TensorHandle tensor_matmul_mlx_streamed(TensorHandle ha, TensorHandle hb, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto a = (Tensor*)ha; auto b = (Tensor*)hb;
    bool rg = a->requires_grad || b->requires_grad;
    auto r = new Tensor(mx::matmul(a->data, b->data), rg);
    if (rg) tape_append(OP_MM, r, a, b, 0);
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_matmul(TensorHandle ha, TensorHandle hb) {
    return tensor_matmul_mlx_streamed(ha, hb, default_stream_tag());
}
