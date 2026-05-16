/* tensor_sub for the mlx backend. See add.cpp for the streamed + tape
 * autograd pattern shared across binary elementwise ops. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"

extern "C" TensorHandle tensor_sub_mlx_streamed(TensorHandle ha, TensorHandle hb, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto a = (Tensor*)ha; auto b = (Tensor*)hb;
    bool rg = a->requires_grad || b->requires_grad;
    auto r = new Tensor(mx::subtract(a->data, b->data), rg);
    if (rg) tape_append(OP_SUB, r, a, b, 0);
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_sub(TensorHandle ha, TensorHandle hb) {
    return tensor_sub_mlx_streamed(ha, hb, default_stream_tag());
}
