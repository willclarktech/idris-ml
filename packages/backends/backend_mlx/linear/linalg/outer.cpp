/* tensor_outer for the mlx backend. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"

extern "C" TensorHandle tensor_outer_mlx_streamed(TensorHandle ha, TensorHandle hb, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto a = (Tensor*)ha; auto b = (Tensor*)hb;
    bool rg = a->requires_grad || b->requires_grad;
    auto r = new Tensor(mx::outer(a->data, b->data), rg);
    if (rg) tape_append(OP_OUTER, r, a, b, 0);
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_outer(TensorHandle ha, TensorHandle hb) {
    return tensor_outer_mlx_streamed(ha, hb, default_stream_tag());
}
