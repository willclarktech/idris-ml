/* tensor_pow for the mlx backend (elementwise base ^ exp). */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"

extern "C" TensorHandle tensor_pow_mlx_streamed(TensorHandle hbase, TensorHandle hexp, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto b = (Tensor*)hbase; auto e = (Tensor*)hexp;
    bool rg = b->requires_grad || e->requires_grad;
    auto r = new Tensor(mx::power(b->data, e->data), rg);
    if (rg) tape_append(OP_POW, r, b, e, 0);
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_pow(TensorHandle hbase, TensorHandle hexp) {
    return tensor_pow_mlx_streamed(hbase, hexp, default_stream_tag());
}
