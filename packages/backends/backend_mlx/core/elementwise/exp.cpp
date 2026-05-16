/* tensor_exp for the mlx backend. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"

extern "C" TensorHandle tensor_exp_mlx_streamed(TensorHandle h, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto t = (Tensor*)h;
    auto r = new Tensor(mx::exp(t->data), t->requires_grad);
    if (t->requires_grad) tape_append(OP_EXP, r, t, nullptr, 0);
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_exp(TensorHandle h) {
    return tensor_exp_mlx_streamed(h, default_stream_tag());
}
