/* tensor_cumprod for the mlx backend. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"

extern "C" TensorHandle tensor_cumprod_mlx_streamed(TensorHandle ht, int dim, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto t = (Tensor*)ht;
    auto result = mx::cumprod(t->data, dim);
    auto r = new Tensor(result, t->requires_grad);
    if (r->requires_grad) tape_append(OP_CUMPROD, r, t, NULL, 0.0);
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_cumprod(TensorHandle ht, int dim) {
    return tensor_cumprod_mlx_streamed(ht, dim, default_stream_tag());
}
