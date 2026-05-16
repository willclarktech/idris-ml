/* tensor_narrow for the mlx backend. Matches tape: flatten then slice
 * a 1D range. `dim` is accepted but ignored (the type-safe Idris
 * surface only narrows axis 0). */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"

extern "C" TensorHandle tensor_narrow_mlx_streamed(TensorHandle h, int dim, int start, int len, int stream_tag) {
    WITH_STREAM(stream_tag);
    (void)dim;
    auto t = (Tensor*)h;
    auto flat = mx::flatten(t->data);
    auto sliced = mx::slice(flat, mx::Shape{start}, mx::Shape{start + len});
    auto r = new Tensor(sliced, t->requires_grad);
    if (t->requires_grad) tape_append(OP_NARROW, r, t, nullptr, (double)start);
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_narrow(TensorHandle h, int dim, int start, int len) {
    return tensor_narrow_mlx_streamed(h, dim, start, len, default_stream_tag());
}
