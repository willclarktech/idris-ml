/* tensor_select for the mlx backend.
 *
 * mx::take(arr, indices_array, axis) returns a slice with that axis
 * removed when indices is a scalar — matches torch's .select() shape.
 * OP_SELECT carries the `index` in scalar_arg for backward replay. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"

extern "C" TensorHandle tensor_select_mlx_streamed(TensorHandle h, int dim, int index, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto t = (Tensor*)h;
    auto r = new Tensor(mx::take(t->data, mx::array(index), dim), t->requires_grad);
    if (t->requires_grad) tape_append(OP_SELECT, r, t, nullptr, (double)index);
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_select(TensorHandle h, int dim, int index) {
    return tensor_select_mlx_streamed(h, dim, index, default_stream_tag());
}
