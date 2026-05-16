/* tensor_masked_fill for the mlx backend.
 *
 * mlx has no native masked_fill — built from mx::where on a same-shape
 * `value` constant. OP_MASKED_FILL replay reproduces the where call. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"

extern "C" TensorHandle tensor_masked_fill_mlx_streamed(TensorHandle h, TensorHandle hmask, double value, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto t = (Tensor*)h; auto mask = (Tensor*)hmask;
    auto val_arr = mx::full(t->data.shape(), value, t->data.dtype());
    auto r = new Tensor(mx::where(mask->data, val_arr, t->data), t->requires_grad);
    if (t->requires_grad) tape_append(OP_MASKED_FILL, r, t, mask, 0);
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_masked_fill(TensorHandle h, TensorHandle hmask, double value) {
    return tensor_masked_fill_mlx_streamed(h, hmask, value, default_stream_tag());
}
