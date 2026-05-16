/* tensor_add for the mlx backend. Streamed variant opens the chosen
 * mx::StreamContext, computes the forward via mx::add, allocates a new
 * Tensor (refcount=0; the FFI wrap caller takes the first retain), and
 * appends the op to the tape when either input requires_grad. Backward
 * is replay-based — mx::vjp dispatches on OP_ADD during tape_backward. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"

extern "C" TensorHandle tensor_add_mlx_streamed(TensorHandle ha, TensorHandle hb, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto a = (Tensor*)ha; auto b = (Tensor*)hb;
    bool rg = a->requires_grad || b->requires_grad;
    auto r = new Tensor(mx::add(a->data, b->data), rg);
    if (rg) tape_append(OP_ADD, r, a, b, 0);
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_add(TensorHandle ha, TensorHandle hb) {
    return tensor_add_mlx_streamed(ha, hb, default_stream_tag());
}
