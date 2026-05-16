/* tensor_dot for the mlx backend.
 *
 * No native mx::dot; expressed as sum(multiply). Backward replays the
 * pair OP_MUL+OP_SUM rather than a dedicated OP_DOT — both share the
 * same vjp shape. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"

extern "C" TensorHandle tensor_dot_mlx_streamed(TensorHandle ha, TensorHandle hb, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto a = (Tensor*)ha; auto b = (Tensor*)hb;
    bool rg = a->requires_grad || b->requires_grad;
    auto r = new Tensor(mx::sum(mx::multiply(a->data, b->data)), rg);
    if (rg) {
        auto prod = new Tensor(mx::multiply(a->data, b->data), rg);
        tape_append(OP_MUL, prod, a, b, 0);
        tape_append(OP_SUM, r, prod, nullptr, 0);
    }
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_dot(TensorHandle ha, TensorHandle hb) {
    return tensor_dot_mlx_streamed(ha, hb, default_stream_tag());
}
