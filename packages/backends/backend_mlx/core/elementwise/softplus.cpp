/* tensor_softplus for the mlx backend.
 *
 * Numerically stable softplus: max(0, x) + log(1 + exp(-|x|)). The
 * naive log(1 + exp(x)) overflows in float32 for x > ~88 — and the
 * NTM addressing path multiplies softplus(x) by cosine_sim and feeds
 * softmax, so an overflow there silently produces ±inf inputs to
 * softmax and the whole chain becomes NaN at the working point. The
 * stable form is correct for all x: for large positive x it reduces
 * to x, for large negative x it reduces to exp(x) ≈ 0. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"
#include "../../precision.h"

extern "C" TensorHandle tensor_softplus_mlx_streamed(TensorHandle h, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto t = (Tensor*)h;
    auto result = mx::add(mx::maximum(t->data, zero_like(t->data)),
                          mx::log(mx::add(one_like(t->data), mx::exp(mx::negative(mx::abs(t->data))))));
    auto r = new Tensor(result, t->requires_grad);
    if (t->requires_grad) tape_append(OP_SOFTPLUS, r, t, nullptr, 0);
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_softplus(TensorHandle h) {
    return tensor_softplus_mlx_streamed(h, default_stream_tag());
}
