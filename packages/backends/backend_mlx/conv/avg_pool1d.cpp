/* tensor_avg_pool1d for the mlx backend.
 *
 * No native avg_pool1d in mlx — built from a strided-slice fold: for
 * each kernel offset kl, slice the input with that offset + stride
 * and accumulate, then divide by kL. */
#include "../tensor.h"
#include "../tape.h"
#include "../stream.h"

extern "C" TensorHandle tensor_avg_pool1d_mlx_streamed(TensorHandle hinput, int kL, int stride, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto inp = (Tensor*)hinput;
    auto dt = inp->data.dtype();
    int C = (int)inp->data.shape(0), L = (int)inp->data.shape(1);
    int oL = (L - kL) / stride + 1;
    mx::array result = mx::zeros({C, oL}, dt);
    for (int kl = 0; kl < kL; kl++) {
        auto sliced = mx::slice(inp->data, {0, kl}, {C, kl + oL * stride}, {1, stride});
        result = mx::add(result, sliced);
    }
    result = mx::divide(result, mx::array((double)kL, dt));
    auto r = new Tensor(result, inp->requires_grad);
    if (inp->requires_grad) tape_append(OP_AVG_POOL1D, r, inp, nullptr, (double)kL + stride * 0.001);
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_avg_pool1d(TensorHandle hinput, int kL, int stride) {
    return tensor_avg_pool1d_mlx_streamed(hinput, kL, stride, default_stream_tag());
}
