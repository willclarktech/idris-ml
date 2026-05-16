/* tensor_conv1d_circular for the mlx backend.
 *
 * Circular correlation: out[i] = sum_j input[(i - k/2 + j + n) % n] * kernel[j].
 * mlx has no native circular conv1d — built from per-shift roll + multiply +
 * accumulate. OP_CONV1D_CIRC replay reproduces the rolls. */
#include "../tensor.h"
#include "../tape.h"
#include "../stream.h"

extern "C" TensorHandle tensor_conv1d_circular_mlx_streamed(TensorHandle hinput, TensorHandle hkernel, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto inp = (Tensor*)hinput; auto kern = (Tensor*)hkernel;
    int n = (int)inp->data.size();
    int k = (int)kern->data.size();

    mx::array result = mx::zeros({n}, inp->data.dtype());
    int half_k = k / 2;
    for (int j = 0; j < k; j++) {
        int shift = half_k - j;
        auto shifted = mx::roll(inp->data, shift);
        auto kern_j = mx::take(kern->data, mx::array(j));
        result = mx::add(result, mx::multiply(shifted, kern_j));
    }

    bool rg = inp->requires_grad || kern->requires_grad;
    auto r = new Tensor(result, rg);
    if (rg) tape_append(OP_CONV1D_CIRC, r, inp, kern, 0);
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_conv1d_circular(TensorHandle hinput, TensorHandle hkernel) {
    return tensor_conv1d_circular_mlx_streamed(hinput, hkernel, default_stream_tag());
}
