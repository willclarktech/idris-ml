/* tensor_max_pool1d for the mlx backend. Strided-slice fold with
 * mx::maximum — same window-by-window pattern as avg_pool. */
#include "../tensor.h"
#include "../tape.h"
#include "../stream.h"

extern "C" TensorHandle tensor_max_pool1d_mlx_streamed(TensorHandle hinput, int kL, int stride, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto inp = (Tensor*)hinput;
    int C = (int)inp->data.shape(0), L = (int)inp->data.shape(1);
    int oL = (L - kL) / stride + 1;

    mx::array result = mx::full({C, oL}, -1e30, inp->data.dtype());
    for (int kl = 0; kl < kL; kl++) {
        auto sliced = mx::slice(inp->data, {0, kl}, {C, kl + oL * stride}, {1, stride});
        result = mx::maximum(result, sliced);
    }

    auto r = new Tensor(result, inp->requires_grad);
    if (inp->requires_grad) {
        int idx = tape_append(OP_MAX_POOL1D, r, inp, nullptr, 0);
        auto* meta = new MaxPool1DReplayMeta();
        meta->C = C; meta->L = L; meta->kL = kL; meta->stride = stride; meta->oL = oL;
        tape[idx].meta = meta;
    }
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_max_pool1d(TensorHandle hinput, int kL, int stride) {
    return tensor_max_pool1d_mlx_streamed(hinput, kL, stride, default_stream_tag());
}
