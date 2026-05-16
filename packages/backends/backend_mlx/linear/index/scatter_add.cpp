/* tensor_scatter_add for the mlx backend.
 *
 * mx::scatter_add's updates shape is indices.shape + base.shape[axis+1:].
 * For a 1D base on axis 0 that's [N, 1] (the trailing 1 is the empty
 * remainder reified as a singleton) — hence the reshape. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"

extern "C" TensorHandle tensor_scatter_add_mlx_streamed(TensorHandle hindex, TensorHandle hsrc, int out_size, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto idx = (Tensor*)hindex;
    auto src = (Tensor*)hsrc;
    auto idx_int = mx::astype(idx->data, mx::int32);
    auto base = mx::zeros({out_size}, src->data.dtype());
    auto updates_2d = mx::reshape(src->data, {(int)src->data.size(), 1});
    auto result = mx::scatter_add(base, {idx_int}, updates_2d, std::vector<int>{0});
    auto r = new Tensor(result, src->requires_grad);
    if (src->requires_grad) tape_append(OP_SCATTER_ADD, r, src, idx, (double)out_size);
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_scatter_add(TensorHandle hindex, TensorHandle hsrc, int out_size) {
    return tensor_scatter_add_mlx_streamed(hindex, hsrc, out_size, default_stream_tag());
}
