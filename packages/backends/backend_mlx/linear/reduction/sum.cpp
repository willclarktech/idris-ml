/* tensor_sum / tensor_sum_dim for the mlx backend.
 *
 * OP_SUM has no metadata (the gradient is a uniform broadcast of the
 * scalar grad); OP_SUM_DIM carries the {dim, keepdim} pair via
 * SumDimReplayMeta so backward replay can unsqueeze + broadcast back. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"

extern "C" TensorHandle tensor_sum_mlx_streamed(TensorHandle h, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto t = (Tensor*)h;
    auto r = new Tensor(mx::sum(t->data), t->requires_grad);
    if (t->requires_grad) tape_append(OP_SUM, r, t, nullptr, 0);
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_sum(TensorHandle h) {
    return tensor_sum_mlx_streamed(h, default_stream_tag());
}

extern "C" TensorHandle tensor_sum_dim_mlx_streamed(TensorHandle h, int dim, int keepdim, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto t = (Tensor*)h;
    int rank = (int)t->data.ndim();
    int normalized = dim < 0 ? dim + rank : dim;
    auto r = new Tensor(
        mx::sum(t->data, std::vector<int>{normalized}, keepdim != 0),
        t->requires_grad);
    if (t->requires_grad) {
        int idx = tape_append(OP_SUM_DIM, r, t, nullptr, 0);
        auto meta = new SumDimReplayMeta{normalized, keepdim != 0 ? 1 : 0};
        tape[idx].meta = meta;
    }
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_sum_dim(TensorHandle h, int dim, int keepdim) {
    return tensor_sum_dim_mlx_streamed(h, dim, keepdim, default_stream_tag());
}
