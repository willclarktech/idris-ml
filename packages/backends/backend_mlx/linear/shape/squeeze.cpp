/* tensor_squeeze / tensor_unsqueeze for the mlx backend.
 *
 * mlx has no native squeeze/unsqueeze — both are implemented as
 * reshape on a recomputed shape. backward replays a single
 * OP_RESHAPE that reconstructs the original shape, so no dedicated
 * OP_SQUEEZE / OP_UNSQUEEZE opcodes are needed. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"

extern "C" TensorHandle tensor_reshape_mlx_streamed(TensorHandle h, int* shape, int rank, int stream_tag);
extern "C" TensorHandle tensor_clone_mlx_streamed(TensorHandle h, int stream_tag);

extern "C" TensorHandle tensor_unsqueeze_mlx_streamed(TensorHandle h, int dim, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto t = (Tensor*)h;
    const auto& orig = t->data.shape();
    int rank = (int)orig.size();
    std::vector<int> new_dims;
    new_dims.reserve(rank + 1);
    for (int i = 0; i <= rank; i++) {
        if (i == dim) new_dims.push_back(1);
        if (i < rank) new_dims.push_back(orig[i]);
    }
    mx::Shape sh(new_dims.begin(), new_dims.end());
    auto r = new Tensor(mx::reshape(t->data, sh), t->requires_grad);
    if (t->requires_grad) tape_append(OP_RESHAPE, r, t, nullptr, 0);
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_unsqueeze(TensorHandle h, int dim) {
    return tensor_unsqueeze_mlx_streamed(h, dim, default_stream_tag());
}

extern "C" TensorHandle tensor_squeeze_mlx_streamed(TensorHandle h, int dim, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto t = (Tensor*)h;
    int rank = (int)t->data.ndim();
    int normalized = dim < 0 ? dim + rank : dim;
    /* No-op if dim is out of range or not size 1 — matches torch's .squeeze(dim) */
    if (normalized < 0 || normalized >= rank || (int)t->data.shape(normalized) != 1) {
        return tensor_clone_mlx_streamed(h, stream_tag);
    }
    std::vector<int> new_shape;
    new_shape.reserve(rank - 1);
    for (int i = 0; i < rank; i++) {
        if (i != normalized) new_shape.push_back((int)t->data.shape(i));
    }
    return tensor_reshape_mlx_streamed(h, new_shape.data(), (int)new_shape.size(), stream_tag);
}

extern "C" TensorHandle tensor_squeeze(TensorHandle h, int dim) {
    return tensor_squeeze_mlx_streamed(h, dim, default_stream_tag());
}
