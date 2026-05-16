/* tensor_embedding for the mlx backend.
 *
 * Indices are cast to int32 (mlx's take expects an integer index
 * array). Output is flattened to [n * embedDim] so the FFI consumer
 * sees a 1D buffer — the Idris layer reshapes back. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"

extern "C" TensorHandle tensor_embedding_mlx_streamed(TensorHandle hweight, TensorHandle hindices, int n, int embedDim, int stream_tag) {
    WITH_STREAM(stream_tag);
    (void)n;
    auto weight = (Tensor*)hweight;
    auto indices = (Tensor*)hindices;
    auto idx_int = mx::astype(indices->data, mx::int32);
    auto rows = mx::take(weight->data, idx_int, 0);
    auto result = mx::flatten(rows);

    auto r = new Tensor(result, weight->requires_grad);
    if (weight->requires_grad) {
        auto idx_t = new Tensor(idx_int, false);
        tape_append(OP_EMBEDDING, r, weight, idx_t, (double)embedDim);
    }
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_embedding(TensorHandle hweight, TensorHandle hindices, int n, int embedDim) {
    return tensor_embedding_mlx_streamed(hweight, hindices, n, embedDim, default_stream_tag());
}
