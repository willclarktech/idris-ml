/* tensor_gather for the mlx backend. mx::take requires int32 indices —
 * coerce on the way in. OP_GATHER replay reproduces the take from the
 * stored index tensor (arg2). */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"

extern "C" TensorHandle tensor_gather_mlx_streamed(TensorHandle hinput, TensorHandle hindex, int n, int stream_tag) {
    WITH_STREAM(stream_tag);
    (void)n;
    auto inp = (Tensor*)hinput;
    auto idx = (Tensor*)hindex;
    auto idx_int = mx::astype(idx->data, mx::int32);
    auto result = mx::take(inp->data, idx_int, 0);
    auto r = new Tensor(result, inp->requires_grad);
    if (inp->requires_grad) tape_append(OP_GATHER, r, inp, idx, 0);
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_gather(TensorHandle hinput, TensorHandle hindex, int n) {
    return tensor_gather_mlx_streamed(hinput, hindex, n, default_stream_tag());
}
