/* tensor_round for the mlx backend. Element-wise round-to-nearest-
 * even (`mx::round`). Inference-only; no tape entry / no replay. */
#include "../../tensor.h"
#include "../../stream.h"

extern "C" TensorHandle tensor_round_mlx_streamed(TensorHandle h, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto t = (Tensor*)h;
    auto r = new Tensor(mx::round(t->data), /*requires_grad=*/false);
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_round(TensorHandle h) {
    return tensor_round_mlx_streamed(h, default_stream_tag());
}
