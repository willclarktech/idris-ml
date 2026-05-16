/* tensor_cross_attention for the mlx backend.
 *
 * Composed from existing ops — mlx replay autograd handles backward.
 * Thread stream_tag through each inner call so the type-level device
 * stays in effect; the unsuffixed sub-op trampolines would each open
 * their own WITH_STREAM(default_stream_tag()) and clobber our scope. */
#include "../../tensor.h"
#include "../../stream.h"

extern "C" TensorHandle tensor_transpose_last2_mlx_streamed(TensorHandle h, int stream_tag);
extern "C" TensorHandle tensor_bmm_3x3_mlx_streamed(TensorHandle ha, TensorHandle hb, int stream_tag);
extern "C" TensorHandle tensor_mul_scalar_mlx_streamed(TensorHandle h, double s, int stream_tag);
extern "C" TensorHandle tensor_masked_fill_mlx_streamed(TensorHandle h, TensorHandle hmask, double value, int stream_tag);
extern "C" TensorHandle tensor_softmax_3d_mlx_streamed(TensorHandle h, int stream_tag);

extern "C" TensorHandle tensor_cross_attention_mlx_streamed(TensorHandle hQ, TensorHandle hK, TensorHandle hV,
                                                            TensorHandle hmask, double scale, int stream_tag) {
    WITH_STREAM(stream_tag);
    TensorHandle KT = tensor_transpose_last2_mlx_streamed(hK, stream_tag);
    TensorHandle scores = tensor_mul_scalar_mlx_streamed(
        tensor_bmm_3x3_mlx_streamed(hQ, KT, stream_tag), scale, stream_tag);
    if (hmask) scores = tensor_masked_fill_mlx_streamed(scores, hmask, -1.0e20, stream_tag);
    TensorHandle attn = tensor_softmax_3d_mlx_streamed(scores, stream_tag);
    return tensor_bmm_3x3_mlx_streamed(attn, hV, stream_tag);
}

extern "C" TensorHandle tensor_cross_attention(TensorHandle hQ, TensorHandle hK, TensorHandle hV,
                                               TensorHandle hmask, double scale) {
    return tensor_cross_attention_mlx_streamed(hQ, hK, hV, hmask, scale, default_stream_tag());
}
