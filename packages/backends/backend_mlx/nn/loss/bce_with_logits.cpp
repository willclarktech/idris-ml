/* tensor_bce_with_logits for the mlx backend.
 *
 * BCE with logits = mean(max(x,0) - x*y + log(1 + exp(-|x|))).
 * Decomposed into primitive ops so each step records its own tape
 * entry — backward flows automatically through replay-based vjp.
 * Without the decomposition the fused result has no tape entry,
 * `tape_idx` stays -1, and `tensor_backward` returns early — params
 * never receive gradients. */
#include "../../tensor.h"
#include "../../stream.h"

extern "C" TensorHandle tensor_clamp_min_mlx_streamed(TensorHandle h, double v, int stream_tag);
extern "C" TensorHandle tensor_mul_mlx_streamed(TensorHandle ha, TensorHandle hb, int stream_tag);
extern "C" TensorHandle tensor_abs_mlx_streamed(TensorHandle h, int stream_tag);
extern "C" TensorHandle tensor_neg_mlx_streamed(TensorHandle h, int stream_tag);
extern "C" TensorHandle tensor_exp_mlx_streamed(TensorHandle h, int stream_tag);
extern "C" TensorHandle tensor_add_scalar_mlx_streamed(TensorHandle h, double s, int stream_tag);
extern "C" TensorHandle tensor_log_mlx_streamed(TensorHandle h, int stream_tag);
extern "C" TensorHandle tensor_sub_mlx_streamed(TensorHandle ha, TensorHandle hb, int stream_tag);
extern "C" TensorHandle tensor_add_mlx_streamed(TensorHandle ha, TensorHandle hb, int stream_tag);
extern "C" TensorHandle tensor_mean_mlx_streamed(TensorHandle h, int stream_tag);

extern "C" TensorHandle tensor_bce_with_logits_mlx_streamed(TensorHandle hinput, TensorHandle htarget, int stream_tag) {
    WITH_STREAM(stream_tag);
    TensorHandle relu_x = tensor_clamp_min_mlx_streamed(hinput, 0.0, stream_tag);
    TensorHandle xy = tensor_mul_mlx_streamed(hinput, htarget, stream_tag);
    TensorHandle abs_x = tensor_abs_mlx_streamed(hinput, stream_tag);
    TensorHandle neg_abs_x = tensor_neg_mlx_streamed(abs_x, stream_tag);
    TensorHandle exp_neg = tensor_exp_mlx_streamed(neg_abs_x, stream_tag);
    TensorHandle one_plus_exp = tensor_add_scalar_mlx_streamed(exp_neg, 1.0, stream_tag);
    TensorHandle log_term = tensor_log_mlx_streamed(one_plus_exp, stream_tag);
    TensorHandle relu_minus_xy = tensor_sub_mlx_streamed(relu_x, xy, stream_tag);
    TensorHandle inner = tensor_add_mlx_streamed(relu_minus_xy, log_term, stream_tag);
    return tensor_mean_mlx_streamed(inner, stream_tag);
}

extern "C" TensorHandle tensor_bce_with_logits(TensorHandle hinput, TensorHandle htarget) {
    return tensor_bce_with_logits_mlx_streamed(hinput, htarget, default_stream_tag());
}
