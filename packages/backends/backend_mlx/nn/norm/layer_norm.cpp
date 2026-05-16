/* tensor_layer_norm_2d for the mlx backend.
 *
 * Computes mean / var over the last dim with keepdim=True, then
 * normalizes with rstd + affine. The replay meta carries gamma/bias
 * pool indices so backward can recover them. */
#include "../../tensor.h"
#include "../../tape.h"
#include "../../stream.h"
#include "../../precision.h"

extern "C" TensorHandle tensor_layer_norm_2d_mlx_streamed(TensorHandle h, TensorHandle hgamma,
                                                          TensorHandle hbias, double eps, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto t = (Tensor*)h;
    auto gamma = (Tensor*)hgamma;
    auto bias = (Tensor*)hbias;

    auto mean = mx::mean(t->data, -1, true);
    auto centered = mx::subtract(t->data, mean);
    auto var = mx::mean(mx::square(centered), -1, true);
    auto rstd = mx::rsqrt(mx::add(var, scalar_like(eps, var)));
    auto x_hat = mx::multiply(centered, rstd);
    auto result = mx::add(mx::multiply(gamma->data, x_hat), bias->data);

    bool rg = t->requires_grad || gamma->requires_grad || bias->requires_grad;
    auto r = new Tensor(result, rg);
    if (rg) {
        int idx = tape_append(OP_LAYER_NORM_2D, r, t, nullptr, eps);
        auto meta = new LayerNormReplayMeta();
        meta->gamma_pool_idx = gamma->pool_idx;
        meta->bias_pool_idx = bias->pool_idx;
        meta->eps = eps;
        tape[idx].meta = meta;
    }
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_layer_norm_2d(TensorHandle h, TensorHandle hgamma,
                                             TensorHandle hbias, double eps) {
    return tensor_layer_norm_2d_mlx_streamed(h, hgamma, hbias, eps, default_stream_tag());
}
