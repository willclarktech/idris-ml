/* tensor_conv2d (per-sample) and tensor_conv2d_batched (batch-native)
 * for the mlx backend.
 *
 * mlx::conv2d expects NHWC layout (input [N, H, W, C_in], weight
 * [C_out, kH, kW, C_in]); the Idris-side ABI uses torch's NCHW so we
 * transpose on both ends. */
#include "../tensor.h"
#include "../tape.h"
#include "../stream.h"

extern "C" TensorHandle tensor_conv2d_mlx_streamed(TensorHandle hinput, TensorHandle hkernel,
                                                   TensorHandle hbias, int padH, int padW,
                                                   int strideH, int strideW, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto inp = (Tensor*)hinput;
    auto ker = (Tensor*)hkernel;
    Tensor* bias = hbias ? (Tensor*)hbias : nullptr;

    int inC = (int)inp->data.shape(0), H = (int)inp->data.shape(1), W = (int)inp->data.shape(2);

    auto inp_hwc  = mx::transpose(inp->data, {1, 2, 0});
    auto inp_nhwc = mx::reshape(inp_hwc, {1, H, W, inC});
    auto ker_mlx  = mx::transpose(ker->data, {0, 2, 3, 1});

    auto out = mx::conv2d(inp_nhwc, ker_mlx, {strideH, strideW}, {padH, padW});
    auto out_sq = mx::squeeze(out, 0);
    auto result = mx::transpose(out_sq, {2, 0, 1});
    if (bias) result = mx::add(result, mx::reshape(bias->data, {-1, 1, 1}));

    bool rg = inp->requires_grad || ker->requires_grad || (bias && bias->requires_grad);
    auto r = new Tensor(result, rg);
    if (rg) {
        int idx = tape_append(OP_CONV2D, r, inp, ker, 0);
        auto* meta = new Conv2DReplayMeta();
        meta->padH = padH; meta->padW = padW;
        meta->strH = strideH; meta->strW = strideW;
        meta->inC = inC; meta->H = H; meta->W = W;
        meta->bias_pool_idx = bias ? bias->pool_idx : -1;
        tape[idx].meta = meta;
    }
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_conv2d(TensorHandle hinput, TensorHandle hkernel,
                                      TensorHandle hbias, int padH, int padW,
                                      int strideH, int strideW) {
    return tensor_conv2d_mlx_streamed(hinput, hkernel, hbias, padH, padW, strideH, strideW, default_stream_tag());
}

extern "C" TensorHandle tensor_conv2d_batched_mlx_streamed(TensorHandle hinput, TensorHandle hkernel,
                                                           TensorHandle hbias, int padH, int padW,
                                                           int strideH, int strideW, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto inp = (Tensor*)hinput;
    auto ker = (Tensor*)hkernel;
    Tensor* bias = hbias ? (Tensor*)hbias : nullptr;

    int B = (int)inp->data.shape(0), inC = (int)inp->data.shape(1);
    int H = (int)inp->data.shape(2), W = (int)inp->data.shape(3);

    auto inp_nhwc = mx::transpose(inp->data, {0, 2, 3, 1});
    auto ker_mlx  = mx::transpose(ker->data, {0, 2, 3, 1});

    auto out = mx::conv2d(inp_nhwc, ker_mlx, {strideH, strideW}, {padH, padW});
    auto result = mx::transpose(out, {0, 3, 1, 2});
    if (bias) result = mx::add(result, mx::reshape(bias->data, {1, -1, 1, 1}));

    bool rg = inp->requires_grad || ker->requires_grad || (bias && bias->requires_grad);
    auto r = new Tensor(result, rg);
    if (rg) {
        int idx = tape_append(OP_CONV2D_BATCHED, r, inp, ker, 0);
        auto* meta = new Conv2DBatchedReplayMeta();
        meta->padH = padH; meta->padW = padW;
        meta->strH = strideH; meta->strW = strideW;
        meta->B = B; meta->inC = inC; meta->H = H; meta->W = W;
        meta->bias_pool_idx = bias ? bias->pool_idx : -1;
        tape[idx].meta = meta;
    }
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_conv2d_batched(TensorHandle hinput, TensorHandle hkernel,
                                              TensorHandle hbias, int padH, int padW,
                                              int strideH, int strideW) {
    return tensor_conv2d_batched_mlx_streamed(hinput, hkernel, hbias, padH, padW, strideH, strideW, default_stream_tag());
}
