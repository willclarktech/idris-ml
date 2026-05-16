/* tensor_avg_pool2d for the mlx backend. Strided-slice fold over the
 * 2D kernel window, then divide by kH*kW — same shape as avg_pool1d. */
#include "../tensor.h"
#include "../tape.h"
#include "../stream.h"

extern "C" TensorHandle tensor_avg_pool2d_mlx_streamed(TensorHandle hinput, int kH, int kW, int strideH, int strideW, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto inp = (Tensor*)hinput;
    auto dt = inp->data.dtype();
    int C = (int)inp->data.shape(0), H = (int)inp->data.shape(1), W = (int)inp->data.shape(2);
    int oH = (H - kH) / strideH + 1;
    int oW = (W - kW) / strideW + 1;
    mx::array result = mx::zeros({C, oH, oW}, dt);
    for (int kh = 0; kh < kH; kh++)
        for (int kw = 0; kw < kW; kw++) {
            auto sliced = mx::slice(inp->data,
                {0, kh, kw}, {C, kh + oH * strideH, kw + oW * strideW}, {1, strideH, strideW});
            result = mx::add(result, sliced);
        }
    result = mx::divide(result, mx::array((double)(kH * kW), dt));
    auto r = new Tensor(result, inp->requires_grad);
    if (inp->requires_grad) {
        int idx = tape_append(OP_AVG_POOL2D, r, inp, nullptr, 0);
        auto* meta = new AvgPool2DReplayMeta();
        meta->C = C; meta->H = H; meta->W = W;
        meta->kH = kH; meta->kW = kW;
        meta->strH = strideH; meta->strW = strideW;
        meta->oH = oH; meta->oW = oW;
        tape[idx].meta = meta;
    }
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_avg_pool2d(TensorHandle hinput, int kH, int kW, int strideH, int strideW) {
    return tensor_avg_pool2d_mlx_streamed(hinput, kH, kW, strideH, strideW, default_stream_tag());
}
