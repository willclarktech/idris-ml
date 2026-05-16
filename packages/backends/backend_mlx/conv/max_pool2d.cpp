/* tensor_max_pool2d for the mlx backend (per-sample variant). */
#include "../tensor.h"
#include "../tape.h"
#include "../stream.h"

extern "C" TensorHandle tensor_max_pool2d_mlx_streamed(TensorHandle hinput, int kH, int kW,
                                                       int strideH, int strideW, int stream_tag) {
    WITH_STREAM(stream_tag);
    auto inp = (Tensor*)hinput;
    int C = (int)inp->data.shape(0), H = (int)inp->data.shape(1), W = (int)inp->data.shape(2);
    int oH = (H - kH) / strideH + 1;
    int oW = (W - kW) / strideW + 1;

    mx::array result = mx::full({C, oH, oW}, -1e30, inp->data.dtype());
    for (int kh = 0; kh < kH; kh++) {
        for (int kw = 0; kw < kW; kw++) {
            auto sliced = mx::slice(inp->data,
                {0, kh, kw}, {C, kh + oH * strideH, kw + oW * strideW}, {1, strideH, strideW});
            result = mx::maximum(result, sliced);
        }
    }

    auto r = new Tensor(result, inp->requires_grad);
    if (inp->requires_grad) {
        int idx = tape_append(OP_MAX_POOL2D, r, inp, nullptr, 0);
        auto* meta = new MaxPool2DReplayMeta();
        meta->C = C; meta->H = H; meta->W = W;
        meta->kH = kH; meta->kW = kW;
        meta->strH = strideH; meta->strW = strideW;
        meta->oH = oH; meta->oW = oW;
        tape[idx].meta = meta;
    }
    return (TensorHandle)r;
}

extern "C" TensorHandle tensor_max_pool2d(TensorHandle hinput, int kH, int kW,
                                          int strideH, int strideW) {
    return tensor_max_pool2d_mlx_streamed(hinput, kH, kW, strideH, strideW, default_stream_tag());
}
