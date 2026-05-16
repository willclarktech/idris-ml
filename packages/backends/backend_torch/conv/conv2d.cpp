/* tensor_conv2d (per-sample) and tensor_conv2d_batched (batch-native)
 * for the torch backend. torch::conv2d already expects [N, C, H, W];
 * the per-sample API just adds/removes the batch dim. */
#include "../tensor.h"

extern "C" TensorHandle tensor_conv2d(TensorHandle hinput, TensorHandle hkernel,
                                      TensorHandle hbias, int padH, int padW,
                                      int strideH, int strideW) {
    auto& inp = *to_tensor(hinput);   /* [inC, H, W] */
    auto& ker = *to_tensor(hkernel);  /* [outC, inC, kH, kW] */

    auto inp_4d = inp.unsqueeze(0);
    at::Tensor bias_t;
    if (hbias) bias_t = *to_tensor(hbias);

    auto out = hbias
        ? torch::conv2d(inp_4d, ker, bias_t, {strideH, strideW}, {padH, padW})
        : torch::conv2d(inp_4d, ker, {},     {strideH, strideW}, {padH, padW});
    return from_tensor(out.squeeze(0));
}

extern "C" TensorHandle tensor_conv2d_batched(TensorHandle hinput, TensorHandle hkernel,
                                              TensorHandle hbias, int padH, int padW,
                                              int strideH, int strideW) {
    auto& inp = *to_tensor(hinput);   /* [B, inC, H, W] */
    auto& ker = *to_tensor(hkernel);  /* [outC, inC, kH, kW] */
    at::Tensor bias_t;
    if (hbias) bias_t = *to_tensor(hbias);
    auto out = hbias
        ? torch::conv2d(inp, ker, bias_t, {strideH, strideW}, {padH, padW})
        : torch::conv2d(inp, ker, {},     {strideH, strideW}, {padH, padW});
    return from_tensor(out);
}
