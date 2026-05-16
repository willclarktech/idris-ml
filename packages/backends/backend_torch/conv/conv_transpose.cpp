/* tensor_conv_transpose1d / tensor_conv_transpose2d for the torch
 * backend. Mirror the conv1d / conv2d unsqueeze + squeeze pattern. */
#include "../tensor.h"

extern "C" TensorHandle tensor_conv_transpose1d(TensorHandle hinput, TensorHandle hkernel,
                                                TensorHandle hbias, int pad, int stride) {
    auto& inp = *to_tensor(hinput);
    auto& ker = *to_tensor(hkernel);
    auto inp_3d = inp.unsqueeze(0);
    at::Tensor bias_t;
    if (hbias) bias_t = *to_tensor(hbias);
    auto out = hbias
        ? torch::conv_transpose1d(inp_3d, ker, bias_t, {stride}, {pad})
        : torch::conv_transpose1d(inp_3d, ker, {},     {stride}, {pad});
    return from_tensor(out.squeeze(0));
}

extern "C" TensorHandle tensor_conv_transpose2d(TensorHandle hinput, TensorHandle hkernel,
                                                TensorHandle hbias, int padH, int padW,
                                                int strideH, int strideW) {
    auto& inp = *to_tensor(hinput);
    auto& ker = *to_tensor(hkernel);
    auto inp_4d = inp.unsqueeze(0);
    at::Tensor bias_t;
    if (hbias) bias_t = *to_tensor(hbias);
    auto out = hbias
        ? torch::conv_transpose2d(inp_4d, ker, bias_t, {strideH, strideW}, {padH, padW})
        : torch::conv_transpose2d(inp_4d, ker, {},     {strideH, strideW}, {padH, padW});
    return from_tensor(out.squeeze(0));
}
