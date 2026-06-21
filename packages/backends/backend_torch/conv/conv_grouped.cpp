/* tensor_conv1d_grouped / tensor_conv2d_grouped for the torch backend.
 * The grouped variant is just `groups != 1` to torch::conv{1,2}d. */
#include "../tensor.h"

extern "C" TensorHandle tensor_conv1d_grouped(TensorHandle hinput, TensorHandle hkernel,
                                              TensorHandle hbias, int pad, int stride, int groups) {
	auto& inp = *to_tensor(hinput);
	auto& ker = *to_tensor(hkernel);
	auto inp_3d = inp.unsqueeze(0);
	at::Tensor bias_t;
	if (hbias != nullptr) bias_t = *to_tensor(hbias);
	auto out = hbias != nullptr ? torch::conv1d(inp_3d, ker, bias_t, {stride}, {pad}, {1}, groups)
	                            : torch::conv1d(inp_3d, ker, {}, {stride}, {pad}, {1}, groups);
	return from_tensor(out.squeeze(0));
}

extern "C" TensorHandle tensor_conv2d_grouped(TensorHandle hinput, TensorHandle hkernel,
                                              TensorHandle hbias, int padH, int padW, int strideH,
                                              int strideW, int groups) {
	auto& inp = *to_tensor(hinput);
	auto& ker = *to_tensor(hkernel);
	auto inp_4d = inp.unsqueeze(0);
	at::Tensor bias_t;
	if (hbias != nullptr) bias_t = *to_tensor(hbias);
	auto out =
	    hbias != nullptr
	        ? torch::conv2d(inp_4d, ker, bias_t, {strideH, strideW}, {padH, padW}, {1, 1}, groups)
	        : torch::conv2d(inp_4d, ker, {}, {strideH, strideW}, {padH, padW}, {1, 1}, groups);
	return from_tensor(out.squeeze(0));
}
