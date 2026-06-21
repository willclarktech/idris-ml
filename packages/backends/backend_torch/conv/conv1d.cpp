/* tensor_conv1d for the torch backend. Per-sample API: unsqueezes the
 * batch dim before calling torch::conv1d (which is batch-native), then
 * squeezes it back off the result. */
#include "../tensor.h"

extern "C" TensorHandle tensor_conv1d(TensorHandle hinput, TensorHandle hkernel, TensorHandle hbias,
                                      int pad, int stride) {
	auto& inp = *to_tensor(hinput);
	auto& ker = *to_tensor(hkernel);
	auto inp_3d = inp.unsqueeze(0);
	at::Tensor bias_t;
	if (hbias != nullptr) bias_t = *to_tensor(hbias);
	auto out = hbias != nullptr ? torch::conv1d(inp_3d, ker, bias_t, {stride}, {pad})
	                            : torch::conv1d(inp_3d, ker, {}, {stride}, {pad});
	return from_tensor(out.squeeze(0));
}
