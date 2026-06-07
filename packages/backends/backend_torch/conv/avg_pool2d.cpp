/* tensor_avg_pool2d for the torch backend. */
#include "../tensor.h"

extern "C" TensorHandle tensor_avg_pool2d(TensorHandle hinput, int kH, int kW, int strideH,
                                          int strideW) {
	auto& inp = *to_tensor(hinput);
	auto inp_4d = inp.unsqueeze(0);
	auto out = torch::avg_pool2d(inp_4d, {kH, kW}, {strideH, strideW});
	return from_tensor(out.squeeze(0));
}
