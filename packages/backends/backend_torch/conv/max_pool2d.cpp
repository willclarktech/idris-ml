/* tensor_max_pool2d (per-sample) and tensor_max_pool2d_batched
 * (batch-native) for the torch backend. */
#include "../tensor.h"

extern "C" TensorHandle tensor_max_pool2d(TensorHandle hinput, int kH, int kW, int strideH,
                                          int strideW) {
	auto& inp = *to_tensor(hinput); /* [C, H, W] */
	auto inp_4d = inp.unsqueeze(0);
	auto out = torch::max_pool2d(inp_4d, {kH, kW}, {strideH, strideW});
	return from_tensor(out.squeeze(0));
}

extern "C" TensorHandle tensor_max_pool2d_batched(TensorHandle hinput, int kH, int kW, int strideH,
                                                  int strideW) {
	auto& inp = *to_tensor(hinput); /* [B, C, H, W] */
	auto out = torch::max_pool2d(inp, {kH, kW}, {strideH, strideW});
	return from_tensor(out);
}
