/* tensor_gather_rows for the torch backend. Indices are coerced to
 * kLong (torch::gather requires int64). Autograd handles backward
 * (scatter to the selected cells) natively. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_gather_rows(TensorHandle hinput, TensorHandle hindex, int b, int n) {
	(void)b;
	(void)n;
	auto& inp = *to_tensor(hinput);
	auto& idx = *to_tensor(hindex);
	auto idx_long = idx.to(torch::kLong).unsqueeze(1);
	return from_tensor(torch::gather(inp, 1, idx_long).squeeze(1));
}
