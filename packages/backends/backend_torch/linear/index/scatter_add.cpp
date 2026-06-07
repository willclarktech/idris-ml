/* tensor_scatter_add for the torch backend.
 *
 * Match src's dtype + device — a hardcoded kFloat64 accumulator trips
 * "scatter(): Expected self.dtype to be equal to src.dtype" against an
 * F32 src (e.g. DNC allocation weighting on an F32 build). */
#include "../../tensor.h"

extern "C" TensorHandle tensor_scatter_add(TensorHandle hindex, TensorHandle hsrc, int out_size) {
	auto& idx = *to_tensor(hindex);
	auto& src = *to_tensor(hsrc);
	auto out = torch::zeros({(int64_t)out_size}, src.options());
	auto idx_long = idx.to(torch::kLong);
	out.scatter_add_(0, idx_long, src);
	return from_tensor(out);
}
