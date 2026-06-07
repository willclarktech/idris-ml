/* tensor_layer_norm_2d for the torch backend.
 *
 * Normalize over the last dim only — n is read from the input shape so
 * callers don't need to thread the size. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_layer_norm_2d(TensorHandle input, TensorHandle gamma,
                                             TensorHandle bias, double eps) {
	auto& t = *to_tensor(input);
	int64_t n = t.size(-1);
	return from_tensor(torch::layer_norm(t, {n}, *to_tensor(gamma), *to_tensor(bias), eps));
}
