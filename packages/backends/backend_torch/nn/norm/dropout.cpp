/* tensor_dropout for the torch backend.
 *
 * `seed` is ignored — torch uses its own thread-local RNG; the
 * Idris-side dropout layer carries the seed for tape-backend
 * reproducibility, which doesn't apply here. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_dropout(TensorHandle hinput, double p, int training,
                                       unsigned int seed) {
	(void)seed;
	auto& inp = *to_tensor(hinput);
	if (!training || p <= 0.0) return hinput;
	return from_tensor(torch::dropout(inp, p, /*train=*/true));
}
