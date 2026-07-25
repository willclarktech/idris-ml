/* tensor_argsort for the torch backend.
 *
 * argsort produces indices — keep them in their natural integer dtype
 * (kLong/I64), not a float. This matches the type-safe `targsort` Idris
 * surface (Tensor _ d I64) and sidesteps two latent bugs the old
 * `.to(kFloat64)` had: precision loss above 2^53 indices, and an MPS
 * abort (Metal has no F64). gather/scatter_add coerce the index to
 * kLong internally, so the untyped DNC path is unaffected. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_argsort(TensorHandle ht, int dim, int descending) {
	auto& t = *to_tensor(ht);
	// stable=true: ties break by ascending index, matching the tape
	// comparators (the DNC allocation weighting depends on the tie order).
	auto result = torch::argsort(t, /*stable=*/true, dim, (bool)descending).to(torch::kLong);
	return from_tensor(result);
}
