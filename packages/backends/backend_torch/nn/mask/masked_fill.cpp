/* tensor_masked_fill for the torch backend. The mask is coerced to
 * kBool — the Idris-side mask tensor may carry an arithmetic dtype
 * (e.g. F32 0/1), but masked_fill_ requires a boolean mask. */
#include "../../tensor.h"

extern "C" TensorHandle tensor_masked_fill(TensorHandle h, TensorHandle mask, double value) {
	return from_tensor(to_tensor(h)->masked_fill(to_tensor(mask)->to(torch::kBool), value));
}
